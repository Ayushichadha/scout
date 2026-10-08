#!/usr/bin/env python3
"""Local validation-only evaluator for Scout/HRM checkpoints.

This script does not train, does not save predictions, and does not modify
checkpoints. It evaluates explicit checkpoint files or known target run
directories and writes one CSV summary row per target.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
HRM_DIR = REPO_ROOT / "HRM"
sys.path.insert(0, str(HRM_DIR))

from puzzle_dataset import (  # noqa: E402
    PuzzleDataset,
    PuzzleDatasetConfig,
    PuzzleDatasetMetadata,
)
from utils.functions import load_model_class  # noqa: E402
from models.losses import accumulate_episode_metrics  # noqa: E402


TARGET_RUN_DIRS = {
    "baseline": HRM_DIR / "checkpoints/hrm_feudal_sweep/feudal_w0.0_p4",
    "p1_w0.05": HRM_DIR / "checkpoints/hrm_feudal_sweep/feudal_w0.05_p1",
    "p3_w0.05": HRM_DIR / "checkpoints/hrm_feudal_sweep/feudal_w0.05_p3",
    "p4_w0.05": HRM_DIR / "checkpoints/hrm_feudal_sweep/feudal_w0.05_p4",
    "p8_w0.05": HRM_DIR / "checkpoints/hrm_feudal_sweep/feudal_w0.05_p8",
}

CSV_FIELDS = [
    "config_name",
    "checkpoint_path",
    "device",
    "num_eval_batches",
    "val_lm_loss",
    "val_token_accuracy",
    "val_exact_accuracy",
    "avg_act_steps",
    "mean_total_interventions",
    "mean_adaptive_interventions",
    "mean_soft_trigger_probability",
    "hard_adaptive_decision_rate",
    "mean_completed_dwell",
    "mean_old_new_goal_cosine",
    "intervention_position_histogram",
    "dwell_histogram",
    "conditional_feature_means",
    "trigger_parameters",
    "notes",
]


@dataclass
class EvalTarget:
    name: str
    checkpoint: Path | None
    run_dir: Path


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def extract_top_level(text: str, key: str, default: Any = None) -> Any:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return parse_scalar(match.group(1)) if match else default


def extract_arch_scalar(text: str, key: str, default: Any = None) -> Any:
    match = re.search(rf"^  {re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return parse_scalar(match.group(1)) if match else default


def extract_loss_scalar(text: str, key: str, default: Any = None) -> Any:
    match = re.search(rf"^    {re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return parse_scalar(match.group(1)) if match else default


def extract_omegaconf_val(text: str, key: str, default: Any = None) -> Any:
    lines = text.splitlines()
    start_re = re.compile(
        rf"^\s{{6}}{re.escape(key)}:\s+!!python/object:omegaconf\.nodes\.AnyNode"
    )
    for idx, line in enumerate(lines):
        if not start_re.match(line):
            continue
        for inner in lines[idx + 1 : idx + 40]:
            if re.match(r"^\s{6}[A-Za-z0-9_]+:", inner):
                break
            match = re.match(r"^\s*_val:\s*(.+)$", inner)
            if match:
                return parse_scalar(match.group(1))
    return default


def load_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text()

    # Checkpoints save the resolved Pydantic config with PyYAML. Preserve the
    # complete architecture/loss mappings so new scientific controls cannot be
    # silently dropped by an incomplete hand-maintained field list.
    try:
        raw = yaml.load(text, Loader=yaml.UnsafeLoader)
    except (yaml.YAMLError, ValueError, TypeError):
        raw = None
    if isinstance(raw, dict) and isinstance(raw.get("arch"), (dict, DictConfig)):
        raw_arch = raw["arch"]
        if isinstance(raw_arch, DictConfig):
            arch = OmegaConf.to_container(raw_arch, resolve=False)
        else:
            arch = {
                key: (
                    OmegaConf.to_container(value, resolve=False)
                    if isinstance(value, DictConfig)
                    else value
                )
                for key, value in raw_arch.items()
            }
        if not isinstance(arch, dict):
            raise TypeError("saved checkpoint arch must reconstruct to a mapping")
        return {
            "arch": arch,
            "data_path": raw.get("data_path", "data/conceptarc-mini"),
            "global_batch_size": int(raw.get("global_batch_size", 8)),
            "run_name": raw.get("run_name", config_path.parent.name),
            "seed": int(raw.get("seed", 0)),
        }

    # Legacy text-format fallback for older experiment summaries.
    loss_cfg = {
        "name": extract_loss_scalar(text, "name", "losses@ACTLossHead"),
        "loss_type": extract_loss_scalar(text, "loss_type", "stablemax_cross_entropy"),
        "feudal_loss_weight": extract_loss_scalar(text, "feudal_loss_weight", 0.0),
        "intervention_weight": extract_loss_scalar(text, "intervention_weight", 0.0),
    }

    hidden_size = extract_arch_scalar(text, "hidden_size", 512)
    subgoal_cfg = {
        "hidden_size": extract_omegaconf_val(text, "hidden_size", hidden_size),
        "goal_dim": extract_omegaconf_val(text, "goal_dim", hidden_size),
        "manager_period": extract_omegaconf_val(text, "manager_period", 4),
        "temperature": extract_omegaconf_val(text, "temperature", 1.0),
        "projection_bias": extract_omegaconf_val(text, "projection_bias", True),
        "normalize_goal": extract_omegaconf_val(text, "normalize_goal", True),
        "goal_scale": extract_omegaconf_val(text, "goal_scale", 1.0),
        "gating": extract_omegaconf_val(text, "gating", True),
        "detach_goals": extract_omegaconf_val(text, "detach_goals", True),
        "inject_subgoal": extract_omegaconf_val(text, "inject_subgoal", True),
        "use_alignment_loss": extract_omegaconf_val(text, "use_alignment_loss", True),
        "random_directions": extract_omegaconf_val(text, "random_directions", False),
        "directional_displacement": extract_omegaconf_val(
            text, "directional_displacement", False
        ),
        "initial_goal_only": extract_omegaconf_val(text, "initial_goal_only", False),
        "replan_mode": extract_omegaconf_val(text, "replan_mode", "fixed"),
        "trigger_threshold": extract_omegaconf_val(text, "trigger_threshold", 0.5),
        "trigger_stochastic_train": extract_omegaconf_val(
            text, "trigger_stochastic_train", True
        ),
    }

    arch = {
        "name": extract_arch_scalar(
            text, "name", "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1"
        ),
        "loss": loss_cfg,
        "halt_exploration_prob": extract_arch_scalar(
            text, "halt_exploration_prob", 0.1
        ),
        "halt_max_steps": extract_arch_scalar(text, "halt_max_steps", 16),
        "fixed_refinement_steps": extract_arch_scalar(
            text, "fixed_refinement_steps", None
        ),
        "H_cycles": extract_arch_scalar(text, "H_cycles", 2),
        "L_cycles": extract_arch_scalar(text, "L_cycles", 2),
        "H_layers": extract_arch_scalar(text, "H_layers", 4),
        "L_layers": extract_arch_scalar(text, "L_layers", 4),
        "hidden_size": hidden_size,
        "num_heads": extract_arch_scalar(text, "num_heads", 8),
        "expansion": extract_arch_scalar(text, "expansion", 4),
        "puzzle_emb_ndim": extract_arch_scalar(text, "puzzle_emb_ndim", hidden_size),
        "pos_encodings": extract_arch_scalar(text, "pos_encodings", "rope"),
        "subgoal_head": subgoal_cfg,
    }

    return {
        "arch": arch,
        "data_path": extract_top_level(text, "data_path", "data/conceptarc-mini"),
        "global_batch_size": int(extract_top_level(text, "global_batch_size", 8)),
        "run_name": extract_top_level(text, "run_name", config_path.parent.name),
        "seed": int(extract_top_level(text, "seed", 0)),
    }


def selected_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def print_device_info() -> None:
    device = selected_device("auto")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
    print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    print(f"selected_device: {device}")


def latest_step_file(run_dir: Path) -> Path | None:
    candidates = [
        p for p in run_dir.glob("step_*") if p.is_file() and p.name[5:].isdigit()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.removeprefix("step_")))


def discover_targets(names: list[str]) -> list[EvalTarget]:
    targets = []
    for name in names:
        run_dir = TARGET_RUN_DIRS[name]
        targets.append(
            EvalTarget(name=name, checkpoint=latest_step_file(run_dir), run_dir=run_dir)
        )
    return targets


def explicit_targets(paths: list[str]) -> list[EvalTarget]:
    targets = []
    for raw_path in paths:
        checkpoint = Path(raw_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (REPO_ROOT / checkpoint).resolve()
        targets.append(
            EvalTarget(
                name=checkpoint.parent.name,
                checkpoint=checkpoint,
                run_dir=checkpoint.parent,
            )
        )
    return targets


def dataset_path_from_config(config: dict[str, Any]) -> Path:
    data_path = Path(str(config["data_path"]))
    if data_path.is_absolute():
        return data_path
    hrm_relative = HRM_DIR / data_path
    if hrm_relative.exists():
        return hrm_relative
    return REPO_ROOT / data_path


def build_model(
    config: dict[str, Any],
    metadata: PuzzleDatasetMetadata,
    batch_size: int,
    device: str,
):
    arch = dict(config["arch"])
    loss_cfg = dict(arch.pop("loss"))
    arch_name = arch.pop("name")
    loss_name = loss_cfg.pop("name")

    subgoal_cfg = arch.get("subgoal_head")
    if isinstance(subgoal_cfg, DictConfig):
        subgoal_cfg = OmegaConf.to_container(subgoal_cfg, resolve=False)
    if isinstance(subgoal_cfg, dict):
        subgoal_cfg = dict(subgoal_cfg)
        hidden_size = arch.get("hidden_size")
        for key in ("hidden_size", "goal_dim"):
            value = subgoal_cfg.get(key)
            if isinstance(value, str) and value.startswith("${"):
                subgoal_cfg[key] = hidden_size
        arch["subgoal_head"] = subgoal_cfg

    if isinstance(arch.get("puzzle_emb_ndim"), str):
        arch["puzzle_emb_ndim"] = arch["hidden_size"]

    model_cfg = {
        **arch,
        "batch_size": batch_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "causal": False,
    }
    model_cls = load_model_class(arch_name)
    loss_cls = load_model_class(loss_name)
    model = loss_cls(model_cls(model_cfg), **loss_cfg)
    return model.to(device)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def load_weights(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    state = torch.load(checkpoint, map_location=device)
    if all(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def evaluate_checkpoint(
    target: EvalTarget,
    requested_device: str,
    batch_size_override: int | None,
    max_batches: int | None,
) -> dict[str, Any]:
    if target.checkpoint is None:
        return {
            "config_name": target.name,
            "checkpoint_path": "",
            "device": selected_device(requested_device),
            "num_eval_batches": 0,
            "val_lm_loss": "",
            "val_token_accuracy": "",
            "val_exact_accuracy": "",
            "avg_act_steps": "",
            "notes": f"no step_* checkpoint file found in {target.run_dir}",
        }
    if not target.checkpoint.exists():
        return {
            "config_name": target.name,
            "checkpoint_path": str(target.checkpoint),
            "device": selected_device(requested_device),
            "num_eval_batches": 0,
            "val_lm_loss": "",
            "val_token_accuracy": "",
            "val_exact_accuracy": "",
            "avg_act_steps": "",
            "notes": "checkpoint file does not exist",
        }

    config_path = target.run_dir / "all_config.yaml"
    if not config_path.exists():
        return {
            "config_name": target.name,
            "checkpoint_path": str(target.checkpoint),
            "device": selected_device(requested_device),
            "num_eval_batches": 0,
            "val_lm_loss": "",
            "val_token_accuracy": "",
            "val_exact_accuracy": "",
            "avg_act_steps": "",
            "notes": f"missing config {config_path}",
        }

    device = selected_device(requested_device)
    try:
        return _evaluate_checkpoint_on_device(
            target, config_path, device, batch_size_override, max_batches
        )
    except RuntimeError as exc:
        if device == "mps":
            print(f"MPS evaluation failed for {target.name}: {exc}")
            print("Retrying this checkpoint on CPU.")
            row = _evaluate_checkpoint_on_device(
                target, config_path, "cpu", batch_size_override, max_batches
            )
            row["notes"] = f"MPS failed; retried on CPU: {str(exc).splitlines()[0]}"
            return row
        raise


def _evaluate_checkpoint_on_device(
    target: EvalTarget,
    config_path: Path,
    device: str,
    batch_size_override: int | None,
    max_batches: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_path = dataset_path_from_config(config)
    metadata_path = dataset_path / "test/dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing validation metadata: {metadata_path}")

    metadata = PuzzleDatasetMetadata(**json.loads(metadata_path.read_text()))
    batch_size = batch_size_override or int(config["global_batch_size"])
    model = build_model(config, metadata, batch_size, device)
    load_weights(model, target.checkpoint, device)  # type: ignore[arg-type]
    model.eval()

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=int(config["seed"]),
            dataset_path=str(dataset_path),
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    set_ids = {name: idx for idx, name in enumerate(metadata.sets)}
    metric_keys: list[str] | None = None
    metric_values: torch.Tensor | None = None
    num_batches = 0

    with torch.inference_mode():
        for set_name, batch, _global_batch_size in dataloader:
            if max_batches is not None and num_batches >= max_batches:
                break
            batch = move_batch(batch, device)
            carry = model.initial_carry(batch)
            episode_metrics = None
            while True:
                carry, _loss, metrics, _preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=[]
                )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
                if all_finish:
                    break
            metrics = episode_metrics

            if metric_keys is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)), dtype=torch.float32, device=device
                )

            metric_values[set_ids[set_name]] += torch.stack(
                [metrics[k] for k in metric_keys]
            )
            num_batches += 1

    if metric_values is None or metric_keys is None:
        raise RuntimeError("No validation batches were processed.")

    totals = {key: 0.0 for key in metric_keys}
    arr = metric_values.cpu()
    for set_id in range(len(set_ids)):
        for metric_id, key in enumerate(metric_keys):
            totals[key] += float(arr[set_id, metric_id])

    count = totals.pop("count", 0.0)
    if count <= 0:
        raise RuntimeError("Validation produced zero valid examples.")

    values = {key: value / count for key, value in totals.items()}
    completed = totals.get("completed_episodes", count)
    eligible = totals.get("trigger_probability_count", 0.0)
    dwell_count = totals.get("completed_dwell_count", 0.0)
    replacements = totals.get("old_new_goal_cosine_count", 0.0)

    def ratio(key: str, denominator: float):
        return totals.get(key, 0.0) / denominator if denominator > 0 else ""

    position_histogram = {
        str(position): totals.get(f"intervention_position_{position}", 0.0)
        for position in range(1, 8)
    }
    dwell_histogram = {
        str(dwell): totals.get(f"completed_dwell_length_{dwell}", 0.0)
        for dwell in range(1, 8)
    }
    conditional_features = {}
    for condition in ("intervene", "retain"):
        denominator = totals.get(f"trigger_feature_{condition}_count", 0.0)
        conditional_features[condition] = {
            feature: ratio(f"trigger_feature_{feature}_{condition}_sum", denominator)
            for feature in ("c", "d", "rho", "dwell", "q", "gate")
        }
    trigger_parameters = {
        feature: totals.get(f"trigger_weight_{feature}", "")
        for feature in ("c", "d", "rho", "dwell", "q", "gate")
    }
    trigger_parameters["bias"] = totals.get("trigger_bias", "")
    return {
        "config_name": target.name,
        "checkpoint_path": str(target.checkpoint),
        "device": device,
        "num_eval_batches": num_batches,
        "val_lm_loss": values.get("lm_loss", ""),
        "val_token_accuracy": values.get("accuracy", ""),
        "val_exact_accuracy": values.get("exact_accuracy", ""),
        "avg_act_steps": values.get("steps", ""),
        "mean_total_interventions": ratio("manager_interventions", completed),
        "mean_adaptive_interventions": ratio("adaptive_interventions", completed),
        "mean_soft_trigger_probability": ratio("trigger_probability_sum", eligible),
        "hard_adaptive_decision_rate": ratio("adaptive_interventions", eligible),
        "mean_completed_dwell": ratio("completed_dwell_sum", dwell_count),
        "mean_old_new_goal_cosine": ratio("old_new_goal_cosine_sum", replacements),
        "intervention_position_histogram": json.dumps(position_histogram),
        "dwell_histogram": json.dumps(dwell_histogram),
        "conditional_feature_means": json.dumps(conditional_features),
        "trigger_parameters": json.dumps(trigger_parameters),
        "notes": "",
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="Print Python/Torch/MPS info and exit.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint file to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_RUN_DIRS),
        help="Known target configs to evaluate by latest step_* in their run dirs.",
    )
    parser.add_argument(
        "--all-targets", action="store_true", help="Evaluate all known target configs."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Validation batch size for local Mac eval.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit validation batches for smoke tests.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs/local_eval_results.csv"),
        help="CSV output path.",
    )
    args = parser.parse_args()

    if args.device_info:
        print_device_info()
        return

    targets: list[EvalTarget] = []
    if args.checkpoint:
        targets.extend(explicit_targets(args.checkpoint))
    if args.all_targets:
        targets.extend(discover_targets(list(TARGET_RUN_DIRS)))
    if args.targets:
        targets.extend(discover_targets(args.targets))
    if not targets:
        parser.error("Pass --checkpoint, --targets, --all-targets, or --device-info.")

    rows = [
        evaluate_checkpoint(
            target,
            requested_device=args.device,
            batch_size_override=args.batch_size,
            max_batches=args.max_batches,
        )
        for target in targets
    ]
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_csv(rows, output)
    for row in rows:
        print(row)
    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    main()
