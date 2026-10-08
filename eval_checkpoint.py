#!/usr/bin/env python3
"""
Eval a single HRM checkpoint on the test split.

Usage:
    python eval_checkpoint.py \
        --checkpoint "HRM/checkpoints/Conceptarc-mini ACT-torch/A_full/step_30321" \
        --data HRM/data/conceptarc-mini \
        --batch-size 64

Results are printed to stdout and saved as JSON next to the checkpoint file.
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
HRM_DIR = os.path.join(REPO_ROOT, "HRM")
sys.path.insert(0, HRM_DIR)

from omegaconf import DictConfig, OmegaConf  # noqa: E402
from puzzle_dataset import (  # noqa: E402
    PuzzleDataset,
    PuzzleDatasetConfig,
    PuzzleDatasetMetadata,
)
from utils.functions import load_model_class  # noqa: E402
from models.losses import accumulate_episode_metrics  # noqa: E402


def load_arch_from_yaml(yaml_path):
    """
    Load all_config.yaml and return (arch_dict, global_batch_size).

    The yaml was written with yaml.dump(config.model_dump(), f).
    ArchConfig.model_dump() produces a plain dict whose top-level values
    are mostly plain Python scalars, except subgoal_head which is stored
    as an OmegaConf DictConfig object (serialized with !!python/object: tags).

    yaml.UnsafeLoader reconstructs those OmegaConf objects in-process.
    The resulting raw["arch"] is a plain dict with H_cycles/H_layers/etc.
    as plain Python ints/floats and subgoal_head as a DictConfig.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.load(f, Loader=yaml.UnsafeLoader)

    raw_arch = raw["arch"]

    # raw_arch is typically a plain dict (from ArchConfig.model_dump()) where
    # most values are scalars and subgoal_head is a DictConfig.
    # Guard against the case where YAML anchor resolution causes raw_arch itself
    # to resolve to a DictConfig (e.g. if PyYAML sees duplicate 'arch' keys and
    # takes the last one, which could be the OmegaConf parent DictConfig).
    if isinstance(raw_arch, DictConfig):
        arch = dict(OmegaConf.to_container(raw_arch, resolve=False))
    else:
        arch = {}
        for k, v in raw_arch.items():
            if isinstance(v, DictConfig):
                arch[k] = OmegaConf.to_container(v, resolve=False)
            else:
                arch[k] = v

    # global_batch_size is a flat field at the top level of the yaml
    global_batch_size = int(raw.get("global_batch_size", 64))
    return arch, global_batch_size


def build_model(arch, metadata, batch_size, device):
    """
    Construct model + loss head from a plain arch dict.
    Mirrors create_model() in pretrain.py without optimizers or torch.compile.
    """
    arch = dict(arch)

    # Resolve subgoal_head interpolations — same logic as create_model()
    if "subgoal_head" in arch:
        sg = arch["subgoal_head"]
        if isinstance(sg, DictConfig):
            sg = OmegaConf.to_container(sg, resolve=False)
        sg = dict(sg)
        hidden_size = arch.get("hidden_size")
        if hidden_size is not None:
            for key in ("hidden_size", "goal_dim", "puzzle_emb_ndim"):
                val = sg.get(key, "")
                if isinstance(val, str) and val.startswith("${"):
                    sg[key] = hidden_size
        arch["subgoal_head"] = sg

    if isinstance(arch.get("puzzle_emb_ndim", ""), str) and arch.get(
        "puzzle_emb_ndim", ""
    ).startswith("${"):
        arch["puzzle_emb_ndim"] = arch.get("hidden_size")

    # Pull out loss config and arch name before building model_cfg
    loss_raw = arch.pop("loss")
    arch_name = arch.pop("name")

    if isinstance(loss_raw, DictConfig):
        loss_cfg = dict(OmegaConf.to_container(loss_raw, resolve=False))
    else:
        loss_cfg = dict(loss_raw)

    loss_name = loss_cfg.pop("name")
    loss_extra = loss_cfg  # remaining keys are kwargs to ACTLossHead

    model_cfg = dict(
        **arch,
        batch_size=batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(arch_name)
    loss_cls = load_model_class(loss_name)
    model = model_cls(model_cfg)
    model = loss_cls(model, **loss_extra)

    if device == "cuda":
        model = model.cuda()

    return model


def main():
    parser = argparse.ArgumentParser(description="Eval HRM checkpoint on test split")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the weights file, e.g. HRM/checkpoints/.../A_full/step_30321",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset root, e.g. HRM/data/conceptarc-mini",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from checkpoint config's global_batch_size)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    checkpoint_file = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_file)

    # Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load config from alongside the checkpoint
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: all_config.yaml not found in {checkpoint_dir}")

    print(f"Config: {config_path}")
    arch, ckpt_batch_size = load_arch_from_yaml(config_path)

    batch_size = args.batch_size if args.batch_size is not None else ckpt_batch_size
    print(f"Batch size: {batch_size}  (checkpoint config: {ckpt_batch_size})")

    # Verify inject_subgoal/use_alignment_loss/random_directions are in config.
    # These flags change the forward pass, so wrong values give wrong eval numbers.
    sg = arch.get("subgoal_head")
    if sg is not None:
        print(
            f"subgoal_head flags: inject_subgoal={sg.get('inject_subgoal', '(default=True)')}"
            f"  use_alignment_loss={sg.get('use_alignment_loss', '(default=True)')}"
            f"  random_directions={sg.get('random_directions', '(default=False)')}"
        )
        print("  ^ Verify these match the intended ablation condition.")

    # 2. Dataset metadata
    meta_path = os.path.join(args.data, "test", "dataset.json")
    if not os.path.exists(meta_path):
        sys.exit(f"ERROR: test/dataset.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))
    print(
        f"Dataset: vocab_size={metadata.vocab_size}  seq_len={metadata.seq_len}"
        f"  test_groups={metadata.total_groups}  sets={metadata.sets}"
    )

    # 3. Build model (no torch.compile for eval)
    print("Building model...")
    model = build_model(arch, metadata, batch_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # 4. Load checkpoint weights
    if not os.path.exists(checkpoint_file):
        sys.exit(f"ERROR: checkpoint file not found: {checkpoint_file}")

    print(f"Loading: {checkpoint_file}")
    state = torch.load(checkpoint_file, map_location=device)

    # Inspect keys as requested — do not assume raw state_dict
    all_keys = list(state.keys())
    print(f"Checkpoint: {len(all_keys)} keys. First 5: {all_keys[:5]}")

    if all_keys and all_keys[0].startswith("_orig_mod."):
        # torch.compile wraps in OptimizedModule; .state_dict() on it produces
        # _orig_mod.-prefixed keys in some PyTorch 2.x builds.
        print("Stripping '_orig_mod.' prefix (torch.compile artifact)")
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    print("Weights loaded OK.")

    # 5. Eval dataloader — matches pretrain.py test mode exactly
    #    test_set_mode=True uses _iter_test: sequential, no shuffle, pads last batch.
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_path=args.data,
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    dl = DataLoader(dataset, batch_size=None, num_workers=0)

    # 6. Eval loop — mirrors evaluate() in pretrain.py line 493-581 exactly.
    #
    # NOTE on normalization: evaluate() divides ALL metrics (including lm_loss)
    # by `count` (= halted samples with valid labels). train_batch() divides
    # loss metrics by global_batch_size instead. If count < global_batch_size,
    # eval lm_loss will be inflated relative to the train snapshot. At near-full
    # convergence this difference is small but train vs eval lm_loss are not on
    # the same scale. This script replicates evaluate() exactly so A/B/E are
    # comparable to each other.
    set_ids = {k: idx for idx, k in enumerate(metadata.sets)}
    metric_keys = None
    metric_values = None
    n_batches = 0

    print("Running eval...")
    with torch.inference_mode():
        for set_name, batch, _global_bs in dl:
            if device == "cuda":
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to("cpu") for k, v in batch.items()}

            carry = model.initial_carry(batch)

            episode_metrics = None
            while True:
                carry, _, metrics, _preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=[]
                )
                episode_metrics = accumulate_episode_metrics(episode_metrics, metrics)
                if all_finish:
                    break
            metrics = episode_metrics

            set_id = set_ids[set_name]
            if metric_keys is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_device = "cuda" if device == "cuda" else "cpu"
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)),
                    dtype=torch.float32,
                    device=metric_device,
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            n_batches += 1

    print(f"Processed {n_batches} batches.")

    # Normalize: divide all by count, following evaluate() exactly
    results = {}
    if metric_values is not None:
        arr = metric_values.cpu().numpy()
        for set_id, set_name in enumerate(set_ids):
            row = {
                metric_keys[i]: float(arr[set_id, i]) for i in range(len(metric_keys))
            }
            count = row.pop("count")
            if count == 0:
                print(f"WARNING: count=0 for set '{set_name}', skipping")
                continue
            values = {k: v / count for k, v in row.items()}
            completed = row.get("completed_episodes", count)
            eligible = row.get("trigger_probability_count", 0.0)
            dwell_count = row.get("completed_dwell_count", 0.0)
            replacements = row.get("old_new_goal_cosine_count", 0.0)
            if completed > 0:
                values["mean_total_interventions_per_episode"] = (
                    row.get("manager_interventions", 0.0) / completed
                )
                values["mean_adaptive_interventions_per_episode"] = (
                    row.get("adaptive_interventions", 0.0) / completed
                )
            if eligible > 0:
                values["mean_soft_trigger_probability"] = (
                    row.get("trigger_probability_sum", 0.0) / eligible
                )
                values["hard_adaptive_decision_rate"] = (
                    row.get("adaptive_interventions", 0.0) / eligible
                )
            if dwell_count > 0:
                values["mean_completed_dwell"] = (
                    row.get("completed_dwell_sum", 0.0) / dwell_count
                )
            if replacements > 0:
                values["mean_old_new_goal_cosine"] = (
                    row.get("old_new_goal_cosine_sum", 0.0) / replacements
                )
            results[set_name] = values

    # Print summary
    print("\n=== Eval results ===")
    for sname, m in results.items():
        print(f"  [{sname}]")
        for key in ("lm_loss", "accuracy", "exact_accuracy"):
            if key in m:
                print(f"    {key}: {m[key]:.6f}")
        for key in sorted(m):
            if key not in ("lm_loss", "accuracy", "exact_accuracy"):
                print(f"    {key}: {m[key]:.6f}")

    # Save JSON next to the checkpoint file
    out_path = checkpoint_file + "_eval.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_file,
                "data": args.data,
                "batch_size": batch_size,
                "device": device,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
