from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import sys
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

# Prefer adam_atan2 optimizer if available; otherwise fall back to torch Adam
try:
    from adam_atan2 import AdamATan2  # type: ignore
except Exception:  # ImportError or backend load errors
    from torch.optim import Adam as AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    loss: LossConfig


class QuickArchConfig(pydantic.BaseModel):
    hidden_size: int = 32
    n_layers: int = 1
    n_heads: int = 2


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Device and runtime controls
    device: str = "cuda"  # "cuda" or "cpu"
    max_steps: Optional[int] = None  # Optional clamp for quick smoke tests
    enable_wandb: bool = True  # Disable to avoid W&B in smoke tests
    overfit_one_batch: bool = (
        False  # Repeat the first train batch for quick sanity check
    )
    # Ultra-light CPU quick sanity toggle + helpers
    quick_overfit: bool = False
    quick_max_steps: int = 15
    quick_batch_size: int = 16
    quick_arch: QuickArchConfig = QuickArchConfig()

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    # Whether to run an evaluation pass at the end
    final_eval: bool = True


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(
    config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs
):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    # CPU quick-mode should not use workers/pinned memory
    use_cpu = config.device != "cuda"
    num_workers = 0 if use_cpu else 1
    pin_memory = False if use_cpu else True
    persistent_workers = num_workers > 0

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # Only pass prefetch_factor when we have workers
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 8

    dataloader = DataLoader(**dl_kwargs)
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int
):
    # Resolve OmegaConf interpolations before passing to Pydantic
    arch_extra_raw = config.arch.__pydantic_extra__  # type: ignore

    # Convert to dict if it's a DictConfig
    if isinstance(arch_extra_raw, DictConfig):
        arch_extra = OmegaConf.to_container(arch_extra_raw, resolve=False)
    else:
        arch_extra = (
            arch_extra_raw.copy()
            if isinstance(arch_extra_raw, dict)
            else dict(arch_extra_raw)
        )

    # Manually resolve interpolations in subgoal_head config
    if "subgoal_head" in arch_extra:
        subgoal_cfg_raw = arch_extra["subgoal_head"]
        # Convert to dict if it's a DictConfig
        if isinstance(subgoal_cfg_raw, DictConfig):
            subgoal_cfg = OmegaConf.to_container(subgoal_cfg_raw, resolve=False)
        else:
            subgoal_cfg = (
                subgoal_cfg_raw.copy()
                if isinstance(subgoal_cfg_raw, dict)
                else dict(subgoal_cfg_raw)
            )

        # Resolve ${.hidden_size} -> arch_extra["hidden_size"]
        if "hidden_size" in arch_extra:
            hidden_size = arch_extra["hidden_size"]
            if isinstance(subgoal_cfg.get("hidden_size"), str) and subgoal_cfg.get(
                "hidden_size", ""
            ).startswith("${"):
                subgoal_cfg["hidden_size"] = hidden_size
            if isinstance(subgoal_cfg.get("goal_dim"), str) and subgoal_cfg.get(
                "goal_dim", ""
            ).startswith("${"):
                subgoal_cfg["goal_dim"] = hidden_size
            if isinstance(subgoal_cfg.get("puzzle_emb_ndim"), str) and subgoal_cfg.get(
                "puzzle_emb_ndim", ""
            ).startswith("${"):
                subgoal_cfg["puzzle_emb_ndim"] = hidden_size
        arch_extra["subgoal_head"] = subgoal_cfg

    # Resolve puzzle_emb_ndim interpolation
    if isinstance(arch_extra.get("puzzle_emb_ndim"), str) and arch_extra.get(
        "puzzle_emb_ndim", ""
    ).startswith("${"):
        if "hidden_size" in arch_extra:
            arch_extra["puzzle_emb_ndim"] = arch_extra["hidden_size"]

    model_cfg = dict(
        **arch_extra,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    # Create model on requested device
    model: nn.Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    # Move model to device
    if config.device == "cuda":
        model = model.cuda()
        # Compile only on CUDA for speed; CPU compile can be very slow
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size,
        ),
        AdamATan2(
            model.parameters(),
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        ),
    ]
    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def validate_training_config(
    config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int = 0
):
    """Validate training configuration and print warnings/errors."""
    errors = []
    warnings = []

    # Check data path exists
    if not os.path.exists(config.data_path):
        errors.append(f"Data path does not exist: {config.data_path}")

    # Check batch size
    if config.global_batch_size <= 0:
        errors.append(
            f"global_batch_size must be positive, got {config.global_batch_size}"
        )

    # Check learning rates
    if config.lr <= 0:
        errors.append(f"lr must be positive, got {config.lr}")
    if config.puzzle_emb_lr <= 0:
        errors.append(f"puzzle_emb_lr must be positive, got {config.puzzle_emb_lr}")

    # Check epochs
    if config.epochs <= 0:
        errors.append(f"epochs must be positive, got {config.epochs}")

    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA requested but not available, falling back to CPU")
        config.device = "cpu"

    # Check subgoal head configuration if present
    arch_extra = config.arch.__pydantic_extra__  # type: ignore
    if "subgoal_head" in arch_extra:
        subgoal_cfg = arch_extra["subgoal_head"]
        hidden_size = arch_extra.get("hidden_size", 512)
        goal_dim = subgoal_cfg.get("goal_dim", hidden_size)
        if goal_dim > hidden_size:
            warnings.append(
                f"subgoal_head.goal_dim ({goal_dim}) > hidden_size ({hidden_size}), this may cause issues"
            )

    # Check feudal loss weight
    loss_extra = config.arch.loss.__pydantic_extra__  # type: ignore
    feudal_loss_weight = loss_extra.get("feudal_loss_weight", 0.0)
    if feudal_loss_weight > 0 and "subgoal_head" not in arch_extra:
        warnings.append(
            "feudal_loss_weight > 0 but no subgoal_head configured, feudal loss will be 0"
        )

    # Print warnings and errors (rank 0 only)
    if rank == 0:
        for warning in warnings:
            print(f"⚠️  WARNING: {warning}")
        for error in errors:
            print(f"❌ ERROR: {error}")

    if errors:
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s)")

    return config


def init_train_state(
    config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int
):
    # Estimated total training steps
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )
    if config.max_steps is not None:
        total_steps = min(total_steps, int(config.max_steps))

    # Model
    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    try:
        # To device
        if config.device == "cuda":
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to("cpu") for k, v in batch.items()}

        # Init carry if it is None
        if train_state.carry is None:
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

        # Forward
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )

        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            if rank == 0:
                print(
                    f"⚠️  WARNING: Non-finite loss detected at step {train_state.step}: {loss.item()}"
                )
            # Skip this batch
            return None

        ((1 / global_batch_size) * loss).backward()
    except Exception as e:
        if rank == 0:
            print(f"❌ ERROR: Training batch failed at step {train_state.step}: {e}")
            import traceback

            traceback.print_exc()
        raise

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(
            sorted(metrics.keys())
        )  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            if config.device == "cuda":
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to("cpu") for k, v in batch.items()}
            carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=config.eval_save_outputs
                )

                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(
                            v.cpu()
                        )  # Move to CPU for saving GPU memory

            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_device = "cuda" if config.device == "cuda" else "cpu"
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device=metric_device,
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(
                all_preds,
                os.path.join(
                    config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"
                ),
            )

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {
                        k: v / count for k, v in metrics.items()
                    }

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    # Log code (if enabled)
    try:
        if config.enable_wandb:
            import wandb as _wandb  # Lazy import

            if _wandb.run is not None:
                _wandb.run.log_code(config.checkpoint_path)
    except Exception:
        pass


def load_synced_config(
    hydra_config: DictConfig, rank: int, world_size: int
) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = (
                f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
            )
        if config.run_name is None:
            config.run_name = (
                f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
            )
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        backend = "nccl" if hydra_config.get("device", "cuda") == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        if hydra_config.get("device", "cuda") == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Support CLI alias --quick without exposing it to Hydra
    if os.environ.get("HRM_PRETRAIN_QUICK", "0") == "1":
        config.quick_overfit = True

    # Validate configuration (before quick mode modifications)
    # We'll validate again after quick mode if needed

    # Print resolved Hydra config once and list key groups (rank 0 only)
    if RANK == 0:
        try:
            print("==== Resolved Hydra Config (once) ====")
            print(OmegaConf.to_yaml(hydra_config))
            # Key groups
            arch_keys = []
            try:
                arch_keys = list(hydra_config.arch.keys())  # type: ignore[attr-defined]
            except Exception:
                pass
            optimizer_keys = [
                k
                for k in (
                    "lr",
                    "lr_min_ratio",
                    "lr_warmup_steps",
                    "beta1",
                    "beta2",
                    "weight_decay",
                    "puzzle_emb_lr",
                    "puzzle_emb_weight_decay",
                )
                if k in hydra_config
            ]
            eval_keys = [
                k
                for k in (
                    "eval_interval",
                    "checkpoint_every_eval",
                    "final_eval",
                    "eval_save_outputs",
                )
                if k in hydra_config
            ]
            regularizer_keys = [
                k
                for k in ("weight_decay", "puzzle_emb_weight_decay")
                if k in hydra_config
            ]
            seed_key = "seed" if "seed" in hydra_config else None
            print("Keys — arch:", arch_keys)
            print("Keys — optimizer:", optimizer_keys)
            print("Keys — eval:", eval_keys)
            print("Keys — regularizers:", regularizer_keys)
            if seed_key is not None:
                print("Keys — seed:", [seed_key])
        except Exception:
            pass

    # Quick ultra-light sanity mode: force minimal CPU settings
    if getattr(config, "quick_overfit", False):
        # Cap CPU threads and hard-disable wandb
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ["WANDB_DISABLED"] = "true"
        try:
            torch.set_num_threads(int(os.getenv("PY_NUM_THREADS", "1")))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Force CPU-only, disable W&B, and overfit a single batch
        config.device = "cpu"
        config.enable_wandb = False
        config.overfit_one_batch = True

        # Fill from quick defaults if unset
        if config.max_steps is None:
            config.max_steps = int(config.quick_max_steps)
        # global_batch_size: set from quick default only if not overridden
        if os.environ.get("HRM_PRETRAIN_QUICK_SET_GBS", "0") == "1":
            config.global_batch_size = int(config.quick_batch_size)

        # Small, safe learning rates for monotonic loss decrease
        config.lr = min(float(config.lr), 1e-3)
        config.puzzle_emb_lr = min(float(config.puzzle_emb_lr), 1e-2)
        config.lr_warmup_steps = 0
        config.lr_min_ratio = 0.0

        # Set tiny arch from quick_arch unless already overridden
        arch = config.arch.__pydantic_extra__  # type: ignore
        if "hidden_size" not in arch:
            arch["hidden_size"] = int(config.quick_arch.hidden_size)
        if "num_heads" not in arch:
            arch["num_heads"] = int(config.quick_arch.n_heads)
        arch.setdefault("H_layers", int(config.quick_arch.n_layers))
        arch.setdefault("L_layers", int(config.quick_arch.n_layers))
        arch.setdefault("H_cycles", 1)
        arch.setdefault("L_cycles", 1)
        arch.setdefault("expansion", 2)

        # Print a concise summary line
        if RANK == 0:
            print(
                f"[Quick] steps={config.max_steps} bs={config.global_batch_size} "
                f"arch(h={arch.get('hidden_size')},L={arch.get('H_layers')},H={arch.get('num_heads')})"
            )

    # If CPU device, shrink model for quick smoke tests unless explicitly overridden
    if config.device == "cpu":
        # Use much smaller defaults for a quick CPU run
        arch = config.arch.__pydantic_extra__  # type: ignore
        arch.setdefault("hidden_size", 64)
        arch.setdefault("num_heads", 2)
        arch.setdefault("expansion", 2)
        arch.setdefault("H_layers", 1)
        arch.setdefault("L_layers", 1)
        arch.setdefault("H_cycles", 1)
        arch.setdefault("L_cycles", 1)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset / eval cadence
    skip_eval = False
    if config.eval_interval is None:
        train_epochs_per_iter = config.epochs
    else:
        try:
            if int(config.eval_interval) == 0:
                # Special case: 0 means no intermediate evals
                skip_eval = True
                train_epochs_per_iter = config.epochs
            else:
                train_epochs_per_iter = int(config.eval_interval)
        except Exception:
            train_epochs_per_iter = config.epochs
    # Allow non-divisible eval intervals by rounding up and cutting off by total_steps
    total_iters = (config.epochs + train_epochs_per_iter - 1) // train_epochs_per_iter

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    # Validate configuration after all modifications
    config = validate_training_config(config, train_metadata, rank=RANK)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if config.enable_wandb:
            try:
                import wandb as _wandb

                _wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=_wandb.Settings(_disable_stats=True))  # type: ignore
                _wandb.log(
                    {
                        "num_params": sum(
                            x.numel() for x in train_state.model.parameters()
                        )
                    },
                    step=0,
                )
            except Exception:
                pass
        save_code_and_config(config)

    # Optional: Overfit one batch sanity mode
    if config.overfit_one_batch:
        if RANK == 0:
            print("[Overfit] Repeating the first train batch for all steps.")
        # Cache the very first batch from the train loader
        first_set_name, first_batch, first_global_batch_size = next(iter(train_loader))

        # Train on the same batch until reaching total_steps
        train_state.model.train()
        while train_state.step < train_state.total_steps:
            metrics = train_batch(
                config,
                train_state,
                first_batch,
                first_global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )
            # Skip logging if batch was skipped (e.g., non-finite loss)
            if metrics is None:
                continue
            if RANK == 0 and metrics is not None:
                if config.enable_wandb:
                    try:
                        import wandb as _wandb

                        _wandb.log(metrics, step=train_state.step)
                    except Exception:
                        pass
                # Show loss inline for instant feedback
                try:
                    if progress_bar is not None and "train/lm_loss" in metrics:
                        progress_bar.set_postfix({"loss": f"{metrics['train/lm_loss']:.4f}"})  # type: ignore
                except Exception:
                    pass
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        # Optional final evaluation
        if config.final_eval:
            train_state.model.eval()
            metrics = evaluate(
                config,
                train_state,
                eval_loader,
                eval_metadata,
                rank=RANK,
                world_size=WORLD_SIZE,
            )
            if RANK == 0 and metrics is not None:
                if config.enable_wandb:
                    try:
                        import wandb as _wandb

                        _wandb.log(metrics, step=train_state.step)
                    except Exception:
                        pass
            if RANK == 0 and config.checkpoint_every_eval:
                save_train_state(config, train_state)

        # finalize
        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            if config.enable_wandb:
                import wandb as _wandb

                _wandb.finish()
        except Exception:
            pass
        return

    # Training Loop
    for _iter_id in range(total_iters):

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            # Skip logging if batch was skipped (e.g., non-finite loss)
            if metrics is None:
                continue

            if RANK == 0 and metrics is not None:
                if config.enable_wandb:
                    try:
                        import wandb as _wandb

                        _wandb.log(metrics, step=train_state.step)
                    except Exception:
                        pass
                # Show loss inline for instant feedback
                try:
                    if progress_bar is not None and "train/lm_loss" in metrics:
                        progress_bar.set_postfix({"loss": f"{metrics['train/lm_loss']:.4f}"})  # type: ignore
                except Exception:
                    pass
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

            # Stop early if reached total steps
            if train_state.step >= train_state.total_steps:
                break

        # If we've reached the target number of steps, stop outer loop as well
        # and skip the final evaluation to exit immediately.
        if train_state.step >= train_state.total_steps:
            break

        ############ Evaluation (optional)
        if not skip_eval and config.final_eval:
            train_state.model.eval()
            metrics = evaluate(
                config,
                train_state,
                eval_loader,
                eval_metadata,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            if RANK == 0 and metrics is not None:
                if config.enable_wandb:
                    try:
                        import wandb as _wandb

                        _wandb.log(metrics, step=train_state.step)
                    except Exception:
                        pass

            ############ Checkpointing
            if RANK == 0 and (
                config.checkpoint_every_eval or (_iter_id == total_iters - 1)
            ):
                save_train_state(config, train_state)

        # If we've reached the target number of steps, stop outer loop
        if train_state.step >= train_state.total_steps:
            break

    # finalize
    if progress_bar is not None:
        try:
            progress_bar.close()
        except Exception:
            pass
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        if config.enable_wandb:
            import wandb as _wandb

            _wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    # Lightweight CLI shim to support flags like:
    #   --max_steps 50 --batch_size 8 --data_root PATH --device cpu --no_wandb
    # Convert them to Hydra overrides, then call launch()
    argv = sys.argv[1:]
    overrides = []
    i = 0
    quick_flag = False
    while i < len(argv):
        arg = argv[i]
        if arg == "--quick":
            # Prefer a supported Hydra override and also set an env flag
            quick_flag = True
            overrides.append("quick_overfit=true")
            i += 1
            continue
        if arg == "--overfit_one":
            overrides.append("overfit_one_batch=true")
            i += 1
            continue
        if arg == "--max_steps" and i + 1 < len(argv):
            overrides.append(f"max_steps={argv[i+1]}")
            i += 2
            continue
        if arg == "--batch_size" and i + 1 < len(argv):
            overrides.append(f"global_batch_size={argv[i+1]}")
            i += 2
            continue
        if arg == "--data_root" and i + 1 < len(argv):
            overrides.append(f"data_path={argv[i+1]}")
            i += 2
            continue
        if arg == "--device" and i + 1 < len(argv):
            overrides.append(f"device={argv[i+1]}")
            i += 2
            continue
        if arg == "--no_wandb":
            overrides.append("enable_wandb=false")
            i += 1
            continue
        # Pass through other Hydra-style overrides as-is
        if "=" in arg:
            overrides.append(arg)
        i += 1

    if quick_flag:
        os.environ["HRM_PRETRAIN_QUICK"] = "1"
        # If user did not override batch size, allow quick to set a tiny one
        has_bs_override = any(s.startswith("global_batch_size=") for s in overrides)
        if not has_bs_override:
            # Pass an explicit override so --quick works even alone
            # and without relying on environment variables.
            overrides.append("global_batch_size=${quick_batch_size}")
            os.environ["HRM_PRETRAIN_QUICK_SET_GBS"] = "1"
    # Rebuild sys.argv for Hydra to strip handled flags like --quick,
    # even when there are no explicit overrides
    sys.argv = [sys.argv[0]] + overrides
    launch()
