#!/usr/bin/env python3
"""Quick training experiment script for feudal loss integration."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hrm_dir = root / "HRM"
sys.path.insert(0, str(hrm_dir))
sys.path.insert(0, str(root))

import os  # noqa: E402

# Set environment variables for quick CPU test
os.environ["DISABLE_COMPILE"] = "1"  # Disable torch.compile for CPU

from pretrain import launch  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def run_quick_experiment():
    """Run a quick training experiment with feudal loss."""
    print("=" * 60)
    print("Feudal Loss Training Experiment")
    print("=" * 60)
    print()

    # Create a minimal config for quick testing
    config = OmegaConf.create(
        {
            "arch": {
                "name": "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
                "loss": {
                    "name": "losses@ACTLossHead",
                    "loss_type": "softmax_cross_entropy",
                    "feudal_loss_weight": 0.1,
                },
                "halt_exploration_prob": 0.1,
                "halt_max_steps": 4,  # Small for quick test
                "H_cycles": 1,
                "L_cycles": 1,
                "H_layers": 1,
                "L_layers": 1,
                "hidden_size": 32,
                "num_heads": 2,
                "expansion": 2,
                "puzzle_emb_ndim": 32,
                "pos_encodings": "rope",
                "subgoal_head": {
                    "hidden_size": 32,
                    "goal_dim": 32,
                    "manager_period": 2,
                    "temperature": 1.0,
                    "projection_bias": True,
                    "normalize_goal": True,
                    "goal_scale": 1.0,
                    "gating": True,
                    "detach_goals": True,
                },
            },
            "data_path": "data/arc-aug-1000",
            "device": "cpu",  # Use CPU for quick test
            "max_steps": 10,  # Very small for quick test
            "enable_wandb": False,  # Disable wandb for quick test
            "seed": 42,
            "global_batch_size": 4,
            "epochs": 1,
            "lr": 1e-4,
            "lr_min_ratio": 1.0,
            "lr_warmup_steps": 0,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "puzzle_emb_weight_decay": 0.1,
            "puzzle_emb_lr": 1e-2,
            "eval_interval": None,
            "checkpoint_every_eval": False,
            "final_eval": False,
        }
    )

    # Initialize Hydra
    GlobalHydra.instance().clear()

    # Convert to DictConfig for Hydra compatibility
    hydra_config = DictConfig(config)

    print("Config:")
    print(OmegaConf.to_yaml(config))
    print()
    print("Starting training experiment...")
    print()

    try:
        # Run training
        launch(hydra_config)
        print()
        print("=" * 60)
        print("✅ Training experiment completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Training experiment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_quick_experiment()
