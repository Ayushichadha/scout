#!/usr/bin/env python3
"""Baseline comparison script: HRM with vs without feudal loss."""

import subprocess
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

root = Path(__file__).resolve().parents[1]
hrm_dir = root / "HRM"


def run_baseline_experiment(
    use_feudal: bool,
    feudal_loss_weight: float = 0.1,
    manager_period: int = 4,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    run_name: str = None,
    **kwargs,
):
    """Run baseline experiment (with or without feudal loss)."""
    if run_name is None:
        run_name = "baseline_feudal" if use_feudal else "baseline_no_feudal"

    cmd = [
        sys.executable,
        str(hrm_dir / "pretrain.py"),
        f"device={device}",
        f"max_steps={max_steps}",
        f"enable_wandb={enable_wandb}",
        f"run_name={run_name}",
    ]

    if use_feudal:
        cmd.extend(
            [
                f"loss.feudal_loss_weight={feudal_loss_weight}",
                f"arch.subgoal_head.manager_period={manager_period}",
            ]
        )
    else:
        # Disable feudal loss
        cmd.append("loss.feudal_loss_weight=0")

    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    print(f"\n{'='*60}")
    print(f"Running experiment: {run_name}")
    print(f"  Feudal loss: {'Enabled' if use_feudal else 'Disabled'}")
    if use_feudal:
        print(f"  Feudal loss weight: {feudal_loss_weight}")
        print(f"  Manager period: {manager_period}")
    print(f"  Max steps: {max_steps}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(hrm_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Experiment {run_name} failed!")
        print(result.stderr)
        return False

    print(f"✅ Experiment {run_name} completed!")
    return True


def run_comparison(
    feudal_loss_weight: float = 0.1,
    manager_period: int = 4,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Run both baseline and feudal experiments for comparison."""
    results = []

    # Run baseline (no feudal loss)
    print("\n" + "=" * 60)
    print("Running BASELINE (no feudal loss)")
    print("=" * 60)
    success_baseline = run_baseline_experiment(
        use_feudal=False,
        max_steps=max_steps,
        device=device,
        enable_wandb=enable_wandb,
        run_name="baseline_no_feudal",
        **kwargs,
    )
    results.append(
        {
            "experiment": "baseline_no_feudal",
            "use_feudal": False,
            "feudal_loss_weight": 0.0,
            "manager_period": None,
            "success": success_baseline,
        }
    )

    # Run with feudal loss
    print("\n" + "=" * 60)
    print("Running WITH FEUDAL LOSS")
    print("=" * 60)
    success_feudal = run_baseline_experiment(
        use_feudal=True,
        feudal_loss_weight=feudal_loss_weight,
        manager_period=manager_period,
        max_steps=max_steps,
        device=device,
        enable_wandb=enable_wandb,
        run_name="baseline_feudal",
        **kwargs,
    )
    results.append(
        {
            "experiment": "baseline_feudal",
            "use_feudal": True,
            "feudal_loss_weight": feudal_loss_weight,
            "manager_period": manager_period,
            "success": success_feudal,
        }
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline HRM with and without feudal loss"
    )
    parser.add_argument(
        "--feudal-weight",
        type=float,
        default=0.1,
        help="Feudal loss weight (when enabled)",
    )
    parser.add_argument(
        "--manager-period",
        type=int,
        default=4,
        help="Manager period (when feudal enabled)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum training steps"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device"
    )
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable WandB logging"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Run comparison
    results = run_comparison(
        feudal_loss_weight=args.feudal_weight,
        manager_period=args.manager_period,
        max_steps=args.max_steps,
        device=args.device,
        enable_wandb=args.enable_wandb,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
    else:
        # Default output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = root / "experiments" / f"baseline_comparison_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        feudal_status = "Enabled" if r["use_feudal"] else "Disabled"
        print(f"  {status} {r['experiment']}: Feudal loss {feudal_status}")
        if r["use_feudal"]:
            print(
                f"      Weight: {r['feudal_loss_weight']}, Period: {r['manager_period']}"
            )


if __name__ == "__main__":
    main()
