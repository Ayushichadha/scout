#!/usr/bin/env python3
"""Hyperparameter sweep script for feudal loss experiments."""

import subprocess
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

root = Path(__file__).resolve().parents[1]
hrm_dir = root / "HRM"


def run_experiment(
    feudal_loss_weight: float,
    manager_period: int,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    run_name: str = None,
    **kwargs,
):
    """Run a single training experiment with given hyperparameters."""
    if run_name is None:
        run_name = f"feudal_w{feudal_loss_weight}_p{manager_period}"

    cmd = [
        sys.executable,
        str(hrm_dir / "pretrain.py"),
        f"device={device}",
        f"max_steps={max_steps}",
        f"enable_wandb={enable_wandb}",
        f"loss.feudal_loss_weight={feudal_loss_weight}",
        f"arch.subgoal_head.manager_period={manager_period}",
        f"run_name={run_name}",
    ]

    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    print(f"\n{'='*60}")
    print(f"Running experiment: {run_name}")
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


def sweep_feudal_loss_weight(
    weights: list,
    manager_period: int = 4,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Sweep over feudal loss weights."""
    results = []
    for weight in weights:
        run_name = f"feudal_w{weight}_p{manager_period}"
        success = run_experiment(
            feudal_loss_weight=weight,
            manager_period=manager_period,
            max_steps=max_steps,
            device=device,
            enable_wandb=enable_wandb,
            run_name=run_name,
            **kwargs,
        )
        results.append(
            {
                "feudal_loss_weight": weight,
                "manager_period": manager_period,
                "run_name": run_name,
                "success": success,
            }
        )
    return results


def sweep_manager_period(
    periods: list,
    feudal_loss_weight: float = 0.1,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Sweep over manager periods."""
    results = []
    for period in periods:
        run_name = f"feudal_w{feudal_loss_weight}_p{period}"
        success = run_experiment(
            feudal_loss_weight=feudal_loss_weight,
            manager_period=period,
            max_steps=max_steps,
            device=device,
            enable_wandb=enable_wandb,
            run_name=run_name,
            **kwargs,
        )
        results.append(
            {
                "feudal_loss_weight": feudal_loss_weight,
                "manager_period": period,
                "run_name": run_name,
                "success": success,
            }
        )
    return results


def grid_search(
    feudal_loss_weights: list,
    manager_periods: list,
    max_steps: int = 100,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Grid search over feudal loss weights and manager periods."""
    results = []
    for weight in feudal_loss_weights:
        for period in manager_periods:
            run_name = f"feudal_w{weight}_p{period}"
            success = run_experiment(
                feudal_loss_weight=weight,
                manager_period=period,
                max_steps=max_steps,
                device=device,
                enable_wandb=enable_wandb,
                run_name=run_name,
                **kwargs,
            )
            results.append(
                {
                    "feudal_loss_weight": weight,
                    "manager_period": period,
                    "run_name": run_name,
                    "success": success,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for feudal loss")
    parser.add_argument(
        "--sweep-type",
        type=str,
        choices=["feudal_weight", "manager_period", "grid"],
        default="feudal_weight",
        help="Type of sweep to run",
    )
    parser.add_argument(
        "--feudal-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        help="Feudal loss weights to sweep",
    )
    parser.add_argument(
        "--manager-periods",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Manager periods to sweep",
    )
    parser.add_argument(
        "--manager-period",
        type=int,
        default=4,
        help="Manager period (for feudal_weight sweep)",
    )
    parser.add_argument(
        "--feudal-weight",
        type=float,
        default=0.1,
        help="Feudal loss weight (for manager_period sweep)",
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

    # Run sweep
    if args.sweep_type == "feudal_weight":
        results = sweep_feudal_loss_weight(
            weights=args.feudal_weights,
            manager_period=args.manager_period,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=args.enable_wandb,
        )
    elif args.sweep_type == "manager_period":
        results = sweep_manager_period(
            periods=args.manager_periods,
            feudal_loss_weight=args.feudal_weight,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=args.enable_wandb,
        )
    elif args.sweep_type == "grid":
        results = grid_search(
            feudal_loss_weights=args.feudal_weights,
            manager_periods=args.manager_periods,
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
        output_path = root / "experiments" / f"sweep_{args.sweep_type}_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Sweep Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print("\nResults:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(
            f"  {status} {r['run_name']}: w={r['feudal_loss_weight']}, p={r['manager_period']}"
        )


if __name__ == "__main__":
    main()
