#!/usr/bin/env python3
"""Hyperparameter sweep script for feudal loss experiments."""

import subprocess
import sys
from pathlib import Path
import argparse
import json
import re
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
        "+project_name=hrm_feudal_sweep",
        f"+run_name={run_name}",
        "data_path=data/conceptarc-mini",
        # Set explicit values to avoid interpolation issues
        "arch.hidden_size=32",
        "arch.subgoal_head.hidden_size=32",
        "arch.subgoal_head.goal_dim=32",
        # Optimize for speed
        "arch.halt_max_steps=4",
        "arch.H_cycles=1",
        "arch.L_cycles=1",
        "arch.H_layers=1",
        "arch.L_layers=1",
        "arch.num_heads=2",
        "arch.expansion=2",
        "arch.puzzle_emb_ndim=32",
        "global_batch_size=4",
        "epochs=1",
        "lr_warmup_steps=0",
        "eval_interval=null",
        "+final_eval=false",
        f"arch.loss.feudal_loss_weight={feudal_loss_weight}",
        f"arch.subgoal_head.manager_period={manager_period}",
    ]

    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    print(f"\n{'='*60}")
    print(f"Running experiment: {run_name}")
    print(f"  Feudal loss weight: {feudal_loss_weight}")
    print(f"  Manager period: {manager_period}")
    print(f"  Max steps: {max_steps}")
    print(f"  WandB: {'Enabled' if enable_wandb else 'Disabled'}")
    print(f"{'='*60}\n")

    # Capture both stdout and stderr
    result = subprocess.run(cmd, cwd=str(hrm_dir), capture_output=True, text=True)

    # Save training logs
    log_dir = root / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"❌ Experiment {run_name} failed!")
        print(result.stderr[:500])
        return {
            "success": False,
            "error": result.stderr[:500],
            "log_file": str(log_file),
        }

    # Parse final loss values from output
    metrics = parse_training_metrics(result.stdout, result.stderr)

    print(f"✅ Experiment {run_name} completed!")
    if metrics:
        print("   Final metrics:")
        for key, value in metrics.items():
            print(f"     {key}: {value}")

    return {"success": True, "metrics": metrics, "log_file": str(log_file)}


def parse_training_metrics(stdout: str, stderr: str) -> dict:
    """Parse final training metrics from output."""
    metrics = {}
    text = stdout + "\n" + stderr

    # Try to find final loss values from progress bar or logs
    loss_patterns = [
        r"loss=([\d.]+)",  # Progress bar format
        r"train/lm_loss[:\s]+([\d.]+)",  # Log format
    ]

    for pattern in loss_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                metrics["final_lm_loss"] = float(matches[-1])
                break
            except ValueError:
                continue

    # Try to find feudal loss
    feudal_patterns = [
        r"train/feudal_loss[:\s]+([\d.]+)",
        r"feudal_loss[:\s]+([\d.]+)",
    ]
    for pattern in feudal_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                metrics["final_feudal_loss"] = float(matches[-1])
                break
            except ValueError:
                continue

    # Try to find accuracy
    acc_patterns = [
        r"train/accuracy[:\s]+([\d.]+)",
        r"accuracy[:\s]+([\d.]+)",
    ]
    for pattern in acc_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                metrics["final_accuracy"] = float(matches[-1])
                break
            except ValueError:
                continue

    return metrics


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
        result_dict = {
            "feudal_loss_weight": weight,
            "manager_period": manager_period,
            "run_name": run_name,
            "success": (
                success.get("success", False) if isinstance(success, dict) else success
            ),
        }
        if isinstance(success, dict):
            result_dict["metrics"] = success.get("metrics", {})
            result_dict["log_file"] = success.get("log_file")
        results.append(result_dict)
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
        result_dict = {
            "feudal_loss_weight": feudal_loss_weight,
            "manager_period": period,
            "run_name": run_name,
            "success": (
                success.get("success", False) if isinstance(success, dict) else success
            ),
        }
        if isinstance(success, dict):
            result_dict["metrics"] = success.get("metrics", {})
            result_dict["log_file"] = success.get("log_file")
        results.append(result_dict)
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
            result_dict = {
                "feudal_loss_weight": weight,
                "manager_period": period,
                "run_name": run_name,
                "success": (
                    success.get("success", False)
                    if isinstance(success, dict)
                    else success
                ),
            }
            if isinstance(success, dict):
                result_dict["metrics"] = success.get("metrics", {})
                result_dict["log_file"] = success.get("log_file")
            results.append(result_dict)
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
        "--enable-wandb",
        action="store_true",
        help="Enable WandB logging (default: True)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # WandB is enabled by default unless --no-wandb is passed
    enable_wandb = args.enable_wandb or not args.no_wandb

    # Print sweep details
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER SWEEP: {args.sweep_type.upper()}")
    print("=" * 60)
    if args.sweep_type == "feudal_weight":
        print(f"  Sweeping feudal loss weights: {args.feudal_weights}")
        print(f"  Fixed manager period: {args.manager_period}")
    elif args.sweep_type == "manager_period":
        print(f"  Fixed feudal loss weight: {args.feudal_weight}")
        print(f"  Sweeping manager periods: {args.manager_periods}")
    elif args.sweep_type == "grid":
        print("  Grid search:")
        print(f"    Feudal weights: {args.feudal_weights}")
        print(f"    Manager periods: {args.manager_periods}")
    print("\nCommon settings:")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Device: {args.device}")
    print(f"  - WandB: {'Enabled' if enable_wandb else 'Disabled'}")
    print("  - Model: Small (hidden_size=32, 1 layer each)")
    print("=" * 60 + "\n")

    # Run sweep
    if args.sweep_type == "feudal_weight":
        results = sweep_feudal_loss_weight(
            weights=args.feudal_weights,
            manager_period=args.manager_period,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=enable_wandb,
        )
    elif args.sweep_type == "manager_period":
        results = sweep_manager_period(
            periods=args.manager_periods,
            feudal_loss_weight=args.feudal_weight,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=enable_wandb,
        )
    elif args.sweep_type == "grid":
        results = grid_search(
            feudal_loss_weights=args.feudal_weights,
            manager_periods=args.manager_periods,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=enable_wandb,
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
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print("\nResults:")
    for r in results:
        status = "✅" if r.get("success", False) else "❌"
        metrics_str = ""
        if r.get("metrics"):
            m = r["metrics"]
            if "final_lm_loss" in m:
                metrics_str = f" | LM Loss: {m['final_lm_loss']:.4f}"
            if "final_accuracy" in m:
                metrics_str += f" | Accuracy: {m['final_accuracy']:.4f}"
        print(
            f"  {status} {r['run_name']}: w={r['feudal_loss_weight']}, p={r['manager_period']}{metrics_str}"
        )

    # Find best hyperparameters
    if successful > 0:
        print(f"\n{'='*60}")
        print("Best Hyperparameters")
        print(f"{'='*60}")
        # Find best by LM loss (lower is better)
        valid_results = [
            r
            for r in results
            if r.get("success") and r.get("metrics", {}).get("final_lm_loss")
        ]
        if valid_results:
            best = min(valid_results, key=lambda x: x["metrics"]["final_lm_loss"])
            print(f"  Best LM Loss: {best['run_name']}")
            print(f"    Feudal weight: {best['feudal_loss_weight']}")
            print(f"    Manager period: {best['manager_period']}")
            print(f"    Final LM Loss: {best['metrics']['final_lm_loss']:.4f}")
            if "final_accuracy" in best["metrics"]:
                print(f"    Final Accuracy: {best['metrics']['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
