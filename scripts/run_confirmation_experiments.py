#!/usr/bin/env python3
"""Run confirmation experiments: replication runs and manager period sweep."""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add scripts directory to path for imports
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
from run_hyperparameter_sweep import run_experiment, sweep_manager_period  # noqa: E402

root = Path(__file__).resolve().parents[1]
hrm_dir = root / "HRM"


def run_replication_experiments(
    feudal_loss_weight: float = 0.05,
    manager_period: int = 4,
    num_replications: int = 5,
    max_steps: int = 200,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Run multiple replications of the same experiment to confirm results."""
    print("\n" + "=" * 60)
    print("REPLICATION EXPERIMENTS")
    print("=" * 60)
    print("Configuration to replicate:")
    print(f"  Feudal loss weight: {feudal_loss_weight}")
    print(f"  Manager period: {manager_period}")
    print(f"  Number of replications: {num_replications}")
    print(f"  Max steps: {max_steps}")
    print("=" * 60 + "\n")

    results = []
    for i in range(num_replications):
        run_name = f"replication_{i+1}_w{feudal_loss_weight}_p{manager_period}"
        print(f"\n{'='*60}")
        print(f"Replication {i+1}/{num_replications}")
        print(f"{'='*60}")

        success = run_experiment(
            feudal_loss_weight=feudal_loss_weight,
            manager_period=manager_period,
            max_steps=max_steps,
            device=device,
            enable_wandb=enable_wandb,
            run_name=run_name,
            **kwargs,
        )

        result_dict = {
            "replication": i + 1,
            "feudal_loss_weight": feudal_loss_weight,
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


def run_manager_period_sweep(
    feudal_loss_weight: float = 0.05,
    manager_periods: list = None,
    max_steps: int = 200,
    device: str = "cpu",
    enable_wandb: bool = False,
    **kwargs,
):
    """Run manager period sweep with fixed feudal weight."""
    if manager_periods is None:
        manager_periods = [1, 2, 3, 4, 5, 6, 8]

    print("\n" + "=" * 60)
    print("MANAGER PERIOD SWEEP")
    print("=" * 60)
    print(f"Fixed feudal loss weight: {feudal_loss_weight}")
    print(f"Manager periods to test: {manager_periods}")
    print(f"Max steps: {max_steps}")
    print("=" * 60 + "\n")

    return sweep_manager_period(
        periods=manager_periods,
        feudal_loss_weight=feudal_loss_weight,
        max_steps=max_steps,
        device=device,
        enable_wandb=enable_wandb,
        **kwargs,
    )


def print_summary(all_results: dict):
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Replication results
    if "replications" in all_results:
        reps = all_results["replications"]
        print("\n📊 REPLICATION EXPERIMENTS")
        print("-" * 80)
        successful = [r for r in reps if r.get("success", False)]
        print(f"Successful: {len(successful)}/{len(reps)}")

        if successful:
            # Extract metrics
            lm_losses = [
                r["metrics"].get("final_lm_loss")
                for r in successful
                if r.get("metrics", {}).get("final_lm_loss")
            ]
            accuracies = [
                r["metrics"].get("final_accuracy")
                for r in successful
                if r.get("metrics", {}).get("final_accuracy")
            ]
            feudal_losses = [
                r["metrics"].get("final_feudal_loss")
                for r in successful
                if r.get("metrics", {}).get("final_feudal_loss")
            ]

            if lm_losses:
                avg_lm = sum(lm_losses) / len(lm_losses)
                min_lm = min(lm_losses)
                max_lm = max(lm_losses)
                print("\nLM Loss:")
                print(f"  Mean: {avg_lm:.4f}")
                print(f"  Min:  {min_lm:.4f}")
                print(f"  Max:  {max_lm:.4f}")
                print(
                    f"  Std:  {(sum((x - avg_lm)**2 for x in lm_losses) / len(lm_losses))**0.5:.4f}"
                )

            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                min_acc = min(accuracies)
                max_acc = max(accuracies)
                print("\nAccuracy:")
                print(f"  Mean: {avg_acc:.4f}")
                print(f"  Min:  {min_acc:.4f}")
                print(f"  Max:  {max_acc:.4f}")
                print(
                    f"  Std:  {(sum((x - avg_acc)**2 for x in accuracies) / len(accuracies))**0.5:.4f}"
                )

            if feudal_losses:
                avg_feudal = sum(feudal_losses) / len(feudal_losses)
                print("\nFeudal Loss:")
                print(f"  Mean: {avg_feudal:.4f}")

            print("\nIndividual Results:")
            for r in successful:
                metrics_str = ""
                if r.get("metrics"):
                    m = r["metrics"]
                    if "final_lm_loss" in m:
                        metrics_str += f"LM Loss: {m['final_lm_loss']:.4f}"
                    if "final_accuracy" in m:
                        metrics_str += f" | Accuracy: {m['final_accuracy']:.4f}"
                    if "final_feudal_loss" in m:
                        metrics_str += f" | Feudal Loss: {m['final_feudal_loss']:.4f}"
                print(f"  ✅ Replication {r['replication']}: {metrics_str}")

    # Manager period sweep results
    if "manager_period_sweep" in all_results:
        mps = all_results["manager_period_sweep"]
        print("\n\n📈 MANAGER PERIOD SWEEP RESULTS")
        print("-" * 80)
        successful = [r for r in mps if r.get("success", False)]
        print(f"Successful: {len(successful)}/{len(mps)}")

        if successful:
            print("\nResults by Manager Period:")
            print(
                f"{'Period':<10} {'LM Loss':<12} {'Accuracy':<12} {'Feudal Loss':<12}"
            )
            print("-" * 50)

            # Sort by manager period
            successful_sorted = sorted(successful, key=lambda x: x["manager_period"])
            for r in successful_sorted:
                period = r["manager_period"]
                metrics = r.get("metrics", {})
                lm_loss = metrics.get("final_lm_loss", "N/A")
                accuracy = metrics.get("final_accuracy", "N/A")
                feudal_loss = metrics.get("final_feudal_loss", "N/A")

                lm_str = (
                    f"{lm_loss:.4f}" if isinstance(lm_loss, float) else str(lm_loss)
                )
                acc_str = (
                    f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
                )
                feudal_str = (
                    f"{feudal_loss:.4f}"
                    if isinstance(feudal_loss, float)
                    else str(feudal_loss)
                )

                print(f"{period:<10} {lm_str:<12} {acc_str:<12} {feudal_str:<12}")

            # Find best
            valid_results = [
                r for r in successful if r.get("metrics", {}).get("final_accuracy")
            ]
            if valid_results:
                best = max(valid_results, key=lambda x: x["metrics"]["final_accuracy"])
                print("\n🏆 Best Configuration:")
                print(f"  Manager Period: {best['manager_period']}")
                print(f"  Feudal Weight: {best['feudal_loss_weight']}")
                print(f"  Accuracy: {best['metrics']['final_accuracy']:.4f}")
                if "final_lm_loss" in best["metrics"]:
                    print(f"  LM Loss: {best['metrics']['final_lm_loss']:.4f}")
                if "final_feudal_loss" in best["metrics"]:
                    print(f"  Feudal Loss: {best['metrics']['final_feudal_loss']:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run confirmation experiments: replications and manager period sweep"
    )
    parser.add_argument(
        "--feudal-weight",
        type=float,
        default=0.05,
        help="Feudal loss weight (default: 0.05)",
    )
    parser.add_argument(
        "--manager-period",
        type=int,
        default=4,
        help="Manager period for replication experiments (default: 4)",
    )
    parser.add_argument(
        "--num-replications",
        type=int,
        default=5,
        help="Number of replication runs (default: 5)",
    )
    parser.add_argument(
        "--manager-periods",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8],
        help="Manager periods to sweep (default: 1 2 3 4 5 6 8)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps (default: 200)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device"
    )
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable WandB logging"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--skip-replications",
        action="store_true",
        help="Skip replication experiments",
    )
    parser.add_argument(
        "--skip-manager-sweep",
        action="store_true",
        help="Skip manager period sweep",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # WandB is enabled by default unless --no-wandb is passed
    enable_wandb = args.enable_wandb or not args.no_wandb

    all_results = {}

    # Run replication experiments
    if not args.skip_replications:
        all_results["replications"] = run_replication_experiments(
            feudal_loss_weight=args.feudal_weight,
            manager_period=args.manager_period,
            num_replications=args.num_replications,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=enable_wandb,
        )

    # Run manager period sweep
    if not args.skip_manager_sweep:
        all_results["manager_period_sweep"] = run_manager_period_sweep(
            feudal_loss_weight=args.feudal_weight,
            manager_periods=args.manager_periods,
            max_steps=args.max_steps,
            device=args.device,
            enable_wandb=enable_wandb,
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            root / "experiments" / f"confirmation_experiments_{timestamp}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
