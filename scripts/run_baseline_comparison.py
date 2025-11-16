#!/usr/bin/env python3
"""Baseline comparison script: HRM with vs without feudal loss."""

import subprocess
import sys
from pathlib import Path
import argparse
import json
import re
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
        "+project_name=hrm_feudal_baseline",
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
    ]

    if use_feudal:
        cmd.extend(
            [
                f"arch.loss.feudal_loss_weight={feudal_loss_weight}",
                f"arch.subgoal_head.manager_period={manager_period}",
            ]
        )
    else:
        # Disable feudal loss by setting weight to 0 (subgoal_head still exists but loss is zero)
        cmd.append("arch.loss.feudal_loss_weight=0.0")

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
    print(f"📝 Training logs saved to: {log_file}")

    if result.returncode != 0:
        print(f"❌ Experiment {run_name} failed!")
        print(result.stderr)
        return {"success": False, "error": result.stderr[:500]}

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
    # Pattern: loss=X.XXXX at the end
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
    baseline_result = run_baseline_experiment(
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
            "success": baseline_result.get("success", False),
            "metrics": baseline_result.get("metrics", {}),
            "log_file": baseline_result.get("log_file"),
        }
    )

    # Run with feudal loss
    print("\n" + "=" * 60)
    print("Running WITH FEUDAL LOSS")
    print("=" * 60)
    feudal_result = run_baseline_experiment(
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
            "success": feudal_result.get("success", False),
            "metrics": feudal_result.get("metrics", {}),
            "log_file": feudal_result.get("log_file"),
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

    # Print experiment details
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON EXPERIMENTS")
    print("=" * 60)
    print("\nExperiment 1: BASELINE (no feudal loss)")
    print("  - Feudal loss weight: 0.0 (disabled)")
    print("  - Subgoal head: Present but loss weight is 0")
    print("  - Purpose: Baseline performance without feudal loss")
    print("\nExperiment 2: WITH FEUDAL LOSS")
    print(f"  - Feudal loss weight: {args.feudal_weight}")
    print(f"  - Manager period: {args.manager_period}")
    print("  - Purpose: Test if feudal loss improves performance")
    print("\nCommon settings:")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Device: {args.device}")
    print(f"  - WandB: {'Enabled' if enable_wandb else 'Disabled'}")
    print("  - Model: Small (hidden_size=32, 1 layer each)")
    print("=" * 60 + "\n")

    # Run comparison
    results = run_comparison(
        feudal_loss_weight=args.feudal_weight,
        manager_period=args.manager_period,
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
        if r.get("metrics"):
            metrics = r["metrics"]
            if "final_lm_loss" in metrics:
                print(f"      Final LM Loss: {metrics['final_lm_loss']:.4f}")
            if "final_feudal_loss" in metrics:
                print(f"      Final Feudal Loss: {metrics['final_feudal_loss']:.4f}")
            if "final_accuracy" in metrics:
                print(f"      Final Accuracy: {metrics['final_accuracy']:.4f}")
        if r.get("log_file"):
            print(f"      Log: {r['log_file']}")

    # Comparison
    if len(results) == 2 and all(r["success"] for r in results):
        baseline_metrics = results[0].get("metrics", {})
        feudal_metrics = results[1].get("metrics", {})
        print(f"\n{'='*60}")
        print("Loss Comparison")
        print(f"{'='*60}")
        if "final_lm_loss" in baseline_metrics and "final_lm_loss" in feudal_metrics:
            baseline_loss = baseline_metrics["final_lm_loss"]
            feudal_loss = feudal_metrics["final_lm_loss"]
            diff = feudal_loss - baseline_loss
            pct = (diff / baseline_loss * 100) if baseline_loss > 0 else 0
            print(f"  Baseline LM Loss: {baseline_loss:.4f}")
            print(f"  Feudal LM Loss:   {feudal_loss:.4f}")
            print(f"  Difference:       {diff:+.4f} ({pct:+.2f}%)")
            if diff < 0:
                print("  ✅ Feudal loss IMPROVES performance!")
            elif diff > 0:
                print("  ⚠️  Feudal loss increases loss")
            else:
                print("  ➡️  No significant difference")


if __name__ == "__main__":
    main()
