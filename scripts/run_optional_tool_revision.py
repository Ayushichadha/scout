"""Optional zero-or-one tool revision in the existing synthetic service simulator."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.run_tool_revision_pilot import HORIZON, POSITIONS, features, generate


# Last column means retaining the primary service for the complete horizon.
CHOICES = np.append(POSITIONS, HORIZON)


def schedule_rewards(data):
    _, primary, backup, required = data
    correct = np.column_stack(
        [primary[:, :k].sum(1) + backup[:, k + 1 :].sum(1) for k in POSITIONS]
        + [primary.sum(1)]
    )
    return (correct >= required[:, None]).astype(float)


def fit(data, rewards, state, penalty):
    future = rewards[:, -1].copy()
    models = {}
    for j in reversed(range(len(POSITIONS))):
        k = int(POSITIONS[j])
        x = features(data, k, state)
        regularizer = np.eye(x.shape[1]) * penalty
        regularizer[0, 0] = 0
        gram = np.einsum("ni,nj->ij", x, x)
        rhs = np.einsum("ni,n->i", x, rewards[:, j] - future)
        beta = np.linalg.solve(gram + regularizer, rhs)
        if not np.isfinite(beta).all():
            raise FloatingPointError("Nonfinite supervisor parameters")
        models[k] = beta
        future = np.where(np.einsum("ni,i->n", x, beta) > 0, rewards[:, j], future)
    return models


def select(data, models, state):
    choices = np.full(len(data[0]), HORIZON)
    for k in POSITIONS:
        score = np.einsum("ni,i->n", features(data, k, state), models[int(k)])
        choices[(choices == HORIZON) & (score > 0)] = k
    return choices


def reactive(data, streak):
    choices = np.full(len(data[0]), HORIZON)
    for k in POSITIONS:
        if k >= streak:
            failed = ~data[1][:, k - streak : k].any(1)
            choices[(choices == HORIZON) & failed] = k
    return choices


def score(rewards, choices):
    columns = np.searchsorted(CHOICES, choices)
    return rewards[np.arange(len(choices)), columns]


def run(output, seed_offset):
    begin = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    seeds = dict(
        train=2026111301 + seed_offset,
        validation=2026111302 + seed_offset,
        test=2026111303 + seed_offset,
    )
    train, val, test = [
        generate(seeds[name], size)
        for name, size in [("train", 2000), ("validation", 500), ("test", 1000)]
    ]
    tr, vr, te = [schedule_rewards(data) for data in (train, val, test)]
    clock = int(CHOICES[vr.mean(0).argmax()])
    choices = {
        "validation_fixed": np.full(len(test[0]), clock),
        "retain_primary": np.full(len(test[0]), HORIZON),
    }
    settings = {}
    for name, state in [("learned_time", False), ("learned_state", True)]:
        candidates = []
        for penalty in (0.1, 1.0, 10.0):
            model = fit(train, tr, state, penalty)
            candidates.append(
                (score(vr, select(val, model, state)).mean(), model, penalty)
            )
        _, model, penalty = max(candidates, key=lambda item: item[0])
        choices[name] = select(test, model, state)
        settings[name] = {
            "penalty": penalty,
            "coefficients": {str(k): values.tolist() for k, values in model.items()},
        }
    streak = max((1, 2, 3), key=lambda n: score(vr, reactive(val, n)).mean())
    choices["failure_streak"] = reactive(test, streak)
    predictions = {name: score(te, selected) for name, selected in choices.items()}
    report = {
        "scope": "synthetic Bernoulli service simulator; optional zero-or-one revision",
        "generator_seeds": seeds,
        "sizes": [2000, 500, 1000],
        "fixed_switch_after_call": clock if clock < HORIZON else None,
        "failure_streak": streak,
        "supervisors": settings,
        "test_completion": {name: float(y.mean()) for name, y in predictions.items()},
        "compute": {},
        "paired": {},
        "regimes": {},
    }
    for name, selected in choices.items():
        # Keep the existing fixed twelve-call budget. A switched run spends one
        # call restoring the cursor and eleven calls transferring records.
        decision_counts = np.where(
            selected < HORIZON, selected - POSITIONS[0] + 1, len(POSITIONS)
        )
        report["compute"][name] = {
            "environment_calls": HORIZON,
            "mean_cursor_restore_calls": float((selected < HORIZON).mean()),
            "mean_transfer_calls": float(HORIZON - (selected < HORIZON).mean()),
            "mean_revisions": float((selected < HORIZON).mean()),
            "mean_supervisor_evaluations": (
                float(decision_counts.mean()) if name.startswith("learned") else 0.0
            ),
            "mean_reactive_checks": (
                float(decision_counts.mean()) if name == "failure_streak" else 0.0
            ),
        }
    rng = np.random.default_rng(20261113)
    for name in (
        "validation_fixed",
        "learned_time",
        "failure_streak",
        "retain_primary",
    ):
        delta = predictions["learned_state"] - predictions[name]
        means = np.concatenate(
            [
                delta[rng.integers(len(delta), size=(100, len(delta)))].mean(1)
                for _ in range(100)
            ]
        )
        report["paired"][name] = {
            "rescued": int((delta > 0).sum()),
            "lost": int((delta < 0).sum()),
            "difference": float(delta.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        }
    for regime in np.unique(test[0]):
        mask = test[0] == regime
        report["regimes"][regime] = {
            "n": int(mask.sum()),
            "completion": {
                name: float(y[mask].mean()) for name, y in predictions.items()
            },
        }
    trace_path = output / "trace.csv.gz"
    with gzip.open(trace_path, "wt") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "regime",
                "required",
                "primary",
                "backup",
                "policy",
                "switch_after_call",
                "records",
                "solved",
            ]
        )
        for name, selected in choices.items():
            for i, k in enumerate(selected):
                primary, backup = test[1][i], test[2][i]
                count = int(primary[:k].sum())
                if k < HORIZON:
                    count += int(backup[k + 1 :].sum())
                assert bool(count >= test[3][i]) == bool(predictions[name][i])

                def encode(a):
                    return "".join(str(int(x)) for x in a)

                writer.writerow(
                    [
                        i,
                        test[0][i],
                        test[3][i],
                        encode(primary),
                        encode(backup),
                        name,
                        k,
                        count,
                        int(predictions[name][i]),
                    ]
                )
    # Independently replay saved records call by call, including the restore call.
    with gzip.open(trace_path, "rt") as file:
        for row in csv.DictReader(file):
            k = int(row["switch_after_call"])
            records = 0
            for call in range(HORIZON):
                if call < k:
                    records += int(row["primary"][call])
                elif call > k:
                    records += int(row["backup"][call])
            assert records == int(row["records"])
            assert int(records >= int(row["required"])) == int(row["solved"])
    report["verified_trace_rows"] = len(test[0]) * len(choices)
    report["wall_seconds"] = perf_counter() - begin
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    generator_path = Path(__file__).with_name("run_tool_revision_pilot.py")
    report["generator_script_sha256"] = hashlib.sha256(
        generator_path.read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"completion": report["test_completion"], "paired": report["paired"]}
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/optional-tool-revision")
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.seed_offset)
