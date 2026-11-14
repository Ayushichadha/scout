"""Synthetic tool-service workflow. Run from repo root with the project Python."""

import argparse
import csv
import gzip
import json
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np


HORIZON = 12
# Exactly one tool revision, after an observed response on calls 2 through 10.
POSITIONS = np.arange(2, 11)


def generate(seed, count):
    """Each workflow has its own reproducible service responses and requirement."""
    rng = np.random.default_rng(seed)
    terrain = rng.choice(["stable", "transient", "persistent"], count)
    onset = rng.integers(2, 8, count)
    duration = rng.integers(1, 4, count)
    position = np.arange(HORIZON)
    outage = (position >= onset[:, None]) & (
        (terrain[:, None] == "persistent")
        | ((terrain[:, None] == "transient") & (position < (onset + duration)[:, None]))
    )
    primary = rng.random((count, HORIZON)) < np.where(outage, 0.08, 0.94)
    backup = rng.random((count, HORIZON)) < 0.72
    required = rng.integers(6, 10, count)
    return terrain, primary, backup, required


def evaluate_schedules(data):
    """A success checkpoints one record. The first backup call restores the cursor."""
    _, primary, backup, required = data
    return np.stack(
        [
            primary[:, :k].sum(1) + backup[:, k + 1 :].sum(1) >= required
            for k in POSITIONS
        ],
        axis=1,
    ).astype(float)


def features(data, k, use_state):
    _, primary, _, required = data
    history = primary[:, :k]
    base = np.ones((len(primary), 1))
    if not use_state:
        return base
    return np.column_stack(
        (
            base,
            history[:, -1],
            history[:, -2:].mean(1),
            history.mean(1),
            (required - history.sum(1)) / HORIZON,
            required / HORIZON,
        )
    )


def fit(data, rewards, use_state, penalty):
    # At each pass compare switching now to following the fitted future policy.
    value = rewards[:, -1]
    models = {}
    for j in reversed(range(len(POSITIONS) - 1)):
        x = features(data, POSITIONS[j], use_state)
        regularizer = np.eye(x.shape[1]) * penalty
        regularizer[0, 0] = 0
        gram = np.einsum("ni,nj->ij", x, x)
        rhs = np.einsum("ni,n->i", x, rewards[:, j] - value)
        beta = np.linalg.solve(gram + regularizer, rhs)
        if not np.isfinite(beta).all():
            raise FloatingPointError("Nonfinite fitted supervisor")
        models[int(POSITIONS[j])] = beta
        value = np.where(np.einsum("ni,i->n", x, beta) > 0, rewards[:, j], value)
    return models


def select(data, models, use_state):
    decisions = np.full(len(data[0]), POSITIONS[-1])
    for k in POSITIONS[:-1]:
        advantage = np.einsum("ni,i->n", features(data, k, use_state), models[int(k)])
        decisions[(decisions == POSITIONS[-1]) & (advantage > 0)] = k
    return decisions


def outcomes(rewards, choices):
    return rewards[np.arange(len(choices)), choices - POSITIONS[0]]


def run(output, seed_offset=0):
    output.mkdir(parents=True, exist_ok=True)
    begin = perf_counter()
    train, val, test = (
        generate(seed, size)
        for seed, size in [
            (2026111201 + seed_offset, 2000),
            (2026111202 + seed_offset, 500),
            (2026111203 + seed_offset, 1000),
        ]
    )
    train_scores, val_scores, test_scores = map(evaluate_schedules, [train, val, test])
    k = POSITIONS[np.argmax(val_scores.mean(0))]
    choices = {"validation_clock": np.full(len(test[0]), k)}
    settings = {}
    for name, state in [("learned_clock", False), ("state_supervisor", True)]:
        candidates = []
        for penalty in (0.1, 1.0, 10.0):
            model = fit(train, train_scores, state, penalty)
            score = outcomes(val_scores, select(val, model, state)).mean()
            candidates.append((score, model, penalty))
        _, model, penalty = max(candidates, key=lambda item: item[0])
        choices[name] = select(test, model, state)
        settings[name] = {
            "ridge_penalty": penalty,
            "coefficients": {str(k): v.tolist() for k, v in model.items()},
        }

    # Calibrate a transparent rule under the same horizon and action space.
    def rule(data, threshold):
        decisions = np.full(len(data[0]), POSITIONS[-1])
        for k in POSITIONS[:-1]:
            if k >= threshold:
                stuck = ~data[1][:, k - threshold : k].any(1)
                decisions[(decisions == POSITIONS[-1]) & stuck] = k
        return decisions

    threshold = max((1, 2, 3), key=lambda t: outcomes(val_scores, rule(val, t)).mean())
    choices["failure_streak"] = rule(test, threshold)
    predictions = {name: outcomes(test_scores, ks) for name, ks in choices.items()}
    report = {
        "scope": "synthetic tool-routing pilot",
        "generator_seeds": {
            "train": 2026111201 + seed_offset,
            "validation": 2026111202 + seed_offset,
            "test": 2026111203 + seed_offset,
        },
        "sizes": [2000, 500, 1000],
        "model_settings": settings,
        "failure_streak_threshold": threshold,
        "validation_clock": int(k),
        "test_rates": {p: float(y.mean()) for p, y in predictions.items()},
        "oracle_rate": float(test_scores.max(1).mean()),
        "paired": {},
        "actual_calls_per_episode": HORIZON,
        "actual_revisions_per_episode": 1,
        "test_regimes": {},
    }
    rng = np.random.default_rng(777)
    for name in ["validation_clock", "learned_clock", "failure_streak"]:
        delta = predictions["state_supervisor"] - predictions[name]
        draws = [
            delta[rng.integers(0, len(delta), (100, len(delta)))].mean(1)
            for _ in range(100)
        ]
        report["paired"][name] = {
            "rescued": int((delta > 0).sum()),
            "lost": int((delta < 0).sum()),
            "difference": float(delta.mean()),
            "paired_95ci": np.quantile(np.concatenate(draws), [0.025, 0.975]).tolist(),
        }
    for regime in np.unique(test[0]):
        mask = test[0] == regime
        report["test_regimes"][str(regime)] = {
            "n": int(mask.sum()),
            "solve_rates": {p: float(y[mask].mean()) for p, y in predictions.items()},
        }
    with gzip.open(output / "test_trace.csv.gz", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "regime",
                "required",
                "primary_responses",
                "backup_responses",
                "policy",
                "switch_after_call",
                "correct_records",
                "solved",
            ]
        )
        for name, positions in choices.items():
            for i, k in enumerate(positions):
                primary, backup = test[1][i], test[2][i]
                correct = primary[:k].sum() + backup[k + 1 :].sum()
                assert bool(correct >= test[3][i]) == bool(predictions[name][i])

                def encode(row):
                    return "".join("1" if x else "0" for x in row)

                writer.writerow(
                    [
                        i,
                        test[0][i],
                        test[3][i],
                        encode(primary),
                        encode(backup),
                        name,
                        k,
                        correct,
                        int(predictions[name][i]),
                    ]
                )
    report["wall_seconds"] = perf_counter() - begin
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("test_rates", "paired", "test_regimes", "wall_seconds")
            }
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tool-revision-pilot")
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.seed_offset)
