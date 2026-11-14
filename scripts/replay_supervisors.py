"""Fit exact-budget stopping policies using seed-0 frozen schedule outcomes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.build_revision_split import build_manifest


def predict(regressor, x):
    # Explicit row reductions avoid spurious floating-point warnings from the
    # platform BLAS matrix-vector path for this six-feature linear model.
    scaler, ridge = regressor.steps[0][1], regressor.steps[1][1]
    scores = (scaler.transform(x) * ridge.coef_).sum(axis=1) + ridge.intercept_
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite linear score")
    return scores


def checked_csv(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Archived CSV digest mismatch: {path}")
    return pd.read_csv(path)


def fit_policy(features, correct, valid, penalty, state_aware):
    # Learn the value of revising now versus the fitted later stopping policy.
    future = correct[:, -1].copy()
    regressors = {}
    for k in reversed(range(5)):
        x = features[:, k] if state_aware else np.zeros((len(valid), 1))
        target = (correct[:, k] - future) / valid
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=penalty))
        regressor.fit(x, target, ridge__sample_weight=valid / valid.mean())
        regressors[k] = regressor
        future = np.where(predict(regressor, x) > 0, correct[:, k], future)
    return regressors


def select(features, regressors, state_aware):
    choice = np.full(len(features), 5)
    for k in range(5):
        x = features[:, k] if state_aware else np.zeros((len(features), 1))
        choice[(choice == 5) & (predict(regressors[k], x) > 0)] = k
    return choice


def main():
    decisions = checked_csv(
        "papers/beyond-the-clock/results/decision_level_metrics.csv",
        "07f3d5eb047ff8bece7c61ba7c816eaf420d7e74fd6713c6a86ddd57ed86099d",
    )
    if decisions.hard_intervene.any():
        raise ValueError(
            "Pre-revision observations require persist-only reference trajectories"
        )
    outcomes = checked_csv(
        "experiments/schedule_sweep_k2/per_episode_matrix.csv",
        "9599a9b57d7c1496455903945bcb2bf1dbd8f9873c53bf697f86a18e8e855e7e",
    )
    matrix = outcomes.pivot(
        index="episode_id", columns="schedule_k", values="tokens_correct"
    )
    if list(matrix) != list(range(2, 8)) or matrix.isna().any().any():
        raise ValueError("Incomplete schedule outcomes")
    if (outcomes.groupby("episode_id").tokens_valid.nunique() != 1).any():
        raise ValueError("The endpoint token denominator changed with schedule")
    if not (outcomes.passes_executed == 8).all():
        raise ValueError("Replay requires eight outer passes")
    if (outcomes.tokens_valid <= 0).any() or not np.array_equal(
        outcomes.tokens_correct == outcomes.tokens_valid,
        outcomes.exact_correct.astype(bool),
    ):
        raise ValueError("Invalid archived endpoint labels")
    state = decisions.set_index(["episode_id", "eligible_pass"])
    index = pd.MultiIndex.from_product([matrix.index, range(2, 8)])
    state = state.loc[index]
    if len(state) != len(index) or not np.isfinite(state.beta).all():
        raise ValueError("Incomplete or duplicated pre-decision state")
    features = (
        state[["c", "d", "rho", "dwell_over_m", "q", "alpha"]]
        .to_numpy()
        .reshape(-1, 6, 6)
    )
    meta = pd.DataFrame(build_manifest(Path("HRM/data/conceptarc-mini"))["rows"])
    meta.index = meta["set"] + ":" + meta.row.astype(str).str.zfill(8)
    meta = meta.loc[matrix.index]
    dev = meta.split.eq("calibration").to_numpy()
    test = ~dev
    perm = np.random.default_rng(20261112).permutation(
        sorted(meta.source[dev].unique())
    )
    if set(meta.source[test]) & set(meta.source[dev]):
        raise ValueError("Development and evaluation share a source puzzle")
    train = meta.source.isin(perm[:60]).to_numpy()
    tune = dev & ~train
    valid = (
        outcomes.drop_duplicates("episode_id")
        .set_index("episode_id")
        .loc[matrix.index]
        .tokens_valid.to_numpy()
    )
    correct = matrix.to_numpy()
    clock = correct[dev].sum(axis=0).argmax()
    policies = {"fixed": np.full(len(meta), clock)}
    # Original operating point restricted to one replacement, compulsory at pass 7.
    trigger = state.beta.to_numpy().reshape(-1, 6) > 0.4982
    trigger[:, -1] = True
    policies["original_trigger_exact_budget"] = trigger.argmax(axis=1)
    penalties = {}
    for name, aware in [("pass_only", False), ("pass_plus_state", True)]:
        best = -1
        for alpha in [1.0, 10.0, 100.0]:
            model = fit_policy(
                features[train], correct[train], valid[train], alpha, aware
            )
            choices = select(features, model, aware)
            score = correct[np.arange(len(meta)), choices][tune].sum()
            if score > best:
                best = score
                penalties[name] = alpha
                policies[name] = choices
    report = {}
    for name, choices in policies.items():
        ncorrect = correct[np.arange(len(meta)), choices]
        paired = (
            pd.DataFrame(
                {
                    "group": meta.source[test],
                    "diff": (ncorrect - correct[:, clock])[test],
                    "tokens": valid[test],
                }
            )
            .groupby("group")
            .sum()
            .to_numpy()
        )
        indices = np.random.default_rng(20261112).integers(
            len(paired), size=(10000, len(paired))
        )
        sums = paired[indices].sum(axis=1)
        report[name] = {
            "micro": ncorrect[test].sum() / valid[test].sum(),
            "solves": int((ncorrect[test] == valid[test]).sum()),
            "gap_vs_fixed_pp": 100 * paired[:, 0].sum() / paired[:, 1].sum(),
            "ci95_pp": np.percentile(
                100 * sums[:, 0] / sums[:, 1], [2.5, 97.5]
            ).tolist(),
            "revision_pass_counts": np.bincount(choices[test], minlength=6).tolist(),
        }
    result = {
        "scope": "retrospective frozen seed-0 replay; exact two-emission action space",
        "checkpoint_sha256": "165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889",
        "fit_sources": int(meta.source[train].nunique()),
        "tuning_sources": int(meta.source[tune].nunique()),
        "evaluation_sources": int(meta.source[test].nunique()),
        "evaluation_rows": int(test.sum()),
        "penalties": penalties,
        "policies": report,
    }
    path = Path("analysis/grouped_supervisors_seed0.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
