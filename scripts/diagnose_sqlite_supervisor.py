"""Audit persistent-outage errors and test one added observed failure-duration feature."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np

from scripts import run_sqlite_revision as sql
from scripts.run_optional_tool_revision import reactive, score, select
from scripts.run_tool_revision_pilot import POSITIONS, features


def observations(data, k):
    base = features(data, k, True)
    history = data[1][:, :k]
    # The first success when reading backward terminates the observed failure run.
    streak = np.where(history.any(axis=1), history[:, ::-1].argmax(axis=1), k)
    return np.column_stack((base, streak / sql.HORIZON))


def fit_duration(data, rewards, penalty):
    future = rewards[:, -1].copy()
    models = {}
    for j in reversed(range(len(POSITIONS))):
        k = int(POSITIONS[j])
        x = observations(data, k)
        regularizer = np.eye(x.shape[1]) * penalty
        regularizer[0, 0] = 0
        beta = np.linalg.solve(
            np.einsum("ni,nj->ij", x, x) + regularizer,
            np.einsum("ni,n->i", x, rewards[:, j] - future),
        )
        assert np.isfinite(beta).all()
        models[k] = beta
        future = np.where(np.einsum("ni,i->n", x, beta) > 0, rewards[:, j], future)
    return models


def choose_duration(data, models):
    choices = np.full(len(data[0]), sql.HORIZON)
    for k in POSITIONS:
        switch = np.einsum("ni,i->n", observations(data, k), models[int(k)]) > 0
        choices[(choices == sql.HORIZON) & switch] = k
    return choices


def compare(y, baseline, rng):
    delta = y - baseline
    draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return dict(
        rescued=int((delta > 0).sum()),
        lost=int((delta < 0).sum()),
        gap=float(delta.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
    )


def historical_diagnosis(root):
    report = []
    for trace in sorted(root.glob("seed*/trace.jsonl")):
        paired = {}
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            if row["task"]["regime"] == "persistent":
                paired.setdefault(row["task"]["id"], {})[row["policy"]] = row
        losses = [
            p
            for p in paired.values()
            if not p["state"]["solved"] and p["reactive"]["solved"]
        ]
        rescues = [
            p
            for p in paired.values()
            if p["state"]["solved"] and not p["reactive"]["solved"]
        ]
        report.append(
            dict(
                run=trace.parent.name,
                persistent_tasks=len(paired),
                losses=len(losses),
                rescues=len(rescues),
                loss_switches=[
                    dict(
                        state=p["state"]["switch"],
                        reactive=p["reactive"]["switch"],
                        onset=p["state"]["task"]["onset"],
                        required=p["state"]["task"]["required"],
                    )
                    for p in losses
                ],
            )
        )
    return report


def run(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    # Reuse original generator, partitions, exhaustive SQL target collection and tuning.
    with contextlib.redirect_stdout(io.StringIO()):
        original = sql.run(output, seed, 1000, 300, 2000)
    parts = [
        sql.build_tasks(seed + i, n, name)
        for i, (name, n) in enumerate(
            [("train", 1000), ("validation", 300), ("test", 2000)]
        )
    ]
    db = sqlite3.connect(output / "workflow.sqlite")
    (train, tr), (val, vr), (test, te) = [sql.collect(db, p) for p in parts]
    candidates = []
    for penalty in [0.1, 1.0, 10.0]:
        model = fit_duration(train, tr, penalty)
        candidates.append(
            (score(vr, choose_duration(val, model)).mean(), model, penalty)
        )
    _, model, penalty = max(candidates, key=lambda c: c[0])
    selections = choose_duration(test, model)
    outcomes = []
    with (output / "duration_trace.jsonl").open("w") as file:
        for task, selected in zip(parts[2], selections):
            workflow = sql.Workflow(db, task)
            pending = False
            for _ in range(sql.HORIZON):
                workflow.step(switch=pending)
                pending = False
                k = workflow.calls
                if workflow.revisions or k not in POSITIONS:
                    continue
                observed = np.zeros((1, sql.HORIZON), dtype=bool)
                observed[0, :k] = workflow.responses
                data = (None, observed, None, np.array([task["required"]]))
                pending = bool(
                    np.einsum("ni,i->n", observations(data, k), model[k])[0] > 0
                )
            switch = next(
                (e["call"] - 1 for e in workflow.trace if e["revision"]), sql.HORIZON
            )
            solved = workflow.finish()
            assert switch == selected
            outcomes.append(float(solved))
            file.write(
                json.dumps(
                    dict(
                        task=task,
                        policy="duration",
                        switch=int(switch),
                        solved=bool(solved),
                        calls=workflow.calls,
                        revisions=workflow.revisions,
                        trace=workflow.trace,
                    )
                )
                + "\n"
            )
    assert np.array_equal(outcomes, score(te, selections))
    verified = sql.verify_traces(output / "duration_trace.jsonl")
    db.commit()
    db.close()
    streak = original["streak"]
    reactive_out = score(te, reactive(test, streak))
    # Reconstruct original state policy from its validation-selected coefficients.
    state_model = {
        int(k): np.array(v)
        for k, v in original["supervisors"]["state"]["coefficients"].items()
    }
    state_out = score(te, select(test, state_model, True))
    y = np.array(outcomes)
    rng = np.random.default_rng(seed + 100)
    result = dict(
        seed=seed,
        penalty=penalty,
        verified_duration_traces=verified,
        completion=dict(
            duration=float(y.mean()),
            state=float(state_out.mean()),
            reactive=float(reactive_out.mean()),
        ),
        vs_state=compare(y, state_out, rng),
        vs_reactive=compare(y, reactive_out, rng),
        regimes={},
    )
    for regime in ["stable", "transient", "persistent"]:
        mask = test[0] == regime
        result["regimes"][regime] = dict(
            n=int(mask.sum()),
            completion=dict(
                duration=float(y[mask].mean()),
                state=float(state_out[mask].mean()),
                reactive=float(reactive_out[mask].mean()),
            ),
            vs_reactive=compare(y[mask], reactive_out[mask], rng),
        )
    (output / "duration_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sqlite-duration-audit")
    )
    args = parser.parse_args()
    # This feature ablation was fixed before inspecting these evaluation outcomes.
    reports = [
        run(args.output / f"seed{seed}", seed)
        for seed in [2026112401, 2026112501, 2026112601]
    ]
    result = dict(
        scope="SQLite workflow, unchanged synthetic fault generator",
        historical=historical_diagnosis(Path("outputs/sqlite-large")),
        fresh_runs=reports,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for r in reports:
        print(r["seed"], r["completion"], r["vs_reactive"], r["regimes"]["persistent"])


if __name__ == "__main__":
    main()
