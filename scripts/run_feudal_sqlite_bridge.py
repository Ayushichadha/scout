"""Train a latent-goal-conditioned worker and audit goal refresh in SQLite tasks.

The manager emits a normalized directional latent goal. The worker chooses
service failover from observed history plus its cached goal. A separate supervisor
chooses when to refresh that goal once. Service faults remain synthetic.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "HRM"))
from models.subgoal_head import SubgoalHead, SubgoalHeadConfig  # noqa: E402
from scripts.run_sqlite_revision import (  # noqa: E402
    HORIZON,
    Workflow,
    build_tasks,
    collect,
)
from scripts.run_tool_revision_pilot import features  # noqa: E402
from scripts.run_optional_tool_revision import reactive, score  # noqa: E402


REFRESH = tuple(range(3, 11))


class Worker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 16), nn.Tanh())
        self.manager = SubgoalHead(SubgoalHeadConfig(hidden_size=16, goal_dim=16))
        self.worker = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))

    def propose(self, observation):
        encoded = self.encoder(observation)
        return self.manager._compute_goal(encoded), self.manager._compute_gate(encoded)

    def forward(self, observation, goal, gate):
        return self.worker(self.encoder(observation) + gate * goal).squeeze(-1)


def observation(data, k):
    return torch.tensor(
        np.column_stack((features(data, k, True), np.full(len(data[1]), k / HORIZON))),
        dtype=torch.float32,
    )


def refresh_features(model, data, k, old_goal):
    obs = observation(data, k)
    goal, _ = model.propose(obs)
    return np.column_stack(
        (np.ones(len(obs)), obs.numpy()[:, 1:], (goal - old_goal).norm(dim=-1).numpy())
    )


@torch.no_grad()
def choices(model, data, refresh=None):
    goal, gate = model.propose(observation(data, 2))
    selected = np.full(len(data[1]), HORIZON)
    for k in range(2, 11):
        if refresh is not None and k == refresh:
            goal, gate = model.propose(observation(data, k))
        scores = model(observation(data, k), goal, gate)
        selected[(selected == HORIZON) & (scores.numpy() > 0)] = k
    return selected


def run(output, seed, steps):
    begin = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "workflow.sqlite")
    db.executescript(
        "CREATE TABLE IF NOT EXISTS source(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
        "CREATE TABLE IF NOT EXISTS delivered(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
    )
    partitions = [
        build_tasks(seed + i, n, p)
        for i, (p, n) in enumerate([("train", 500), ("validation", 200), ("test", 500)])
    ]
    for tasks in partitions:
        for task in tasks:
            db.executemany(
                "INSERT OR REPLACE INTO source VALUES (?,?,?)",
                [(task["id"], i, f'{task["id"]}-{i}') for i in range(HORIZON)],
            )
    collected = [collect(db, tasks) for tasks in partitions]
    (train, tr), (val, vr), (test, te) = collected
    model = Worker()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    # Endpoint target: switch now minus retaining primary for the entire episode.
    # The random emission age teaches the worker to consume stale commitments.
    for _ in range(steps):
        k = int(rng.integers(2, 11))
        age = int(rng.integers(2, k + 1))
        goal, gate = model.propose(observation(train, age))
        prediction = model(observation(train, k), goal, gate)
        target = torch.tensor(tr[:, k - 2] - tr[:, -1], dtype=torch.float32)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    manager_grads = [p.grad for p in model.manager.parameters() if p.grad is not None]
    assert manager_grads and all(torch.isfinite(g).all() for g in manager_grads)
    assert sum(g.abs().sum().item() for g in manager_grads) > 0
    model.eval()
    with torch.no_grad():
        initial, g = model.propose(observation(train, 2))
        fresh, h = model.propose(observation(train, 5))
        delta = (
            (
                model(observation(train, 5), initial, g)
                - model(observation(train, 5), fresh, h)
            )
            .abs()
            .max()
            .item()
        )
        assert delta > 0 and torch.equal(
            model(observation(train, 5), initial, g),
            model(observation(train, 5), initial, g),
        )
        # Freeze the worker; training data alone supplies the refresh targets.
        rewards = np.column_stack(
            [score(tr, choices(model, train, k)) for k in REFRESH]
            + [score(tr, choices(model, train))]
        )
        old_goal, _ = model.propose(observation(train, 2))
        future = rewards[:, -1].copy()
        refresh_models = {}
        for j in reversed(range(len(REFRESH))):
            k = REFRESH[j]
            x = refresh_features(model, train, k, old_goal)
            regularizer = np.eye(x.shape[1]) * 10
            regularizer[0, 0] = 0
            beta = np.linalg.solve(
                np.einsum("ni,nj->ij", x, x) + regularizer,
                np.einsum("ni,n->i", x, rewards[:, j] - future),
            )
            refresh_models[k] = beta
            future = np.where(np.einsum("ni,i->n", x, beta) > 0, rewards[:, j], future)
        time_future = rewards[:, -1].copy()
        time_refresh = {}
        for j in reversed(range(len(REFRESH))):
            k = REFRESH[j]
            time_refresh[k] = float((rewards[:, j] - time_future).mean())
            if time_refresh[k] > 0:
                time_future = rewards[:, j].copy()
        time_choice = next((k for k in REFRESH if time_refresh[k] > 0), None)
        val_rewards = [
            score(vr, choices(model, val, k)).mean() for k in (*REFRESH, None)
        ]
        best = (*REFRESH, None)[int(np.argmax(val_rewards))]
        streak = max((1, 2, 3), key=lambda n: score(vr, reactive(val, n)).mean())
        policies = {
            "persist": None,
            "validation_refresh_clock": best,
            "state_refresh": "state",
            "time_refresh": time_choice,
        }
        trace_path = output / "trace.jsonl"
        outcomes = {}
        with trace_path.open("w") as file:
            for policy, refresh in policies.items():
                outcomes[policy] = []
                expected = None
                if refresh != "state":
                    expected = choices(model, test, refresh)
                for task in partitions[2]:
                    workflow = Workflow(db, task)
                    goal = gate = None
                    refreshed = False
                    pending = False
                    refresh_positions = []
                    for _ in range(HORIZON):
                        workflow.step(switch=pending)
                        pending = False
                        k = workflow.calls
                        if k not in range(2, 11) or workflow.revisions:
                            continue
                        history = np.zeros((1, HORIZON), dtype=bool)
                        history[0, :k] = workflow.responses
                        data = (None, history, None, np.array([task["required"]]))
                        obs = observation(data, k)
                        if goal is None:
                            goal, gate = model.propose(obs)
                            goal, gate = goal.detach(), gate.detach()
                        revise = False
                        if not refreshed and k in REFRESH:
                            if refresh == "state":
                                x = refresh_features(model, data, k, goal)
                                revise = bool(
                                    np.einsum("ni,i->n", x, refresh_models[k])[0] > 0
                                )
                            else:
                                revise = k == refresh
                        if revise:
                            goal, gate = model.propose(obs)
                            goal, gate = goal.detach(), gate.detach()
                            refreshed = True
                            refresh_positions.append(k)
                        pending = bool(model(obs, goal, gate)[0] > 0)
                    solved = workflow.finish()
                    if expected is not None:
                        actual = next(
                            (
                                event["call"] - 1
                                for event in workflow.trace
                                if event["revision"]
                            ),
                            HORIZON,
                        )
                        assert actual == int(expected[len(outcomes[policy])])
                    outcomes[policy].append(float(solved))
                    assert (
                        workflow.calls == 12
                        and workflow.revisions <= 1
                        and len(refresh_positions) <= 1
                    )
                    file.write(
                        json.dumps(
                            dict(
                                task=task,
                                policy=policy,
                                solved=solved,
                                goal_refresh=refresh_positions,
                                events=workflow.trace,
                            )
                        )
                        + "\n"
                    )
        verified = 0
        with trace_path.open() as file:
            for line in file:
                row = json.loads(line)
                cursor = 0
                switches = 0
                for i, event in enumerate(row["events"]):
                    assert event["call"] == i + 1
                    cursor += event["status"] == "copied"
                    switches += event["revision"]
                    assert event["cursor"] == cursor
                assert len(row["events"]) == HORIZON and switches <= 1
                assert row["solved"] == (cursor >= row["task"]["required"])
                assert len(row["goal_refresh"]) <= 1
                verified += 1
        outcomes["reactive"] = score(te, reactive(test, streak)).tolist()
        persist_outcome = score(te, choices(model, test)).astype(bool)
        paired_by_pass = {}
        for k in REFRESH:
            revise_outcome = score(te, choices(model, test, k)).astype(bool)
            paired_by_pass[k] = dict(
                workflows=len(persist_outcome),
                only_refresh=int((revise_outcome & ~persist_outcome).sum()),
                only_persist=int((persist_outcome & ~revise_outcome).sum()),
                both=int((persist_outcome & revise_outcome).sum()),
                neither=int((~persist_outcome & ~revise_outcome).sum()),
            )
        report = {
            "scope": "latent goal injection in learned SQLite worker; synthetic faults; no directional auxiliary objective",
            "seed": seed,
            "steps": steps,
            "sizes": [500, 200, 500],
            "verified_trace_rows": verified,
            "task_target": "switch-now endpoint success minus never-switch success",
            "goal_refresh_clock": best,
            "time_refresh_choice": time_choice,
            "time_refresh_advantages": time_refresh,
            "paired_outcomes_by_pass": paired_by_pass,
            "reactive_streak": streak,
            "refresh_supervisor_penalty": 10,
            "refresh_supervisor_coefficients": {
                str(k): v.tolist() for k, v in refresh_models.items()
            },
            "tool_calls_per_episode": HORIZON,
            "max_tool_switches": 1,
            "max_optional_goal_refreshes": 1,
            "causal_logit_delta": delta,
            "completion": {k: float(np.mean(v)) for k, v in outcomes.items()},
            "paired": {},
        }
        for name in ["persist", "validation_refresh_clock", "time_refresh", "reactive"]:
            diff = np.array(outcomes["state_refresh"]) - outcomes[name]
            draws = diff[rng.integers(len(diff), size=(10000, len(diff)))].mean(1)
            report["paired"][name] = {
                "rescued": int((diff > 0).sum()),
                "lost": int((diff < 0).sum()),
                "difference": float(diff.mean()),
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
    db.commit()
    db.close()
    report["wall_seconds"] = time.perf_counter() - begin
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), output / "worker.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/feudal-sqlite-bridge")
    )
    parser.add_argument("--seed", type=int, default=2026113001)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    run(args.output, args.seed, args.steps)
