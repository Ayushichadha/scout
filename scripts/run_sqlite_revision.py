"""Learn optional tool failover using actual SQLite cursor and record-copy operations.

Faults are synthetic. Each call either copies one real row or restores the backup
cursor. Source and result databases are local files, with independently keyed tasks.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from scripts.run_optional_tool_revision import CHOICES, fit, reactive, score, select
from scripts.run_tool_revision_pilot import POSITIONS, features


HORIZON = 12


def uniform(task, service, call):
    raw = hashlib.sha256(f"{task}:{service}:{call}".encode()).digest()
    return int.from_bytes(raw[:8], "little") / 2**64


def build_tasks(seed, count, partition):
    rng = np.random.default_rng(seed)
    return [
        dict(
            id=f"{partition}-{seed}-{i}",
            regime=str(rng.choice(["stable", "transient", "persistent"])),
            onset=int(rng.integers(2, 8)),
            duration=int(rng.integers(1, 4)),
            required=int(rng.integers(6, 10)),
        )
        for i in range(count)
    ]


class Workflow:
    def __init__(self, connection, task):
        self.db = connection
        self.task = task
        self.cursor = 0
        self.service = "primary"
        self.calls = 0
        self.revisions = 0
        self.responses = []
        self.trace = []
        self.db.execute("DELETE FROM delivered WHERE task=?", (task["id"],))

    def step(self, switch=False):
        call = self.calls
        self.calls += 1
        if switch:
            if self.revisions:
                raise ValueError("Only one revision allowed")
            self.revisions = 1
            self.service = "backup"
            row = self.db.execute(
                "SELECT COALESCE(MAX(position)+1,0) FROM delivered WHERE task=?",
                (self.task["id"],),
            ).fetchone()
            self.cursor = row[0]
            status = "cursor_restored"
        else:
            outage = call >= self.task["onset"] and (
                self.task["regime"] == "persistent"
                or self.task["regime"] == "transient"
                and call < self.task["onset"] + self.task["duration"]
            )
            probability = (
                0.72 if self.service == "backup" else (0.08 if outage else 0.94)
            )
            if uniform(self.task["id"], self.service, call) >= probability:
                status = "unavailable"
            else:
                row = self.db.execute(
                    "SELECT position,payload FROM source WHERE task=? AND position=?",
                    (self.task["id"], self.cursor),
                ).fetchone()
                if row:
                    self.db.execute(
                        "INSERT INTO delivered VALUES (?,?,?)", (self.task["id"], *row)
                    )
                    self.cursor += 1
                    status = "copied"
                else:
                    status = "complete"
        self.responses.append(status == "copied")
        self.trace.append(
            dict(
                call=self.calls,
                service=self.service,
                status=status,
                cursor=self.cursor,
                revision=bool(switch),
            )
        )
        return status

    def finish(self):
        mismatches = self.db.execute(
            "SELECT COUNT(*) FROM delivered d LEFT JOIN source s ON d.task=s.task AND d.position=s.position "
            "WHERE d.task=? AND (s.payload IS NULL OR d.payload != s.payload)",
            (self.task["id"],),
        ).fetchone()[0]
        count = self.db.execute(
            "SELECT COUNT(*) FROM delivered WHERE task=?", (self.task["id"],)
        ).fetchone()[0]
        assert not mismatches and count == self.cursor
        return count >= self.task["required"]


def execute(db, task, switch):
    workflow = Workflow(db, task)
    for call in range(HORIZON):
        workflow.step(switch=call == switch and switch < HORIZON)
    solved = workflow.finish()
    return workflow, float(solved)


def execute_policy(db, task, policy, selected, models=None):
    """Choose after observing call k, then spend call k+1 restoring the cursor."""
    workflow = Workflow(db, task)
    pending = False
    for _ in range(HORIZON):
        workflow.step(switch=pending)
        pending = False
        k = workflow.calls
        if workflow.revisions or k not in POSITIONS:
            continue
        if policy in ("time", "state"):
            observed = np.zeros((1, HORIZON), dtype=bool)
            observed[0, :k] = workflow.responses
            data = (None, observed, None, np.array([task["required"]]))
            x = features(data, k, policy == "state")
            pending = bool(np.einsum("ni,i->n", x, models[k])[0] > 0)
        elif policy == "reactive":
            pending = k >= selected and not any(workflow.responses[-selected:])
        elif policy == "fixed":
            pending = k == selected
    return workflow, float(workflow.finish())


def verify_traces(path):
    """Validate saved events independently of the schedule outcome matrix."""
    count = 0
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            cursor = 0
            revisions = 0
            for i, event in enumerate(row["trace"]):
                assert event["call"] == i + 1
                assert event["status"] in (
                    "copied",
                    "unavailable",
                    "complete",
                    "cursor_restored",
                )
                if event["status"] == "copied":
                    cursor += 1
                if event["revision"]:
                    revisions += 1
                    assert event["status"] == "cursor_restored"
                    assert i == row["switch"] and i in POSITIONS
                assert event["cursor"] == cursor
                assert event["service"] == ("backup" if revisions else "primary")
            assert len(row["trace"]) == row["calls"] == HORIZON
            assert revisions == row["revisions"] <= 1
            assert bool(cursor >= row["task"]["required"]) == row["solved"]
            count += 1
    return count


def collect(db, tasks):
    primary, rewards = [], []
    for task in tasks:
        persist, solved = execute(db, task, HORIZON)
        primary.append(persist.responses)
        row = []
        for switch in CHOICES[:-1]:
            _, result = execute(db, task, int(switch))
            row.append(result)
        rewards.append(row + [solved])
    data = (
        np.array([t["regime"] for t in tasks]),
        np.array(primary),
        None,
        np.array([t["required"] for t in tasks]),
    )
    return data, np.array(rewards)


def run(output, seed, train_size=400, validation_size=100, test_size=200):
    if min(train_size, validation_size, test_size) < 1:
        raise ValueError("Partition sizes must be positive")
    begin = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    database = output / "workflow.sqlite"
    db = sqlite3.connect(database)
    db.executescript(
        "CREATE TABLE IF NOT EXISTS source(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
        "CREATE TABLE IF NOT EXISTS delivered(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
    )
    partitions = [
        build_tasks(seed + i, n, name)
        for i, (name, n) in enumerate(
            [
                ("train", train_size),
                ("validation", validation_size),
                ("test", test_size),
            ]
        )
    ]
    partition_ids = [set(task["id"] for task in tasks) for tasks in partitions]
    assert sum(map(len, partition_ids)) == len(set.union(*partition_ids))
    for tasks in partitions:
        for task in tasks:
            db.executemany(
                "INSERT OR REPLACE INTO source VALUES (?,?,?)",
                [
                    (
                        task["id"],
                        i,
                        hashlib.sha256(f'{task["id"]}-{i}'.encode()).hexdigest(),
                    )
                    for i in range(HORIZON)
                ],
            )
    collected = [collect(db, tasks) for tasks in partitions]
    (train, tr), (val, vr), (test, te) = collected
    fixed = int(CHOICES[vr.mean(0).argmax()])
    choices = {
        "fixed": np.full(len(test[0]), fixed),
        "retain": np.full(len(test[0]), HORIZON),
    }
    settings = {}
    selected_models = {}
    for name, state in [("time", False), ("state", True)]:
        candidates = []
        for penalty in [0.1, 1.0, 10.0]:
            model = fit(train, tr, state, penalty)
            candidates.append(
                (score(vr, select(val, model, state)).mean(), model, penalty)
            )
        _, model, penalty = max(candidates, key=lambda x: x[0])
        choices[name] = select(test, model, state)
        selected_models[name] = model
        settings[name] = {
            "penalty": penalty,
            "coefficients": {str(k): v.tolist() for k, v in model.items()},
        }
    streak = max([1, 2, 3], key=lambda x: score(vr, reactive(val, x)).mean())
    choices["reactive"] = reactive(test, streak)
    predictions = {}
    trace_path = output / "trace.jsonl"
    with trace_path.open("w") as file:
        for name, decisions in choices.items():
            predictions[name] = []
            for i, (task, switch) in enumerate(zip(partitions[2], decisions)):
                setting = streak if name == "reactive" else fixed
                workflow, solved = execute_policy(
                    db, task, name, setting, selected_models.get(name)
                )
                actual_switch = next(
                    (e["call"] - 1 for e in workflow.trace if e["revision"]), HORIZON
                )
                assert actual_switch == int(switch)
                assert solved == score(te, decisions)[i]
                predictions[name].append(solved)
                file.write(
                    json.dumps(
                        dict(
                            task=task,
                            policy=name,
                            switch=int(switch),
                            solved=bool(solved),
                            calls=workflow.calls,
                            revisions=workflow.revisions,
                            trace=workflow.trace,
                        )
                    )
                    + "\n"
                )
    verified_trace_rows = verify_traces(trace_path)
    db.commit()
    db.close()
    result = {
        "scope": "actual SQLite operations with synthetic service faults; explicit tool commitments",
        "seed": seed,
        "verified_trace_rows": verified_trace_rows,
        "partition_id_hashes": [
            hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
            for ids in partition_ids
        ],
        "sizes": [train_size, validation_size, test_size],
        "horizon": HORIZON,
        "fixed": fixed,
        "streak": streak,
        "supervisors": settings,
        "completion": {k: float(np.mean(v)) for k, v in predictions.items()},
        "paired": {},
        "regimes": {},
        "wall_seconds": perf_counter() - begin,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    rng = np.random.default_rng(seed + 100)
    for name in ["fixed", "time", "reactive"]:
        diff = np.array(predictions["state"]) - predictions[name]
        samples = diff[rng.integers(len(diff), size=(10000, len(diff)))].mean(1)
        result["paired"][name] = {
            "rescued": int((diff > 0).sum()),
            "lost": int((diff < 0).sum()),
            "difference": float(diff.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    for regime in ["stable", "transient", "persistent"]:
        mask = test[0] == regime
        if not mask.any():
            continue
        state = np.asarray(predictions["state"])[mask]
        baseline = np.asarray(predictions["reactive"])[mask]
        delta = state - baseline
        draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        result["regimes"][regime] = {
            "n": int(mask.sum()),
            "completion": {
                name: float(np.asarray(values)[mask].mean())
                for name, values in predictions.items()
            },
            "state_vs_reactive": {
                "rescued": int((delta > 0).sum()),
                "lost": int((delta < 0).sum()),
                "difference": float(delta.mean()),
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            },
        }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/sqlite-revision"))
    parser.add_argument("--seed", type=int, default=2026111401)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=200)
    args = parser.parse_args()
    run(args.output, args.seed, args.train_size, args.validation_size, args.test_size)
