"""Cursor restoration and retry semantics for the local tool workflow."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sqlite_revision import Workflow


def test_failover_restores_cursor_without_duplicate_records(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE source(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
        "CREATE TABLE delivered(task TEXT,position INTEGER,payload TEXT,PRIMARY KEY(task,position));"
    )
    db.executemany(
        "INSERT INTO source VALUES ('task',?,?)", [(i, str(i)) for i in range(12)]
    )
    task = dict(id="task", regime="stable", onset=3, duration=2, required=6)
    monkeypatch.setattr("scripts.run_sqlite_revision.uniform", lambda *_: 0.0)
    worker = Workflow(db, task)
    worker.step()
    worker.step()
    assert worker.cursor == 2
    worker.step(switch=True)
    assert worker.cursor == 2 and worker.calls == 3 and worker.revisions == 1
    worker.step()
    assert worker.cursor == 3
    assert db.execute(
        "SELECT position FROM delivered ORDER BY position"
    ).fetchall() == [(0,), (1,), (2,)]
    assert not worker.finish()


def test_unavailable_call_keeps_cursor(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE source(task TEXT,position INTEGER,payload TEXT);"
        "CREATE TABLE delivered(task TEXT,position INTEGER,payload TEXT);"
    )
    task = dict(id="task", regime="stable", onset=3, duration=2, required=6)
    monkeypatch.setattr("scripts.run_sqlite_revision.uniform", lambda *_: 1.0)
    worker = Workflow(db, task)
    assert worker.step() == "unavailable"
    assert worker.cursor == 0 and worker.calls == 1


def test_features_ignore_future_responses_and_regime():
    import numpy as np

    from scripts.run_tool_revision_pilot import features

    responses = np.array([[True, False, True] + [False] * 9])
    first = (np.array(["stable"]), responses, None, np.array([8]))
    changed = responses.copy()
    changed[:, 3:] = True
    second = (np.array(["persistent"]), changed, None, np.array([8]))
    np.testing.assert_array_equal(features(first, 3, True), features(second, 3, True))


def test_trace_checker_rejects_cursor_corruption(tmp_path):
    import json

    import pytest

    from scripts.run_sqlite_revision import verify_traces

    trace = [
        dict(
            call=i + 1,
            service="primary",
            status="unavailable",
            cursor=0,
            revision=False,
        )
        for i in range(12)
    ]
    row = dict(
        task=dict(required=6),
        switch=12,
        solved=False,
        calls=12,
        revisions=0,
        trace=trace,
    )
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert verify_traces(path) == 1
    trace[3]["cursor"] = 1
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        verify_traces(path)
