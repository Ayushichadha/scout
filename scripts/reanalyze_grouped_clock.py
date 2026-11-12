"""Retrospective calibration and evaluation on the archived seed-0 schedule matrix."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_revision_split import build_manifest


def main():
    manifest = build_manifest(Path("HRM/data/conceptarc-mini"))
    meta = pd.DataFrame(manifest["rows"])
    meta["episode_id"] = meta["set"] + ":" + meta["row"].astype(str).str.zfill(8)
    matrix = pd.read_csv("experiments/schedule_sweep_k2/per_episode_matrix.csv")
    rows = matrix.merge(meta, on="episode_id", validate="many_to_one", how="left")
    if (
        rows["source"].isna().any()
        or rows.duplicated(["episode_id", "schedule_k"]).any()
    ):
        raise ValueError("Episode identity mismatch")
    outcomes = rows.pivot(
        index="episode_id", columns="schedule_k", values="tokens_correct"
    )
    if list(outcomes) != list(range(2, 8)) or outcomes.isna().any().any():
        raise ValueError("Incomplete six-schedule matrix")
    if rows.groupby("episode_id").tokens_valid.nunique().max() != 1:
        raise ValueError("Schedules disagree on target length")
    meta = (
        rows.drop_duplicates("episode_id").set_index("episode_id").loc[outcomes.index]
    )
    cal = meta.split == "calibration"
    if set(meta.source[cal]) & set(meta.source[~cal]):
        raise ValueError("Shared source ID")
    clock = int(outcomes.loc[cal].sum().idxmax())
    test = meta.loc[~cal, ["source", "tokens_valid"]].copy()
    test["clock_correct"] = outcomes.loc[~cal, clock]
    test["oracle_correct"] = outcomes.loc[~cal].max(axis=1)
    test["delta"] = test.oracle_correct - test.clock_correct
    grouped = test.groupby("source")[["delta", "tokens_valid"]].sum().to_numpy()
    rng = np.random.default_rng(20261112)
    sums = grouped[rng.integers(len(grouped), size=(10000, len(grouped)))].sum(axis=1)
    result = dict(
        seed=0,
        scope="retrospective, archived final episodes only",
        calibration_sources=int(meta.source[cal].nunique()),
        calibration_rows=int(cal.sum()),
        evaluation_sources=len(grouped),
        evaluation_rows=len(test),
        selected_clock=clock,
        calibration_micro=(
            outcomes.loc[cal].sum() / meta.tokens_valid[cal].sum()
        ).to_dict(),
        evaluation_micro=(outcomes.loc[~cal].sum() / test.tokens_valid.sum()).to_dict(),
        fixed_exact=int((test.clock_correct == test.tokens_valid).sum()),
        oracle_exact=int((test.oracle_correct == test.tokens_valid).sum()),
        headroom_pp=100 * test.delta.sum() / test.tokens_valid.sum(),
        headroom_95ci_pp=np.percentile(
            100 * sums[:, 0] / sums[:, 1], [2.5, 97.5]
        ).tolist(),
    )
    output = Path("analysis/grouped_clock_seed0.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
