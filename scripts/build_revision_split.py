"""Build a puzzle-grouped evaluation manifest shared by HRM and TRM experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def build_manifest(dataset):
    """Partition the existing test corpus using source names, before scoring."""
    names = json.loads((dataset / "identifiers.json").read_text())
    test = dataset / "test"
    metadata = json.loads((test / "dataset.json").read_text())
    rows = []
    hashes = {}
    for subset in metadata["sets"]:

        def load(field):
            path = test / f"{subset}__{field}.npy"
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            return np.load(path, mmap_mode="r", allow_pickle=False)

        ids = load("puzzle_identifiers")
        puzzles = load("puzzle_indices")
        groups = load("group_indices")
        inputs, labels = load("inputs"), load("labels")
        if (
            inputs.shape != labels.shape
            or len(puzzles) != len(ids) + 1
            or groups[0] != 0
            or groups[-1] != len(ids)
            or puzzles[0] != 0
            or puzzles[-1] != len(inputs)
            or np.any(np.diff(groups.astype(int)) <= 0)
            or np.any(np.diff(puzzles.astype(int)) <= 0)
        ):
            raise ValueError(f"Inconsistent dataset boundaries in {subset}")
        for start, end in zip(groups[:-1], groups[1:]):
            sources = {
                re.sub(r"_t[0-7]_[0-9]{10}$", "", names[i]) for i in ids[start:end]
            }
            if len(sources) != 1 or "<blank>" in sources:
                raise ValueError(f"Ambiguous source IDs in group {start}: {sources}")
            source = sources.pop()
            for i in range(puzzles[start], puzzles[end]):
                rows.append({"set": subset, "row": i, "source": source})

    sources = sorted({row["source"] for row in rows})
    if len(sources) < 2:
        raise ValueError("Need multiple source puzzles")
    rng = np.random.Generator(np.random.Philox(20260819))
    calibration = set(rng.permutation(sources)[: max(1, round(len(sources) * 0.2))])
    for row in rows:
        row["split"] = "calibration" if row["source"] in calibration else "evaluation"
    hashes["identifiers.json"] = hashlib.sha256(
        (dataset / "identifiers.json").read_bytes()
    ).hexdigest()
    return {"seed": 20260819, "source_files_sha256": hashes, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
