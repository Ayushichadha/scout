"""Replay the original 64 diagnostic episodes with a single forced revision or none."""

import csv
from pathlib import Path
import time

import torch

from scripts import run_frozen_schedule_control as control
from scripts.run_adaptive_matched_budget import _batch_episodes, _set_threshold


def main():
    torch.set_num_threads(2)
    start = time.monotonic()
    if control.sha256_file(control.CHECKPOINT) != control.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Historical checkpoint mismatch")
    config = control.load_config(control.CONFIG)
    metadata, episodes = control.load_held_out_episodes(
        control.dataset_path_from_config(config)
    )
    model = control.build_model(config, metadata, 4, "cpu")
    control.load_weights(model, control.CHECKPOINT, "cpu")
    model.eval()
    path = Path("analysis/counterfactual_persistence_v2_diagnostic/decision_states.csv")
    with path.open() as file:
        states = {
            (row["episode_id"], int(row["pass_position"])): row
            for row in csv.DictReader(file)
        }
    records = []
    with torch.inference_mode():
        for batch, valid in _batch_episodes(
            episodes[:64], batch_size=4, metadata=metadata
        ):
            labels = batch["labels"]
            for revision in (None, *range(2, 8)):
                carry = model.model.initial_carry(batch)
                for step in range(1, 9):
                    _set_threshold(model, 0.0 if step == revision else 1.0)
                    carry, outputs = model.model(carry, batch)
                    assert (
                        outputs["subgoal_updated"].bool() == (step in (1, revision))
                    ).all()
                solved = (
                    (outputs["logits"].argmax(-1) == labels)
                    | (labels == control.IGNORE_LABEL_ID)
                ).all(-1)
                if revision is None:
                    persist = solved
                    continue
                for i, episode in enumerate(valid):
                    row = states[episode.episode_id, revision]
                    records.append(
                        dict(
                            episode=episode.episode_id,
                            puzzle_id=episode.puzzle_identifier,
                            step=revision,
                            local_delta=row["advantage"],
                            persist=int(persist[i]),
                            revise=int(solved[i]),
                        )
                    )
    output = Path("outputs/revision-endpoints.csv")
    output.parent.mkdir(exist_ok=True)
    with output.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} pairs in {time.monotonic()-start:.1f}s: {output}")


if __name__ == "__main__":
    main()
