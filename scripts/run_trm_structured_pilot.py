"""Measure revision headroom on held-out pointer graphs using reference TRM."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.run_reference_trm_scout import PIN, ScoutTRM, load_models


def partition_graphs(seed):
    """Hold out entire six-node permutation graphs before creating queries."""
    graphs = torch.tensor(list(itertools.permutations(range(6))), dtype=torch.long)
    order = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(seed))
    return graphs, dict(
        train=order[:504], development=order[504:612], evaluation=order[612:]
    )


def queries(graphs, ids):
    records = [
        (int(i), start, depth)
        for i in ids
        for start in range(6)
        for depth in range(1, 4)
    ]
    inputs, labels = [], []
    for graph_id, start, depth in records:
        graph = graphs[graph_id].tolist()
        answer = start
        for _ in range(depth):
            answer = graph[answer]
        inputs.append(graph + [start, depth + 6])
        labels.append([answer] + [-100] * 7)
    return {
        "inputs": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "puzzle_identifiers": torch.zeros(len(records), dtype=torch.long),
    }, records


def rollout(model, data, revision, feudal_weight):
    carry, commitment = model.initial(data), None
    auxiliaries = []
    for step in range(1, 5):
        carry, logits, _, auxiliary = model.transition(
            carry, data, commitment, return_auxiliary=True
        )
        if commitment is not None:
            auxiliaries.append(auxiliary)
        if step == 1 or step == revision:
            commitment = model.propose(carry)
    auxiliary = torch.stack(auxiliaries).mean()
    return logits, auxiliary * feudal_weight


@torch.no_grad()
def evaluate(model, data, records):
    predictions = []
    for revision in [None, 2, 3]:
        correct = []
        for offset in range(0, len(records), 64):
            batch = {key: value[offset : offset + 64] for key, value in data.items()}
            logits, _ = rollout(model, batch, revision, 0)
            predicted = logits[:, 0].argmax(-1)
            correct.extend((predicted == batch["labels"][:, 0]).tolist())
        predictions.append(correct)
    outcomes = torch.tensor(predictions).T
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--feudal-weight", type=float, default=0.05)
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/trm-pointer-pilot")
    )
    args = parser.parse_args()
    if args.steps < 1 or args.feudal_weight < 0:
        parser.error("steps must be positive and feudal weight nonnegative")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    reference, scout = load_models(args.source.resolve())
    config = dict(
        batch_size=32,
        seq_len=8,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=10,
        H_cycles=1,
        L_cycles=2,
        H_layers=0,
        L_layers=2,
        hidden_size=64,
        expansion=2,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
    )
    model = ScoutTRM(reference, scout, config)
    graphs, parts = partition_graphs(20261113)
    assert not set(parts["train"].tolist()) & set(parts["development"].tolist())
    assert not set(parts["train"].tolist()) & set(parts["evaluation"].tolist())
    assert not set(parts["development"].tolist()) & set(parts["evaluation"].tolist())
    datasets = {name: queries(graphs, ids) for name, ids in parts.items()}
    train = datasets["train"][0]
    if args.overfit_check:
        # Deterministic examples drawn exclusively from the training partition.
        ids = torch.randperm(
            len(train["inputs"]), generator=torch.Generator().manual_seed(7)
        )[:8]
        train = {key: value[ids] for key, value in train.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(args.seed + 1)
    start = time.perf_counter()
    loss_history = []
    for step in range(args.steps):
        indices = torch.randint(len(train["inputs"]), (32,), generator=generator)
        batch = {key: value[indices] for key, value in train.items()}
        # Balance fixed schedules during training, independent of task labels.
        revision = [None, 2, 3][step % 3]
        logits, auxiliary = rollout(model, batch, revision, args.feudal_weight)
        loss = (
            F.cross_entropy(logits.reshape(-1, 10), batch["labels"].reshape(-1))
            + auxiliary
        )
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
        if step % 250 == 0:
            loss_history.append(dict(step=step, loss=float(loss.detach())))
    model.eval()
    if args.overfit_check:
        outcomes = evaluate(model, train, list(range(8)))
        report = dict(
            scope="eight-example training-only overfit diagnostic",
            reference_commit=PIN,
            seed=args.seed,
            steps=args.steps,
            feudal_weight=args.feudal_weight,
            exact=outcomes.float().mean(0).tolist(),
            loss_history=loss_history,
            wall_seconds=time.perf_counter() - start,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    dev_data, dev_records = datasets["development"]
    dev = evaluate(model, dev_data, dev_records)
    best = int(dev.float().mean(0).argmax())
    opportunity = int((dev.any(1) & ~dev[:, best]).sum())
    # Predeclared pilot gate: above chance competence and at least 20 rescues.
    gate = bool(dev[:, best].float().mean() >= 0.25 and opportunity >= 20)
    report = dict(
        reference_commit=PIN,
        seed=args.seed,
        steps=args.steps,
        config=config,
        feudal_weight=args.feudal_weight,
        graph_split_seed=20261113,
        graph_ids={k: v.tolist() for k, v in parts.items()},
        loss_history=loss_history,
        development_rates=dev.float().mean(0).tolist(),
        schedules=["persist", "revise_after_2", "revise_after_3"],
        development_selected_schedule=best,
        development_oracle_rescues=opportunity,
        development_gate_passed=gate,
        gate_definition="development exact >=25% and >=20 oracle rescues",
        scope="six-node permutation pointer following; held-out graphs; official TRM with Scout",
    )
    rows = []
    for i, record in enumerate(dev_records):
        rows.append(
            dict(
                split="development",
                graph=record[0],
                start=record[1],
                depth=record[2],
                persist=int(dev[i, 0]),
                revise2=int(dev[i, 1]),
                revise3=int(dev[i, 2]),
            )
        )
    if gate:
        data, records = datasets["evaluation"]
        outcomes = evaluate(model, data, records)
        report["evaluation_rates"] = outcomes.float().mean(0).tolist()
        report["evaluation_oracle_rescues"] = int(
            (outcomes.any(1) & ~outcomes[:, best]).sum()
        )
        for i, record in enumerate(records):
            rows.append(
                dict(
                    split="evaluation",
                    graph=record[0],
                    start=record[1],
                    depth=record[2],
                    persist=int(outcomes[i, 0]),
                    revise2=int(outcomes[i, 1]),
                    revise3=int(outcomes[i, 2]),
                )
            )
    report["wall_seconds"] = time.perf_counter() - start
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "outcomes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.output / "weights.pt")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["graph_ids", "config"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
