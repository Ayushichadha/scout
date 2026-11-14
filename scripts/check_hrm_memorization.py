"""Replay consumed training batches and fine-tune on the first four examples."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts import run_frozen_schedule_control as control
from pretrain import PretrainConfig, create_dataloader, init_train_state, train_batch


def decode(row):
    symbols = {i + 2: str(i) for i in range(10)}
    symbols.update({control.IGNORE_LABEL_ID: ".", 0: "_", 1: "|"})
    return [
        "".join(symbols[token] for token in line)
        for line in row.reshape(30, 30).tolist()
    ]


@torch.no_grad()
def score(model, batches, grid_report=None):
    model.eval()
    correct = tokens = solved = examples = 0
    for batch in batches:
        carry = model.initial_carry(batch)
        for _ in range(8):
            carry, outputs = model.model(carry, batch)
        labels = batch["labels"]
        mask = labels != control.IGNORE_LABEL_ID
        predictions = outputs["logits"].argmax(-1)
        matches = predictions == labels
        if grid_report is not None:
            for target, prediction, identifier in zip(
                labels, predictions, batch["puzzle_identifiers"]
            ):

                scored = target != control.IGNORE_LABEL_ID
                errors = scored & (target != prediction)
                grid_report.append(
                    dict(
                        puzzle_id=int(identifier),
                        target=decode(target),
                        prediction=decode(
                            prediction.masked_fill(~scored, control.IGNORE_LABEL_ID)
                        ),
                        errors=[
                            dict(
                                row=i // 30,
                                col=i % 30,
                                expected=int(target[i]),
                                predicted=int(prediction[i]),
                            )
                            for i in errors.nonzero().flatten().tolist()
                        ],
                    )
                )
        assert mask.any(dim=1).all()
        correct += (matches & mask).sum().item()
        tokens += mask.sum().item()
        solved += (matches | ~mask).all(dim=1).sum().item()
        examples += len(labels)
    return dict(solved=solved, examples=examples, token_accuracy=correct / tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 8:
        parser.error("--steps must be a positive multiple of eight")
    torch.set_num_threads(2)
    raw = control.load_config(control.CONFIG)
    config = PretrainConfig(**yaml.safe_load(control.CONFIG.read_text()))
    torch.manual_seed(config.seed)
    config.data_path = str(control.dataset_path_from_config(raw))
    loader, metadata = create_dataloader(
        config,
        "train",
        0,
        1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    config.epochs = max(config.epochs, args.steps)
    config.max_steps = args.steps
    state = init_train_state(config, metadata, 1)
    assert control.sha256_file(control.CHECKPOINT) == control.EXPECTED_CHECKPOINT_SHA256
    control.load_weights(state.model, control.CHECKPOINT, "cpu")
    if args.resume:
        control.load_weights(state.model, args.resume, "cpu")
    batches = []
    for index, (_, batch, _) in enumerate(loader):
        if index >= 400:
            break
        if index % 8 == 0:
            batches.append(batch)
    assert len(batches) == 50
    print("training replay", json.dumps(score(state.model, batches)), flush=True)
    for step in range(args.steps + 1):
        if step in (0, 80, 200, args.steps):
            report = [] if step == args.steps else None
            print(step, json.dumps(score(state.model, batches[:1], report)), flush=True)
            if report is not None:
                path = Path("outputs/hrm-memorization-grids.json")
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
                torch.save(
                    state.model.state_dict(), "outputs/hrm-memorization-weights.pt"
                )
                Path("outputs/hrm-memorization-run.json").write_text(
                    json.dumps(
                        dict(
                            steps=args.steps,
                            resume=str(args.resume),
                            config=config.model_dump(),
                        ),
                        indent=2,
                    )
                    + "\n"
                )
        if step < args.steps:
            state.model.train()
            metrics = train_batch(config, state, batches[0], 4, 0, 1)
            if metrics is None:
                raise RuntimeError(f"Skipped training step {step + 1}")


if __name__ == "__main__":
    main()
