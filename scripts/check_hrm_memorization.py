"""Replay consumed training batches and fine-tune on the first four examples."""

import json

import torch
import yaml

from scripts import run_frozen_schedule_control as control
from pretrain import PretrainConfig, create_dataloader, init_train_state, train_batch


@torch.no_grad()
def score(model, batches):
    model.eval()
    correct = tokens = solved = examples = 0
    for batch in batches:
        carry = model.initial_carry(batch)
        for _ in range(8):
            carry, outputs = model.model(carry, batch)
        labels = batch["labels"]
        mask = labels != control.IGNORE_LABEL_ID
        matches = outputs["logits"].argmax(-1) == labels
        assert mask.any(dim=1).all()
        correct += (matches & mask).sum().item()
        tokens += mask.sum().item()
        solved += (matches | ~mask).all(dim=1).sum().item()
        examples += len(labels)
    return dict(solved=solved, examples=examples, token_accuracy=correct / tokens)


def main():
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
    state = init_train_state(config, metadata, 1)
    assert control.sha256_file(control.CHECKPOINT) == control.EXPECTED_CHECKPOINT_SHA256
    control.load_weights(state.model, control.CHECKPOINT, "cpu")
    batches = []
    for index, (_, batch, _) in enumerate(loader):
        if index >= 400:
            break
        if index % 8 == 0:
            batches.append(batch)
    assert len(batches) == 50
    print("training replay", json.dumps(score(state.model, batches)), flush=True)
    for step in range(401):
        if step in (0, 80, 200, 400):
            print(step, json.dumps(score(state.model, batches[:1])), flush=True)
        if step < 400:
            state.model.train()
            metrics = train_batch(config, state, batches[0], 4, 0, 1)
            if metrics is None:
                raise RuntimeError(f"Skipped training step {step + 1}")


if __name__ == "__main__":
    main()
