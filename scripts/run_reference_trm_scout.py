"""Scout persistence on the official TRM recurrent core, with a CPU task pilot."""

import argparse
import csv
import hashlib
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PIN = "c01103738605ba39d1430519b1ee0c62f4c707f8"


def load_models(source):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise ValueError(f"Expected reference TRM {PIN}, found {revision}")
    sys.path.insert(0, str(source))
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1,
    )

    spec = importlib.util.spec_from_file_location(
        "scout_subgoal_reference", ROOT / "HRM/models/subgoal_head.py"
    )
    scout = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scout
    spec.loader.exec_module(scout)
    loss_spec = importlib.util.spec_from_file_location(
        "scout_losses_reference", ROOT / "HRM/models/losses.py"
    )
    losses = importlib.util.module_from_spec(loss_spec)
    loss_spec.loader.exec_module(losses)
    scout.feudal_loss = losses.feudal_loss
    return TinyRecursiveReasoningModel_ACTV1, scout


@dataclass
class Commitment:
    goal: torch.Tensor
    gate: torch.Tensor
    emission: torch.Tensor
    anchor: torch.Tensor


class ScoutTRM(nn.Module):
    """Inject one commitment only into worker updates of the reference core.

    This wrapper calls the unchanged reference inner forward. A temporary hook
    adds gate*goal to its input injection on L updates, skipping the answer update.
    Calls must be sequential because the temporary hook is scoped to a transition.
    """

    def __init__(self, reference, scout, config):
        super().__init__()
        self.reference = reference(config)
        self.feudal_loss = scout.feudal_loss
        self.manager = scout.SubgoalHead(
            scout.SubgoalHeadConfig(
                hidden_size=config["hidden_size"], goal_dim=config["hidden_size"]
            )
        )

    def initial(self, batch):
        inner = self.reference.inner
        flags = torch.ones(len(batch["inputs"]), dtype=torch.bool)
        return inner.reset_carry(flags, inner.empty_carry(len(flags)))

    def propose(self, carry):
        emission = carry.z_H[:, 0].detach()
        return Commitment(
            self.manager._compute_goal(emission).detach(),
            self.manager._compute_gate(emission).detach(),
            emission,
            carry.z_L.mean(dim=1).detach(),
        )

    def transition(
        self, carry, batch, commitment=None, inject=True, return_auxiliary=False
    ):
        inner = self.reference.inner
        hooks = []
        counter = 0
        worker_output = None
        consuming = inject and commitment is not None
        if consuming:
            # A zero-valued recomputation supplies a local manager gradient.
            # Cached goals, gates, and emission anchors persist as detached values.
            fresh_goal = self.manager._compute_goal(commitment.emission)
            fresh_gate = self.manager._compute_gate(commitment.emission)
            goal = commitment.goal + (fresh_goal - fresh_goal.detach())
            gate = commitment.gate + (fresh_gate - fresh_gate.detach())

            def add_goal(module, args, kwargs):
                nonlocal counter
                worker = counter % (inner.config.L_cycles + 1) < inner.config.L_cycles
                counter += 1
                if worker:
                    injection = args[1] + (gate * goal).unsqueeze(1)
                    return (args[0], injection), kwargs
                return args, kwargs

            def capture_worker(module, args, output):
                nonlocal worker_output
                worker = (counter - 1) % (
                    inner.config.L_cycles + 1
                ) < inner.config.L_cycles
                if worker:
                    worker_output = output

            hooks.append(
                inner.L_level.register_forward_pre_hook(add_goal, with_kwargs=True)
            )
            if return_auxiliary:
                hooks.append(inner.L_level.register_forward_hook(capture_worker))
        try:
            result = inner(carry, batch)
            if not return_auxiliary:
                return result
            if consuming:
                assert worker_output is not None
                auxiliary = self.feudal_loss(
                    worker_state=worker_output,
                    manager_goal=goal,
                    gate=gate,
                    anchor=commitment.anchor,
                    active_mask=torch.ones(len(batch["inputs"]), dtype=torch.bool),
                    reduction="mean",
                )
            else:
                # The bootstrap pass has not consumed any commitment.
                auxiliary = result[1].sum() * 0
            return (*result, auxiliary)
        finally:
            for hook in hooks:
                hook.remove()


def batch(pattern_ids):
    """Decode all eight-bit task patterns without sampling duplicate inputs."""
    pattern_ids = torch.as_tensor(pattern_ids, dtype=torch.long)
    bits = (pattern_ids.unsqueeze(1) >> torch.arange(7)) & 1
    reverse = ((pattern_ids >> 7) & 1).unsqueeze(1)
    inputs = torch.cat((reverse + 2, bits), dim=1)
    target = torch.where(reverse.bool(), bits.flip(1), bits)
    labels = torch.cat((reverse + 2, target), dim=1)
    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": torch.zeros(len(pattern_ids), dtype=torch.long),
    }


def partitions():
    generator = torch.Generator().manual_seed(20261112)
    ids = torch.randperm(256, generator=generator)
    train, dev, test = ids[:192], ids[192:224], ids[224:]
    assert len(set(train.tolist()) | set(dev.tolist()) | set(test.tolist())) == 256
    return train, dev, test


def rollout(model, data, revision=2, inject=True, return_auxiliary=False):
    carry = model.initial(data)
    goal = None
    auxiliary_losses = []
    for step in range(1, 5):
        if return_auxiliary:
            carry, logits, _, auxiliary = model.transition(
                carry, data, goal, inject=inject, return_auxiliary=True
            )
            if goal is not None and inject:
                auxiliary_losses.append(auxiliary)
        else:
            carry, logits, _ = model.transition(carry, data, goal, inject=inject)
        if step == 1 or step == revision:
            goal = model.propose(carry)
    if return_auxiliary:
        auxiliary = (
            torch.stack(auxiliary_losses).mean()
            if auxiliary_losses
            else logits.sum() * 0
        )
        return logits, auxiliary
    return logits


def verify(model):
    data = batch([1, 2, 131, 167])
    carry = model.initial(data)
    base = model.reference.inner(carry, data)
    disabled = model.transition(carry, data, inject=False)
    assert torch.equal(base[1], disabled[1])
    carry = base[0]
    goal = model.propose(carry)
    disabled_with_cache = model.transition(carry, data, goal, inject=False)
    assert torch.equal(disabled_with_cache[1], model.reference.inner(carry, data)[1])
    assert not goal.goal.requires_grad and not goal.emission.requires_grad
    assert not goal.anchor.requires_grad
    assert torch.equal(goal.anchor, carry.z_L.mean(dim=1))
    bootstrap_auxiliary = model.transition(
        model.initial(data), data, return_auxiliary=True
    )[3]
    assert bootstrap_auxiliary.item() == 0
    disabled_auxiliary = model.transition(
        carry, data, goal, inject=False, return_auxiliary=True
    )[3]
    assert disabled_auxiliary.item() == 0
    decision_carry = model.transition(carry, data, goal)[0]
    replacement = model.propose(decision_carry)
    a = model.transition(decision_carry, data, goal)
    b = model.transition(decision_carry, data, goal)
    assert torch.equal(a[1], b[1])
    replaced = model.transition(decision_carry, data, replacement)
    delta = float((a[1] - replaced[1]).abs().max().detach())
    assert delta > 0
    F.cross_entropy(a[1].reshape(-1, 4), data["labels"].reshape(-1)).backward()
    for module in [model.manager, model.reference.inner.L_level]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    model.zero_grad(set_to_none=True)
    auxiliary = model.transition(decision_carry, data, goal, return_auxiliary=True)[3]
    assert torch.isfinite(auxiliary) and auxiliary.item() >= 0
    auxiliary.backward()
    for module in [model.manager, model.reference.inner.L_level]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    model.zero_grad(set_to_none=True)
    return {
        "disabled_equals_reference": True,
        "cached_replay_equal": True,
        "replacement_logit_delta": delta,
        "manager_and_core_gradient": True,
        "directional_auxiliary_gradient": True,
        "bootstrap_and_disabled_auxiliary_zero": True,
        "anchor_matches_emission_worker": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--disable-injection", action="store_true")
    parser.add_argument("--feudal-weight", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reference-trm-scout")
    )
    args = parser.parse_args()
    if args.feudal_weight < 0 or not torch.isfinite(torch.tensor(args.feudal_weight)):
        parser.error("--feudal-weight must be finite and nonnegative")
    if args.disable_injection and args.feudal_weight > 0:
        parser.error("Positive feudal loss requires consumed goal injection")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    reference, scout = load_models(args.source.resolve())
    config = dict(
        batch_size=32,
        seq_len=8,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=4,
        H_cycles=1,
        L_cycles=2,
        H_layers=0,
        L_layers=2,
        hidden_size=32,
        expansion=2,
        num_heads=2,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
    )
    model = ScoutTRM(reference, scout, config)
    checks = verify(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start = time.perf_counter()
    train_ids, dev_ids, test_ids = partitions()
    training_generator = torch.Generator().manual_seed(args.seed + 1)
    for step in range(args.steps):
        indices = torch.randint(len(train_ids), (32,), generator=training_generator)
        data = batch(train_ids[indices])
        if args.feudal_weight > 0:
            logits, auxiliary = rollout(model, data, inject=True, return_auxiliary=True)
        else:
            logits = rollout(model, data, inject=not args.disable_injection)
            auxiliary = logits.sum() * 0
        loss = F.cross_entropy(logits.reshape(-1, 4), data["labels"].reshape(-1))
        loss = loss + args.feudal_weight * auxiliary
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    scores = {}
    predictions = []
    with torch.no_grad():
        dev = batch(dev_ids)
        test = batch(test_ids)
        for name, data in [("development", dev), ("evaluation", test)]:
            ids = dev_ids if name == "development" else test_ids
            scores[name] = {}
            for revision in [None, 2, 3]:
                logits = rollout(
                    model, data, revision, inject=not args.disable_injection
                )
                predicted = logits.argmax(-1)
                for index, pattern in enumerate(ids.tolist()):
                    predictions.append(
                        {
                            "split": name,
                            "pattern_id": pattern,
                            "revision": str(revision),
                            "prediction": "".join(map(str, predicted[index].tolist())),
                            "target": "".join(map(str, data["labels"][index].tolist())),
                            "exact": bool(
                                (predicted[index] == data["labels"][index]).all()
                            ),
                        }
                    )
                scores[name][str(revision)] = float(
                    (logits.argmax(-1) == data["labels"]).all(-1).float().mean()
                )
    result = dict(
        reference_commit=PIN,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config=config,
        seed=args.seed,
        steps=args.steps,
        checks=checks,
        exact=scores,
        goal_injection=not args.disable_injection,
        directional_auxiliary_loss=args.feudal_weight > 0,
        feudal_weight=args.feudal_weight,
        partition_seed=20261112,
        pattern_ids={
            "train": train_ids.tolist(),
            "development": dev_ids.tolist(),
            "evaluation": test_ids.tolist(),
        },
        wall_seconds=time.perf_counter() - start,
        scope="synthetic copy/reverse task with exhaustive disjoint input partitions; "
        "engineering pilot with local reconstructed manager gradients",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.state_dict(), args.output / "weights.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
