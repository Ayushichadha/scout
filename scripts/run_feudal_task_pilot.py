"""Train a shared-core Scout-goal prototype on synthetic four-class decisions.

This is a task-level engineering pilot, distinct from reference TRM and HRM.
Goals receive full-trajectory gradients rather than HRM's local-credit training.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HRM"))
from models.subgoal_head import SubgoalHead, SubgoalHeadConfig


class GoalReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(8, 32)
        self.core = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.head = nn.Linear(32, 4)
        self.manager = SubgoalHead(SubgoalHeadConfig(hidden_size=32, goal_dim=32))

    def forward(self, inputs, revision):
        context = self.encode(inputs)
        hidden = torch.zeros_like(context)
        goal = torch.zeros_like(context)
        gate = torch.zeros(len(inputs), 1)
        auxiliary = torch.zeros(())
        for step in range(1, 7):
            previous = hidden
            hidden = self.core(context + hidden + gate * goal)
            if step > 1:
                auxiliary = (
                    auxiliary
                    + (1 - F.cosine_similarity(hidden - previous, goal, dim=-1)).mean()
                )
            if step == 1 or step == revision:
                goal = self.manager._compute_goal(hidden)
                gate = self.manager._compute_gate(hidden)
        return self.head(hidden), auxiliary / 5


def data(seed, count):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 8, generator=generator)
    # The two bits are independently verifiable threshold decisions.
    y = (x[:, :4].sum(1) > 0).long() * 2 + (x[:, 4:].sum(1) > 0).long()
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/feudal-task-pilot")
    )
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    train_x, train_y = data(args.seed + 1, 4096)
    dev_x, dev_y = data(args.seed + 2, 512)
    test_x, test_y = data(args.seed + 3, 1024)
    model = GoalReasoner()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for step in range(600):
        indices = torch.randint(len(train_x), (128,))
        logits, auxiliary = model(train_x[indices], 2 + step % 4)
        loss = F.cross_entropy(logits, train_y[indices]) + 0.01 * auxiliary
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            history.append({"step": step, "loss": loss.item()})
    model.eval()
    with torch.inference_mode():
        schedules = [None, 2, 3, 4, 5]
        dev = [model(dev_x, k)[0].argmax(1).eq(dev_y) for k in schedules]
        selected = max(range(len(schedules)), key=lambda i: dev[i].sum().item())
        logits = [model(test_x, k)[0] for k in schedules]
        correct = [out.argmax(1).eq(test_y) for out in logits]
        replay, _ = model(test_x, schedules[selected])
        torch.testing.assert_close(replay, logits[selected], rtol=0, atol=0)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    with (args.output / "outcomes.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "schedule", "prediction", "target", "solved"])
        for k, out, solved in zip(schedules, logits, correct):
            for i in range(len(test_y)):
                writer.writerow(
                    [i, str(k), int(out[i].argmax()), int(test_y[i]), int(solved[i])]
                )
    results = dict(
        scope="synthetic threshold task, shared-core prototype using Scout goal projections",
        training="full-trajectory gradient; random fixed revision schedule; 600 Adam steps",
        feudal_weight=0.01,
        seed=args.seed,
        train_n=4096,
        development_n=512,
        test_n=1024,
        selected_schedule=schedules[selected],
        history=history,
        development_solves={str(k): int(v.sum()) for k, v in zip(schedules, dev)},
        test_solves={str(k): int(v.sum()) for k, v in zip(schedules, correct)},
        causal_class_changes={
            str(k): int((logits[0].argmax(1) != v.argmax(1)).sum())
            for k, v in zip(schedules, logits)
        },
        revision_only_solves={
            str(k): int((v & ~correct[0]).sum()) for k, v in zip(schedules, correct)
        },
        persistence_only_solves={
            str(k): int((correct[0] & ~v).sum()) for k, v in zip(schedules, correct)
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter() - start,
    )
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
