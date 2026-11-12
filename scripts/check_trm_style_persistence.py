"""CPU feasibility probe on a randomly initialized TRM-style shared recurrent core."""

from pathlib import Path
import sys
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HRM"))
from models.subgoal_head import SubgoalHead, SubgoalHeadConfig


class SharedCoreProbe(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.manager = SubgoalHead(SubgoalHeadConfig(hidden_size=width, goal_dim=width))

    def refine(self, inputs, answer, latent, commitment):
        for _ in range(2):
            latent = self.core(inputs + answer + latent + commitment)
        return self.core(answer + latent), latent

    def propose(self, answer):
        return self.manager._compute_gate(answer) * self.manager._compute_goal(answer)


@torch.inference_mode()
def main():
    torch.manual_seed(17)
    width = 16
    model = SharedCoreProbe(width).eval()
    inputs = torch.randn(4, width)
    empty = torch.zeros_like(inputs)
    answer, latent = model.refine(inputs, empty, empty, empty)
    active = model.propose(answer)
    answer, latent = model.refine(inputs, answer, latent, active)
    candidate = model.propose(answer)
    persist, _ = model.refine(inputs, answer, latent, active)
    revise, _ = model.refine(inputs, answer, latent, candidate)
    replay, _ = model.refine(inputs, answer.clone(), latent.clone(), active.clone())
    torch.testing.assert_close(persist, replay, rtol=0, atol=0)
    effect = (persist - revise).abs().amax(dim=-1)
    assert torch.isfinite(effect).all() and (effect > 0).all()
    print("Next-pass max absolute answer changes per example:", effect.tolist())


if __name__ == "__main__":
    main()
