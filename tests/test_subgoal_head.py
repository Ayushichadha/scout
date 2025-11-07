import torch
from scout.subgoals.subgoal_head import SubgoalHead, SubgoalHeadConfig


def test_forward_shapes():
    cfg = SubgoalHeadConfig(d_model=32, num_subgoals=7)
    head = SubgoalHead(cfg)
    z = torch.randn(4, 32)
    logits = head(z)
    assert logits.shape == (4, 7)


def test_forward_with_timepool():
    cfg = SubgoalHeadConfig(d_model=16, num_subgoals=5)
    head = SubgoalHead(cfg)
    z = torch.randn(2, 3, 16)  # B, T, D
    logits = head(z)
    assert logits.shape == (2, 5)
