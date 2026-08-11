"""Verify parameter count is identical across all 8 flag combinations."""

import sys
import itertools

sys.path.insert(0, "HRM")

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead

BASE_CFG = dict(
    batch_size=4,
    seq_len=16,
    puzzle_emb_ndim=0,
    num_puzzle_identifiers=10,
    vocab_size=100,
    H_cycles=2,
    L_cycles=2,
    H_layers=2,
    L_layers=2,
    hidden_size=64,
    expansion=2,
    num_heads=4,
    pos_encodings="rope",
    halt_max_steps=4,
    halt_exploration_prob=0.1,
    subgoal_head=dict(
        hidden_size=64,
        goal_dim=64,
        manager_period=4,
    ),
)

counts = {}
for inject, align, rand in itertools.product([True, False], repeat=3):
    cfg = {**BASE_CFG}
    cfg["subgoal_head"] = {
        **BASE_CFG["subgoal_head"],
        "inject_subgoal": inject,
        "use_alignment_loss": align,
        "random_directions": rand,
    }
    inner = HierarchicalReasoningModel_ACTV1(cfg)
    model = ACTLossHead(
        inner, loss_type="stablemax_cross_entropy", feudal_loss_weight=0.1
    )
    n = sum(p.numel() for p in model.parameters())
    key = (inject, align, rand)
    counts[key] = n
    print(f"inject={inject!s:5} align={align!s:5} rand={rand!s:5}  params={n:,}")

values = list(counts.values())
assert all(v == values[0] for v in values), f"PARAM COUNT MISMATCH: {counts}"
print(f"\nAll 8 combinations: {values[0]:,} parameters. PASS.")
