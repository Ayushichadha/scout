"""CPU step-time for Sudoku-shaped (81 cells) and Maze-shaped (900 cells) inputs, plain HRM model, random tokens (timing only)."""
import sys, time, os, torch
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HRM")); sys.path.append(R); os.chdir(R)
from utils.functions import load_model_class
for name, L, V, h, nl, c, B in [("sudoku", 81, 11, 128, 2, 1, 64), ("sudoku", 81, 11, 256, 2, 1, 64), ("sudoku", 81, 11, 512, 4, 2, 64),
                                ("maze", 900, 6, 128, 2, 1, 32), ("maze", 900, 6, 256, 2, 1, 32)]:
    cfg = dict(batch_size=B, seq_len=L, vocab_size=V, num_puzzle_identifiers=1, puzzle_emb_ndim=h, H_cycles=c, L_cycles=c, H_layers=nl, L_layers=nl,
               hidden_size=h, expansion=4, num_heads=max(2, h // 64), pos_encodings="rope", halt_max_steps=8, halt_exploration_prob=0.0,
               fixed_refinement_steps=8, forward_dtype="float32", subgoal_head=None, causal=False)
    m = load_model_class("losses@ACTLossHead")(load_model_class("hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1")(cfg), loss_type="stablemax_cross_entropy", feudal_loss_weight=0.0, intervention_weight=0.0)
    opt = torch.optim.AdamW([p for n_, p in m.named_parameters() if "puzzle_emb" not in n_], lr=1e-4)
    batch = {"inputs": torch.randint(1, V, (B, L), dtype=torch.int32), "labels": torch.randint(1, V, (B, L), dtype=torch.int32), "puzzle_identifiers": torch.zeros(B, dtype=torch.int32)}
    carry = m.initial_carry(batch); m.train(); ts = []
    for i in range(8):
        t = time.time(); carry, loss, *_ = m(carry=carry, batch=batch, return_keys=[]); (loss / B).backward(); opt.step(); opt.zero_grad(); ts.append(time.time() - t)
    t = sorted(ts[3:])[len(ts[3:]) // 2]
    print(f"{name} seq {L} h{h} {nl}+{nl} layers cycles {c} batch {B}: {t:.3f} s/step -> {B / t:.0f} example-passes/s", flush=True)
