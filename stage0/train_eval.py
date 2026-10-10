#!/usr/bin/env python3
"""Stage 0 driver: plain HRM-style model (NO goals, NO controller) trained and evaluated end to end.

Uses the repo's own model (models/hrm/hrm_act_v1.py with subgoal_head=None), loss head, dataset class and sparse
puzzle-embedding optimiser, read-only from a repo copy. Logs a CSV row every --eval-every steps with
 micro token accuracy, non-background colour accuracy, changed-cell accuracy, exact match
 and the copy-input baseline for the same episodes.

Definitions (valid tokens = target-grid colour cells + the EOS border; PAD ignored; token id = colour + 2, EOS = 1):
 micro      = correct valid tokens / valid tokens (summed over episodes)
 nonbg      = accuracy on valid colour cells whose label is not colour 0 (token 2) and not EOS
 changed    = accuracy on valid cells whose label differs from the input cell at the same position
 exact      = fraction of episodes with every valid token right
 copy_*     = the same metrics for the prediction "output = input" (changed-cell accuracy is 0 by construction)
One 'step' = one refinement pass over a batch (as in HRM); an example needs --passes steps (default 8) to finish.
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--hrm", required=True, help="HRM folder of the repo copy (read-only)")
p.add_argument("--data", required=True, help="folder made by build_data.py")
p.add_argument("--out", required=True, help="output folder (metrics.csv, config.json, ckpt.pt)")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"],
               help="auto = bfloat16 on CUDA GPUs that support it, else float32 (P100/T4/CPU)")
p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA >= 7.0 only)")
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--layers", type=int, default=4, help="layers per level (H and L)")
p.add_argument("--cycles", type=int, default=2, help="H_cycles = L_cycles")
p.add_argument("--heads", type=int, default=8)
p.add_argument("--expansion", type=float, default=4)
p.add_argument("--passes", type=int, default=8, help="fixed refinement passes per example")
p.add_argument("--batch", type=int, default=768)
p.add_argument("--steps", type=int, default=1000, help="total training steps (refinement passes)")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=2000)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--puzzle-lr", type=float, default=1e-2)
p.add_argument("--puzzle-wd", type=float, default=0.1)
p.add_argument("--optim", default="adamw", choices=["adamw", "adam_atan2"],
               help="adamw works everywhere; adam_atan2 needs the adam-atan2 package (HRM's choice)")
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--eval-n", type=int, default=2000, help="episodes in the fixed evaluation subset")
p.add_argument("--eval-batch", type=int, default=128)
p.add_argument("--eval-set", default="final", choices=["final", "calibration"])
p.add_argument("--threads", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
a = p.parse_args()

# repo modules are imported read-only; real adam_atan2 (site-packages) must win over the repo's CPU stub
a.data = str(Path(a.data).resolve()); a.out = str(Path(a.out).resolve())
repo = str(Path(a.hrm).resolve())
sys.path.append(repo)
os.chdir(repo)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, IGNORE_LABEL_ID
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
if dev == "cuda" and not torch.cuda.is_available():
    sys.exit("CUDA requested but not available")
dtype = a.dtype
if dtype == "auto":
    dtype = "bfloat16" if (dev == "cuda" and torch.cuda.is_bf16_supported()) else "float32"
if dev == "cpu" and dtype == "bfloat16":
    print("note: bfloat16 on CPU is slow and numerically unreliable on some Xeons; using float32"); dtype = "float32"
if a.threads: torch.set_num_threads(a.threads)
torch.manual_seed(a.seed); np.random.seed(a.seed)

out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
data = Path(a.data).resolve()

# ---------------- data ----------------
ds = PuzzleDataset(PuzzleDatasetConfig(seed=a.seed, dataset_path=str(data), global_batch_size=a.batch,
                                       test_set_mode=False, epochs_per_iter=20, rank=0, num_replicas=1), split="train")
md = ds.metadata
tin = np.load(data / "test/all__inputs.npy", mmap_mode="r"); tlab = np.load(data / "test/all__labels.npy", mmap_mode="r")
tpid = np.load(data / "test/all__puzzle_identifiers.npy"); tpi = np.load(data / "test/all__puzzle_indices.npy")
sp = np.load(data / "split.npz")
pool = sp["final_idx"] if a.eval_set == "final" else np.flatnonzero(sp["is_calibration"])
if a.eval_n and a.eval_n < len(pool):
    ev = np.sort(np.random.default_rng(12345).choice(pool, a.eval_n, replace=False))   # fixed subset, same every run
else:
    ev = np.sort(pool)
ev_in = np.asarray(tin[ev]).astype(np.int64); ev_lab = np.asarray(tlab[ev]).astype(np.int64)
ev_pid = tpid[np.searchsorted(tpi, ev, side="right") - 1].astype(np.int64)
ev_lab = np.where(ev_lab == 0, IGNORE_LABEL_ID, ev_lab)
EOS = 1

def cell_masks(inp, lab):
    valid = lab != IGNORE_LABEL_ID
    return dict(valid=valid, changed=valid & (lab != inp), nonbg=valid & (lab != EOS) & (lab != 2))

def score(pred, inp, lab):
    m = cell_masks(inp, lab); corr = m["valid"] & (pred == lab)
    n = {k: float(v.sum()) for k, v in m.items()}
    return dict(micro=corr.sum() / n["valid"], nonbg=(corr & m["nonbg"]).sum() / max(n["nonbg"], 1),
                changed=(corr & m["changed"]).sum() / max(n["changed"], 1),
                exact=float((corr.sum(1) == m["valid"].sum(1)).mean()),
                n_valid=n["valid"], n_nonbg=n["nonbg"], n_changed=n["changed"])
copy = score(ev_in, ev_in, ev_lab)
print(f"eval subset: {len(ev)} episodes ({a.eval_set} split); copy-input micro {copy['micro']:.4f}, "
      f"non-bg {copy['nonbg']:.4f}, changed {copy['changed']:.4f}, exact {copy['exact']:.4f}", flush=True)

# ---------------- model ----------------
cfg = dict(batch_size=a.batch, seq_len=md.seq_len, vocab_size=md.vocab_size, num_puzzle_identifiers=md.num_puzzle_identifiers,
           puzzle_emb_ndim=a.hidden, H_cycles=a.cycles, L_cycles=a.cycles, H_layers=a.layers, L_layers=a.layers,
           hidden_size=a.hidden, expansion=a.expansion, num_heads=a.heads, pos_encodings="rope",
           halt_max_steps=a.passes, halt_exploration_prob=0.0, fixed_refinement_steps=a.passes,
           forward_dtype=dtype, subgoal_head=None, causal=False)
core = load_model_class("hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1")(cfg)
model = load_model_class("losses@ACTLossHead")(core, loss_type="stablemax_cross_entropy",
                                                feudal_loss_weight=0.0, intervention_weight=0.0)
assert core.subgoal_head is None, "goals/controller must be off"
if dev == "cuda": model = model.cuda()
if a.compile and dev == "cuda": model = torch.compile(model, dynamic=False)
pe = core.puzzle_emb; pe_ids = {id(q) for q in pe.parameters()}
if a.optim == "adam_atan2":
    from adam_atan2 import AdamATan2 as Dense
else:
    from torch.optim import AdamW as Dense
opts = [CastedSparseEmbeddingSignSGD_Distributed(list(pe.parameters()) + list(pe.buffers()), lr=0,
                                                 weight_decay=a.puzzle_wd, world_size=1),
        Dense([q for q in model.parameters() if id(q) not in pe_ids], lr=0, weight_decay=a.wd, betas=(0.9, 0.95))]
base_lrs = [a.puzzle_lr, a.lr]
n_dense = sum(q.numel() for q in model.parameters() if id(q) not in pe_ids)
n_emb = pe.weights.numel()
print(f"device {dev}, dtype {dtype}, dense params {n_dense:,}, puzzle-embedding table {n_emb:,} "
      f"({md.num_puzzle_identifiers:,} ids x {a.hidden}), torch {torch.__version__}", flush=True)

def lr_at(base, step):
    return base * min(1.0, step / max(1, a.warmup)) if a.warmup else base

# ---------------- eval ----------------
@torch.inference_mode()
def evaluate():
    model.eval()
    preds = np.zeros_like(ev_in); lm = 0.0
    for i in range(0, len(ev), a.eval_batch):
        s = slice(i, min(i + a.eval_batch, len(ev))); nv = s.stop - s.start
        def pad(x, v):
            x = torch.as_tensor(x[s]); 
            if nv < a.eval_batch: x = torch.cat([x, torch.full((a.eval_batch - nv,) + tuple(x.shape[1:]), v, dtype=x.dtype)])
            return x.to(torch.int32).to(dev)
        batch = {"inputs": pad(ev_in, 0), "labels": pad(ev_lab, IGNORE_LABEL_ID), "puzzle_identifiers": pad(ev_pid, 0)}
        carry = model.initial_carry(batch)
        for _ in range(a.passes):
            carry, _l, _m, outs, _d = model(carry=carry, batch=batch, return_keys=["logits"])
        logits = outs["logits"][:nv].float()
        preds[s] = logits.argmax(-1).cpu().numpy()
    model.train()
    return score(preds, ev_in, ev_lab) | dict(pred_eq_input=float(((preds == ev_in) & (ev_lab != IGNORE_LABEL_ID)).sum() / copy["n_valid"]))

# ---------------- train ----------------
ckpt = out / "ckpt.pt"; step = 0
if a.resume and ckpt.exists():
    st = torch.load(ckpt, map_location=dev if dev == "cuda" else "cpu", weights_only=False)
    model.load_state_dict(st["model"]); [o.load_state_dict(s) for o, s in zip(opts, st["opts"])]; step = st["step"]
    print(f"resumed from step {step}", flush=True)
ds._iters = step
(out / "config.json").write_text(json.dumps(dict(args=vars(a), dtype=dtype, device=dev, dense_params=n_dense,
                                                 emb_table=int(n_emb), copy_input=copy, eval_n=len(ev),
                                                 model="plain HRM (subgoal_head=None), fixed passes"), indent=1, default=float))
cols = ["step", "example_passes", "wall_s", "train_lm_loss", "eval_micro", "eval_nonbg", "eval_changed", "eval_exact",
        "eval_pred_eq_input", "copy_micro", "copy_nonbg", "copy_changed", "copy_exact", "eval_n"]
new = not (out / "metrics.csv").exists() or not a.resume
f = open(out / "metrics.csv", "a" if (a.resume and not new) else "w", newline=""); w = csv.writer(f)
if new: w.writerow(cols)
model.train(); carry = None; t0 = time.time(); loss_acc, loss_n = 0.0, 0
it = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0))

def log_row():
    r = evaluate()
    row = [step, step * a.batch, round(time.time() - t0, 1), round(loss_acc / max(loss_n, 1), 5), r["micro"], r["nonbg"],
           r["changed"], r["exact"], r["pred_eq_input"], copy["micro"], copy["nonbg"], copy["changed"], copy["exact"], len(ev)]
    w.writerow(row); f.flush()
    print(f"step {step:6d} | {row[2]:7.0f}s | loss {row[3]:.4f} | micro {r['micro']:.4f} (copy {copy['micro']:.4f}) | "
          f"nonbg {r['nonbg']:.4f} | changed {r['changed']:.4f} | exact {r['exact']:.4f}", flush=True)
    torch.save(dict(model=model.state_dict(), opts=[o.state_dict() for o in opts], step=step), ckpt)

if step == 0: log_row()
while step < a.steps:
    try: _, batch, _ = next(it)
    except StopIteration: it = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)); continue
    batch = {k: v.to(dev) for k, v in batch.items()}
    if carry is None: carry = model.initial_carry(batch)
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    if not torch.isfinite(loss): print(f"non-finite loss at step {step}, skipping", flush=True); step += 1; continue
    (loss / a.batch).backward()
    step += 1
    for o, b in zip(opts, base_lrs):
        for g in o.param_groups: g["lr"] = lr_at(b, step)
        o.step(); o.zero_grad()
    loss_acc += float(metrics["lm_loss"]) / a.batch; loss_n += 1
    if step % a.eval_every == 0 or step == a.steps:
        log_row(); loss_acc, loss_n = 0.0, 0
f.close(); print("done", flush=True)
