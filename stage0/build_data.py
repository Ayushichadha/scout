#!/usr/bin/env python3
"""Stage 0 data builder: upstream HRM ARC builder (unchanged) with N augmentations, written to a NEW folder.

* Uses the repo's own dataset/build_arc_dataset.py (read-only import), seed 42, default folders
  (ARC-AGI/data + ConceptARC/corpus), directory listing sorted (so the build is reproducible on any machine).
* Afterwards writes <out>/split.npz: the same deterministic calibration/final split logic as the Stage A study
  (Philox seed 20260819, 20% calibration) over the held-out ("test") episodes, plus a fixed evaluation subset
  used by train_eval.py (so evaluation during training stays cheap even with 300 augmentations).
"""
import argparse, os, sys, glob as globmod
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--hrm", required=True, help="path to the HRM folder of the repo copy (contains dataset/)")
ap.add_argument("--raw", required=True, help="folder containing ARC-AGI/data and ConceptARC/corpus")
ap.add_argument("--out", required=True)
ap.add_argument("--num-aug", type=int, default=300)
ap.add_argument("--dirs", default="ARC-AGI/data,ConceptARC/corpus")
ap.add_argument("--eval-subset", type=int, default=2000, help="size of the fixed evaluation subset of the final split")
ap.add_argument("--split-seed", type=int, default=20260819)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.hrm) / "dataset"))
import build_arc_dataset as B

_scandir, _glob = os.scandir, globmod.glob
class _SD:
    def __init__(self, p): self.items = sorted(list(_scandir(p)), key=lambda e: e.name)
    def __iter__(self): return iter(self.items)
    def __enter__(self): return self
    def __exit__(self, *x): pass
B.os.scandir = lambda p: _SD(p)
B.glob = lambda pat: sorted(_glob(pat), key=lambda p: os.path.basename(p))

cfg = B.DataProcessConfig(dataset_dirs=[str(Path(a.raw) / d) for d in a.dirs.split(",")],
                          output_dir=a.out, seed=42, num_aug=a.num_aug)
B.convert_dataset(cfg)

# ---- split (same logic as scripts/run_adaptive_matched_budget.deterministic_split) ----
out = Path(a.out)
n = len(np.load(out / "test" / "all__inputs.npy", mmap_mode="r"))
n_cal = min(max(int(round(n * 0.20)), 1), n - 1)
perm = np.random.Generator(np.random.Philox(a.split_seed)).permutation(n)
is_cal = np.zeros(n, bool); is_cal[perm[:n_cal]] = True
final_idx = np.flatnonzero(~is_cal)
rng = np.random.default_rng(a.split_seed + 1)
k = min(a.eval_subset, len(final_idx))
eval_idx = np.sort(rng.choice(final_idx, size=k, replace=False))
np.savez(out / "split.npz", is_calibration=is_cal, final_idx=final_idx, eval_idx=eval_idx)
print(f"held-out episodes {n}: calibration {int(is_cal.sum())} / final {len(final_idx)}; fixed eval subset {k}")
