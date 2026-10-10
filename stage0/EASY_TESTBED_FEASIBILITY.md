# Easier test-bed: Sudoku-Extreme / Maze-Hard (report only, 10 Oct 2026)

**Builders in the repo copy [measured, read from files]:** `HRM/dataset/build_sudoku_dataset.py` (downloads `sapientinc/sudoku-extreme` from Hugging Face:
train.csv 719 MB, test.csv 79 MB; options `--subsample-size`, `--num-aug`, `--min-difficulty`) and `HRM/dataset/build_maze_dataset.py` (downloads `sapientinc/maze-30x30-hard-1k`,
1.8 MB per split). Both need `huggingface_hub` (not in the box venv; I installed it into the box venv only). The Maze build **ran on the box** in seconds
(1,000 train groups, seq 900, vocab 6, one puzzle id, no augmentation) into `stage0/data/maze-1k`. The Sudoku build was not run (719 MB download plus a CPU-heavy parse).
Sudoku: 81 cells, vocab 11.

**CPU step time on this box, plain HRM model, float32, random tokens [measured, `seq_timing.py`]:**
| Task | Config | Example-passes/s |
|---|---|---|
| Sudoku (81 cells) | h128, 2+2 layers, 1x1 | 505 |
| Sudoku | h256, 2+2 layers, 1x1 | 249 |
| Sudoku | h512, 4+4 layers, 2x2 (HRM size) | 19 |
| Maze (900 cells) | h128, 2+2 layers, 1x1 | 35 |
| Maze | h256, 2+2 layers, 1x1 | 14 |
For comparison ARC (901 tokens) was 10 at h128 and 3.9 at h256. Sudoku is about 25x cheaper per example than ARC; Maze costs the same as ARC per example.

**What a "small" run would need [estimate, not measured to convergence].** HRM's published Sudoku-1k recipe is `epochs=20000`, batch 384, about 5x10^4 steps, about 2x10^7 example-passes.
At 82 tokens the HRM-size model needs about 2.8x10^10 FLOPs per example-pass, so about 6x10^17 FLOPs: roughly 1 H100-hour at the effective 180 TFLOP/s used in COMPUTE_PLAN.md, and
a few hours on a mid-range GPU (HRM's README says about 10 h on an RTX 4070 laptop GPU, which is consistent in order of magnitude). A h256 model costs about 10x less per example-pass:
2x10^7 example-passes would take about 22 h on this CPU (249/s), about 15-20 min on an RTX 4090 and a few hours on a Kaggle P100 [estimate].
**Whether a small model reaches non-trivial accuracy within that budget is unverified:** TRM reports 87% on Sudoku-Extreme with a 5M model but needs about 18 h on one L40S (its README); HRM reports 55%.
I know of no published small-budget learning curve. Quick CPU-only (a few hours) runs can probably show rising per-cell accuracy on blanks, but exact-grid accuracy needs the GPU-scale budget.
Maze at seq 900 is expensive per example (HRM README: about 1 h on 8 GPUs; TRM: under 24 h on 4 L40S); Sudoku is the cheaper test-bed.

**To use either with `train_eval.py`:** it is ARC-specific (puzzle-id table, `split.npz`, EOS/colour metrics). A Sudoku/Maze mode needs about 30-60 minutes of driver work: different metrics
(cell accuracy on blanks, exact grid) and the builder's own train/test folders. Not done.
