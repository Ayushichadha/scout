# Stage 0: a plain HRM-style model on ARC, end to end (no goals, no controller)

**What this is.** One script that builds the data, trains a plain HRM-style model and evaluates it every N steps.
It writes a CSV you can read like a learning curve. Goals and the re-planning controller are switched off
(`subgoal_head=None`), so any result is about the base model only. The scripts live in this repo under `stage0/`.
They read `HRM/` and write only into the data and run folders you set.

**What it logs** (one row per evaluation, `metrics.csv`), always next to the copy-input baseline on the same episodes:

| Column | Meaning |
|---|---|
| `eval_micro` | share of valid output tokens (target grid cells + the EOS border) that are right. The headline metric. |
| `eval_nonbg` | accuracy on cells whose answer is a coloured cell (not black background, not EOS). |
| `eval_changed` | accuracy on cells where the answer differs from the input cell. Copy-input scores 0 here by construction. |
| `eval_exact` | share of episodes with every token right. |
| `copy_*` | the same four numbers for "output = input". The model must beat `copy_micro` and get `eval_exact` above 0. |
| `eval_pred_eq_input` | share of valid tokens where the model's prediction equals the input. High means "it has learned to copy". |

**Target** (from `COMPUTE_PLAN.md`): `eval_micro` above `copy_micro`, `eval_exact` above 0, and `eval_nonbg` /
`eval_changed` clearly above a copy-everything model (changed-cell accuracy of 5% or more is a first sign of real learning).

## Files
- `build_data.py`: builds the data with N augmentations (default 300) into a new folder, and writes `split.npz` with the same
  deterministic calibration/final split logic as Stage A (Philox seed 20260819, 20% calibration) plus a fixed evaluation subset
  (`eval_idx`). On the final split, `train_eval.py` evaluates that subset.
- `train_eval.py`: the driver (model, optimiser, loop, evaluation, CSV, checkpoint and `--resume`).
- `train_eval.sh <preset>`: wrapper: builds the data if missing, then trains. Presets: `smoke`, `small`, `hrm`.

## Presets (the settings are my first guesses, not tuned)
| Preset | Model | Batch | Steps | Example-passes | Where |
|---|---|---|---|---|---|
| `smoke` | hidden 64, 1+1 layers | 32 | 2,400 | 77k | CPU, about 15 min |
| `small` | hidden 256, 2+2 layers (3.4M weights) | 256 | 40,000 | 10M | free GPU (Kaggle) or a cheap rented GPU |
| `hrm` | hidden 512, 4+4 layers, 2x2 cycles (27M weights) | 768 | 52,000 | 40M (about 10% of the HRM recipe) | rented A100/4090/H100 |

One "step" is one refinement pass over a batch (HRM's convention); each example needs 8 passes. Steps are counted in passes.

## Run it on a free Kaggle GPU (about 15 minutes of your time)
1. **Account.** kaggle.com, free. To use a GPU and the internet in a notebook, Kaggle asks you to verify a phone number.
   (I have not signed up for anything. This is for you to do, or to tell me to prepare.)
2. **New notebook.** Settings: Accelerator = *GPU P100* (or *T4 x2*), Internet = *On*, Persistence = *Files only*.
3. **Get the code and the raw data** (first cell). The Stage 0 scripts are already in this repo at `stage0/`. Clone this branch (`cursor/stage0-scripts-c699`). After the pull request is merged, clone `main` and skip the checkout.
   ```
   !git clone https://github.com/Ayushichadha/scout /kaggle/working/scout
   !cd /kaggle/working/scout && git checkout cursor/stage0-scripts-c699
   !git clone https://github.com/fchollet/ARC-AGI /kaggle/working/raw/ARC-AGI
   !cd /kaggle/working/raw/ARC-AGI && git checkout 3990304
   !git clone https://github.com/victorvikram/ConceptARC /kaggle/working/raw/ConceptARC
   !cd /kaggle/working/raw/ConceptARC && git checkout b22ef52
   !pip -q install einops coolname pydantic argdantic omegaconf hydra-core
   ```
   (The ARC-AGI and ConceptARC commits are the ones used in Stage A. If a URL has moved, the `HRM/.gitmodules` file in the repo names the sources.
   The repo's own `HRM/requirements.txt` lists everything; `wandb`, `adam-atan2` and `huggingface_hub` are not needed here.)
4. **Run** from the repo root (second cell). Kaggle's P100 has no bf16, so use float32; `DTYPE=auto` already picks that on a P100. `REPO` defaults to this repo's `HRM/`.
   ```
   !cd /kaggle/working/scout && RAW=/kaggle/working/raw \
      DATA=/kaggle/working/arc300 OUT=/kaggle/working/run_small DEVICE=cuda \
      bash stage0/train_eval.sh small
   ```
   The first run builds the data (about 2.2 GB, 10-20 minutes, about 6 GB RAM; Kaggle gives about 29 GB). It then trains.
5. **Sessions end** (about 9-12 h at most; weekly GPU quota about 30 h). From the repo root, re-run the same command with `--resume` and the same `OUT=`:
   `RAW=/kaggle/working/raw DATA=/kaggle/working/arc300 OUT=/kaggle/working/run_small DEVICE=cuda bash stage0/train_eval.sh small --resume`.
   It continues from `ckpt.pt`, and `wall_s` keeps counting from the previous session. Save `run_small/metrics.csv`
   (the Output tab, or "Save Version") before the session closes.
6. **Read the CSV.** Open `metrics.csv`; compare `eval_micro` with `copy_micro`, and look at `eval_changed` and `eval_exact`.
   If the training loss (`train_lm_loss`) is not falling after a few thousand steps, stop and tell me; that points to a bug or a
   settings problem, not a lack of compute.

## Run it on a rented GPU (Vast.ai / RunPod / Lambda), about 20 minutes of your time
Only after you have decided to spend money; I will not sign up or pay. A single RTX 4090 or A100 is enough for `small` and `hrm`.
1. Rent one GPU with a PyTorch image (CUDA 12.x). Vast.ai and RunPod bill per second; set a small prepaid top-up as a hard cap.
2. SSH in, then run the same clone/pip lines as above (paths of your choice). From the repo root:
   ```
   RAW=~/raw DEVICE=cuda bash stage0/train_eval.sh hrm
   ```
   Add `--compile` for a speed-up on A100/H100/4090 (not on P100/T4). On Ampere or newer GPUs `DTYPE=auto` picks bf16.
3. When it finishes (or the curve has clearly plateaued), copy `metrics.csv` back and **destroy the instance** so it stops billing.
4. If you would rather not touch the terminal, give me the instance's SSH details and a spend cap, and say go; I can run the same
   command and check the CSV (that needs your explicit approval first).

## Things worth knowing before you read the numbers
- **Puzzle embeddings.** Each augmented puzzle has its own learned embedding (282,368 of them at 300 augmentations; the table is
  a few hundred MB at hidden 512). A held-out puzzle can only be solved well if its embedding was trained, and each one is visited
  rarely. In short runs most embeddings are untouched, which holds the scores down. If your budget is under about 10M example-passes,
  try `NUM_AUG=100` (a second built folder) before spending more; this is untested.
- **Held-out episodes** are the 400 ARC-AGI-1 evaluation tasks in all their augmented views (123,294 episodes at 300 augmentations).
  On the final split, training evaluates `split.npz`'s `eval_idx` (2,000 episodes by default). A smaller `--eval-n` uses a prefix of that subset; a larger one keeps `eval_idx` and fills the rest from the final split. Use `--eval-n 0` for every episode in the split.
- **Optimiser.** `adamw` by default (works anywhere). HRM used `adam-atan2`; add `--optim adam_atan2` after `pip install adam-atan2` on a GPU.
- **Some tasks have fewer than 300 distinct augmentations** (the builder prints "augmentation not full"); that is the upstream builder's behaviour.
- **Short runs will not beat copy-input.** A model first learns to copy the input, then slowly learns the changes. Watch `eval_changed` and the gap.
