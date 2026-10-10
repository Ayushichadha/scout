#!/usr/bin/env bash
# Stage 0: build data (if missing) -> train a plain HRM-style model (goals/controller OFF) -> evaluate every N steps -> CSV.
#   bash stage0/train_eval.sh <preset> [extra train_eval.py flags]
# presets: smoke (CPU, ~25 min) | small (3.4M params, free-GPU stage) | hrm (27M params, ~10% of the HRM recipe)
# environment variables (all optional):
#   REPO=<path to this repo's HRM folder>   default: ../HRM (stage0/ sits next to HRM/)
#   RAW=<folder with ARC-AGI/data and ConceptARC/corpus>   default: ../rawlink
#   NUM_AUG=300   DATA=<built-data folder>   OUT=<run folder>   DEVICE=auto|cpu|cuda   DTYPE=auto|float32|bfloat16
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-smoke}"; shift || true
REPO="${REPO:-$HERE/../HRM}"; RAW="${RAW:-$HERE/../rawlink}"
NUM_AUG="${NUM_AUG:-300}"; DATA="${DATA:-$HERE/data/arc${NUM_AUG}}"
OUT="${OUT:-$HERE/runs/${PRESET}_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-auto}"; DTYPE="${DTYPE:-auto}"
[ "$DEVICE" = "cuda" ] || export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI   # CPU-only numerics fix (see REPORT.md)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [ ! -f "$DATA/split.npz" ]; then
  echo ">> building data with $NUM_AUG augmentations into $DATA (about 2.2 GB, 10-20 min, needs ~6 GB RAM at 300)"
  python "$HERE/build_data.py" --hrm "$REPO" --raw "$RAW" --out "$DATA" --num-aug "$NUM_AUG"
fi

case "$PRESET" in
  smoke) ARGS="--hidden 64 --layers 1 --cycles 1 --heads 2 --expansion 2 --batch 32 --steps 2400 --eval-every 240 --eval-n 1000 --lr 1e-3 --warmup 100" ;;
  small) ARGS="--hidden 256 --layers 2 --cycles 1 --heads 4 --expansion 4 --batch 256 --steps 40000 --eval-every 2000 --eval-n 2000 --lr 3e-4 --warmup 2000" ;;
  hrm)   ARGS="--hidden 512 --layers 4 --cycles 2 --heads 8 --expansion 4 --batch 768 --steps 52000 --eval-every 2000 --eval-n 2000 --lr 1e-4 --warmup 2000" ;;
  *) echo "unknown preset $PRESET"; exit 2 ;;
esac
mkdir -p "$OUT"
echo ">> preset $PRESET, output $OUT"
# --resume lets you re-run the same command (same OUT=...) after a session limit and carry on from ckpt.pt
python "$HERE/train_eval.py" --hrm "$REPO" --data "$DATA" --out "$OUT" --device "$DEVICE" --dtype "$DTYPE" $ARGS "$@" 2>&1 | tee -a "$OUT/log.txt"
echo ">> finished. Metrics: $OUT/metrics.csv"
