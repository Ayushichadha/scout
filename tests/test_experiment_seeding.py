from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HRM_ROOT = ROOT / "HRM"
if str(HRM_ROOT) not in sys.path:
    sys.path.insert(0, str(HRM_ROOT))

from puzzle_dataset import _sample_batch  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


def _rng_draws(seed: int) -> tuple[float, float, torch.Tensor]:
    seed_everything(seed)
    return random.random(), float(np.random.random()), torch.rand(4)


def test_seed_everything_controls_python_numpy_and_torch():
    first = _rng_draws(17)
    second = _rng_draws(17)
    different = _rng_draws(18)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert first[0] != different[0]
    assert first[1] != different[1]
    assert not torch.equal(first[2], different[2])


def _sample(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.Philox(seed=seed))
    _, example_indices, puzzle_ids = _sample_batch(
        rng,
        group_order=np.array([0, 1], dtype=np.int32),
        puzzle_indices=np.array([0, 4, 8], dtype=np.int32),
        group_indices=np.array([0, 1, 2], dtype=np.int32),
        start_index=0,
        global_batch_size=6,
    )
    return example_indices, puzzle_ids


def test_sample_batch_uses_only_the_explicit_generator():
    np.random.seed(1)
    first = _sample(23)

    # Disturb global NumPy state. The explicit generator must isolate sampling.
    np.random.seed(999)
    np.random.random(1000)
    second = _sample(23)
    different = _sample(24)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], different[0])
