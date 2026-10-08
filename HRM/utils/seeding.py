"""Experiment RNG initialization helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed all RNGs used by training and return the effective process seed."""
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def seed_data_worker(worker_id: int) -> None:
    """Initialize Python, NumPy, and torch RNGs in a DataLoader worker."""
    del worker_id  # The DataLoader-provided seed already incorporates worker_id.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
