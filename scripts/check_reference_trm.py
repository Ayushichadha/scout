"""Verify the official external TRM checkout runs on CPU before Scout integration."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/reference_trm_check.json")
    )
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import torch
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    torch.manual_seed(20261112)
    torch.set_num_threads(2)
    config = dict(
        batch_size=4,
        seq_len=16,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=6,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=2,
        hidden_size=32,
        expansion=2,
        num_heads=2,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
    )
    model = TinyRecursiveReasoningModel_ACTV1(config).eval()
    batch = {
        "inputs": torch.randint(0, 6, (4, 16)),
        "labels": torch.randint(0, 6, (4, 16)),
        "puzzle_identifiers": torch.zeros(4, dtype=torch.long),
    }
    carry = model.initial_carry(batch)
    with torch.no_grad():
        for step in range(4):
            carry, outputs = model(carry, batch)
            assert torch.isfinite(outputs["logits"]).all()
            assert torch.all(carry.steps == step + 1)
            assert torch.all(carry.halted == (step == 3))
    # Check an actual supervised gradient reaches the reference recurrent core.
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    loss = torch.nn.functional.cross_entropy(
        outputs["logits"].reshape(-1, 6), batch["labels"].reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in model.inner.L_level.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    code = source / "models/recursive_reasoning/trm.py"
    result = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "finite_logits": True,
        "four_pass_halting": True,
        "core_gradient": True,
        "scope": "official source, random initialization, synthetic tokens; task learning and Scout integration pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
