"""
CPU smoke test for all 5 ablation cells.
Run from repo root: python scripts/verify_ablation_flags.py

Two-pass design per cell:
  Pass 1 (no_grad warmup): SubgoalHead step=1, manager_period=1 → goal updates
    from zeros to a real vector. With zeros goal, inject=True and inject=False
    are numerically identical (tensor + 0 == tensor), so we cannot test injection
    on pass 1.
  Pass 2 (main, with grad): goal_tensor is now non-zero; inject=True adds it,
    inject=False forces goal=None. Hidden states diverge → test is meaningful.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRM_DIR = os.path.join(REPO_ROOT, "HRM")
sys.path.insert(0, HRM_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from utils.functions import load_model_class  # noqa: E402

# ── Synthetic data constants ──────────────────────────────────────────────────
VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH_SIZE = 4
SEED = 0

# ── Tiny arch + determinism overrides shared by all cells ─────────────────────
BASE_OVERRIDES = [
    "data_path=DUMMY",
    "device=cpu",
    "enable_wandb=false",
    "global_batch_size=4",
    "arch.hidden_size=32",
    "arch.num_heads=4",
    "arch.expansion=2",
    "arch.H_layers=2",
    "arch.L_layers=2",
    "arch.H_cycles=2",
    "arch.L_cycles=2",
    "arch.halt_max_steps=4",
    "arch.halt_exploration_prob=0.0",
    "arch.puzzle_emb_ndim=0",
    # manager_period=1 so the goal updates on the very first step (step % 1 == 0)
    # Without this, the default period=4 keeps goal=zeros through the warmup pass.
    "arch.subgoal_head.manager_period=1",
]

CELL_DEFS: dict[str, dict] = {
    "A": dict(inject=True, align=True, rand=False),
    "B": dict(inject=False, align=False, rand=False),
    "C": dict(inject=True, align=False, rand=False),
    "D": dict(inject=False, align=True, rand=False),
    "E": dict(inject=True, align=True, rand=True),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _flag_overrides(flags: dict) -> list[str]:
    def b(v: bool) -> str:
        return "true" if v else "false"

    return [
        f"arch.subgoal_head.inject_subgoal={b(flags['inject'])}",
        f"arch.subgoal_head.use_alignment_loss={b(flags['align'])}",
        f"arch.subgoal_head.random_directions={b(flags['rand'])}",
    ]


def _load_arch_dict(overrides: list[str]) -> dict:
    config_dir = os.path.abspath(os.path.join(HRM_DIR, "config"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    GlobalHydra.instance().clear()
    return arch  # type: ignore[return-value]


def _build_model(arch: dict) -> torch.nn.Module:
    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg |= {
        "batch_size": BATCH_SIZE,
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "num_puzzle_identifiers": 10,
        "causal": False,
    }
    loss_cfg = arch["loss"]
    inner = load_model_class(arch["name"])(model_cfg)
    loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
    return load_model_class(loss_cfg["name"])(inner, **loss_kwargs)


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
        "labels": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
        "puzzle_identifiers": torch.zeros(BATCH_SIZE, dtype=torch.long),
    }


def _run_cell(model: torch.nn.Module, batch: dict) -> dict:
    model.train()
    model.zero_grad()

    # Pass 1 — warmup, no grad.
    # After this, subgoal_state.goal is a real non-zero vector (manager_period=1
    # guarantees update_mask=True on step 1).
    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, _, _, _, _ = model(return_keys=[], carry=carry, batch=batch)

    # Pass 2 — main forward with gradient tracking.
    # inject=True now adds a non-zero goal; inject=False forces goal=None.
    carry, total_loss, metrics, outs, _ = model(
        return_keys=["worker_hidden", "subgoal_goal"],
        carry=carry,
        batch=batch,
    )

    # Backward — confirms gradient flow end to end.
    total_loss.backward()

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "total_loss": total_loss.item(),
        "align_loss": metrics.get("feudal_loss", torch.tensor(0.0)).item(),
        "align_present": "feudal_loss" in metrics,
        "worker_hidden": outs.get("worker_hidden"),  # detached [B, T, D]
        "subgoal_goal": outs.get("subgoal_goal"),  # detached [B, D]
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(SEED)
    batch = _make_batch()

    results: dict[str, dict] = {}
    for name, flags in CELL_DEFS.items():
        label = f"inject={flags['inject']} align={flags['align']} rand={flags['rand']}"
        print(f"  cell {name}  {label} ... ", end="", flush=True)
        try:
            arch = _load_arch_dict(BASE_OVERRIDES + _flag_overrides(flags))
            torch.manual_seed(SEED)  # identical init weights for all 5 cells
            model = _build_model(arch)
            results[name] = _run_cell(model, batch)
            print("ok")
        except Exception:
            import traceback

            msg = traceback.format_exc()
            print(f"FAIL\n{msg}")
            results[name] = {"error": msg}

    # ── Assertions ────────────────────────────────────────────────────────────
    failures: list[str] = []

    # 1. Param count identical across all cells.
    counts = {k: v["params"] for k, v in results.items() if "params" in v}
    if counts:
        ref = next(iter(counts.values()))
        for cell, n in counts.items():
            if n != ref:
                failures.append(f"[param] cell {cell}: {n:,} ≠ ref {ref:,}")

    # 2. Alignment loss absent (key not in metrics) for cells B and C.
    for cell in ("B", "C"):
        r = results.get(cell, {})
        if "error" not in r and r.get("align_present"):
            failures.append(
                f"[{cell}] feudal_loss present in metrics (expected absent); "
                f"value={r['align_loss']:.6f} — use_alignment_loss gate not working"
            )

    # 3. Injection: worker hidden must differ between A (inject=on) and D (inject=off).
    wh_A = results.get("A", {}).get("worker_hidden")
    wh_D = results.get("D", {}).get("worker_hidden")
    if wh_A is not None and wh_D is not None:
        if torch.allclose(wh_A, wh_D, atol=1e-6):
            failures.append(
                "[A vs D] worker_hidden allclose — inject_subgoal gate had no effect; "
                "check that warmup populated a non-zero goal"
            )
    else:
        failures.append("[A vs D] worker_hidden missing from A or D outputs")

    # 4. Cell A: cosine sim between worker_hidden and subgoal_goal is finite ∈ [-1, 1].
    g_A = results.get("A", {}).get("subgoal_goal")
    if g_A is not None and wh_A is not None:
        wh_repr = wh_A.mean(dim=1) if wh_A.dim() == 3 else wh_A  # [B, D]
        cos_A = (
            F.cosine_similarity(
                F.normalize(wh_repr.float(), dim=-1),
                F.normalize(g_A.float(), dim=-1),
                dim=-1,
            )
            .mean()
            .item()
        )
        results["A"]["cos_A"] = cos_A
        if not (-1.0 <= cos_A <= 1.0) or (cos_A != cos_A):
            failures.append(f"[A] cosine_sim={cos_A} not finite in [-1, 1]")
    else:
        failures.append("[A] subgoal_goal or worker_hidden missing")

    # 5a. Cell E: subgoal_goal norms ≈ 1.0 (random unit vectors).
    g_E = results.get("E", {}).get("subgoal_goal")
    if g_E is not None:
        norms = g_E.float().norm(dim=-1)  # [B]
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            failures.append(
                f"[E] subgoal_goal norms {[f'{x:.6f}' for x in norms.tolist()]} "
                "not ≈ 1.0 — random_directions normalization broken"
            )
        results["E"]["e_norm"] = norms.mean().item()
    else:
        failures.append("[E] subgoal_goal missing from outputs")

    # 5b. Cell E goal vs cell A goal: |cos_sim| < 0.3 (independent vectors).
    if g_E is not None and g_A is not None:
        cos_EA = (
            F.cosine_similarity(
                F.normalize(g_E.float(), dim=-1),
                F.normalize(g_A.float(), dim=-1),
                dim=-1,
            )
            .abs()
            .mean()
            .item()
        )
        results["E"]["cos_vs_A"] = cos_EA
        if cos_EA >= 0.3:
            failures.append(
                f"[E vs A] |cos_sim|={cos_EA:.4f} ≥ 0.3 — random goal unexpectedly "
                "correlated with real goal; check random_directions code path"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    def _cell_has_failure(cell: str) -> bool:
        prefixes = {f"[{cell}]", "[param]"}
        if cell == "D":
            prefixes.add("[A vs D]")
        if cell == "E":
            prefixes.add("[E vs A]")
        return any(any(f.startswith(p) for p in prefixes) for f in failures)

    SEP = "─" * 85
    print(f"\n{SEP}")
    print(
        f"{'Cell':<4}  {'params':>8}  {'total_loss':>10}  {'align_loss':>10}  "
        f"{'inject_diff':>11}  {'E_norm':>6}  {'E_vs_A_cos':>10}  status"
    )
    print(SEP)

    for cell in ("A", "B", "C", "D", "E"):
        r = results.get(cell, {})
        if "error" in r:
            print(f"{cell:<4}  ERROR (see above)")
            continue

        params = r.get("params", -1)
        t_loss = r.get("total_loss", float("nan"))
        a_loss = r.get("align_loss", float("nan"))

        if cell == "D" and wh_A is not None and wh_D is not None:
            inj_col = "SAME!  " if torch.allclose(wh_A, wh_D, atol=1e-6) else "DIFFERS"
        else:
            inj_col = "      -"

        e_norm = f"{r['e_norm']:.4f}" if "e_norm" in r else "     -"
        cos_ea = f"{r['cos_vs_A']:.4f}" if "cos_vs_A" in r else "         -"
        status = "FAIL" if _cell_has_failure(cell) else "PASS"

        print(
            f"{cell:<4}  {params:>8,}  {t_loss:>10.4f}  {a_loss:>10.4f}  "
            f"{inj_col:>11}  {e_norm:>6}  {cos_ea:>10}  {status}"
        )

    print(SEP)

    if failures:
        print()
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("\nAll invariants pass.")


if __name__ == "__main__":
    main()
