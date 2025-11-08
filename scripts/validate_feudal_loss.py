#!/usr/bin/env python3
"""Quick validation script for feudal loss integration."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hrm_dir = root / "HRM"
sys.path.insert(0, str(hrm_dir))
sys.path.insert(0, str(root))

import torch  # noqa: E402
from models.subgoal_head import SubgoalHead, SubgoalHeadConfig  # noqa: E402
from models.losses import feudal_loss, ACTLossHead  # noqa: E402
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # noqa: E402


def test_feudal_loss_function():
    """Test the feudal_loss function directly."""
    print("Testing feudal_loss function...")

    batch_size = 4
    hidden_dim = 8

    # Create random worker state and manager goal
    worker_state = torch.randn(batch_size, hidden_dim)
    manager_goal = torch.randn(batch_size, hidden_dim)
    gate = torch.ones(batch_size, 1)

    # Compute feudal loss
    loss = feudal_loss(worker_state, manager_goal, gate=gate, reduction="sum")

    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert loss.item() >= 0, f"Feudal loss should be non-negative, got {loss.item()}"

    # Test with perfect alignment (should give low loss)
    worker_state_aligned = manager_goal.clone()
    loss_aligned = feudal_loss(worker_state_aligned, manager_goal, reduction="sum")
    assert (
        loss_aligned.item() < 0.1
    ), f"Aligned states should give low loss, got {loss_aligned.item()}"

    print("✅ Feudal loss function works correctly")


def test_subgoal_head_integration():
    """Test subgoal head state management."""
    print("Testing subgoal head integration...")

    cfg = SubgoalHeadConfig(hidden_size=8, goal_dim=8, manager_period=2)
    head = SubgoalHead(cfg)

    batch_size = 4
    device = torch.device("cpu")
    state = head.initial_state(batch_size=batch_size, device=device)

    # Simulate manager updates
    z_h = torch.randn(batch_size, 8)

    # First step: should not update (step=1, period=2)
    state, output = head(z_h, state)
    assert output.updated.sum().item() == 0, "First step should not update"

    # Second step: should update (step=2, period=2)
    state, output = head(z_h, state)
    assert output.updated.sum().item() == batch_size, "Second step should update all"

    print("✅ Subgoal head integration works correctly")


def test_model_with_subgoal_head():
    """Test that model can be instantiated with subgoal head."""
    print("Testing model instantiation with subgoal head...")

    # Minimal config for smoke test
    config_dict = {
        "batch_size": 2,
        "seq_len": 10,
        "vocab_size": 100,
        "num_puzzle_identifiers": 10,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "subgoal_head": {
            "hidden_size": 32,
            "goal_dim": 32,
            "manager_period": 2,
            "gating": True,
        },
    }

    model = HierarchicalReasoningModel_ACTV1(config_dict)

    # Check subgoal head exists
    assert model.subgoal_head is not None, "Subgoal head should be instantiated"

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 100, (2, 10)),
        "labels": torch.randint(0, 100, (2, 10)),
        "puzzle_identifiers": torch.randint(0, 10, (2,)),
    }

    # Initialize carry
    carry = model.initial_carry(batch)
    assert carry.subgoal_state is not None, "Subgoal state should be initialized"

    # Forward pass
    model.eval()
    with torch.no_grad():
        new_carry, outputs = model(carry, batch)

        # Check outputs contain subgoal information
        assert "subgoal_goal" in outputs, "Outputs should contain subgoal_goal"
        assert "worker_hidden" in outputs, "Outputs should contain worker_hidden"
        assert "manager_hidden" in outputs, "Outputs should contain manager_hidden"

    print("✅ Model with subgoal head works correctly")


def test_loss_head_with_feudal_loss():
    """Test that loss head computes feudal loss."""
    print("Testing loss head with feudal loss...")

    # Create minimal model
    config_dict = {
        "batch_size": 2,
        "seq_len": 10,
        "vocab_size": 100,
        "num_puzzle_identifiers": 10,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 32,
        "expansion": 2,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "subgoal_head": {
            "hidden_size": 32,
            "goal_dim": 32,
            "manager_period": 2,
            "gating": True,
        },
    }

    model = HierarchicalReasoningModel_ACTV1(config_dict)
    loss_head = ACTLossHead(
        model, loss_type="softmax_cross_entropy", feudal_loss_weight=0.1
    )

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 100, (2, 10)),
        "labels": torch.randint(0, 100, (2, 10)),
        "puzzle_identifiers": torch.randint(0, 10, (2,)),
    }

    # Initialize carry
    carry = loss_head.initial_carry(batch)

    # Forward pass
    loss_head.eval()
    with torch.no_grad():
        new_carry, loss, metrics, _, _ = loss_head(
            carry=carry, batch=batch, return_keys=[]
        )

        # Check feudal loss is in metrics
        assert "feudal_loss" in metrics, "Metrics should contain feudal_loss"
        assert metrics["feudal_loss"] >= 0, "Feudal loss should be non-negative"

    print("✅ Loss head with feudal loss works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Feudal Loss Integration Validation")
    print("=" * 60)
    print()

    try:
        test_feudal_loss_function()
        print()
        test_subgoal_head_integration()
        print()
        test_model_with_subgoal_head()
        print()
        test_loss_head_with_feudal_loss()
        print()
        print("=" * 60)
        print("✅ All validation tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
