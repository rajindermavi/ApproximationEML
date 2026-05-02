"""
Tests for step 4: regularization effects and debug logging.

Each class tests one concrete claim:
  - DiagnosticsIntegration: collect_diagnostics key wires into history records
  - VerboseLogging: verbose=True prints to stdout without crashing
  - LeafEntropyPenalty: lambda_leaf > 0 concentrates leaf distributions
  - GatePenaltyEffect: lambda_gate > 0 increases gate sigmoid values
  - SafetyPenaltyStability: lambda_safe > 0 keeps training finite
"""
import torch
import pytest
from approximation_eml.data import make_dataset, target_identity, target_sum
from approximation_eml.model import EMLTree
from approximation_eml.train import train_model
from approximation_eml.utils import set_seed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean_max_leaf_prob(model) -> float:
    tau = model.tau.item()
    leaves = model.leaf_modules()
    return sum(leaf.probs(tau).max().item() for leaf in leaves) / len(leaves)


def _mean_gate_value(model) -> float:
    tau = model.tau.item()
    vals = []
    for node in model.node_modules():
        vals.append(torch.sigmoid(node.g_left / tau).item())
        vals.append(torch.sigmoid(node.g_right / tau).item())
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Diagnostics integration
# ---------------------------------------------------------------------------

class TestDiagnosticsIntegration:
    def test_enabled_adds_leaf_probs_to_history(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={
            "epochs": 5, "log_every": 5, "collect_diagnostics": True
        })
        assert "leaf_0_probs" in history[-1]
        assert "leaf_0_argmax" in history[-1]

    def test_enabled_adds_gate_values_for_gated_model(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        history = train_model(model, x, y, config={
            "epochs": 5, "log_every": 5, "collect_diagnostics": True
        })
        assert "node_0_gate_left" in history[-1]
        assert "node_0_gate_right" in history[-1]

    def test_disabled_by_default(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={"epochs": 5, "log_every": 5})
        assert "leaf_0_probs" not in history[-1]

    def test_all_leaves_present(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=2)   # 4 leaves
        history = train_model(model, x, y, config={
            "epochs": 5, "log_every": 5, "collect_diagnostics": True
        })
        for i in range(4):
            assert f"leaf_{i}_probs" in history[-1]
            assert f"leaf_{i}_argmax" in history[-1]

    def test_probs_are_lists_of_floats(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={
            "epochs": 5, "log_every": 5, "collect_diagnostics": True
        })
        probs = history[-1]["leaf_0_probs"]
        assert isinstance(probs, list)
        assert len(probs) == 3   # input_dim + 1
        assert all(isinstance(v, float) for v in probs)


# ---------------------------------------------------------------------------
# Verbose logging
# ---------------------------------------------------------------------------

class TestVerboseLogging:
    def test_verbose_prints_epoch_and_mse(self, capsys):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=1)
        model = EMLTree(input_dim=1, depth=1)
        train_model(model, x, y, config={"epochs": 5, "log_every": 5, "verbose": True})
        out = capsys.readouterr().out
        assert "epoch" in out
        assert "mse" in out

    def test_verbose_prints_leaf_entropy(self, capsys):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=1)
        model = EMLTree(input_dim=1, depth=1)
        train_model(model, x, y, config={"epochs": 5, "log_every": 5, "verbose": True})
        out = capsys.readouterr().out
        assert "H_leaf" in out

    def test_verbose_false_produces_no_output(self, capsys):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=1)
        model = EMLTree(input_dim=1, depth=1)
        train_model(model, x, y, config={"epochs": 5, "log_every": 5, "verbose": False})
        out = capsys.readouterr().out
        assert out == ""

    def test_verbose_with_val_shows_val_mse(self, capsys):
        set_seed(0)
        x, y = make_dataset(target_identity, n=64, input_dim=1)
        model = EMLTree(input_dim=1, depth=1)
        train_model(model, x[:50], y[:50], x_val=x[50:], y_val=y[50:],
                    config={"epochs": 5, "log_every": 5, "verbose": True})
        out = capsys.readouterr().out
        assert "val_mse" in out


# ---------------------------------------------------------------------------
# Leaf entropy penalty
# ---------------------------------------------------------------------------

class TestLeafEntropyPenalty:
    def test_leaf_entropy_appears_in_history(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=64, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={"epochs": 10, "log_every": 5})
        assert "leaf_entropy" in history[0]

    def test_high_lambda_leaf_decreases_entropy(self):
        """With a strong entropy penalty, leaf distributions should concentrate over training."""
        set_seed(3)
        x, y = make_dataset(target_identity, n=128, input_dim=2)
        model = EMLTree(input_dim=2, depth=2)
        history = train_model(model, x, y, config={
            "epochs": 500, "lr": 1e-3, "lambda_leaf": 0.2, "log_every": 100,
        })
        assert history[-1]["leaf_entropy"] < history[0]["leaf_entropy"]

    def test_high_lambda_leaf_concentrates_leaf_distributions(self):
        """Model trained with entropy penalty should have higher max-leaf-prob than without."""
        x, y = make_dataset(target_identity, n=128, input_dim=2)

        set_seed(5)
        model_reg = EMLTree(input_dim=2, depth=2)
        train_model(model_reg, x, y, config={
            "epochs": 500, "lr": 1e-3, "lambda_leaf": 0.5, "log_every": 500,
        })

        set_seed(5)
        model_noreg = EMLTree(input_dim=2, depth=2)
        train_model(model_noreg, x, y, config={
            "epochs": 500, "lr": 1e-3, "lambda_leaf": 0.0, "log_every": 500,
        })

        assert _mean_max_leaf_prob(model_reg) > _mean_max_leaf_prob(model_noreg)


# ---------------------------------------------------------------------------
# Gate penalty effect
# ---------------------------------------------------------------------------

class TestGatePenaltyEffect:
    def test_high_lambda_gate_raises_gate_values(self):
        """Gate sigmoid values should be higher when gate penalty is applied."""
        x, y = make_dataset(target_sum, n=128, input_dim=2)

        set_seed(9)
        model_gated = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model_gated, x, y, config={
            "epochs": 400, "lr": 5e-3, "lambda_gate": 0.5, "log_every": 400,
        })

        set_seed(9)
        model_ungated = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model_ungated, x, y, config={
            "epochs": 400, "lr": 5e-3, "lambda_gate": 0.0, "log_every": 400,
        })

        assert _mean_gate_value(model_gated) > _mean_gate_value(model_ungated)

    def test_gate_penalty_in_history(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        history = train_model(model, x, y, config={"epochs": 5, "log_every": 5})
        assert "gate_penalty" in history[0]


# ---------------------------------------------------------------------------
# Safety penalty stability
# ---------------------------------------------------------------------------

class TestSafetyPenaltyStability:
    def test_training_finite_with_safety_penalty(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=64, input_dim=2)
        model = EMLTree(input_dim=2, depth=2)
        history = train_model(model, x, y, config={
            "epochs": 200, "lr": 1e-3, "lambda_safe": 0.1, "log_every": 100,
        })
        assert all(torch.isfinite(torch.tensor(r["loss"])).item() for r in history)

    def test_safety_penalty_in_history(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={
            "epochs": 5, "log_every": 5, "lambda_safe": 0.1,
        })
        assert "safety_penalty" in history[0]
