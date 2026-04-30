import torch
import pytest
from approximation_eml.data import make_dataset, target_identity, target_sum, target_square_first, train_val_split
from approximation_eml.model import EMLTree
from approximation_eml.train import train_model, compute_tau
from approximation_eml.utils import set_seed


def _mse(model, x, y):
    model.eval()
    with torch.no_grad():
        return ((model(x) - y) ** 2).mean().item()


class TestComputeTau:
    def test_starts_at_tau_start(self):
        assert compute_tau(0, 100, 1.0, 0.1) == pytest.approx(1.0)

    def test_ends_at_tau_end(self):
        assert compute_tau(99, 100, 1.0, 0.1) == pytest.approx(0.1, rel=1e-5)

    def test_monotone_decreasing(self):
        taus = [compute_tau(e, 100, 1.0, 0.1) for e in range(100)]
        assert all(taus[i] >= taus[i + 1] for i in range(len(taus) - 1))


class TestHistoryFormat:
    def test_keys_present(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=32, input_dim=1)
        model = EMLTree(input_dim=1, depth=1)
        history = train_model(model, x, y, config={"epochs": 10, "log_every": 5})
        assert len(history) >= 2
        assert {"epoch", "tau", "mse", "loss"} <= history[0].keys()

    def test_val_keys_present(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=64, input_dim=1)
        x_train, y_train, x_val, y_val = train_val_split(x, y)
        model = EMLTree(input_dim=1, depth=1)
        history = train_model(model, x_train, y_train, x_val=x_val, y_val=y_val,
                              config={"epochs": 10, "log_every": 5})
        assert "val_mse" in history[0]

    def test_minibatch_runs(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=64, input_dim=2)
        model = EMLTree(input_dim=2, depth=1)
        history = train_model(model, x, y, config={"epochs": 10, "log_every": 10, "batch_size": 16})
        assert len(history) >= 1


class TestFitLinear:
    """Model should reduce MSE to well below the naive constant-prediction baseline."""

    CONFIG = {"epochs": 600, "lr": 1e-3, "log_every": 600,
              "lambda_leaf": 0.01, "lambda_gate": 0.01, "lambda_param": 1e-3}

    def test_fit_identity(self):
        # f(x) = x_1  over [-1, 1]^2
        set_seed(0)
        x, y = make_dataset(target_identity, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        initial = _mse(model, x, y)
        train_model(model, x, y, config=self.CONFIG)
        final = _mse(model, x, y)
        assert final < initial * 0.5, f"identity: {initial:.4f} -> {final:.4f}"

    def test_fit_sum(self):
        # f(x) = x_1 + x_2  over [-1, 1]^2
        set_seed(1)
        x, y = make_dataset(target_sum, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        initial = _mse(model, x, y)
        train_model(model, x, y, config=self.CONFIG)
        final = _mse(model, x, y)
        assert final < initial * 0.5, f"sum: {initial:.4f} -> {final:.4f}"


class TestFitNonlinear:
    """f(x, y) = x^2 + y^2  over [-1, 1]^2."""

    def test_fit_square_sum(self):
        set_seed(2)
        x, y = make_dataset(target_square_first, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        initial = _mse(model, x, y)
        train_model(model, x, y, config={
            "epochs": 800, "lr": 1e-3, "log_every": 800,
            "lambda_leaf": 0.01, "lambda_gate": 0.01, "lambda_param": 1e-3,
        })
        final = _mse(model, x, y)
        assert final < initial * 0.5, f"x^2+y^2: {initial:.4f} -> {final:.4f}"
