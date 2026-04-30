import torch
import pytest
from approximation_eml.model import EMLTree


class TestEMLTree:
    def test_forward_shape_depth1(self):
        model = EMLTree(input_dim=2, depth=1)
        x = torch.randn(8, 2)
        assert model(x).shape == (8,)

    def test_forward_shape_depth2(self):
        model = EMLTree(input_dim=2, depth=2)
        x = torch.randn(16, 2)
        assert model(x).shape == (16,)

    def test_forward_shape_depth3(self):
        model = EMLTree(input_dim=4, depth=3)
        x = torch.randn(32, 4)
        assert model(x).shape == (32,)

    def test_forward_finite(self):
        model = EMLTree(input_dim=3, depth=3)
        x = torch.randn(32, 3)
        assert torch.isfinite(model(x)).all()

    def test_node_leaf_counts(self):
        for depth in [1, 2, 3]:
            model = EMLTree(input_dim=2, depth=depth)
            assert len(model.leaf_modules()) == 2 ** depth
            assert len(model.node_modules()) == 2 ** depth - 1

    def test_update_tau(self):
        model = EMLTree(input_dim=2, depth=2)
        model.update_tau(0.1)
        assert model.tau.item() == pytest.approx(0.1)

    def test_tau_buffer_not_in_parameters(self):
        model = EMLTree(input_dim=2, depth=2)
        param_names = {name for name, _ in model.named_parameters()}
        assert "tau" not in param_names

    def test_backward_pass(self):
        model = EMLTree(input_dim=2, depth=2)
        x = torch.randn(32, 2)
        y = x[:, 0]
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        # All parameters should have gradients after backward
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_small_optimization_step(self):
        torch.manual_seed(42)
        model = EMLTree(input_dim=2, depth=2)
        x = torch.randn(64, 2)
        y = x[:, 0]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        # Loss should decrease (or at least not explode)
        assert torch.isfinite(torch.tensor(final_loss))
        assert final_loss < initial_loss * 1.5
