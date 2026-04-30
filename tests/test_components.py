import torch
import pytest
from approximation_eml.components import SoftLeaf, EMLNode


class TestSoftLeaf:
    def test_output_shape(self):
        leaf = SoftLeaf(input_dim=3)
        x = torch.randn(8, 3)
        assert leaf(x).shape == (8,)

    def test_output_finite(self):
        leaf = SoftLeaf(input_dim=3)
        x = torch.randn(16, 3)
        assert torch.isfinite(leaf(x)).all()

    def test_probs_normalized(self):
        leaf = SoftLeaf(input_dim=3)
        probs = leaf.probs()
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        assert (probs >= 0).all()

    def test_probs_normalized_low_tau(self):
        leaf = SoftLeaf(input_dim=3)
        probs = leaf.probs(tau=0.1)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_constant_primitive_selectable(self):
        # With logits heavily favoring index 0 (constant 1), output should be near 1
        leaf = SoftLeaf(input_dim=2)
        with torch.no_grad():
            leaf.logits[0] = 10.0
            leaf.logits[1] = -10.0
            leaf.logits[2] = -10.0
        x = torch.zeros(4, 2)
        out = leaf(x, tau=0.1)
        assert torch.allclose(out, torch.ones(4), atol=0.01)

    def test_coordinate_primitive_selectable(self):
        # With logits favoring index 1 (x_1), output should match x[:, 0]
        leaf = SoftLeaf(input_dim=2)
        with torch.no_grad():
            leaf.logits[0] = -10.0
            leaf.logits[1] = 10.0
            leaf.logits[2] = -10.0
        x = torch.randn(8, 2)
        out = leaf(x, tau=0.1)
        assert torch.allclose(out, x[:, 0], atol=0.01)


class TestEMLNode:
    def test_output_shape_no_gates(self):
        node = EMLNode(use_gates=False)
        left = torch.randn(8)
        right = torch.randn(8)
        assert node(left, right).shape == (8,)

    def test_output_shape_with_gates(self):
        node = EMLNode(use_gates=True)
        left = torch.randn(8)
        right = torch.randn(8)
        assert node(left, right).shape == (8,)

    def test_output_finite_at_zero(self):
        node = EMLNode(use_gates=False)
        left = torch.zeros(16)
        right = torch.zeros(16)
        assert torch.isfinite(node(left, right)).all()

    def test_output_finite_random(self):
        node = EMLNode(use_gates=False)
        left = torch.randn(32)
        right = torch.randn(32)
        assert torch.isfinite(node(left, right)).all()

    def test_log_argument_positive(self):
        node = EMLNode(use_gates=False)
        right = torch.randn(16)
        log_arg = node.log_argument(right)
        assert (log_arg > 0).all()

    def test_gates_collapse_to_constant(self):
        # When gate logits are very large, s → 1 and both inputs should become ~1
        node = EMLNode(use_gates=True)
        with torch.no_grad():
            node.g_left.fill_(100.0)
            node.g_right.fill_(100.0)
        left = torch.zeros(4)   # would give different result without gate
        right = torch.zeros(4)
        out_gated = node(left, right, tau=1.0)

        node_ref = EMLNode(use_gates=False)
        with torch.no_grad():
            node_ref.a.copy_(node.a)
            node_ref.b.copy_(node.b)
            node_ref.c.copy_(node.c)
            node_ref.d.copy_(node.d)
        out_ref = node_ref(torch.ones(4), torch.ones(4), tau=1.0)
        assert torch.allclose(out_gated, out_ref, atol=1e-3)
