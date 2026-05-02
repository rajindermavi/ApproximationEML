import torch
import pytest
from approximation_eml.components import SoftLeaf, EMLNode
from approximation_eml.model import EMLTree
from approximation_eml.export import snap_leaf, snap_gate, summarize_structure, export_tree, TreeSummary
from approximation_eml.data import make_dataset, target_identity
from approximation_eml.train import train_model
from approximation_eml.utils import set_seed


class TestSnapGate:
    def test_high_prob_collapsed(self):
        assert snap_gate(0.95) is True

    def test_low_prob_open(self):
        assert snap_gate(0.03) is False

    def test_ambiguous_none(self):
        assert snap_gate(0.5) is None

    def test_at_threshold_boundary(self):
        assert snap_gate(0.9) is True
        assert snap_gate(0.05) is False   # clearly below 1 - 0.9 = 0.1

    def test_custom_threshold(self):
        assert snap_gate(0.8, threshold=0.75) is True
        assert snap_gate(0.8, threshold=0.85) is None


class TestSnapLeaf:
    def test_dominant_constant(self):
        leaf = SoftLeaf(input_dim=2)
        with torch.no_grad():
            leaf.logits[0] = 10.0
            leaf.logits[1] = -10.0
            leaf.logits[2] = -10.0
        assert snap_leaf(leaf) == "1"

    def test_dominant_first_feature(self):
        leaf = SoftLeaf(input_dim=3)
        with torch.no_grad():
            leaf.logits.fill_(-10.0)
            leaf.logits[1] = 10.0
        assert snap_leaf(leaf) == "x_1"

    def test_dominant_second_feature(self):
        leaf = SoftLeaf(input_dim=3)
        with torch.no_grad():
            leaf.logits.fill_(-10.0)
            leaf.logits[2] = 10.0
        assert snap_leaf(leaf) == "x_2"

    def test_uniform_logits_no_snap(self):
        # Uniform logits → softmax ~1/(p+1) per entry → below any reasonable threshold
        leaf = SoftLeaf(input_dim=3)
        assert snap_leaf(leaf, threshold=0.9) is None

    def test_custom_threshold(self):
        leaf = SoftLeaf(input_dim=1)
        with torch.no_grad():
            leaf.logits[0] = 1.5
            leaf.logits[1] = 0.5
        # softmax([1.5, 0.5]) ≈ [0.731, 0.269] — below 0.9 but above 0.7
        assert snap_leaf(leaf, threshold=0.9) is None
        assert snap_leaf(leaf, threshold=0.7) == "1"


class TestSummarizeStructure:
    def test_leaf_probs_shape(self):
        model = EMLTree(input_dim=2, depth=2)
        summary = summarize_structure(model)
        assert len(summary.leaf_probs) == 4          # 2^depth leaves
        assert all(len(p) == 3 for p in summary.leaf_probs)  # p+1 = 3 probs

    def test_gate_values_count(self):
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        summary = summarize_structure(model)
        # 3 internal nodes × 2 gates each
        assert len(summary.gate_values) == 6
        assert len(summary.gate_snap) == 6

    def test_no_gates_empty_gate_lists(self):
        model = EMLTree(input_dim=2, depth=2, use_gates=False)
        summary = summarize_structure(model)
        assert summary.gate_values == []
        assert summary.gate_snap == []

    def test_node_params_count(self):
        model = EMLTree(input_dim=2, depth=2)
        summary = summarize_structure(model)
        assert len(summary.node_params) == 3   # 2^depth - 1 nodes
        assert all({"a", "b", "c", "d"} <= set(p.keys()) for p in summary.node_params)

    def test_expression_empty(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = summarize_structure(model)
        assert summary.expression == ""


class TestExportTree:
    def test_expression_non_empty(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = export_tree(model)
        assert len(summary.expression) > 0

    def test_expression_contains_exp_log(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = export_tree(model)
        assert "exp(" in summary.expression
        assert "log(" in summary.expression

    def test_collapsed_gate_shows_1_in_expression(self):
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        # Force left gate to be fully collapsed
        with torch.no_grad():
            model.nodes[0].g_left.fill_(100.0)
        summary = export_tree(model, gate_threshold=0.9)
        assert summary.gate_snap[0] is True
        # "1" should appear where the left child was
        assert "1" in summary.expression

    def test_open_gate_shows_child_expr(self):
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        # Force left gate to be fully open (s ≈ 0)
        with torch.no_grad():
            model.nodes[0].g_left.fill_(-100.0)
            # Force leaf 0 to snap to x_1
            model.leaves[0].logits.fill_(-10.0)
            model.leaves[0].logits[1] = 10.0
        summary = export_tree(model, gate_threshold=0.9)
        assert summary.gate_snap[0] is False
        assert "x_1" in summary.expression


class TestGateCollapseInTraining:
    """Show that training with high gate penalty collapses unnecessary branches."""

    def test_high_gate_penalty_increases_gate_values(self):
        """Gate sigmoid values should increase toward 1 under strong gate penalty."""
        set_seed(7)
        x, y = make_dataset(target_identity, n=128, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)

        before = summarize_structure(model)
        mean_before = sum(before.gate_values) / len(before.gate_values)

        train_model(model, x, y, config={
            "epochs": 400,
            "lr": 5e-3,
            "lambda_gate": 0.5,   # strong push toward collapse
            "lambda_leaf": 0.01,
            "log_every": 400,
        })

        after = summarize_structure(model)
        mean_after = sum(after.gate_values) / len(after.gate_values)
        assert mean_after > mean_before, (
            f"Gate values did not increase: {mean_before:.3f} → {mean_after:.3f}"
        )

    def test_snapped_gates_present_after_hardening(self):
        """At least one gate should snap to True with aggressive hardening."""
        set_seed(13)
        x, y = make_dataset(target_identity, n=128, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)

        train_model(model, x, y, config={
            "epochs": 600,
            "lr": 5e-3,
            "tau_start": 1.0,
            "tau_end": 0.05,
            "lambda_gate": 0.5,
            "lambda_leaf": 0.02,
            "log_every": 600,
        })

        summary = export_tree(model, gate_threshold=0.7)
        n_collapsed = sum(1 for s in summary.gate_snap if s is True)
        assert n_collapsed >= 1, (
            f"No gates snapped; gate values: {[f'{v:.3f}' for v in summary.gate_values]}"
        )
