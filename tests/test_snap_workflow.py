"""
Step 5 tests: end-to-end snap/export workflow.

Each class tests one concrete claim:
  - UncertainLeafExpression: uncertain leaves show top candidate, not "?"
  - ConstantNodeCollapse: fully-constant subtrees evaluate to a number
  - FormatSummary: format_summary produces readable multi-line text
  - EndToEndSnap: train → export produces interpretable discrete structure
"""
import math
import torch
import pytest
from approximation_eml.components import SoftLeaf
from approximation_eml.model import EMLTree
from approximation_eml.export import (
    snap_leaf, snap_gate, summarize_structure, export_tree,
    format_summary, _leaf_expr, _build_expression,
)
from approximation_eml.data import make_dataset, target_identity, target_sum, target_square_first
from approximation_eml.train import train_model
from approximation_eml.utils import set_seed


# ---------------------------------------------------------------------------
# Uncertain leaf expression
# ---------------------------------------------------------------------------

class TestUncertainLeafExpression:
    def test_uncertain_leaf_shows_tilde_prefix(self):
        leaf = SoftLeaf(input_dim=2)
        # uniform logits → no snap
        expr = _leaf_expr(None, [0.33, 0.34, 0.33])
        assert expr.startswith("~")

    def test_uncertain_leaf_shows_top_label(self):
        # probs[2] is highest → x_2
        expr = _leaf_expr(None, [0.1, 0.2, 0.7])
        assert "x_2" in expr

    def test_uncertain_leaf_shows_percentage(self):
        expr = _leaf_expr(None, [0.1, 0.2, 0.7])
        assert "70%" in expr

    def test_uncertain_leaf_constant_top(self):
        # probs[0] is highest → "1"
        expr = _leaf_expr(None, [0.8, 0.1, 0.1])
        assert "~1" in expr

    def test_snapped_leaf_no_tilde(self):
        expr = _leaf_expr("x_1", [0.02, 0.96, 0.02])
        assert expr == "x_1"
        assert "~" not in expr

    def test_expression_no_question_marks(self):
        """After training, expression should never contain bare '?'."""
        model = EMLTree(input_dim=2, depth=2)
        summary = export_tree(model)
        assert "?" not in summary.expression


# ---------------------------------------------------------------------------
# Constant node collapse
# ---------------------------------------------------------------------------

class TestConstantNodeCollapse:
    def _force_all_gates_collapsed(self, model):
        with torch.no_grad():
            for node in model.node_modules():
                node.g_left.fill_(100.0)
                node.g_right.fill_(100.0)

    def test_fully_collapsed_tree_is_a_number(self):
        """When every gate is snapped, the root expression should parse as float."""
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        self._force_all_gates_collapsed(model)
        summary = export_tree(model, gate_threshold=0.5)
        # All gate_snap should be True
        assert all(s is True for s in summary.gate_snap)
        # Expression should be a single numeric value
        assert _try_parseable_float(summary.expression), (
            f"Expected numeric expression, got: {summary.expression!r}"
        )

    def test_partially_collapsed_root_is_formula(self):
        """If only one child is collapsed, the root stays a formula."""
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        with torch.no_grad():
            model.nodes[0].g_left.fill_(100.0)   # left → "1"
            model.nodes[0].g_right.fill_(-100.0)  # right → open
            # Force right child leaf to stay uncertain
            model.leaves[1].logits.fill_(0.0)
        summary = export_tree(model, gate_threshold=0.5)
        assert "exp(" in summary.expression

    def test_constant_value_is_finite(self):
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        self._force_all_gates_collapsed(model)
        summary = export_tree(model, gate_threshold=0.5)
        if _try_parseable_float(summary.expression):
            val = float(summary.expression)
            assert math.isfinite(val)


def _try_parseable_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_returns_non_empty_string(self):
        model = EMLTree(input_dim=2, depth=2)
        summary = export_tree(model)
        out = format_summary(summary)
        assert isinstance(out, str) and len(out) > 0

    def test_contains_leaves_header(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = export_tree(model)
        assert "Leaves" in format_summary(summary)

    def test_contains_all_leaf_lines(self):
        model = EMLTree(input_dim=2, depth=2)   # 4 leaves
        summary = export_tree(model)
        out = format_summary(summary)
        for i in range(4):
            assert f"leaf_{i}" in out

    def test_contains_gates_header_when_gated(self):
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        summary = export_tree(model)
        assert "Gates" in format_summary(summary)

    def test_no_gates_section_when_ungated(self):
        model = EMLTree(input_dim=2, depth=2, use_gates=False)
        summary = export_tree(model)
        assert "Gates" not in format_summary(summary)

    def test_contains_expression_section(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = export_tree(model)
        assert "Expression" in format_summary(summary)

    def test_collapsed_gate_shown_in_summary(self):
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        with torch.no_grad():
            model.nodes[0].g_left.fill_(100.0)
        summary = export_tree(model, gate_threshold=0.5)
        out = format_summary(summary)
        assert "collapsed" in out

    def test_open_gate_shown_in_summary(self):
        model = EMLTree(input_dim=2, depth=1, use_gates=True)
        with torch.no_grad():
            model.nodes[0].g_left.fill_(-100.0)
        summary = export_tree(model, gate_threshold=0.5)
        out = format_summary(summary)
        assert "open" in out

    def test_snapped_leaf_shown_without_tilde(self):
        leaf = SoftLeaf(input_dim=2)
        with torch.no_grad():
            leaf.logits.fill_(-10.0)
            leaf.logits[1] = 10.0
        model = EMLTree(input_dim=2, depth=1)
        model.leaves[0] = leaf
        summary = summarize_structure(model)
        out = format_summary(summary)
        assert "[snapped]" in out

    def test_uncertain_leaf_shown_with_uncertain_label(self):
        model = EMLTree(input_dim=2, depth=1)
        summary = summarize_structure(model)   # uniform logits → all uncertain
        out = format_summary(summary)
        assert "[uncertain]" in out

    def test_is_multiline(self):
        model = EMLTree(input_dim=2, depth=2)
        summary = export_tree(model)
        assert "\n" in format_summary(summary)


# ---------------------------------------------------------------------------
# End-to-end snap workflow
# ---------------------------------------------------------------------------

class TestEndToEndSnap:
    """Train a model on simple targets, then verify the exported structure is interpretable."""

    # lr=5e-3 and lambda_gate=1.0 confirmed to reliably push gate logits positive
    # across multiple seeds; lambda_leaf=0.3 concentrates leaves within 600 epochs.
    SNAP_CONFIG = {
        "epochs": 600,
        "lr": 5e-3,
        "tau_start": 1.0,
        "tau_end": 0.05,
        "lambda_leaf": 0.3,
        "lambda_gate": 1.0,
        "log_every": 600,
    }

    def test_at_least_one_leaf_snaps_after_training(self):
        set_seed(0)
        x, y = make_dataset(target_identity, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model, x, y, config=self.SNAP_CONFIG)

        summary = export_tree(model, leaf_threshold=0.75)
        snapped = [s for s in summary.leaf_snap if s is not None]
        assert len(snapped) >= 1, (
            f"No leaves snapped at threshold=0.75. "
            f"Max probs per leaf: {[max(p) for p in summary.leaf_probs]}"
        )

    def test_dominant_feature_appears_in_expression(self):
        set_seed(1)
        x, y = make_dataset(target_identity, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model, x, y, config=self.SNAP_CONFIG)

        summary = export_tree(model, leaf_threshold=0.75)
        # f(x) = x_1: the primary feature must appear somewhere
        assert "x_1" in summary.expression or "~x_1" in summary.expression, (
            f"x_1 not found in expression:\n{summary.expression}"
        )

    def test_at_least_one_gate_snaps_after_training(self):
        set_seed(2)
        x, y = make_dataset(target_identity, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model, x, y, config=self.SNAP_CONFIG)

        summary = export_tree(model, gate_threshold=0.7)
        collapsed = [s for s in summary.gate_snap if s is True]
        assert len(collapsed) >= 1, (
            f"No gates snapped at threshold=0.7. "
            f"Gate values: {[f'{v:.3f}' for v in summary.gate_values]}"
        )

    def test_format_summary_after_training_is_readable(self):
        set_seed(3)
        x, y = make_dataset(target_sum, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model, x, y, config=self.SNAP_CONFIG)

        summary = export_tree(model, leaf_threshold=0.75)
        out = format_summary(summary)
        # Must contain structural sections
        assert "Leaves" in out
        assert "Gates" in out
        assert "Expression" in out
        # Must not contain bare question marks
        assert "?" not in out

    def test_no_uncertain_leaves_after_strong_regularization(self):
        """With strong leaf penalty, all leaves should commit to a single primitive."""
        set_seed(7)
        x, y = make_dataset(target_identity, n=256, input_dim=2)
        model = EMLTree(input_dim=2, depth=2, use_gates=True)
        train_model(model, x, y, config={
            **self.SNAP_CONFIG,
            "lambda_leaf": 0.8,
        })
        summary = export_tree(model, leaf_threshold=0.75)
        uncertain = [s for s in summary.leaf_snap if s is None]
        assert len(uncertain) == 0, (
            f"Still {len(uncertain)} uncertain leaf(ves). "
            f"Max probs: {[max(p) for p in summary.leaf_probs]}"
        )
