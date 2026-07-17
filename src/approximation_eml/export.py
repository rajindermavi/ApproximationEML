from dataclasses import dataclass, field
import math
import torch

from .config import DEFAULT_CONFIG


@dataclass
class TreeSummary:
    leaf_probs: list[list[float]]
    leaf_snap: list[str | None]
    gate_values: list[float]
    gate_snap: list[bool | None]
    node_params: list[dict]
    expression: str = ""


def snap_leaf(leaf, threshold: float = DEFAULT_CONFIG.leaf_threshold, tau: float = 1.0) -> str | None:
    """Return the primitive name if argmax prob >= threshold, else None.

    Uses tau=1.0 by default for standalone calls.  summarize_structure passes
    the model's current tau so snapping reflects the model's actual inference
    distribution rather than the untempered logits.
    """
    probs = leaf.probs(tau=tau)
    max_prob, idx = probs.max(dim=0)
    if max_prob.item() < threshold:
        return None
    return "1" if idx.item() == 0 else f"x_{idx.item()}"


def snap_gate(prob: float, threshold: float = DEFAULT_CONFIG.gate_threshold) -> bool | None:
    """True = collapsed to 1, False = open (using child), None = uncertain."""
    if prob >= threshold:
        return True
    if prob <= 1.0 - threshold:
        return False
    return None


def summarize_structure(model, leaf_threshold: float = DEFAULT_CONFIG.leaf_threshold, gate_threshold: float = DEFAULT_CONFIG.gate_threshold) -> TreeSummary:
    """Inspect learned parameters and return a TreeSummary with expression left empty."""
    tau = model.tau.item()

    leaf_probs = [leaf.probs(tau).detach().cpu().tolist() for leaf in model.leaf_modules()]
    leaf_snap_list = [snap_leaf(leaf, leaf_threshold, tau=tau) for leaf in model.leaf_modules()]

    gate_values: list[float] = []
    gate_snap_list: list[bool | None] = []
    if model.use_gates:
        for node in model.node_modules():
            s_l = torch.sigmoid(node.g_left / tau).item()
            s_r = torch.sigmoid(node.g_right / tau).item()
            gate_values.extend([s_l, s_r])
            gate_snap_list.extend([snap_gate(s_l, gate_threshold), snap_gate(s_r, gate_threshold)])

    node_params = [
        {"a": n.a.item(), "b": n.b.item(), "c": n.c.item(), "d": n.d.item()}
        for n in model.node_modules()
    ]

    return TreeSummary(
        leaf_probs=leaf_probs,
        leaf_snap=leaf_snap_list,
        gate_values=gate_values,
        gate_snap=gate_snap_list,
        node_params=node_params,
    )


# ---------------------------------------------------------------------------
# Expression builder
# ---------------------------------------------------------------------------

def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _softplus(x: float) -> float:
    # numerically stable: log(1 + exp(x))
    return math.log1p(math.exp(min(x, 20.0))) if x < 20.0 else x


def _leaf_expr(snap: str | None, probs: list[float]) -> str:
    """Return a display token for a leaf: the snapped name, or ~top(pct%) if uncertain."""
    if snap is not None:
        return snap
    max_prob = max(probs)
    max_idx = probs.index(max_prob)
    label = "1" if max_idx == 0 else f"x_{max_idx}"
    return f"~{label}({max_prob:.0%})"


def _build_expression(model, summary: TreeSummary) -> str:
    """Recursively build a readable text expression for the learned tree.

    Uncertain leaves show their top candidate with confidence (~x_1(62%)).
    When both inputs to a node resolve to numeric constants, the node value
    is computed and shown as a number rather than a formula.
    """
    n_internal = 2 ** model.depth - 1
    n_leaves = 2 ** model.depth
    exprs: list[str | None] = [None] * (n_internal + n_leaves)

    for i in range(n_leaves):
        exprs[n_internal + i] = _leaf_expr(summary.leaf_snap[i], summary.leaf_probs[i])

    for bfs_idx in range(n_internal - 1, -1, -1):
        left_expr = exprs[2 * bfs_idx + 1]
        right_expr = exprs[2 * bfs_idx + 2]

        if model.use_gates:
            g = bfs_idx * 2
            if g < len(summary.gate_snap):
                if summary.gate_snap[g] is True:
                    left_expr = "1"
                if summary.gate_snap[g + 1] is True:
                    right_expr = "1"

        p = summary.node_params[bfs_idx]
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]

        l_val = _try_float(left_expr)
        r_val = _try_float(right_expr)
        if l_val is not None and r_val is not None:
            eps = model.nodes[bfs_idx].eps
            exp_val = math.exp(min(a * l_val + b, 80.0))
            log_val = math.log(_softplus(c * r_val + d) + eps)
            exprs[bfs_idx] = f"{exp_val - log_val:.4g}"
        else:
            exp_part = f"exp({a:.3f}*{left_expr} + {b:.3f})"
            log_part = f"log(softplus({c:.3f}*{right_expr} + {d:.3f}))"
            exprs[bfs_idx] = f"{exp_part} - {log_part}"

    return exprs[0] or ""


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

def _gate_label(value: float, snap: bool | None) -> str:
    if snap is True:
        return f"collapsed({value:.2f})"
    if snap is False:
        return f"open({value:.2f})"
    return f"uncertain({value:.2f})"


def format_summary(summary: TreeSummary) -> str:
    """Return a multi-line human-readable description of a TreeSummary."""
    lines: list[str] = []

    n_leaves = len(summary.leaf_probs)
    lines.append(f"Leaves ({n_leaves}):")
    for i, (probs, snap) in enumerate(zip(summary.leaf_probs, summary.leaf_snap)):
        max_prob = max(probs)
        max_idx = probs.index(max_prob)
        top_label = "1" if max_idx == 0 else f"x_{max_idx}"
        probs_str = "[" + " ".join(f"{p:.2f}" for p in probs) + "]"
        if snap is not None:
            status = f"{snap:<6} [snapped]"
        else:
            status = f"~{top_label}({max_prob:.0%}) [uncertain]"
        lines.append(f"  leaf_{i}  {status}  probs={probs_str}")

    if summary.gate_values:
        n_nodes = len(summary.gate_values) // 2
        lines.append(f"\nGates ({n_nodes} nodes):")
        for i in range(n_nodes):
            sl = summary.gate_values[2 * i]
            sr = summary.gate_values[2 * i + 1]
            snap_l = summary.gate_snap[2 * i]
            snap_r = summary.gate_snap[2 * i + 1]
            lines.append(
                f"  node_{i}  left={_gate_label(sl, snap_l):22s}  right={_gate_label(sr, snap_r)}"
            )

    if summary.expression:
        lines.append(f"\nExpression:\n  {summary.expression}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_tree(model, leaf_threshold: float = DEFAULT_CONFIG.leaf_threshold, gate_threshold: float = DEFAULT_CONFIG.gate_threshold) -> TreeSummary:
    """Build a TreeSummary and populate its expression field with a readable text tree."""
    summary = summarize_structure(model, leaf_threshold, gate_threshold)
    summary.expression = _build_expression(model, summary)
    return summary
