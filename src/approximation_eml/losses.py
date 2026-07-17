import torch
import torch.nn.functional as F

from .config import DEFAULT_CONFIG


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true)


def leaf_entropy_penalty(model) -> torch.Tensor:
    """Mean Shannon entropy of leaf distributions. Minimising pushes leaves toward discrete selection."""
    tau = model.tau.item()
    entropies = []
    for leaf in model.leaf_modules():
        p = leaf.probs(tau)
        entropies.append(-(p * torch.log(p + 1e-12)).sum())
    return torch.stack(entropies).mean()


def gate_penalty(model) -> torch.Tensor:
    """Mean (1 - gate_prob) across all gates. Minimising pushes gates toward constant-1 substitution."""
    if not model.use_gates:
        return torch.tensor(0.0)
    tau = model.tau.item()
    gate_probs = []
    for node in model.node_modules():
        gate_probs.append(torch.sigmoid(node.g_left / tau))
        gate_probs.append(torch.sigmoid(node.g_right / tau))
    return (1.0 - torch.cat(gate_probs)).mean()


def parameter_penalty(model) -> torch.Tensor:
    """Sum of squared node parameters a, b, c, d."""
    params = []
    for node in model.node_modules():
        params.extend([node.a, node.b, node.c, node.d])
    return torch.stack([p.pow(2).sum() for p in params]).sum()


def safety_penalty(model, x: torch.Tensor, margin: float = DEFAULT_CONFIG.safe_margin) -> torch.Tensor:
    """Penalise log arguments that fall below margin: mean(ReLU(margin - softplus(cv+d)))."""
    tau = model.tau.item()
    n_internal = 2 ** model.depth - 1
    n_leaves = 2 ** model.depth
    values: list[torch.Tensor | None] = [None] * (n_internal + n_leaves)

    for i, leaf in enumerate(model.leaf_modules()):
        values[n_internal + i] = leaf(x, tau)

    penalties = []
    for bfs_idx in range(n_internal - 1, -1, -1):
        left_idx = 2 * bfs_idx + 1
        right_idx = 2 * bfs_idx + 2
        node = model.nodes[bfs_idx]
        left, right = values[left_idx], values[right_idx]
        _, right_gated = node.apply_gates(left, right, tau)
        log_arg = node.log_argument(right_gated)
        penalties.append(F.relu(margin - log_arg).mean())
        values[bfs_idx] = node(left, right, tau)

    if not penalties:
        return torch.tensor(0.0, device=x.device)
    return torch.stack(penalties).mean()


def total_loss(model, x: torch.Tensor, y: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    """Return total scalar loss and a metrics dict."""
    epoch = config.get("_epoch", 0)
    hardening_epoch = config.get("hardening_epoch", None)

    if hardening_epoch is not None and epoch >= hardening_epoch:
        lambda_leaf = config.get("lambda_leaf_hard", DEFAULT_CONFIG.lambda_leaf_hard)
        lambda_gate = config.get("lambda_gate_hard", DEFAULT_CONFIG.lambda_gate_hard)
    else:
        lambda_leaf = config.get("lambda_leaf", DEFAULT_CONFIG.lambda_leaf)
        lambda_gate = config.get("lambda_gate", DEFAULT_CONFIG.lambda_gate)

    lambda_param = config.get("lambda_param", DEFAULT_CONFIG.lambda_param)
    lambda_safe = config.get("lambda_safe", DEFAULT_CONFIG.lambda_safe)
    safe_margin = config.get("safe_margin", DEFAULT_CONFIG.safe_margin)

    y_pred = model(x)
    l_mse = mse_loss(y_pred, y)
    l_leaf = leaf_entropy_penalty(model)
    l_gate = gate_penalty(model)
    l_param = parameter_penalty(model)

    loss = l_mse + lambda_leaf * l_leaf + lambda_gate * l_gate + lambda_param * l_param

    if lambda_safe > 0.0:
        l_safe = safety_penalty(model, x, margin=safe_margin)
        loss = loss + lambda_safe * l_safe
    else:
        l_safe = torch.tensor(0.0)

    metrics = {
        "loss": loss.item(),
        "mse": l_mse.item(),
        "leaf_entropy": l_leaf.item(),
        "gate_penalty": l_gate.item(),
        "param_penalty": l_param.item(),
        "safety_penalty": l_safe.item(),
    }
    return loss, metrics
