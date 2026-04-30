import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLeaf(nn.Module):
    """Softly selects one primitive from {1, x_1, ..., x_p}."""

    def __init__(self, input_dim: int):
        super().__init__()
        # Uniform initialization → uniform softmax at start
        self.logits = nn.Parameter(torch.zeros(input_dim + 1))

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Return one scalar per batch row using temperature-scaled softmax."""
        weights = F.softmax(self.logits / tau, dim=0)          # (p+1,)
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        candidates = torch.cat([ones, x], dim=1)               # (batch, p+1)
        return (candidates * weights).sum(dim=1)               # (batch,)

    def probs(self, tau: float = 1.0) -> torch.Tensor:
        """Return softmax(logits / tau) probabilities over primitives."""
        return F.softmax(self.logits / tau, dim=0)


class EMLNode(nn.Module):
    """Internal node computing exp(a*u+b) - log(softplus(c*v+d)+eps)."""

    def __init__(self, use_gates: bool = True, eps: float = 1e-6):
        super().__init__()
        self.use_gates = use_gates
        self.eps = eps

        # Conservative initialization per spec: a,c near 1; b,d near 0
        self.a = nn.Parameter(torch.empty(1).normal_(1.0, 0.05))
        self.b = nn.Parameter(torch.empty(1).normal_(0.0, 0.05))
        self.c = nn.Parameter(torch.empty(1).normal_(1.0, 0.05))
        self.d = nn.Parameter(torch.empty(1).normal_(0.0, 0.05))

        if use_gates:
            # sigmoid(-1.73) ≈ 0.15 → gates start mostly open (using child output)
            self.g_left = nn.Parameter(torch.full((1,), -1.73))
            self.g_right = nn.Parameter(torch.full((1,), -1.73))

    def apply_gates(self, left: torch.Tensor, right: torch.Tensor, tau: float = 1.0):
        """Interpolate child values with constant 1 using sigmoid(g / tau)."""
        if not self.use_gates:
            return left, right
        s_left = torch.sigmoid(self.g_left / tau)
        s_right = torch.sigmoid(self.g_right / tau)
        left = (1 - s_left) * left + s_left * 1.0
        right = (1 - s_right) * right + s_right * 1.0
        return left, right

    def log_argument(self, right: torch.Tensor) -> torch.Tensor:
        """Return softplus(c*v+d) — the positive argument passed to log before epsilon floor."""
        return F.softplus(self.c * right + self.d)

    def forward(self, left: torch.Tensor, right: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Compute exp(a*u+b) - log(softplus(c*v+d)+eps) with temperature-gated inputs."""
        left, right = self.apply_gates(left, right, tau)
        exp_branch = torch.exp(self.a * left + self.b)
        log_arg = self.log_argument(right) + self.eps
        log_branch = torch.log(log_arg)
        return exp_branch - log_branch
