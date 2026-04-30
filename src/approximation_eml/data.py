import torch


def sample_box(n: int, input_dim: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    return torch.empty(n, input_dim).uniform_(low, high)


def make_dataset(fn, n: int, input_dim: int, low: float = -1.0, high: float = 1.0):
    """Return X, y for a callable target function."""
    x = sample_box(n, input_dim, low, high)
    y = fn(x)
    return x, y


def target_identity(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0]


def target_sum(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] + x[:, 1]


def target_square_first(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum of squared features, i.e. x_1^2 + x_2^2 + ..."""
    return (x ** 2).sum(dim=1)


def train_val_split(x: torch.Tensor, y: torch.Tensor, frac: float = 0.8):
    n = x.shape[0]
    split = int(n * frac)
    return x[:split], y[:split], x[split:], y[split:]
