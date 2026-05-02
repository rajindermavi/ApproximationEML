import torch

from .losses import total_loss
from .utils import collect_diagnostics


def compute_tau(epoch: int, epochs: int, tau_start: float, tau_end: float) -> float:
    """Exponential decay: tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))."""
    return tau_start * (tau_end / tau_start) ** (epoch / max(epochs - 1, 1))


def _print_log(record: dict) -> None:
    epoch = record["epoch"]
    tau = record["tau"]
    loss = record.get("loss", float("nan"))
    mse = record.get("mse", float("nan"))
    parts = [f"epoch {epoch:5d}  tau={tau:.4f}  loss={loss:.6f}  mse={mse:.6f}"]
    if "leaf_entropy" in record:
        parts.append(f"H_leaf={record['leaf_entropy']:.4f}")
    if "gate_penalty" in record:
        parts.append(f"gate_pen={record['gate_penalty']:.4f}")
    if "val_mse" in record:
        parts.append(f"val_mse={record['val_mse']:.6f}")
    print("  ".join(parts))


def train_step(model, optimizer, x_batch: torch.Tensor, y_batch: torch.Tensor, config: dict) -> dict:
    model.train()
    optimizer.zero_grad()
    loss, metrics = total_loss(model, x_batch, y_batch, config)
    loss.backward()
    grad_clip = config.get("grad_clip", 1.0)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return metrics


def evaluate(model, x: torch.Tensor, y: torch.Tensor, config: dict) -> dict:
    model.eval()
    with torch.no_grad():
        _, metrics = total_loss(model, x, y, config)
    return metrics


def train_model(model, x_train: torch.Tensor, y_train: torch.Tensor, x_val=None, y_val=None, config: dict | None = None):
    if config is None:
        config = {}

    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 1000)
    batch_size = config.get("batch_size", None)
    tau_start = config.get("tau_start", 1.0)
    tau_end = config.get("tau_end", 0.1)
    log_every = config.get("log_every", 100)
    verbose = config.get("verbose", False)
    do_diagnostics = config.get("collect_diagnostics", False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    n_train = x_train.shape[0]

    for epoch in range(epochs):
        tau = compute_tau(epoch, epochs, tau_start, tau_end)
        model.update_tau(tau)
        cfg = {**config, "_epoch": epoch}

        if batch_size is None:
            metrics = train_step(model, optimizer, x_train, y_train, cfg)
        else:
            perm = torch.randperm(n_train)
            batch_metrics: list[dict] = []
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                batch_metrics.append(train_step(model, optimizer, x_train[idx], y_train[idx], cfg))
            keys = batch_metrics[0].keys()
            metrics = {k: sum(m[k] for m in batch_metrics) / len(batch_metrics) for k in keys}

        if epoch % log_every == 0 or epoch == epochs - 1:
            record = {"epoch": epoch, "tau": tau, **metrics}
            if x_val is not None:
                val_metrics = evaluate(model, x_val, y_val, cfg)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
            if do_diagnostics:
                record.update(collect_diagnostics(model, x_train))
            if verbose:
                _print_log(record)
            history.append(record)

    return history
