import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_python_float_dict(metrics: dict) -> dict:
    return {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics.items()}


def collect_diagnostics(model, x: torch.Tensor) -> dict:
    tau = model.tau.item()
    diag: dict = {"tau": tau}

    for i, leaf in enumerate(model.leaf_modules()):
        probs = leaf.probs(tau).detach().cpu().tolist()
        diag[f"leaf_{i}_probs"] = probs
        diag[f"leaf_{i}_argmax"] = int(torch.tensor(probs).argmax().item())

    if model.use_gates:
        for i, node in enumerate(model.node_modules()):
            diag[f"node_{i}_gate_left"] = torch.sigmoid(node.g_left / tau).item()
            diag[f"node_{i}_gate_right"] = torch.sigmoid(node.g_right / tau).item()

    return diag
