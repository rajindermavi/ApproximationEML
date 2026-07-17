"""Single source of truth for every tunable constant in the EML tree.

Mirrors the reference tables in config.md. Defaults scattered across
components.py, model.py, losses.py, train.py, and export.py are sourced
from here rather than re-hardcoded, so the two can't drift apart.
"""

from dataclasses import dataclass


@dataclass
class EMLConfig:
    # --- Model construction (EMLTree.__init__) ------------------------------
    use_gates: bool = True
    eps: float = 1e-6

    # --- Node parameter initialization (EMLNode.__init__) --------------------
    affine_scale_mean: float = 1.0   # a, c ~ N(affine_scale_mean, affine_init_std)
    affine_shift_mean: float = 0.0   # b, d ~ N(affine_shift_mean, affine_init_std)
    affine_init_std: float = 0.05
    gate_init_logit: float = -1.73   # sigmoid(-1.73) ~= 0.15 -> gates start mostly open

    # --- Temperature schedule ------------------------------------------------
    tau_start: float = 1.0
    tau_end: float = 0.1

    # --- Optimizer -------------------------------------------------------
    lr: float = 1e-3
    grad_clip: float = 1.0

    # --- Data / batching ---------------------------------------------------
    batch_size: int | None = None
    epochs: int = 1000

    # --- Loss weights: phase 1 (soft training) --------------------------------
    lambda_leaf: float = 1e-2
    lambda_gate: float = 1e-2
    lambda_param: float = 1e-3
    lambda_safe: float = 0.0

    # --- Loss weights: phase 2 (hardening, optional) --------------------------
    hardening_epoch: int | None = None
    lambda_leaf_hard: float = 0.1
    lambda_gate_hard: float = 0.1

    # --- Safety penalty ----------------------------------------------------
    safe_margin: float = 1e-3

    # --- Snapping / export -------------------------------------------------
    leaf_threshold: float = 0.9
    gate_threshold: float = 0.9

    # --- Diagnostics / logging ----------------------------------------------
    log_every: int = 100
    verbose: bool = False
    collect_diagnostics: bool = False

    def to_dict(self) -> dict:
        """Return the plain dict accepted by total_loss / train_model as `config`."""
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "tau_start": self.tau_start,
            "tau_end": self.tau_end,
            "grad_clip": self.grad_clip,
            "lambda_leaf": self.lambda_leaf,
            "lambda_gate": self.lambda_gate,
            "lambda_param": self.lambda_param,
            "lambda_safe": self.lambda_safe,
            "safe_margin": self.safe_margin,
            "hardening_epoch": self.hardening_epoch,
            "lambda_leaf_hard": self.lambda_leaf_hard,
            "lambda_gate_hard": self.lambda_gate_hard,
            "log_every": self.log_every,
            "verbose": self.verbose,
            "collect_diagnostics": self.collect_diagnostics,
        }


DEFAULT_CONFIG = EMLConfig()
