# Training Configuration Reference

This document defines every key accepted in the `config: dict` passed to
`total_loss`, `train_step`, `evaluate`, and `train_model`.

All keys are optional. Missing keys fall back to the defaults listed here.

---

## Model construction

These are passed to `EMLTree.__init__` and are not part of the training config,
but are recorded here for completeness.

| Key | Type | Default | Description |
|---|---|---|---|
| `input_dim` | `int` | — | Number of input features `p`. Required. |
| `depth` | `int` | — | Tree depth. Required. |
| `use_gates` | `bool` | `True` | Enable constant-substitution gates at internal nodes. |
| `eps` | `float` | `1e-6` | Epsilon floor inside the log branch of each node. |

---

## Temperature schedule

`tau` is updated at the start of each epoch using exponential decay and stored
on the model as a non-trainable buffer via `model.update_tau(tau)`.

| Key | Type | Default | Description |
|---|---|---|---|
| `tau_start` | `float` | `1.0` | Initial temperature. Standard softmax/sigmoid at `1.0`. |
| `tau_end` | `float` | `0.1` | Final temperature. Distributions are strongly peaked at `0.1`. |

The schedule is: `τ(t) = tau_start * (tau_end / tau_start) ** (t / (epochs - 1))`

---

## Optimizer

| Key | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `str` | `"adam"` | Optimizer name. Only `"adam"` supported in v1. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `grad_clip` | `float` | `1.0` | Max gradient norm for clipping. Set to `0.0` to disable. |

---

## Data / batching

| Key | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` or `None` | `None` | Mini-batch size. `None` means full-batch. |
| `epochs` | `int` | `1000` | Total training epochs. |

---

## Loss weights (Phase 1 — soft training)

| Key | Type | Default | Description |
|---|---|---|---|
| `lambda_leaf` | `float` | `1e-2` | Weight on leaf entropy penalty `L_leaf`. |
| `lambda_gate` | `float` | `1e-2` | Weight on gate sparsity-toward-1 penalty `L_gate`. |
| `lambda_param` | `float` | `1e-3` | Weight on parameter L2 norm penalty `L_param`. |
| `lambda_safe` | `float` | `0.0` | Weight on log-argument safety penalty `L_safe`. Set > 0 if instability observed. |

---

## Loss weights (Phase 2 — hardening)

After `hardening_epoch`, the leaf and gate lambdas are replaced by these
larger values to push distributions toward discrete choices.

| Key | Type | Default | Description |
|---|---|---|---|
| `hardening_epoch` | `int` | `None` | Epoch at which to switch to hardening weights. `None` disables auto-hardening. |
| `lambda_leaf_hard` | `float` | `0.1` | Leaf entropy weight after hardening epoch. |
| `lambda_gate_hard` | `float` | `0.1` | Gate sparsity weight after hardening epoch. |

---

## Safety penalty

| Key | Type | Default | Description |
|---|---|---|---|
| `safe_margin` | `float` | `1e-3` | Margin `delta` in `L_safe = mean(ReLU(delta - softplus(cv+d)))`. |

---

## Snapping / export

These are used by `export_tree` and `summarize_structure`, not during training.

| Key | Type | Default | Description |
|---|---|---|---|
| `leaf_threshold` | `float` | `0.9` | Minimum softmax probability to snap a leaf to its argmax primitive. |
| `gate_threshold` | `float` | `0.9` | Minimum sigmoid value to snap a gate to constant-1 substitution. |

---

## Diagnostics / logging

| Key | Type | Default | Description |
|---|---|---|---|
| `log_every` | `int` | `100` | Print/record diagnostic metrics every N epochs. |
| `collect_diagnostics` | `bool` | `False` | Run `collect_diagnostics` each log step (adds overhead). |

---

## Example: minimal config

```python
config = {}  # all defaults
```

## Example: explicit Phase 1 config

```python
config = {
    "lr": 1e-3,
    "epochs": 2000,
    "tau_start": 1.0,
    "tau_end": 0.1,
    "batch_size": None,
    "grad_clip": 1.0,
    "lambda_leaf": 0.01,
    "lambda_gate": 0.01,
    "lambda_param": 0.001,
    "lambda_safe": 0.0,
    "log_every": 100,
}
```

## Example: config with hardening (temperature schedule + penalty boost)

```python
config = {
    "lr": 1e-3,
    "epochs": 3000,
    "tau_start": 1.0,
    "tau_end": 0.05,          # push further than default for harder snapping
    "hardening_epoch": 2000,
    "lambda_leaf": 0.01,
    "lambda_gate": 0.01,
    "lambda_leaf_hard": 0.1,
    "lambda_gate_hard": 0.1,
    "lambda_param": 0.001,
}
```
