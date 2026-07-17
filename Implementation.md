# MLP Baseline First Implementation Plan

## Goal

Build a first working implementation of a standard, trainable **multilayer
perceptron (MLP)** for scalar-output regression:

[
f: \mathbb{R}^p \to \mathbb{R}
]

using **conventional deep learning components** — dense layers, nonlinear
activations, normalization, and dropout — trained end-to-end with
gradient-based optimization.

This baseline exists to answer a narrow, comparative question:

> On the same toy regression targets used to evaluate the EML tree, how does
> a conventional black-box network compare in fit quality, training
> stability, and parameter efficiency — and does the EML tree's
> interpretability cost it any accuracy relative to this baseline?

This is an exploratory baseline, not a production model-serving system.

---

## Core Design Decisions

### Function class

Exactly the same function class as the EML tree, so the two are directly
comparable:

[
x = (x_1, \dots, x_p) \in \mathbb{R}^p,
\qquad
f(x) \in \mathbb{R}.
]

### Network topology

Use a **fully-connected feedforward network of fixed depth and width**.

* An input layer accepting `p` features.
* A stack of hidden blocks, each `Linear -> Normalization -> Activation ->
  Dropout`.
* A final linear output layer producing one scalar.

We will **not** implement architecture search (width/depth tuning beyond
manual config) in version 1.

### Hidden layer block

Each hidden block is a standard four-stage composition:

1. **Linear**: `nn.Linear(in_features, out_features)`
2. **Normalization**: `nn.BatchNorm1d(out_features)` (optional, toggleable)
3. **Activation**: `ReLU` by default (configurable to `GELU` or `Tanh`)
4. **Dropout**: `nn.Dropout(p=dropout_rate)`

This ordering (`Linear -> Norm -> Activation -> Dropout`) is the conventional
default and keeps normalization statistics computed on pre-activation values.

### Complexity control: gates vs. dropout

The EML tree already has a mechanism that behaves *like* dropout in effect:
constant-substitution gates can learn to collapse a subtree's contribution
to the constant `1`, functionally removing it from the computation. The
effect — a node's contribution disappearing from the output — is a real
analogue to dropping a unit. The mechanism is not the same, though: EML
gates are **deterministic, monotonically-trained soft masks**, evaluated
identically on every forward pass once trained, pushed toward collapse by a
fixed regularization term plus gradient pressure. Dropout is **stochastic**
— a fresh random mask every training step, specifically to inject noise and
discourage co-adaptation. The closer standard-deep-learning analogue to the
EML tree's gates is a learned/soft pruning gate (L0-style or
DARTS-style continuous relaxation of an architectural choice), not Bernoulli
dropout.

The MLP baseline still includes conventional dropout, since it's the
standard regularizer a conventional implementation would reach for — it
just isn't offered as a mechanism-matched analogue to EML gates, only a
functional (if partial) one. Complexity control on the MLP side comes from
three conventional sources:

1. **Dropout** inside every hidden block, randomly zeroing activations
   during training to discourage co-adaptation of units.
2. **Weight decay** (L2 penalty) applied through the optimizer.
3. **Early stopping** driven by validation loss.

The larger and more consequential difference between the two models is not
gates-vs-dropout but raw capacity: see "Matching capacity to the EML tree"
below.

---

## First-Version Scope

### Included

* Real-valued PyTorch implementation
* Fixed-depth, fixed-width fully-connected network
* Configurable hidden activation (`ReLU` default)
* Optional `BatchNorm1d` per hidden layer
* Dropout regularization per hidden layer
* Trainable weights and biases at every linear layer
* Standard supervised regression training loop
* Weight decay and early stopping
* Basic training diagnostics/logging
* Small synthetic experiments, reusing the same toy targets as the EML tree

### Excluded for version 1

* Convolutional, recurrent, or attention layers
* Architecture search (depth/width optimization)
* Learning-rate schedulers beyond a simple optional step/cosine decay
* Ensembling or model averaging
* Symbolic or structural interpretability of any kind
* Multi-output `R^p -> R^m` support
* Full benchmark suite beyond the shared toy target set

---

## Proposed Architecture

## 1. Input handling

The network accepts a batch `X` of shape `(batch_size, p)`. Unlike the EML
tree, the MLP has no primitive dictionary or leaf selection — every input
feature is always consumed by the first linear layer's weight matrix. Input
standardization (zero mean, unit variance per feature) is applied once,
outside the model, before training; the MLP itself does not manage feature
scaling.

## 2. Hidden stack

Given a list of hidden widths `[h_1, h_2, ..., h_k]`, the network builds `k`
hidden blocks:

```
x -> Linear(p, h_1) -> [BatchNorm1d(h_1)] -> ReLU -> Dropout(p_drop)
  -> Linear(h_1, h_2) -> [BatchNorm1d(h_2)] -> ReLU -> Dropout(p_drop)
  -> ...
  -> Linear(h_{k-1}, h_k) -> [BatchNorm1d(h_k)] -> ReLU -> Dropout(p_drop)
```

A depth-2 configuration is the default starting point, matching the EML
tree's two levels of internal composition. Width is **not** picked as a
generic conventional value — see "Matching capacity to the EML tree"
directly below, which derives the default `hidden_dims=[3, 4]` from the
depth-2 EML tree's own parameter count.

## Matching capacity to the EML tree

The single largest confound in comparing these two models is raw parameter
count, not architecture. A depth-2 EML tree with `p` inputs and gates
enabled has:

```
n_leaves    = 2^depth                       # 4 for depth=2
n_internal  = 2^depth - 1                   # 3 for depth=2
leaf_params = n_leaves * (p + 1)            # 3 logits/leaf for p=2
node_params = n_internal * 6                # a,b,c,d + 2 gate logits
total       = leaf_params + node_params
```

For the toy suite's usual `depth=2, p=2`: `4*3 + 3*6 = 12 + 18 = 30`
trainable parameters.

An MLP with two hidden layers of width `h1, h2` and no BatchNorm has:

```
total = (p*h1 + h1) + (h1*h2 + h2) + (h2 + 1)
```

Solving for widths near that 30-parameter budget at `p=2` gives
`hidden_dims=[3, 4]` → `(2*3+3) + (3*4+4) + (4+1) = 9 + 16 + 5 = 30` —
an exact match. **This, not `[32, 32]`, is the default.** If `input_dim`
or `depth` changes for a given experiment, recompute the EML tree's
parameter count from the formula above and re-derive matching widths rather
than reusing `[3, 4]` unchanged.

BatchNorm is off by default (`use_batchnorm=False`) for the same reason:
its affine parameters (`2 * h` per layer) are a large fraction of a
30-parameter budget, it has no EML-tree analogue, and running statistics on
layers this narrow are unreliable with small batches anyway. It remains
available as an opt-in ablation, not part of the parameter-matched default.

## 3. Output layer

A single final `Linear(h_k, 1)` layer with no activation, squeezed to shape
`(batch_size,)` to match the EML tree's output convention.

### Recommended interface

```python
class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [3, 4],
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout: float = 0.1,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...  # shape: (batch_size,)
```

### Why this design

There is no clever structural story here by design — that is the point of a
baseline. A `Linear -> Norm -> Activation -> Dropout` stack is the
conventional, well-understood building block for tabular regression
problems, chosen specifically so that any accuracy or stability gap between
it and the EML tree can be attributed to the EML tree's structural
constraints rather than to an unusual or unfairly weak baseline.

This keeps the implementation:

* fully differentiable,
* real-valued,
* expressive (a sufficiently wide/deep MLP is a universal approximator),
* structurally uninterpretable — trained weights do not correspond to a
  readable formula,
* a fair, conventional point of comparison for the EML tree.

---

## Training Objective

Use supervised regression loss with weight decay handled by the optimizer
rather than as an explicit loss term:

[
\mathcal{L} = \mathcal{L}_{fit}
]

with weight decay `λ_wd` passed to the optimizer, and dropout active only
during training (`model.train()`).

## 1. Fit loss

Standard MSE, identical to the EML tree's fit term so comparisons are
apples-to-apples:

[
\mathcal{L}_{fit} = \frac{1}{N}\sum_i (\hat y_i - y_i)^2
]

## 2. Weight decay

Rather than a hand-written L2 penalty term, use the optimizer's built-in
`weight_decay` argument (decoupled weight decay via `AdamW`). This is the
conventional mechanism for L2-style regularization in modern PyTorch
training loops and keeps the loss function itself simple.

## 3. Dropout as implicit regularization

Dropout is not part of the loss function — it is a stochastic forward-pass
mechanism active only in training mode. It is listed here because, together
with weight decay, it plays the same complexity-control role that the EML
tree's entropy and gate penalties play: keeping the model from overfitting
a fixed amount of capacity to a small dataset.

## 4. No safety penalty needed

Unlike the EML tree, the MLP has no `exp`/`log` branches and therefore no
domain-safety concern to penalize. `ReLU` and `BatchNorm1d` are the only
numerically sensitive components, and both are well-behaved by construction.

---

## Training Strategy

### No temperature schedule

The MLP has no soft-to-discrete relaxation to anneal — every weight is a
plain continuous parameter throughout training, and there is nothing to
"snap" afterward. This is the central procedural difference from the EML
tree's training loop: a single, ordinary training phase replaces the EML
tree's temperature-annealed phase-1/phase-2 structure.

### Single-phase training

Train all parameters continuously for a fixed number of epochs, or until
early stopping triggers.

Recommended initial defaults:

* optimizer: `AdamW`
* learning rate: `1e-3`
* weight decay: `1e-4`
* batch size: full-batch for small synthetic datasets, mini-batch (e.g. 64)
  otherwise
* gradient clipping: `1.0`
* dropout rate: `0.1`
* max epochs: `2000`, with early stopping patience of `100` epochs on
  validation MSE

### Early stopping

Track validation MSE every `log_every` epochs. If it fails to improve by
more than a small tolerance for `patience` consecutive checks, stop training
and restore the best-seen weights. This is the MLP's analogue of the EML
tree's temperature-driven convergence — a way of deciding training is
"done" without a fixed epoch count being the only signal.

### Learning rate schedule (optional)

A simple `ReduceLROnPlateau` or cosine decay schedule may be layered on top
of the base learning rate if training plateaus before convergence. Optional
in the first pass, same as the EML tree's hardening phase was optional.

---

## Initialization

Initialization matters less dramatically here than for the EML tree — there
is no `exp`/`log` composition to destabilize — but sensible defaults still
matter for training speed.

### Linear layer weights

Use **Kaiming (He) initialization**, matched to the `ReLU` activation:

```python
nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

This is PyTorch's default for `nn.Linear` followed by `ReLU` and requires no
custom logic beyond using the standard initializers explicitly (rather than
relying on implicit defaults) for clarity.

### BatchNorm parameters

Standard defaults: scale (`weight`) initialized to `1`, shift (`bias`)
initialized to `0`.

### Output layer

The final linear layer may be initialized with a smaller gain (e.g.
`gain=0.1` on a Xavier/Glorot initialization) so the network's initial
predictions start near zero rather than an arbitrary large value, which
tends to speed up early convergence on standardized regression targets.

---

## Numerical Stability Plan

This is a lighter concern than for the EML tree, but not absent.

### Risks

* exploding gradients in deeper stacks
* dead ReLUs (units stuck outputting zero) if learning rate is too high
* batch statistics becoming unstable on very small batches (a reason
  BatchNorm is off by default at these widths, not just a parameter-count
  concern)
* at a parameter-matched, narrow width (`[3, 4]`), underfitting is a more
  realistic risk than overfitting

### Mitigations

* gradient clipping
* Kaiming initialization matched to `ReLU`
* dropout and weight decay to control overfitting
* conservative default learning rate (`1e-3`) with optional LR scheduling
* `BatchNorm1d` available as an opt-in if training proves unstable, at the
  cost of reintroducing a parameter-count and stabilization asymmetry
  against the EML tree

### Debug metrics to log

Per training run, log:

* fit loss (train and validation)
* gradient norm (pre-clipping)
* parameter norm
* fraction of dead ReLU units (activations ≤ 0 across a full batch), if
  investigating training stalls
* current learning rate (if a scheduler is active)

---

## Export and Interpretation

The MLP produces no symbolic readout — this is a deliberate and expected
contrast with the EML tree, not a missing feature.

## What "export" means here

Rather than a snapped formula, export consists of:

* the trained `state_dict` (weights and biases), for reloading or
  checkpointing,
* summary statistics: parameter count, final train/val MSE, weight norm per
  layer,
* optionally, **permutation feature importance**: shuffle one input column
  at a time on the validation set and measure the resulting increase in MSE,
  giving a coarse per-feature relevance signal without claiming any formula.

## Why no formula

There is no snapping step because there is nothing to snap: every weight in
every layer participates continuously in the output, and no component of
the architecture is designed to collapse toward a discrete, human-readable
structure. Reporting a `TreeSummary`-style structural export for this model
would be misleading, so no equivalent is implemented.

## Comparison readout

For the purpose of comparing against the EML tree, the useful export is a
single row per trial: `{target_name, val_mse, param_count, converged_epoch}`.
This is what the shared experiment suite (see below) actually records.

---

## Repository Structure

Add a small sibling module next to the existing `approximation_eml` package,
reusing its data utilities rather than duplicating them.

```text
ApproximationEML/
  README.md
  Implementation.md          (this document)
  pyproject.toml

  src/
    approximation_eml/       (existing EML tree package — reused, not modified)
      data.py                 <- reused for toy targets and splitting
      utils.py                <- reused for set_seed / get_device

    mlp_baseline/
      __init__.py
      model.py
      train.py
      export.py

  experiments/
    fit_toy_mlp.py
    fit_suite_mlp.py

  tests/
    test_mlp_model.py
    test_mlp_training.py
```

This is intentionally compact and deliberately reuses `approximation_eml.data`
and `approximation_eml.utils` rather than forking them, since the entire
point of this baseline is a controlled comparison on identical data.

### File responsibilities

#### `src/mlp_baseline/__init__.py`

Keep this minimal. Export the main public objects.

Suggested stub:

```python
from .model import MLPRegressor
from .train import train_mlp
from .export import export_summary, MLPSummary
```

#### `src/mlp_baseline/model.py`

Defines the network module.

Responsibilities:

* build a configurable stack of `Linear -> [BatchNorm1d] -> Activation ->
  Dropout` blocks
* build the final scalar output layer
* apply Kaiming initialization explicitly
* expose a `forward` pass returning shape `(batch_size,)`

Suggested stub:

```python
import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLPRegressor(nn.Module):
    """Standard fully-connected regressor: R^p -> R."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [3, 4],
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming init to hidden linears, small-gain init to the output layer."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (batch_size,)."""
        raise NotImplementedError
```

#### `src/mlp_baseline/train.py`

Contains the reusable training loop, structurally parallel to
`approximation_eml.train` but without a temperature schedule.

Responsibilities:

* device placement (`train_mlp` owns this, same convention as
  `approximation_eml.train.train_model`)
* `AdamW` optimizer setup with weight decay
* epoch loop with `model.train()` / `model.eval()` mode switching
* validation evaluation
* early stopping on validation MSE
* gradient clipping
* metric tracking

Suggested stub:

```python
import torch
from approximation_eml.utils import get_device


def train_step(model, optimizer, x_batch, y_batch, config: dict) -> dict:
    raise NotImplementedError


def evaluate(model, x, y) -> dict:
    raise NotImplementedError


def train_mlp(model, x_train, y_train, x_val=None, y_val=None, config: dict | None = None):
    """Move model/data to device, run early-stopped training.

    Each epoch:
      1. run train_step in model.train() mode
      2. evaluate on validation set in model.eval() mode (if provided)
      3. track best validation MSE and best weights
      4. stop early if no improvement for config['patience'] checks
      5. log diagnostics every config['log_every'] epochs

    Return a history dict with per-epoch metrics, and restore best weights
    into `model` before returning.
    """
    raise NotImplementedError
```

#### `src/mlp_baseline/export.py`

Turns a trained model into a comparison-friendly summary. No symbolic
export — see "Export and Interpretation" above for why.

Suggested stub:

```python
from dataclasses import dataclass


@dataclass
class MLPSummary:
    param_count: int
    train_mse: float
    val_mse: float
    converged_epoch: int
    layer_weight_norms: list[float]
    feature_importance: list[float] | None = None


def count_parameters(model) -> int:
    raise NotImplementedError


def permutation_importance(model, x_val, y_val, n_repeats: int = 5) -> list[float]:
    """Shuffle each input column independently and measure the MSE increase."""
    raise NotImplementedError


def export_summary(model, history: list[dict], x_val=None, y_val=None) -> MLPSummary:
    raise NotImplementedError
```

#### `experiments/fit_toy_mlp.py`

Primary small entry-point script, structurally parallel to
`experiments/fit_toy.py`.

Responsibilities:

* instantiate one toy dataset (reusing `approximation_eml.data`)
* create one `MLPRegressor`
* train it
* print metrics and the `MLPSummary`

Suggested skeleton:

```python
from approximation_eml.data import make_dataset, target_square_first, train_val_split
from approximation_eml.utils import set_seed
from mlp_baseline.model import MLPRegressor
from mlp_baseline.train import train_mlp
from mlp_baseline.export import export_summary


def main():
    set_seed(0)
    x, y = make_dataset(target_square_first, n=512, input_dim=2)
    x_train, y_train, x_val, y_val = train_val_split(x, y, frac=0.8)
    model = MLPRegressor(input_dim=2, hidden_dims=[3, 4])
    history = train_mlp(model, x_train, y_train, x_val=x_val, y_val=y_val, config={})
    print(export_summary(model, history, x_val, y_val))


if __name__ == "__main__":
    main()
```

#### `experiments/fit_suite_mlp.py`

Runs the same Stage A–D toy targets used for the EML tree
(`approximation_eml.data.target_*`), training a fresh `MLPRegressor` on
each, and prints a summary table directly comparable to
`experiments/fit_suite.py`'s output.

Responsibilities:

* reuse the exact target function list from the EML tree's suite
* train one MLP config per target (default `hidden_dims=[3, 4]`, matched to
  the corresponding depth-2 EML tree's ~30-parameter budget for `p=2`; other
  `input_dim` values need re-derived widths per "Matching capacity to the
  EML tree")
* record final val MSE and parameter count per target
* print a summary table: rows = targets, columns = `{val_mse, param_count}`

#### `tests/test_mlp_model.py`

Minimal tests for the model module.

Responsibilities:

* forward-pass shape checks
* finite-output checks on random data
* parameter count sanity check for a known config

#### `tests/test_mlp_training.py`

Minimal tests for the training loop.

Responsibilities:

* successful small optimization step (loss decreases)
* early stopping actually halts training before `max_epochs`
* dropout is inactive during `model.eval()` (deterministic output on repeat
  calls)

## Recommended Implementation Order

## Step 1: minimal forward pass

Implement:

* configurable `Linear -> [BatchNorm1d] -> Activation -> Dropout` stack
* output layer
* explicit weight initialization

Success criterion:

* forward pass works for batch input and returns finite outputs, in both
  `train()` and `eval()` mode.

## Step 2: basic training loop

Implement regression training on a small toy target, reusing
`approximation_eml.data`.

Suggested first target:

* `f(x) = x_1`
* then `f(x) = x_1 + x_2`
* then a nonlinear target

Success criterion:

* model can reduce MSE on simple synthetic data.

## Step 3: add early stopping

Implement validation-driven early stopping and best-weight restoration.

Success criterion:

* training halts before `max_epochs` on an easy target without harming
  final validation MSE.

## Step 4: add regularization and logging

Wire up weight decay, dropout, gradient clipping, and debug logging.

Success criterion:

* training is stable across a range of dataset sizes without divergence.

## Step 5: comparison export

Implement `MLPSummary` and permutation importance.

Success criterion:

* able to produce a summary directly comparable to the EML tree's
  `TreeSummary` val-MSE and parameter count, even without a symbolic
  expression.

## Step 6: run toy experiments

Evaluate on the same Stage A–D target families as the EML tree suite, and
diff the two summary tables.

---

## Initial Toy Experiments

Reuse the exact same staged targets as the EML tree's suite
(`Implementation.md` for the EML tree — now `docs/exposition.md` — Stages
A–D), sourced from `approximation_eml.data`, so results are directly
comparable.

## Stage A: sanity checks

1. `f(x) = x_1`
2. `f(x) = 1`
3. `f(x) = x_2`

Goal:

* verify the network can trivially fit degenerate/near-linear targets
* establish a baseline parameter count and convergence speed

## Stage B: simple multivariate combinations

4. `f(x) = x_1 + x_2`
5. `f(x) = x_1 - x_2`
6. `f(x) = 2x_1 + 1`

Goal:

* confirm the network fits affine targets with negligible error, as expected
  of a much higher-capacity model than the EML tree at this stage

## Stage C: nonlinear targets

7. `f(x) = x_1^2`
8. `f(x) = \exp(x_1)`
9. `f(x) = \log(x_1 + 2)` on a safe domain
10. `f(x) = \sin(x_1)` over a bounded interval

Goal:

* measure whether the unconstrained MLP fits these targets faster / to
  lower error than the structurally-constrained EML tree, as a sanity check
  on how much the EML tree's interpretability costs in accuracy

## Stage D: mild multivariate nonlinear targets

11. `f(x) = x_1 x_2`
12. `f(x) = \exp(x_1 - x_2)`
13. `f(x) = \sin(x_1) + x_2`

Goal:

* same comparison as Stage C, extended to multivariate composition

---

## Open Questions for Later Versions

These are intentionally deferred.

### 1. Should a higher-capacity "conventional" MLP also be benchmarked?

Current version: no — `hidden_dims` defaults to a parameter-matched width
(`[3, 4]` for the depth-2/p=2 case, ~30 parameters, see "Matching capacity
to the EML tree"), specifically so raw capacity isn't a confound in the
comparison.

Later option:

* additionally run a conventional, unmatched-capacity MLP (e.g. `[32, 32]`)
  as a second, explicitly-labeled baseline, to separately ask "what does a
  practitioner get without thinking about parameter parity at all" — kept
  out of v1 to avoid conflating two different questions under one default

### 2. Should the baseline include any structural regularization at all?

Current version: no — dropout and weight decay only, no sparsity-inducing
penalty.

Later option:

* add an L1 penalty on first-layer weights as a crude analogue of the EML
  tree's leaf-selection sparsity, for a fairer interpretability comparison

### 3. Should input standardization be learned or fixed?

Current version: fixed, computed once from the training split before
training begins.

Later option:

* a learnable input normalization layer, or per-batch normalization only

### 4. Should the comparison include training-time / compute cost?

Current version: no — only final val MSE and parameter count are compared.

Later option:

* record wall-clock training time and epochs-to-convergence per target

### 5. How much of the accuracy gap (if any) is really about interpretability?

Current version is deliberately a blunt, conventional baseline.

Parameter count is matched by default (see "Matching capacity to the EML
tree"), which rules out the crudest confound. But depth-2 EML nodes have a
fixed fan-in of exactly 2, while a dense layer of matched total width has
much higher connectivity per parameter — so "same parameter count" is not
the same as "same inductive bias." Whether any remaining accuracy gap
reflects a real expressiveness cost of the EML tree's constraints, or this
connectivity difference, is intentionally left open pending Step 6 results.

---

## Clarifications Resolved So Far

These points are now fixed for version 1:

* The target class is `R^p -> R`, identical to the EML tree
* Computation is real-valued, using conventional dense layers
* The network has no soft-to-discrete relaxation and nothing is "snapped"
* Complexity control comes from dropout, weight decay, and early stopping —
  not structural gating
* Default hidden widths are parameter-matched to the corresponding EML
  tree's parameter count (`[3, 4]`, ~30 params, for the depth-2/p=2 case),
  not sized as a generic conventional width; BatchNorm is off by default for
  the same reason
* Network shape (depth/width) is fixed in advance per run
* Toy targets and data utilities are reused unmodified from
  `approximation_eml.data`
* Export produces a comparison summary, not a symbolic expression

---

## Suggested First Milestone

Implement a 2-hidden-layer MLP sized to match the EML tree's parameter
count (`hidden_dims=[3, 4]`, ~30 parameters for `p=2`, BatchNorm off) and
show that it can fit:

* `f(x) = x_1`
* `f(x) = x_1 + x_2`
* `f(x) = x_1^2`

while producing:

* finite activations,
* decreasing training and validation loss,
* early stopping triggering appropriately on easy targets,
* a summary table directly comparable to the EML tree's Stage A–C results.

If that works, the baseline is usable for meaningful comparison.

---

## Final Notes for the Coding Agent

Prioritize a small, readable implementation over generality — this is a
baseline, not the main contribution of the project.

The first implementation should optimize for:

1. clarity of module boundaries,
2. exact reuse of `approximation_eml.data` / `approximation_eml.utils`
   (no forked copies),
3. fair comparability with the EML tree's experiment suite,
4. ease of iteration.

Do not over-engineer architecture search, learning-rate scheduling, or a
symbolic-style export for this model. A correct, conventional, and
directly comparable MLP baseline is more valuable than an elaborately tuned
one that can no longer be cleanly compared against the EML tree.
