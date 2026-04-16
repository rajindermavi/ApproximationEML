# EML Tree First Implementation Plan

## Goal

Build a first working implementation of a trainable **EML tree** for scalar-output regression:

[
f: \mathbb{R}^p \to \mathbb{R}
]

using **real-valued computation only**, a **fixed-depth full binary tree**, and **gradient-based training**.

This first version is meant to answer a narrow question:

> Can a constrained real-valued EML tree learn useful scalar functions from data when its symbolic ingredients are chosen softly and later snapped to discrete structure?

This is an exploratory implementation, not a production symbolic regression system.

---

## Core Design Decisions

### Function class

We will model a scalar-valued function of a vector input:

[
x = (x_1, \dots, x_p) \in \mathbb{R}^p,
\qquad
f(x) \in \mathbb{R}.
]

### Tree topology

Use a **full binary tree of fixed depth**.

* Leaves provide primitive inputs.
* Internal nodes combine child outputs.
* The root produces the final scalar output.

We will **not** implement dynamic tree growth in version 1.

### Leaf primitives

Each leaf will softly choose from the dictionary:

[
{1, x_1, \dots, x_p}.
]

The choice will be parameterized by learnable logits and converted to probabilities with **softmax**.

During evaluation/export, leaves can later be **snapped** to a discrete choice.

### Internal-node composition

Each internal node combines two child values using a real-valued EML-inspired operation.

We are deliberately staying in the real domain, so the log branch must remain strictly positive.

A first practical parameterization is:

[
\text{node}(u,v) = \exp(a u + b) - \log(\phi(c v + d) + \varepsilon)
]

where:

* (a,b,c,d) are trainable scalars for the node,
* (\phi) is a positivity-enforcing function such as `softplus`,
* (\varepsilon > 0) is a small constant.

This is not the exact unrestricted symbolic EML grammar, but it is a practical real-valued surrogate suitable for optimization.

### Complexity mitigation

We want the model to be able to simplify itself.

Version 1 will support this in two ways:

1. **Constant-capable leaves** via the primitive `1`
2. **Mid-tree substitution of constants** via learned gates that can replace a subtree input by `1`

So each internal node should be able to use either:

* the child output, or
* the constant `1`

through a soft gate.

This lets the model collapse unnecessary structure instead of forcing every branch to carry information.

---

## First-Version Scope

### Included

* Real-valued PyTorch implementation
* Fixed-depth full binary tree
* Leaf soft selection over `{1, x_1, ..., x_p}`
* Optional internal gating toward constant `1`
* Trainable affine parameters at internal nodes
* Standard supervised regression training loop
* Basic regularization
* Snapping/export of learned discrete structure
* Small synthetic experiments

### Excluded for version 1

* Complex-valued training
* Dynamic tree growth
* Reinforcement-style architecture search
* Exact symbolic simplification engine
* Multi-output `R^p -> R^m` support
* Full benchmark suite
* Noise-robust symbolic recovery guarantees

---

## Proposed Architecture

## 1. Leaf layer

Each leaf has logits of size `p + 1`, corresponding to:

* index 0: constant `1`
* indices 1..p: coordinates `x_1, ..., x_p`

For input batch `X` of shape `(batch_size, p)`, construct candidate tensor:

* `ones`: shape `(batch_size, 1)`
* `X`: shape `(batch_size, p)`
* `candidates = concat([ones, X], dim=1)` with shape `(batch_size, p+1)`

For each leaf:

* apply softmax to leaf logits
* compute weighted sum over candidates

This yields one scalar output per leaf per batch element.

### Snapping

At export time, each leaf can be snapped to `argmax` if the max probability exceeds a threshold.

---

## 2. Internal node

Each internal node receives two scalar child outputs `(u, v)`.

### Node parameters

Each node has:

* `a, b` for the exponential branch
* `c, d` for the log branch
* optional left gate logit `g_left`
* optional right gate logit `g_right`

### Constant-substitution gates

For each child input, define gate value with sigmoid:

[
s = \sigma(g)
]

and interpolate:

[
\tilde{u} = (1-s_L)u + s_L \cdot 1,
\qquad
\tilde{v} = (1-s_R)v + s_R \cdot 1.
]

So:

* `s = 0` means use the subtree output
* `s = 1` means replace that input by constant `1`

### Node formula

The output is:

[
out = \exp(a \tilde{u} + b) - \log(\phi(c \tilde{v} + d) + \varepsilon)
]

with:

* `phi = softplus`
* `epsilon = 1e-6` or similar

### Why this design

This keeps the implementation:

* fully differentiable,
* real-valued,
* expressive enough to be interesting,
* constrained enough to remain interpretable.

---

## 3. Tree module

The tree module should:

* instantiate `2^depth` leaves
* instantiate `2^depth - 1` internal nodes
* evaluate bottom-up
* return one scalar output per batch row

### Recommended interface

```python
class EMLTree(nn.Module):
    def __init__(self, input_dim: int, depth: int, use_gates: bool = True):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...  # shape: (batch_size,)
```

---

## Training Objective

Use supervised regression loss:

[
\mathcal{L} = \mathcal{L}*{fit} + \lambda*{leaf} \mathcal{L}*{leaf} + \lambda*{gate} \mathcal{L}*{gate} + \lambda*{param} \mathcal{L}*{param} + \lambda*{safe} \mathcal{L}_{safe}
]

## 1. Fit loss

Start with standard MSE:

[
\mathcal{L}_{fit} = \frac{1}{N}\sum_i (\hat y_i - y_i)^2
]

## 2. Leaf regularization

Encourage confident leaf selections.

Options:

* entropy penalty on leaf softmax distributions
* later hardening schedule

A simple first choice is:

* penalize average entropy of leaf distributions

## 3. Gate regularization

Encourage gates toward 0 or 1 and optionally favor pruning.

Options:

* entropy penalty on gate sigmoids
* sparsity bias encouraging some branches to collapse to `1`

## 4. Parameter regularization

Use weight decay or explicit L2 penalty on node parameters `a,b,c,d`.

This reduces wild coefficients and numerical instability.

## 5. Safety regularization

Even with `softplus`, it may help to penalize very small log arguments before the final epsilon floor.

Example:

[
\mathcal{L}_{safe} = \frac{1}{N} \sum \mathrm{ReLU}(\delta - \phi(c\tilde v+d))
]

for some small margin `delta`.

This is optional in the first pass, but worth leaving room for.

---

## Training Strategy

## Phase 1: soft training

Train all parameters continuously:

* leaf logits
* gate logits
* node affine parameters

Use Adam.

Recommended initial defaults:

* optimizer: `Adam`
* learning rate: `1e-3`
* batch size: full-batch for small synthetic datasets, mini-batch otherwise
* gradient clipping: `1.0`

## Phase 2: hardening

After basic convergence:

* increase entropy penalties,
* inspect leaf and gate probabilities,
* optionally freeze/snap confident selections,
* continue training remaining continuous parameters.

This does not need to be fully automated in version 1.

A manual hardening pass is acceptable.

---

## Initialization

Initialization matters because exponentials and logs can destabilize quickly.

### Leaf logits

Initialize near uniform with small noise.

### Gate logits

Initialize slightly toward using child outputs rather than constants.

For example, initialize gate logits so `sigmoid(g)` is around `0.1` to `0.2`.

### Node parameters

Initialize conservatively:

* `a, c` near `1`
* `b, d` near `0`
* small random perturbations

Possible example:

* `a = 1 + 0.05 * noise`
* `c = 1 + 0.05 * noise`
* `b = 0.05 * noise`
* `d = 0.05 * noise`

Avoid large magnitudes initially.

---

## Numerical Stability Plan

This is a first-class concern.

### Risks

* exponential blow-up
* log argument approaching zero
* exploding gradients
* highly redundant parameterizations

### Mitigations

* `softplus` on the log branch
* `epsilon` floor inside the log
* conservative initialization
* gradient clipping
* optional clamping of internal activations for debugging
* regularization on coefficients

### Debug metrics to log

Per training run, log:

* fit loss
* parameter norms
* gate means
* leaf entropy
* min log argument
* fraction of NaN/Inf activations, if any

---

## Export and Interpretation

We want a usable symbolic-ish readout after training.

## Leaf export

For each leaf:

* convert softmax probabilities into chosen primitive
* if confidence below threshold, record as soft/uncertain

## Gate export

For each gate:

* if near 1, replace subtree input with constant `1`
* if near 0, keep child
* otherwise mark as unresolved soft gate

## Expression export

Recursively export the tree into a string form such as:

```text
exp(a*u + b) - log(softplus(c*v + d) + eps)
```

or a symbolic pseudocode tree.

For version 1, plain text export is enough.

A later version can add SymPy conversion.

---

## Repository Structure

Use a small, readable layout with short experiment scripts and most logic imported from a few utility modules.

```text
eml_tree/
  README.md
  pyproject.toml

  src/
    eml_tree/
      __init__.py
      components.py
      model.py
      losses.py
      data.py
      export.py
      train.py
      utils.py

  experiments/
    fit_toy.py
    fit_suite.py

  tests/
    test_components.py
    test_model.py
```

This is intentionally compact. The goal is to keep the project easy to inspect while still separating:

* local mathematical building blocks,
* overall tree composition,
* training,
* toy-data generation,
* export and snapping.

### File responsibilities

#### `src/eml_tree/__init__.py`

Keep this minimal. Export the main public objects.

Suggested contents:

* `EMLTree`
* `train_model`
* `export_tree`

Suggested stub:

```python
from .model import EMLTree
from .train import train_model
from .export import export_tree
```

#### `src/eml_tree/components.py`

Holds the small reusable computational pieces.

Responsibilities:

* leaf soft-selection over `{1, x_1, ..., x_p}`
* internal EML node with affine parameters
* optional gate logic for replacing inputs with constant `1`

Suggested stubs:

```python
import torch
import torch.nn as nn


class SoftLeaf(nn.Module):
    """Softly selects one primitive from {1, x_1, ..., x_p}."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(input_dim + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one scalar per batch row."""
        raise NotImplementedError

    def probs(self) -> torch.Tensor:
        """Return softmax probabilities over primitives."""
        raise NotImplementedError


class EMLNode(nn.Module):
    """Internal node computing a real-valued EML surrogate."""

    def __init__(self, use_gates: bool = True, eps: float = 1e-6):
        super().__init__()
        self.use_gates = use_gates
        self.eps = eps

    def apply_gates(self, left: torch.Tensor, right: torch.Tensor):
        """Optionally interpolate child values with constant 1."""
        raise NotImplementedError

    def log_argument(self, right: torch.Tensor) -> torch.Tensor:
        """Return the positive argument passed to log."""
        raise NotImplementedError

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute exp(a*u+b) - log(softplus(c*v+d)+eps)."""
        raise NotImplementedError
```

#### `src/eml_tree/model.py`

Defines the full fixed-depth binary tree.

Responsibilities:

* instantiate leaves and internal nodes
* perform bottom-up evaluation
* expose helper methods for inspection

Suggested stubs:

```python
import torch
import torch.nn as nn

from .components import SoftLeaf, EMLNode


class EMLTree(nn.Module):
    """Fixed-depth scalar-output EML tree for R^p -> R."""

    def __init__(self, input_dim: int, depth: int, use_gates: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.use_gates = use_gates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (batch_size,)."""
        raise NotImplementedError

    def leaf_modules(self):
        """Return leaves in a stable order for inspection/export."""
        raise NotImplementedError

    def node_modules(self):
        """Return internal nodes in a stable order for inspection/export."""
        raise NotImplementedError
```

#### `src/eml_tree/losses.py`

Holds the main loss terms and regularizers.

Responsibilities:

* fit loss
* leaf entropy penalty
* gate entropy or pruning penalty
* parameter norm penalty
* optional safety penalty for log arguments

Suggested stubs:

```python
import torch


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def leaf_entropy_penalty(model) -> torch.Tensor:
    raise NotImplementedError


def gate_penalty(model) -> torch.Tensor:
    raise NotImplementedError


def parameter_penalty(model) -> torch.Tensor:
    raise NotImplementedError


def safety_penalty(model, x: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
    raise NotImplementedError


def total_loss(model, x: torch.Tensor, y: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    """Return total scalar loss and a metrics dict."""
    raise NotImplementedError
```

#### `src/eml_tree/data.py`

Generates small synthetic regression datasets.

Responsibilities:

* toy target functions
* train/validation splits
* bounded-domain sampling

Suggested stubs:

```python
import torch


def sample_box(n: int, input_dim: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    raise NotImplementedError


def make_dataset(fn, n: int, input_dim: int, low: float = -1.0, high: float = 1.0):
    """Return X, y for a callable target function."""
    raise NotImplementedError


def target_identity(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def target_sum(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def target_square_first(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def train_val_split(x: torch.Tensor, y: torch.Tensor, frac: float = 0.8):
    raise NotImplementedError
```

#### `src/eml_tree/export.py`

Turns the learned model into something readable.

Responsibilities:

* snap leaf softmax choices to discrete primitives
* snap gate decisions when confident
* recursively build a readable expression string
* expose confidence summaries

Suggested stubs:

```python
def snap_leaf(leaf, threshold: float = 0.9):
    raise NotImplementedError


def snap_gate(prob: float, threshold: float = 0.9):
    raise NotImplementedError


def export_tree(model, leaf_threshold: float = 0.9, gate_threshold: float = 0.9) -> str:
    """Return a readable text expression for the current tree."""
    raise NotImplementedError


def summarize_structure(model) -> dict:
    raise NotImplementedError
```

#### `src/eml_tree/train.py`

Contains the reusable training loop.

Responsibilities:

* optimizer setup
* epoch loop
* optional validation
* gradient clipping
* metric tracking

Suggested stubs:

```python
import torch


def train_step(model, optimizer, x_batch: torch.Tensor, y_batch: torch.Tensor, config: dict) -> dict:
    raise NotImplementedError


def evaluate(model, x: torch.Tensor, y: torch.Tensor, config: dict) -> dict:
    raise NotImplementedError


def train_model(model, x_train: torch.Tensor, y_train: torch.Tensor, x_val=None, y_val=None, config: dict | None = None):
    """Train the model and return a history dictionary."""
    raise NotImplementedError
```

#### `src/eml_tree/utils.py`

Small general helpers only.

Responsibilities:

* seeding
* device selection
* simple logging helpers
* formatting metrics

Suggested stubs:

```python
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    raise NotImplementedError


def get_device() -> torch.device:
    raise NotImplementedError


def to_python_float_dict(metrics: dict) -> dict:
    raise NotImplementedError
```

#### `experiments/fit_toy.py`

Primary small entry-point script.

Responsibilities:

* instantiate one toy dataset
* create one model
* train it
* print metrics
* print exported expression

Suggested skeleton:

```python
from eml_tree.data import make_dataset, target_square_first
from eml_tree.model import EMLTree
from eml_tree.train import train_model
from eml_tree.export import export_tree, summarize_structure
from eml_tree.utils import set_seed


def main():
    set_seed(0)
    x, y = make_dataset(target_square_first, n=512, input_dim=2)
    model = EMLTree(input_dim=2, depth=2, use_gates=True)
    history = train_model(model, x, y, config={})
    print(history)
    print(summarize_structure(model))
    print(export_tree(model))


if __name__ == "__main__":
    main()
```

#### `experiments/fit_suite.py`

Optional lightweight script for running several toy targets in sequence.

Responsibilities:

* compare a few targets under the same configuration
* save or print a compact summary

This is optional for the first pass.

#### `tests/test_components.py`

Minimal tests for local modules.

Responsibilities:

* shape checks for leaves and nodes
* finite-output checks
* probability normalization checks

#### `tests/test_model.py`

Minimal tests for the whole tree.

Responsibilities:

* forward-pass shape
* finite outputs on random data
* successful small optimization step

## Recommended Implementation Order

## Step 1: minimal forward pass

Implement:

* leaf module with softmax over `{1, x_1, ..., x_p}`
* internal node with affine EML surrogate
* fixed-depth tree forward pass

Success criterion:

* forward pass works for batch input and returns finite outputs.

## Step 2: basic training loop

Implement regression training on a small toy target.

Suggested first target:

* `f(x) = x_1`
* then `f(x) = x_1 + x_2`
* then a nonlinear target

Success criterion:

* model can reduce MSE on simple synthetic data.

## Step 3: add constant-substitution gates

Implement mid-tree replacement of child values with `1`.

Success criterion:

* unnecessary branches can collapse during training.

## Step 4: add regularization and logging

Implement entropy penalties, parameter penalty, debug logging.

Success criterion:

* training becomes more stable and outputs more interpretable structures.

## Step 5: snapping/export

Implement confidence-based snapping for leaves and gates.

Success criterion:

* able to inspect a mostly discrete learned tree after training.

## Step 6: run toy experiments

Evaluate on small families of functions.

---

## Initial Toy Experiments

Run in increasing order of difficulty.

## Stage A: sanity checks

1. `f(x) = x_1`
2. `f(x) = 1`
3. `f(x) = x_2`

Goal:

* verify leaf selection works
* verify constants can be expressed cleanly

## Stage B: simple multivariate combinations

4. `f(x) = x_1 + x_2`
5. `f(x) = x_1 - x_2`
6. `f(x) = 2x_1 + 1`

Goal:

* see whether affine parameters are being used sensibly

## Stage C: nonlinear targets

7. `f(x) = x_1^2`
8. `f(x) = \exp(x_1)`
9. `f(x) = \log(x_1 + 2)` on a safe domain
10. `f(x) = \sin(x_1)` over a bounded interval

Goal:

* explore approximation behavior and stability

## Stage D: mild multivariate nonlinear targets

11. `f(x) = x_1 x_2`
12. `f(x) = \exp(x_1 - x_2)`
13. `f(x) = \sin(x_1) + x_2`

Goal:

* test whether repeated variable usage and composition are effective

---

## Open Questions for Later Versions

These are intentionally deferred.

### 1. Should leaves also allow learned scalar constants?

Current version: no, except for primitive `1` and mid-tree constant substitution.

Later option:

* allow dedicated learned-constant leaves

### 2. Should the tree grow dynamically?

Current version: no.

Later option:

* overparameterize and prune
* or progressive depth growth

### 3. Should there be asymmetry between the exp branch and log branch?

Current version: only the log branch gets positivity enforcement.

Later option:

* stronger structural restrictions on the right child

### 4. Should we keep the softplus surrogate forever?

Current version: yes.

Later option:

* attempt projection toward a stricter EML form after training

### 5. How symbolic should this remain?

Current version is deliberately between symbolic regression and neural function fitting.

This tension is part of the experiment.

---

## Clarifications Resolved So Far

These points are now fixed for version 1:

* The target class is `R^p -> R`
* Computation is real-valued
* Leaves softly select from `{1, x_1, ..., x_p}`
* Selection will later be snapped
* Internal nodes may substitute constants in the middle of the tree to reduce unnecessary complexity
* Tree shape is fixed in advance
* We are not reproducing the exact complex-domain paper implementation
* We are pursuing a cleaner, more expressive affine-node variant for exploratory purposes

---

## Suggested First Milestone

Implement a depth-2 or depth-3 model and show that it can fit:

* `f(x) = x_1`
* `f(x) = x_1 + x_2`
* `f(x) = x_1^2`

while producing:

* finite activations,
* decreasing training loss,
* interpretable leaf selections,
* at least partial branch collapse through constant substitution.

If that works, the project is viable.

---

## Final Notes for the Coding Agent

Prioritize a small, readable implementation over generality.

The first implementation should optimize for:

1. clarity of module boundaries,
2. numerical safety,
3. inspectability of learned structure,
4. ease of iteration.

Do not over-engineer tree search, symbolic simplification, or benchmarking in the initial pass.

A correct and inspectable depth-2/depth-3 prototype is more valuable than a flexible but opaque framework.
