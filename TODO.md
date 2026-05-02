# TODO

Items collected from Implementation.md after comparing against the current codebase.

---

## Open Questions (explicitly deferred in Implementation.md)

### 1. Should leaves allow learned scalar constants?

Currently leaves select from `{1, x_1, ..., x_p}` only. The constant `1` is a fixed
primitive, not a free parameter. The open question is whether a leaf should also be
able to learn an arbitrary real-valued constant, distinct from the structural `1`.

Options noted in the doc:
- add dedicated learned-constant leaf parameters alongside the softmax selection

### 2. Should the tree grow dynamically?

Currently depth is fixed at construction time. The question is whether a later version
should start overparameterized and prune, or grow progressively during training.

Options noted in the doc:
- overparameterize and prune dead branches post-training
- progressive depth growth (add levels as residual improves)

### 3. Should there be stronger asymmetry between exp and log branches?

Currently only the log (right) branch receives positivity enforcement via softplus.
The left (exp) branch is unconstrained. The question is whether structural restrictions
on the right child should be tightened further.

### 4. Should the softplus surrogate be replaced after training?

The current real-valued node formula uses `softplus` to keep the log argument positive,
which diverges from the strict EML form `e^x - ln(y)`. The question is whether a
post-training projection step should attempt to recover a stricter EML expression.

### 5. How symbolic should the output remain?

The design sits between neural function approximation and symbolic regression. The
tension is intentional but unresolved: how much symbolic interpretability to require
vs. how much approximation to tolerate in practice.

---

## Implementation Gaps (spec specified, code differs)

### 6. Device handling is not in `train_model`

The spec (`Implementation.md`, `train.py` section) says:

> `train_model` calls `get_device()` at the start, moves the model with `model.to(device)`,
> and moves each `x_batch`/`y_batch` to the same device before every forward pass.
> Callers (experiment scripts) never need to call `.to(device)`.

Current `train.py` has no device placement. Everything runs on whatever device the
caller put tensors on (CPU by default). `get_device()` exists in `utils.py` but is
unused by `train_model`.

Resolution: add device logic to `train_model`, or explicitly decide that device
handling stays with the caller and update the spec accordingly.

### 7. `collect_diagnostics` does not return the fields specified

The spec (`utils.py` section) lists the expected return keys:

- `log_args` — per-node log branch arguments before the epsilon floor
- `node_outputs` — per-node scalar outputs in BFS order
- `min_log_arg` — minimum log argument across all nodes and batch elements
- `nan_inf_fraction` — fraction of node outputs that are NaN or Inf
- `gate_means` — mean `sigmoid(g)` per gate, BFS order
- `leaf_entropies` — entropy of each leaf's softmax distribution

Current `collect_diagnostics` returns flat per-leaf and per-node keys
(`leaf_i_probs`, `leaf_i_argmax`, `node_i_gate_left`, `node_i_gate_right`) and
is missing `log_args`, `node_outputs`, `min_log_arg`, and `nan_inf_fraction` entirely.
This means the stability diagnostics (NaN fraction, minimum log argument) that the
spec calls out as important debug metrics are not collected.

### 8. Expression export format differs from spec

The spec shows a hierarchical indented format:

```
node[0]: exp(1.02 * node[1] + 0.01) - log(softplus(-0.98 * node[2] + 0.03) + 1e-6)
  node[1]: exp(0.99 * x_1 + 0.00) - log(softplus(1.01 * 1 + 0.02) + 1e-6)
  node[2]: ...
    leaves: x_1 (conf=0.97), 1 (conf=0.94), ...
```

with unresolved leaves shown as `leaf[i]?`. The current `_build_expression` produces
a single flat inline string by substituting child expressions recursively, and uses
`~x_1(62%)` for uncertain leaves rather than `leaf[i]?`. This is readable but does
not match the spec's intended indented tree format.

### 9. `fit_suite.py` runs one config per target, not a target × config grid

The spec says:

> define a list of named config dicts (e.g. varying depth, tau schedule, lambda weights)
> for each (target, config) pair: train a fresh model ...
> print a compact summary table: rows = targets, columns = configs, cells = val MSE

The current `fit_suite.py` runs each target against a single config (with per-target
epoch overrides) and prints a single-column results table. The multi-config grid
dimension — comparing e.g. depth=2 vs depth=3, or different lambda schedules — is
not implemented.

The spec does note this was intentionally deferred until `fit_toy.py` behavior was
understood, so this is the natural next step once the Stage A–D baselines are available.
