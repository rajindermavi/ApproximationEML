# EML Trees: How the Model Works

This document explains what an EML tree is, how it computes a prediction, and
how training turns a soft continuous relaxation into a discrete, readable
expression. It describes the system as built, not as a plan. 

---

## The idea in one sentence

An EML tree is a fixed-shape binary expression tree whose leaves softly pick
input variables and whose internal nodes combine their children with a single
recurring operator, `exp(u) - log(v)`; training simultaneously fits the data
and pushes every soft choice toward a discrete one, so the trained network can
be read back out as a symbolic formula.

The name comes from the operator it is built around:

```
eml(x, y) = e^x - ln(y)
```

drawn from the result that every elementary function can be expressed using
this single operator, composed with itself. The tree structure is the
composition mechanism; training is how the composition gets discovered from
data instead of hand-derived.

---

## What the model represents

The model is a function `f: R^p -> R`. It is built as a **full binary tree of
fixed depth**, chosen once at construction time (`EMLTree(input_dim, depth,
use_gates)`):

* Leaves sit at the bottom and each produce one scalar value derived from the
  input `x`.
* Internal nodes sit above them and each combine their two children into one
  scalar.
* The root's output is the model's prediction.

A depth-`d` tree has `2^d` leaves and `2^d - 1` internal nodes. Depth 2 (4
leaves, 3 internal nodes) is enough to represent things like `x_1^2` or
`sin(x_1) + x_2`, and is the depth used throughout the toy experiments.

Nodes are addressed by BFS index: the root is `0`, and node `i`'s children are
at `2i+1` and `2i+2`. Leaves occupy the bottom `2^d` BFS slots. This indexing
is what lets the tree be evaluated and exported with simple index arithmetic
rather than an explicit tree data structure.

---

## Leaves: soft selection over a small dictionary

Every leaf chooses among `p + 1` candidates: the constant `1`, and each input
coordinate `x_1 ... x_p`. Concretely, a leaf holds one logit per candidate.
Given a temperature `τ`, it computes

```
weights = softmax(logits / τ)
output  = sum(weights * [1, x_1, ..., x_p])
```

At `τ = 1` this is an ordinary softmax-weighted blend of the candidates — the
leaf's output is genuinely a mixture, not any one variable. As `τ` is annealed
toward 0 over training, the softmax sharpens: one candidate's weight goes to
~1 and the rest to ~0, so the leaf's output converges to literally being that
one variable (or the constant `1`).

This is the mechanism that lets the model *search* over which variable
belongs at a given position using gradient descent, rather than committing to
a variable assignment up front.

---

## Internal nodes: the EML operator

Each internal node receives two scalar children, `left` and `right`, and
produces

```
node(left, right) = exp(a * left + b) - log(softplus(c * right + d) + eps)
```

`a, b, c, d` are trainable scalars private to that node. `softplus` and the
small `eps` (1e-6) exist purely to keep the argument of `log` strictly
positive — real-valued autodiff cannot tolerate `log` of a nonpositive number,
so this is the concession that keeps the exact complex-valued EML operator
usable in a real, differentiable setting.

**The left/right roles are not symmetric, and that asymmetry is intentional.**
The left child always feeds the exponential branch; the right child always
feeds the (softplus-guarded) log branch. This mirrors the reference operator
`eml(x, y) = e^x - ln(y)` exactly, so a fully snapped, trained tree is
structurally a literal EML expression — not an approximation of one written in
different notation.

### Constant-substitution gates

Beyond the `a,b,c,d` affine parameters, each node optionally (`use_gates=True`,
the default) owns two gate logits, `g_left` and `g_right` — one per input
slot. Each gate produces a temperature-scaled sigmoid `s = sigmoid(g / τ)` and
interpolates that slot's value toward the constant `1`:

```
left'  = (1 - s_left)  * left  + s_left  * 1
right' = (1 - s_right) * right + s_right * 1
```

`s = 0` means "use the subtree as computed"; `s = 1` means "ignore that
subtree entirely and substitute the constant `1` instead." This is the
mechanism by which the tree can *prune itself*: if a branch's actual value
isn't needed to fit the data, its gate can drift toward 1 and the whole
subtree beneath it becomes irrelevant to the output, without changing the
tree's fixed topology. This is how the model reduces effective complexity
without dynamic tree growth or pruning logic — the topology is always full and
fixed, but gates make parts of it functionally inert.

---

## Evaluation

`EMLTree.forward` evaluates bottom-up in one pass:

1. Every leaf is evaluated against the input batch, using the current `τ`.
2. Internal nodes are processed from the deepest level up to the root; each
   reads its two already-computed children by BFS index and applies gates
   then the EML formula.
3. The root's value (BFS index 0) is the batch of predictions.

`τ` lives on the model as a non-trainable buffer (`model.tau`), not a training
config value passed around by hand — it is set once per epoch by the training
loop (`model.update_tau(tau)`) and read by every leaf and gate during that
epoch's forward passes. This keeps temperature synchronized across the whole
tree without threading it through every call.

---

## Training: fitting and discretizing at once

A single scalar loss combines five terms:

```
L = L_fit + λ_leaf * L_leaf + λ_gate * L_gate + λ_param * L_param + λ_safe * L_safe
```

* **`L_fit`** — plain MSE between prediction and target. This is the only
  term that cares about actually being *correct*; everything else is a prior
  pushing the soft structure toward something dischargeable into a discrete
  formula.

* **`L_leaf`** — mean Shannon entropy of each leaf's softmax distribution.
  Minimizing entropy pushes leaves toward a peaked, near-one-hot distribution
  — i.e., toward having actually chosen a variable rather than blending
  several.

* **`L_gate`** — mean `(1 - sigmoid(g))` across all gates. This is a constant
  bias toward `s -> 1`, i.e. toward collapsing branches to `1`. It's a prior
  for simplicity, not a correctness requirement: whenever a branch is
  actually load-bearing for fitting the data, gradient pressure from `L_fit`
  overwhelms this bias and the gate stays open.

* **`L_param`** — L2 penalty on each node's `a, b, c, d`. Keeps affine
  coefficients from drifting to extreme values that would make `exp` blow up
  or gradients explode.

* **`L_safe`** (off by default, `λ_safe = 0`) — `ReLU(margin - softplus(c*v+d))`
  averaged over nodes and batch. Penalizes the log argument for approaching
  zero *before* the epsilon floor catches it, giving the optimizer a smooth
  early warning instead of relying solely on the floor.

### The temperature schedule is what actually drives discretization

`τ` decays exponentially over the run, `τ(t) = τ_start * (τ_end/τ_start)^(t/T)`,
by default from `1.0` to `0.1`. This — not the entropy/gate penalties alone —
is the primary mechanism that turns soft mixtures into (near) one-hot choices:
at `τ = 0.1` a leaf with any real preference among its candidates will have
softmax probability above 0.99 on the winner. The entropy and gate penalties
act as a secondary nudge that breaks ties and discourages the model from
sitting on genuinely ambiguous mixtures for as long as possible.

An optional second phase (`hardening_epoch`) can swap in larger
`λ_leaf_hard` / `λ_gate_hard` penalty weights partway through training, for
cases where the temperature schedule alone doesn't produce confident enough
selections. In practice, for well-conditioned toy targets, the temperature
schedule by itself is normally sufficient.

---

## From trained weights to a readable formula

After training, `export_tree(model)` converts the (still technically
continuous) trained network into a `TreeSummary`:

* Each leaf's softmax distribution is inspected. If the top probability is
  above `leaf_threshold` (default 0.9), the leaf **snaps** to that primitive's
  name (e.g. `x_1`, or `1`). Below threshold it stays a labeled mixture like
  `~x_1(62%)`, which is the honest signal that training didn't fully resolve
  that position.
* Each gate's `sigmoid(g/τ)` is checked against `gate_threshold`. Above it,
  the gate is **collapsed** (that input becomes literally `1` in the exported
  formula); below `1 - gate_threshold`, it's **open** (the subtree is kept);
  in between, it's **uncertain**.
* The tree is then walked bottom-up, substituting snapped leaves and
  collapsed-to-1 gates directly into each node's `exp(...) - log(softplus(...))`
  formula, and folding fully-numeric subtrees down to a single number. The
  result is one inline expression string for the whole tree.

Because the left/right EML asymmetry is preserved structurally throughout
training (it's baked into the node formula, not learned), a fully-snapped
exported expression is not just *close to* a valid EML-grammar formula — it
*is* one, term for term.

---

## Numerical stability

`exp` and `log` are both capable of blowing up during optimization, so
several safeguards are structural rather than incidental:

* `softplus` guarantees the log argument is always positive before the
  `eps` floor is even applied.
* Node parameters `a, b, c, d` are initialized conservatively (`a, c` near
  1; `b, d` near 0, with small noise) so the tree starts close to the
  identity-ish region rather than an already-extreme one.
* Gate logits are initialized so `sigmoid(g) ≈ 0.1–0.2` — gates start mostly
  *open* (favoring real subtree values over the constant), leaving the
  "collapse to constant" behavior something training has to actively earn
  via `L_gate` and `L_fit`, not a default.
* Gradient norms are clipped (default max norm 1.0) every step.
* `L_param` discourages coefficient blow-up as a training-time pressure, and
  the optional `L_safe` term gives early warning on log arguments heading
  toward zero.

---

## What has actually been demonstrated

The toy experiment suite (`experiments/fit_suite.py`) runs a depth-2 tree
against thirteen target functions across four difficulty stages:

* **Stage A** (sanity): `x_1`, the constant `1`, `x_2` — checks that leaf
  selection and constant expression work at all.
* **Stage B** (affine): `x_1 + x_2`, `x_1 - x_2`, `2x_1 + 1` — checks that the
  node's affine parameters are used sensibly rather than just memorizing.
* **Stage C** (nonlinear, univariate): `x_1^2`, `exp(x_1)`, `log(x_1 + 2)`,
  `sin(x_1)` — checks approximation quality and stability when the target
  genuinely needs the exp/log machinery.
* **Stage D** (nonlinear, multivariate): `x_1 * x_2`, `exp(x_1 - x_2)`,
  `sin(x_1) + x_2` — checks composition and repeated-variable use across
  branches.

Each trial trains a fresh model, reports final validation MSE, and prints the
exported expression, giving a qualitative read on whether the fitted formula
is also an interpretable one — not just whether the MSE is low.

---

## What this model deliberately does not do (yet)

These are open, not overlooked:

* **No learned scalar constants beyond `1`.** A leaf can express `1` or a raw
  input variable, but not an arbitrary learned number like `2.7`. Affine
  coefficients on nodes give some of this expressiveness indirectly.
* **No dynamic tree shape.** Depth is fixed at construction. Effective
  complexity is only reduced via gates collapsing branches to `1`, never by
  actually growing or shrinking the tree.
* **No exact symbolic simplification.** The exported expression is a direct
  syntactic reading of the snapped tree, not an algebraically simplified one.
* **The softplus surrogate is permanent in this version.** Nothing projects a
  trained tree back toward the unrestricted complex-domain EML grammar after
  training.
* **Single output only.** The model is `R^p -> R`; there is no multi-output
  extension.

---

## Reading a trained tree

A fully-resolved depth-2 tree looks like this shape (values illustrative):

```
exp(0.99*x_1 + 0.00) - log(softplus(1.01*1 + 0.02) + 1e-6)
```

Every piece is traceable back to a specific leaf snap, gate snap, or affine
parameter — there is no hidden computation between "what the tree learned"
and "what the formula says," which is the entire point of constraining the
architecture this tightly in the first place.
