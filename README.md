# EML Feed Forward Tree 

## Status

The EML tree described below is implemented. See [docs/exposition.md](docs/exposition.md)
for how it works as built, and [config.md](config.md) for the tunable defaults
(sourced from `src/approximation_eml/config.py`). [Implementation.md](Implementation.md)
now tracks the plan for an MLP baseline used to compare against this tree; it
no longer documents the EML tree itself.

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