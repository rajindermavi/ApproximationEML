# Quick Reference: *All elementary functions from a single operator*

## What the paper claims

The paper argues that a single binary operator,

[
\operatorname{eml}(x,y) = e^x - \ln(y),
]

together with the constant `1`, is sufficient to generate the usual collection of elementary functions used in scientific calculation.

The headline analogy is to **NAND** in Boolean logic:

* NAND + wiring can generate all Boolean operations.
* EML + the constant `1` is claimed to generate the usual elementary-function repertoire in continuous mathematics.

The paper presents this as a constructive basis rather than just an existence theorem.

---

## Core operator

The central primitive is:

[
\operatorname{eml}(x,y) = e^x - \ln(y).
]

The paper’s point is not merely that EML is expressive, but that repeated composition of this one operator can reproduce familiar constants, arithmetic operations, and standard transcendental functions.

Examples highlighted in the paper include:

* exponentials,
* logarithms,
* arithmetic operations,
* powers,
* trigonometric and hyperbolic functions,
* standard constants such as `e`, `π`, and `i`.

---

## Why this is interesting

Normally we think of elementary mathematics as being built from many distinct primitives:

* addition,
* multiplication,
* division,
* exponentials,
* logarithms,
* trigonometric functions,
* roots.

The paper says these can all be generated from one repeated binary building block.

So the conceptual appeal is:

* a **uniform grammar** for elementary expressions,
* a possible analog of a universal gate for continuous mathematics,
* a new way to think about symbolic expressions as trees of a single repeated node type.

---

## Representation as trees

A key idea is that EML expressions can be written using a very simple grammar:

[
S \to 1 \mid \operatorname{eml}(S,S).
]

That means every expression is a binary tree whose internal nodes are all identical.

This matters because it gives a uniform structure for:

* symbolic representation,
* search over formulas,
* trainable differentiable models,
* possible hardware or compiler-oriented interpretations.

For our project, this tree viewpoint is the most relevant part.

---

## Relation to symbolic regression

The paper does not stop at the algebraic claim.

It also describes using **EML trees as trainable circuits** for symbolic regression.

The basic idea is:

* fix a tree shape,
* allow leaves and/or internal choices to be trainable,
* optimize parameters with gradient-based methods,
* try to recover a closed-form elementary expression from data.

The paper reports shallow-tree experiments in which exact elementary formulas can sometimes be recovered from numerical samples.

This is one of the main reasons the paper is useful for implementation work: it suggests EML is not only a theoretical basis, but also a possible trainable representation.

---

## Important implementation tension

The paper’s theoretical construction naturally lives in a broader setting that includes **complex values**, while many practical machine-learning experiments are easier to run in the **real domain**.

That creates a tension for implementation:

* the exact operator contains `ln(y)`,
* in real-valued code the log input must stay positive,
* naive real-valued training can become numerically unstable.

So practical implementations often need a surrogate or constraint on the log branch.

For our first implementation, this is one of the central design issues.

---

## What matters most for our project

The most relevant takeaways are:

1. **Uniform binary-tree structure**
   EML formulas can be represented as trees with one repeated internal node type.

2. **Leaves as primitives**
   A tree can be built from simple leaf ingredients such as constants and input variables.

3. **Trainable symbolic-ish architecture**
   The tree can be relaxed into a differentiable model and trained with backpropagation.

4. **Potential for exact recovery**
   If the target law is elementary, the trained tree may sometimes recover the actual formula, not just a numerical fit.

5. **Numerical care is essential**
   Exponentials and logarithms make optimization delicate.

---

## How our planned implementation differs

Our first implementation is inspired by the paper but is not a direct reproduction.

We are planning:

* `R^p -> R`
* real-valued computation only
* fixed-depth full binary tree
* leaves that softly select from `{1, x_1, ..., x_p}`
* optional substitution of constants inside the tree to reduce unnecessary complexity
* affine parameters at internal nodes
* a positivity-enforcing surrogate on the log branch for numerical safety

So our version is best understood as an **EML-inspired trainable tree architecture**, not an exact reconstruction of the paper’s full formalism.



