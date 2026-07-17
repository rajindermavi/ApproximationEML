# Changelog

## 2026-07-16

- Added `src/approximation_eml/config.py` (`EMLConfig`/`DEFAULT_CONFIG`) as the
  single source of truth for tunable defaults; `components.py`, `losses.py`,
  `model.py`, and `train.py` now read their defaults from it instead of
  hardcoding the same literals in multiple places.
- Updated `config.md` to state that `config.py` is authoritative and that the
  documented defaults are read from it, not maintained independently.
- Added `docs/exposition.md`, a narrative description of how the EML tree
  works as built (leaf selection, the `exp(u) - log(v)` node operator,
  constant-substitution gates, temperature annealing).
- Rewrote `Implementation.md` from the original EML tree implementation plan
  into a new implementation plan for an MLP baseline model, intended for
  comparison against the EML tree (parameter count matched, same toy
  targets). This is planning only — no MLP code has been added yet.
