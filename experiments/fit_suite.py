"""Toy experiment suite — evaluates EMLTree on staged families of functions.

Stages A–D run in increasing order of difficulty per Implementation.md §Step 6.
"""

from dataclasses import dataclass

from approximation_eml.data import (
    make_dataset,
    train_val_split,
    target_identity,
    target_constant,
    target_x2,
    target_sum,
    target_diff,
    target_affine,
    target_square_x1,
    target_exp_x1,
    target_log_safe,
    target_sin_x1,
    target_product,
    target_exp_diff,
    target_sin_plus_x2,
)
from approximation_eml.export import export_tree
from approximation_eml.model import EMLTree
from approximation_eml.train import train_model
from approximation_eml.utils import set_seed


@dataclass
class TrialResult:
    stage: str
    name: str
    val_mse: float
    expression: str


# Base training config shared by all trials.
_BASE = {
    "lr": 1e-3,
    "epochs": 2000,
    "tau_start": 1.0,
    "tau_end": 0.05,
    "lambda_leaf": 0.01,
    "lambda_gate": 0.01,
    "lambda_param": 1e-3,
    "verbose": True,
    "log_every": 500,
}

# (stage, display_name, target_fn, tree_depth, config_overrides)
SUITE = [
    # Stage A — sanity checks: verify leaf selection and constant expression
    ("A", "f(x)=x_1",          target_identity,    2, {}),
    ("A", "f(x)=1",             target_constant,    2, {}),
    ("A", "f(x)=x_2",           target_x2,          2, {}),
    # Stage B — affine combinations: check that node parameters are used sensibly
    ("B", "f(x)=x_1+x_2",       target_sum,         2, {}),
    ("B", "f(x)=x_1-x_2",       target_diff,        2, {}),
    ("B", "f(x)=2*x_1+1",       target_affine,      2, {}),
    # Stage C — nonlinear targets: approximation quality and stability
    ("C", "f(x)=x_1^2",         target_square_x1,   2, {"epochs": 3000}),
    ("C", "f(x)=exp(x_1)",      target_exp_x1,      2, {}),
    ("C", "f(x)=log(x_1+2)",    target_log_safe,    2, {}),
    ("C", "f(x)=sin(x_1)",      target_sin_x1,      2, {"epochs": 3000}),
    # Stage D — multivariate nonlinear: composition and repeated variable use
    ("D", "f(x)=x_1*x_2",       target_product,     2, {"epochs": 3000}),
    ("D", "f(x)=exp(x_1-x_2)",  target_exp_diff,    2, {}),
    ("D", "f(x)=sin(x_1)+x_2",  target_sin_plus_x2, 2, {"epochs": 3000}),
]


def _status(val_mse: float) -> str:
    if val_mse < 1e-4:
        return "PASS"
    if val_mse < 0.01:
        return "GOOD"
    if val_mse < 0.1:
        return "POOR"
    return "FAIL"


def run_trial(stage: str, name: str, fn, depth: int, config: dict, seed: int = 0) -> TrialResult:
    set_seed(seed)
    x, y = make_dataset(fn, n=512, input_dim=2)
    x_train, y_train, x_val, y_val = train_val_split(x, y, frac=0.8)
    model = EMLTree(input_dim=2, depth=depth, use_gates=True)
    history = train_model(model, x_train, y_train, x_val=x_val, y_val=y_val, config=config)
    final = history[-1]
    summary = export_tree(model)
    return TrialResult(
        stage=stage,
        name=name,
        val_mse=final.get("val_mse", float("nan")),
        expression=summary.expression,
    )


def main():
    results: list[TrialResult] = []
    current_stage = None

    for stage, name, fn, depth, overrides in SUITE:
        if stage != current_stage:
            current_stage = stage
            print(f"\n{'=' * 60}")
            print(f"  Stage {stage}")
            print(f"{'=' * 60}")

        config = {**_BASE, **overrides}
        print(f"\n-- {name}  (depth={depth}, epochs={config['epochs']}) --")

        result = run_trial(stage, name, fn, depth, config)
        results.append(result)

        print(f"   val_mse  : {result.val_mse:.6f}  [{_status(result.val_mse)}]")
        print(f"   expr     : {result.expression}")

    # Final summary table
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'St':<3} {'Name':<22} {'val_mse':>10}  Status")
    print(f"  {'-' * 3} {'-' * 22} {'-' * 10}  {'-' * 6}")
    for r in results:
        print(f"  {r.stage:<3} {r.name:<22} {r.val_mse:>10.6f}  {_status(r.val_mse)}")


if __name__ == "__main__":
    main()
