"""GOLEM-EV, then optionally GOLEM-NV initialised from it (gcastle); subprocess of dagscale.golem.fit_golem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run(args: argparse.Namespace) -> int:
    import numpy as np
    from castle.algorithms import GOLEM

    config = json.loads(Path(args.config).read_text())
    seed = int(config["seed"])
    equal_variances = bool(config["equal_variances"])
    num_iter = float(config["num_iter"])
    learning_rate = float(config["learning_rate"])

    inputs = np.load(args.input)
    x = np.asarray(inputs["x"], dtype=np.float64)

    def _train(lambda1: float, ev_objective: bool, b_init: np.ndarray | None) -> np.ndarray:
        model = GOLEM(
            B_init=b_init,
            lambda_1=lambda1,
            lambda_2=5.0,
            equal_variances=ev_objective,
            learning_rate=learning_rate,
            num_iter=num_iter,
            checkpoint_iter=None,
            seed=seed,
            device_type="cpu",
        )
        model.learn(x)
        return np.asarray(model.weight_causal_matrix, dtype=np.float64)

    ev_solution = _train(2e-2, True, None)
    weights = ev_solution if equal_variances else _train(2e-3, False, ev_solution)

    np.savez(args.output, weights=weights)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(_run(_build_parser().parse_args()))
