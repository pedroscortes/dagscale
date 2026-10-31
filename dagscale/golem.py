"""GOLEM-EV/NV (gcastle) in a subprocess (scripts/run_golem.py), with the interpreter $GOLEM_PYTHON if set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_golem.py"


def fit_golem(
    samples: np.ndarray,
    *,
    equal_variances: bool = False,
    seed: int = 0,
    num_iter: float = 1e5,
    learning_rate: float = 1e-3,
    timeout: float | None = None,
) -> np.ndarray:
    """GOLEM-EV, or GOLEM-NV initialised from GOLEM-EV; returns W[source, target]."""
    x = np.asarray(samples, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="golem-") as tmp:
        tmp_dir = Path(tmp)
        inputs_path = tmp_dir / "inputs.npz"
        output_path = tmp_dir / "output.npz"
        config_path = tmp_dir / "config.json"
        np.savez(inputs_path, x=x)
        config = {
            "seed": seed,
            "equal_variances": equal_variances,
            "num_iter": num_iter,
            "learning_rate": learning_rate,
        }
        config_path.write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            os.environ.get("GOLEM_PYTHON", sys.executable),
            str(_RUNNER),
            "--input",
            str(inputs_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, check=True)
        return np.asarray(np.load(output_path)["weights"], dtype=np.float64)
