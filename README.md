# dagscale

Code and results for P. Cortes dos Santos and R. A. Krohling, "The impact of data scale and barrier schedule on acyclicity in continuous DAG learning".

## Install

```
uv venv --python 3.11
uv pip install -e .
```

This installs CPU builds of jax and torch. TOPO and the log-likelihood DAGMA variant (`topo`, `dagma_logll*` in `scripts/methods.py`) need the dagrad package, which is not redistributed here because of its non-commercial licence; install it by hand with `uv pip install --no-deps "dagrad @ git+https://github.com/Duntrain/dagrad.git@4ce4c5a"` (`--no-deps` keeps the pinned notears). GOLEM runs in a subprocess (`scripts/run_golem.py`); set `GOLEM_PYTHON` to an interpreter with gcastle and torch if the current one should not be used.

## Check data before fitting

```python
from dagscale.check import check, noise_units

print(check(X))          # ok / too small / too large / spread, from the covariance alone
X_fit = noise_units(X)   # divide by the median conditional standard deviation
```

## Reproduce

All commands run from the repository root. Each run script writes to `results/<script name>/`; `WORKERS` (or `--workers`) sets the number of processes.

1. Experiments (seed ranges as in the reported runs):

```
python scripts/scale_window.py                                    # seeds 30-39
SEED_LO=40 SEED_HI=100 OUT=results/scale_window/scale_window_s40_99.csv python scripts/scale_window.py
python scripts/leak_law_check.py
python scripts/predict.py                                         # seeds 30-39
SEED_LO=40 SEED_HI=80 OUT=results/predict/predict_s40_79.csv python scripts/predict.py
python scripts/predict_newton.py
python scripts/predict_lasso.py
python scripts/floor_alarm.py results/floor_alarm/floor_alarm.csv
python scripts/masked_diagonal.py
python scripts/methods.py                                         # seeds 30-39
METHODS=topo,dagma_logll,analytic2 OUT=results/methods/methods_recent.csv python scripts/methods.py
METHODS=dagma_logll VARIANTS=0.01,1,100,resc OUT=results/methods/methods_logll.csv python scripts/methods.py
METHODS=dagma_logll,dagma_logll_zero,dagma_logll_nnorm VARIANTS=0.01,1,100 \
  OUT=results/methods/methods_logll_control.csv python scripts/methods.py
METHODS=sortnregress,r2sortnregress VARIANTS=0.01,1,100,resc,std OUT=results/methods/methods_sorters.csv python scripts/methods.py
METHODS=dagma,dagma_nnorm,notears,colide_ev,colide_nv,topo,dagma_logll,analytic2 VARIANTS=std \
  OUT=results/methods/methods_std.csv python scripts/methods.py
METHODS=golem_ev,golem_nv VARIANTS=std OUT=results/methods/methods_std_golem.csv python scripts/methods.py
python scripts/light_tunnel.py                                    # downloads the data with causalchamber
METHODS=topo,dagma_logll,analytic2 OUT=results/light_tunnel/light_tunnel_recent.csv python scripts/light_tunnel.py
METHODS=dagma,dagma_logll OUT=results/light_tunnel/light_tunnel_rerun.csv python scripts/light_tunnel.py
python scripts/noise_units.py
python scripts/noise_units_real.py > results/noise_units_real/readout.txt
python scripts/spread.py
python scripts/simulate.py --seeds 30-49
python scripts/objective_gap.py
python scripts/schedule.py
python scripts/newton_cg.py
python scripts/selfloop_share.py
python scripts/intro_example.py
```

The files stored gzipped in `results/` (`*.csv.gz`) are the gzipped outputs of the commands above.

2. Tables, in-text numbers and figures from `results/`:

```
python scripts/make_tables.py     # results/tables/*.tex and numbers.txt (runs table_real.py)
python scripts/figures.py         # figures/*.pdf
```

| Paper item | Script |
|---|---|
| Fig. 2, Table S3 (scale window) | scale_window.py |
| Fig. 3 (leak law) | leak_law_check.py |
| Table 3 (prediction) | predict.py, floor_alarm.py, predict_newton.py, predict_lasso.py |
| Fig. 4, Tables 4 and S4, Section S5 (methods) | methods.py |
| Table S5 (noise units) | noise_units.py |
| Tables 5 and S6 (schedule) | simulate.py, schedule.py, objective_gap.py |
| Table S7 (spread) | spread.py |
| Section S6 (Newton-CG) | newton_cg.py |
| Table S8 (masked diagonal) | masked_diagonal.py |
| Table 6 (real data) | methods.py, light_tunnel.py, noise_units_real.py, table_real.py |
| Section 3.4 (self-loop share) | selfloop_share.py |

## Data

`data/sachs/` holds the Sachs et al. (2005) flow-cytometry data. The Causal Chambers light-tunnel data (Gamella, Peters and Buhlmann 2025, CC BY 4.0) is downloaded by `light_tunnel.py` into `data/causal_chambers/` (or `CC_ROOT`).

## Licence

MIT (see LICENSE). `scripts/third_party/colide_model.py` is from github.com/SAMiatto/colide under the Apache License 2.0 (`scripts/third_party/LICENSE-colide`).
