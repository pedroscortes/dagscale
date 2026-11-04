"""Self-loops and cycles between distinct variables in the thresholded DAGMA outputs at x100 (Section 1).

The 90 synthetic data sets of methods.py (ER2/ER4/SF4, weight scale 0.5/1/2, seeds 30-39), multiplied by 100;
threshold 0.3. Usage: python scripts/selfloop_x100.py <workers> results/selfloop_x100/selfloop_x100.json
"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS","cpu")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
def job(a):
    from methods import fit, has_cycle
    from simulate import make_data
    g,w,s=a
    X,A=make_data(g,30,3000,s,w); W=fit("dagma",X*100.0)
    B=(np.abs(W.T)>0.3).astype(int); loops=int(np.trace(B)); np.fill_diagonal(B,0)
    return dict(g=g,w=w,s=s,selfloops=loops,cyclic=bool(has_cycle(B)))
if __name__=="__main__":
    jobs=[(g,w,s) for g in ("ER2","ER4","SF4") for w in (0.5,1.0,2.0) for s in range(30,40)]
    with ProcessPoolExecutor(int(sys.argv[1])) as ex: res=list(ex.map(job,jobs))
    json.dump(res,open(sys.argv[2],"w"))
    print("fits",len(res),"with self-loop",sum(r["selfloops"]>0 for r in res),"cyclic",sum(r["cyclic"] for r in res))
