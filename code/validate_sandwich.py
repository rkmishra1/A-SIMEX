"""Bootstrap validation of the analytic SIMEX sandwich (parallel)."""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import asimex_lib as L

beta0 = np.array([1.0, -0.5, 0.8])
CELLS = [(100, 0.1), (100, 1.0), (500, 0.5), (500, 1.0), (1000, 0.5),
         (1000, 1.0)]
R, NB = 100, 60


def one_cell(cell):
    n, s2 = cell
    ana = np.zeros((R, 3))
    emp = np.zeros((R, 3))
    master = np.random.SeedSequence(20260905, spawn_key=(99, n, int(s2 * 100)))
    seeds = master.spawn(R)
    d = L.combined_weights(L.STD_LAM, 2)
    for r, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        X = rng.standard_normal((n, 3))
        Y = rng.poisson(np.exp(X @ beta0))
        W = X + rng.standard_normal((n, 3)) * np.sqrt(s2)
        me = [0, 1, 2]
        bn = L.poisson_irls(Y, W)
        curve, info, ba = L.simex_curve(Y, W, me, [s2] * 3, L.STD_LAM, 100,
                                        rng, beta_init=bn)
        v, _ = L.simex_sandwich(Y, W, me, [s2] * 3, L.STD_LAM, curve, info,
                                ba, 2, weights=d)
        ana[r] = np.diag(v)
        bb = np.empty((NB, 3))
        for k in range(NB):
            rb = np.random.default_rng(np.random.SeedSequence(
                20260905, spawn_key=(98, n, int(s2 * 100), r, k)))
            Xb = rb.standard_normal((n, 3))
            Yb = rb.poisson(np.exp(Xb @ beta0))
            Wb = Xb + rb.standard_normal((n, 3)) * np.sqrt(s2)
            cb, _, _ = L.simex_curve(Yb, Wb, me, [s2] * 3, L.STD_LAM, 100,
                                     rb, beta_init=L.poisson_irls(Yb, Wb))
            bb[k] = d @ cb
        emp[r] = np.diag(np.cov(bb.T)) * n
    ratio = ana.mean(0) / emp.mean(0)
    return (n, s2, np.round(ratio, 2).tolist())


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        for (n, s2, ratio) in ex.map(one_cell, CELLS):
            print(f"n={n:5d} s2={s2}: analytic/bootstrap variance ratio "
                  f"per coefficient = {ratio}", flush=True)
