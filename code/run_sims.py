"""
Monte Carlo driver for the Adaptive SIMEX simulations.

Scenarios (manuscript Section 5 + additions):
  S1  low-dimensional, strong signal : n in {100,300,500,1000}, p=3,
      sigma2 in {0.1,0.5,1.0}, M=1000            -> Tables 1 / Figure 1
  S2  misspecified error (Laplace DGP, Gaussian assumed), n=500, p=10,
      M=1000                                     -> Table 2 / Figure 2
  S3  scalability timing, n=200, p in {5,15,30,50}, M=100 -> Table 3 / Fig 3
  T4  negative-binomial outcomes (overdispersion), n=500, p=3, M=1000
  T5  estimated error variance from m=2 replicated measures, n=500, p=3

Master seed: 20260905. Every repetition's RNG derives from
SeedSequence(20260905, spawn_key=(scenario_id, cell_id, rep)) so all
results are exactly reproducible.

Usage:  python3 run_sims.py <scenario> [--M 1000]
        python3 run_sims.py all
"""
from __future__ import annotations

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import asimex_lib as L

MASTER_SEED = 20260905
RESULTS = Path(__file__).parent.parent / "results" / "raw"
RESULTS.mkdir(parents=True, exist_ok=True)

BETA1 = np.array([1.0, -0.5, 0.8])
BETA2 = np.array([1.0, 0.7, -0.5, 0.3, 0, 0, 0, 0, 0, 0])
METHODS = ["naive", "rc", "cs", "std_simex", "adaptive", "adaptive_am", "oracle"]

SCENARIOS = {
    "S1": dict(n_list=[100, 300, 500, 1000], sigma2_list=[0.1, 0.5, 1.0],
               p=3, beta=BETA1, M=1000, sid=1),
    "S2": dict(n_list=[500], sigma2_list=[0.5], p=10, beta=BETA2, M=1000,
               sid=2, laplace=True),
    "S3": dict(n_list=[200], sigma2_list=[0.3], p=None, beta=None, M=100,
               sid=3, timing=True, p_list=[5, 15, 30, 50]),
    "T4": dict(n_list=[500], sigma2_list=[0.5], p=3, beta=BETA1, M=1000,
               sid=4, nb_size=2.0),
    "T5": dict(n_list=[500], sigma2_list=[0.5], p=3, beta=BETA1, M=1000,
               sid=5, replicates=2),
}


def generate_data(rng, n, p, beta, sigma2, laplace=False, nb_size=None,
                  replicates=None):
    """Draw one Monte Carlo dataset; returns (Y, W, X, sigma2_used)."""
    X = rng.standard_normal((n, p))
    if nb_size is None:
        Y = rng.poisson(np.exp(np.clip(X @ beta, -30, 30)))
    else:
        mu = np.exp(np.clip(X @ beta, -30, 30))
        Y = rng.negative_binomial(nb_size, nb_size / (nb_size + mu))
    if laplace:
        tau = np.sqrt(np.asarray(sigma2) / 2.0)   # Var = 2 tau^2, per coord
        eps = rng.laplace(0.0, tau[None, :], size=(n, len(sigma2)))
    else:
        eps = rng.standard_normal((n, len(sigma2))) * np.sqrt(sigma2)
    if replicates is None:
        W = X.copy()
        for j, col in enumerate(range(len(sigma2))):
            W[:, col] += eps[:, j]
        return Y, W, X, sigma2
    # replicated measures: Wbar observed, individual error variance estimated
    W1 = X.copy(); W2 = X.copy()
    for j in range(len(sigma2)):
        W1[:, j] += eps[:, j]
        W2[:, j] += rng.standard_normal(n) * np.sqrt(sigma2[j])
    Wbar = (W1 + W2) / 2.0
    d = W1 - W2
    sigma2_hat = np.mean(d ** 2) / 2.0            # estimates sigma2 (per var)
    sigma2_used = np.full(len(sigma2), sigma2_hat / 2.0)  # error var of Wbar
    return Y, Wbar, X, sigma2_used


def run_chunk(job):
    """Run a chunk of MC repetitions for one scenario cell."""
    (scen, cell_key, rep_lo, rep_hi) = job
    cfg = SCENARIOS[scen]
    sid = cfg["sid"]
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=(sid,) + tuple(cell_key))
    rep_seeds = ss.spawn(rep_hi - rep_lo)

    if cfg.get("timing"):
        p = cell_key[0]
        beta = np.array([1.0 / j for j in range(1, p + 1)])
        sigma2 = np.full(p, cfg["sigma2_list"][0])
        n = cfg["n_list"][0]
    else:
        p = cfg["p"]
        beta = cfg["beta"]
        n = cell_key[0]
        sigma2 = np.full(p, cell_key[1] / 100.0)
        if scen == "S2":
            sigma2 = np.zeros(p); sigma2[:3] = cell_key[1] / 100.0

    M = rep_hi - rep_lo
    res = {m: {"b": np.full((M, p), np.nan), "se": np.full((M, p), np.nan),
               "t": np.full(M, np.nan)} for m in METHODS}
    picks = np.full((M, 4), np.nan)
    n_fail = 0

    for r in range(M):
        try:
            rng = np.random.default_rng(rep_seeds[r])
            Y, W, X, s2_used = generate_data(
                rng, n, p, beta, sigma2, laplace=cfg.get("laplace", False),
                nb_size=cfg.get("nb_size"),
                replicates=cfg.get("replicates"))
            me_cols = list(range(len(sigma2)))
            out = {}
            for meth, kw in [
                ("naive", None), ("rc", None), ("cs", None),
                ("std_simex", None), ("adaptive", None)]:
                try:
                    o = L.run_methods(
                        Y, W, s2_used, me_cols, rng, B=100, B_cv=25,
                        use_adaptive=(meth == "adaptive"),
                        only=meth)
                    out.update(o)
                except Exception:
                    pass
            try:
                b_or, se_or = L.oracle_fit(Y, X)
                res["oracle"]["b"][r] = b_or
                res["oracle"]["se"][r] = se_or
            except Exception:
                pass
            if "adaptive_pick" in out:
                picks[r] = out["adaptive_pick"]
            n_ok = 0
            for m in METHODS:
                if m == "oracle":
                    continue
                if m in out:
                    b, se = out[m]
                    if np.all(np.isfinite(b)) and np.all(np.isfinite(se)):
                        res[m]["b"][r] = b
                        res[m]["se"][r] = se
                        res[m]["t"][r] = out.get(f"{m}_time", np.nan)
                        n_ok += 1
            if n_ok == 0:
                n_fail += 1
        except Exception:
            n_fail += 1
            continue

    tag = "_".join(str(k) for k in cell_key)
    np.savez_compressed(
        RESULTS / f"{scen}_{tag}_{rep_lo}_{rep_hi}.npz",
        **{f"{m}_{k}": v for m in METHODS
           for k, v in res[m].items()},
        picks=picks, n_fail=n_fail, cell=json.dumps(cell_key),
        beta=beta, sigma2=sigma2)
    return f"{scen} {tag} reps {rep_lo}-{rep_hi} done ({n_fail} failures)"


def chunks_for(scen, chunk=100):
    cfg = SCENARIOS[scen]
    jobs = []
    if cfg.get("timing"):
        for p in cfg["p_list"]:
            for lo in range(0, cfg["M"], chunk):
                jobs.append((scen, (p,), lo, min(lo + chunk, cfg["M"])))
    else:
        for n in cfg["n_list"]:
            for s2 in cfg["sigma2_list"]:
                for lo in range(0, cfg["M"], chunk):
                    jobs.append((scen, (n, int(round(s2 * 100))), lo,
                                 min(lo + chunk, cfg["M"])))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()
    if args.M:
        for k in (args.scenario,) if args.scenario != "all" else SCENARIOS:
            SCENARIOS[k]["M"] = args.M

    scens = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    jobs = []
    for sc in scens:
        jobs += chunks_for(sc, args.chunk)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(run_chunk, jobs):
            print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)
    print(f"ALL DONE in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
