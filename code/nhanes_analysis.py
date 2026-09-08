"""
NHANES 2017-2018 application: Poisson regression of the number of
prescription medications (RXDCOUNT) on body weight, with self-reported
weight (WHD020, converted to kg) as the error-prone covariate and measured
weight (BMXWT) as the gold-standard anchor.

Design (matching the manuscript's Section 6 plan):
  - classical error variance estimated from the self-report minus measured
    differences: sigma2_hat = Var(W - X)   [the survey is its own
    validation subsample: both reports are observed]
  - methods: naive, regression calibration, Nakamura corrected score,
    standard SIMEX, Adaptive SIMEX (1-SE and argmin), measured-weight fit
    (oracle anchor)
  - sensitivity analysis: sigma2 = m * sigma2_hat, m in {0.25,...,2.0}

Outputs: results/nhanes_table.tex, results/fig4_sensitivity.pdf/png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import asimex_lib as L

DATA = HERE.parent / "data"
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

LB2KG = 0.45359237
MASTER_SEED = 20260905


def load_and_merge():
    demo = pd.read_sas(DATA / "DEMO_J.XPT", format="xport")
    bmx = pd.read_sas(DATA / "BMX_J.XPT", format="xport")
    whq = pd.read_sas(DATA / "WHQ_J.XPT", format="xport")
    rxq = pd.read_sas(DATA / "RXQ_RX_J.XPT", format="xport")
    rxq = rxq.drop_duplicates(subset="SEQN")   # one row per person

    df = (demo[["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR"]]
          .merge(bmx[["SEQN", "BMXWT", "BMXHT"]], on="SEQN", how="inner")
          .merge(whq[["SEQN", "WHD020"]], on="SEQN", how="inner")
          .merge(rxq[["SEQN", "RXDCOUNT"]], on="SEQN", how="left"))

    # clean: positive plausible weights, non-missing covariates and count
    df = df.rename(columns={
        "RIDAGEYR": "age", "RIAGENDR": "sex", "INDFMPIR": "pir",
        "BMXWT": "w_meas", "BMXHT": "height", "WHD020": "w_self_lb",
        "RXDCOUNT": "nmeds"})
    df["w_self"] = df["w_self_lb"] * LB2KG
    df = df[(df["w_self"] > 30) & (df["w_self"] < 350)]
    df = df[(df["w_meas"] > 30) & (df["w_meas"] < 350)]
    df["nmeds"] = df["nmeds"].fillna(0)   # persons without an RXQ record
    df = df.dropna(subset=["age", "sex", "height", "w_meas",
                           "w_self"])
    df["pir_missing"] = df["pir"].isna().astype(float)
    df["pir"] = df["pir"].fillna(df["pir"].median())
    df["nmeds"] = df["nmeds"].fillna(0).clip(upper=60).astype(int)
    return df.reset_index(drop=True)


def poisson_fit(Y, X):
    beta = L.poisson_irls(Y, X)
    var = L.poisson_sandwich(Y, X, beta)
    return beta, np.sqrt(np.diag(var))


def main():
    df = load_and_merge()
    print(f"analysis sample: n = {len(df)}")

    d = df["w_self"] - df["w_meas"]
    sigma2_hat = float(d.var(ddof=1))
    bias_hat = float(d.mean())
    print(f"self-report error: mean {bias_hat:+.2f} kg, "
          f"variance {sigma2_hat:.2f} kg^2, sd {np.sqrt(sigma2_hat):.2f} kg")

    Y = df["nmeds"].to_numpy(float)
    w_self = df["w_self"].to_numpy()
    w_meas = df["w_meas"].to_numpy()
    Z = np.column_stack([
        np.ones(len(df)), df["age"], (df["sex"] == 2.0).astype(float),
        df["height"] / 100.0, df["pir"],
        df["pir_missing"],
    ])
    names = ["Intercept", "Age (10y)", "Female", "Height (m)",
             "Income ratio"]
    names_w = ["Weight (10 kg)"]

    # rescale weight to per-10kg for readability
    w_self10 = w_self / 10.0
    w_meas10 = w_meas / 10.0
    # error variance on the 10kg scale
    sigma2_10 = sigma2_hat / 100.0

    # oracle anchor: measured weight (observed for everyone here)
    X_or = np.column_stack([Z, w_meas10])
    b_or, se_or = poisson_fit(Y, X_or)

    def run_with_sigma2(sigma2_w):
        me_cols = [Z.shape[1]]          # weight is the last column
        W = np.column_stack([Z, w_self10])
        rng = np.random.default_rng(np.random.SeedSequence(
            MASTER_SEED, spawn_key=(77, int(round(sigma2_w * 1e4)))))
        out = {}
        b_naive = L.poisson_irls(Y, W)
        out["naive"] = (b_naive,
                        np.sqrt(np.diag(L.poisson_sandwich(Y, W, b_naive))))
        b_rc, v_rc = L.regression_calibration(Y, W, [sigma2_w], me_cols)
        out["rc"] = (b_rc, np.sqrt(np.diag(v_rc)))
        b_cs, v_cs = L.nakamura_corrected_score(Y, W, [sigma2_w], me_cols,
                                               beta_init=b_naive)
        out["cs"] = (b_cs, np.sqrt(np.diag(v_cs)))
        curve, info, beta_all = L.simex_curve(
            Y, W, me_cols, [sigma2_w], L.STD_LAM, 100, rng,
            beta_init=b_naive)
        d_std = L.combined_weights(L.STD_LAM, 2)
        b_ss = d_std @ curve
        v_ss, _ = L.simex_sandwich(Y, W, me_cols, [sigma2_w], L.STD_LAM,
                                   curve, info, beta_all, 2, weights=d_std)
        out["std_simex"] = (b_ss, np.sqrt(np.diag(v_ss) / len(Y)))
        picks = L.cv_select(Y, W, me_cols, [sigma2_w], rng, B_cv=25)
        lm_1se, dg_1se, lm_am, dg_am, _, _ = picks
        for tag, (lm, dg) in [("adaptive", (lm_1se, dg_1se)),
                              ("adaptive_am", (lm_am, dg_am))]:
            grid = np.arange(0.0, lm + 1e-9, 0.5)
            curve, info, beta_all = L.simex_curve(
                Y, W, me_cols, [sigma2_w], grid, 100, rng,
                beta_init=b_naive)
            dd = L.combined_weights(grid, dg)
            bb = dd @ curve
            vv, _ = L.simex_sandwich(Y, W, me_cols, [sigma2_w], grid,
                                     curve, info, beta_all, dg, weights=dd)
            out[tag] = (bb, np.sqrt(np.diag(vv) / len(Y)))
            out[f"{tag}_pick"] = (lm, dg)
        return out

    base = run_with_sigma2(sigma2_10)

    rows = []
    label = {
        "naive": "Naive (self-reported weight)",
        "rc": "Regression calibration",
        "cs": "Corrected score (Nakamura)",
        "std_simex": "Standard SIMEX",
        "adaptive": "Adaptive SIMEX (1-SE rule)",
        "adaptive_am": "Adaptive SIMEX (argmin)",
        "oracle": "Measured weight (anchor)",
    }
    order = ["naive", "rc", "cs", "std_simex", "adaptive", "adaptive_am",
             "oracle"]
    for m in order:
        if m == "oracle":
            b, se = b_or, se_or
            pick = "--"
        else:
            b, se = base[m]
            pick = base.get(f"{m}_pick", "--")
        wbeta = b[-1] * 10.0          # per-10kg, on 10kg scale coefficient
        wse = se[-1] * 10.0
        rows.append((label[m], wbeta, wse, str(pick)))
        print(f"{label[m]:34s} beta_w = {wbeta:+.3f} "
              f"({wse:.3f})  pick={pick}")

    (OUT / "nhanes_results.json").write_text(json.dumps({
        "n": int(len(df)), "sigma2": sigma2_hat, "bias": bias_hat,
        "rows": [{"label": lab, "b": float(b), "se": float(se)}
                 for lab, b, se, _ in rows]}))
    with open(OUT / "nhanes_results.json") as f:
        pass

    # ---- sensitivity over sigma2 multipliers (adaptive + std) ----
    mults = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sens = []
    for mlt in mults:
        o = run_with_sigma2(sigma2_10 * mlt)
        sens.append((mlt, o["std_simex"][0][-1] * 10.0,
                     o["adaptive"][0][-1] * 10.0,
                     o["cs"][0][-1] * 10.0))
        print(f"  sigma2 x{mlt}: std {sens[-1][1]:+.3f}, "
              f"adaptive {sens[-1][2]:+.3f}, cs {sens[-1][3]:+.3f}")

    # ---- LaTeX table ----
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{NHANES 2017--2018 application: Poisson regression of the",
        "number of prescription medications on body weight (coefficient per",
        "10\\,kg, robust standard errors). The error-prone covariate is",
        "self-reported weight; the classical error variance is estimated",
        "from the self-report minus measured differences",
        f"($\\hat\\sigma^2_\\epsilon = {sigma2_hat:.1f}\\,\\mathrm{{kg}}^2$,",
        "mean difference " f"${bias_hat:+.1f}\\,\\mathrm{{kg}}$). The",
        "measured-weight fit is shown as the gold-standard anchor.}",
        "\\label{tab:nhanes}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Method & $\\hat\\beta$ (10\\,kg) & SE & Selected $(\\lambda_{\\max},$"
        " degree$)$ \\\\", "\\midrule"]
    for lab, b, se, pick in rows:
        pk = "--" if pick == "--" else f"({pick[0]}, {pick[1]})"
        lines.append(f"{lab} & {b:+.3f} & ({se:.3f}) & {pk} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "nhanes_table.tex").write_text("\n".join(lines))

    # ---- sensitivity figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    s = np.array(sens)
    ax.plot(s[:, 0], s[:, 1], "o-", label="Standard SIMEX")
    ax.plot(s[:, 0], s[:, 2], "s-", label="Adaptive SIMEX (1-SE)")
    ax.plot(s[:, 0], s[:, 3], "^-", label="Corrected score")
    ax.axhline(b_or[-1] * 10.0, color="k", ls="--", lw=1,
               label="Measured weight (anchor)")
    ax.axhline(float((base["naive"][0][-1] * 10.0)), color="gray", ls=":",
               lw=1, label="Naive")
    ax.set_xlabel(r"Error-variance multiplier $m$ "
                  r"($\sigma^2_\epsilon = m\,\hat\sigma^2_\epsilon$)")
    ax.set_ylabel(r"$\hat\beta$ per 10 kg")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sensitivity.pdf")
    fig.savefig(OUT / "fig4_sensitivity.png", dpi=150)
    print("saved results/nhanes_table.tex and fig4_sensitivity.*")


if __name__ == "__main__":
    main()
