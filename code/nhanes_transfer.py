"""
Second application: temporal transfer of the calibrated error variance.

NHANES 2015-2016 (cycle H), same design as the 2017-2018 application:
Poisson regression of the number of prescription medications on body
weight, with self-reported weight as the error-prone covariate. The
exercise asks whether the error variance calibrated on the 2017-2018
cycle transfers out of sample, and whether a within-cycle calibration
changes the conclusions.

Outputs: results/transfer_table.tex, results/fig8_transfer.pdf/png,
results/transfer_results.json
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
SIGMA2_1718 = 24.05          # calibrated on the 2017-18 cycle (kg^2)


def load_cycle(suffix):
    demo = pd.read_sas(DATA / f"DEMO_{suffix}.XPT", format="xport")
    bmx = pd.read_sas(DATA / f"BMX_{suffix}.XPT", format="xport")
    whq = pd.read_sas(DATA / f"WHQ_{suffix}.XPT", format="xport")
    rxq = pd.read_sas(DATA / f"RXQ_RX_{suffix}.XPT", format="xport")
    rxq = rxq.drop_duplicates(subset="SEQN")   # one row per person
    df = (demo[["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR"]]
          .merge(bmx[["SEQN", "BMXWT", "BMXHT"]], on="SEQN", how="inner")
          .merge(whq[["SEQN", "WHD020"]], on="SEQN", how="inner")
          .merge(rxq[["SEQN", "RXDCOUNT"]], on="SEQN", how="left"))
    df = df.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex",
                            "INDFMPIR": "pir", "BMXWT": "w_meas",
                            "BMXHT": "height", "WHD020": "w_self_lb",
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


def run_all(Y, Z, w_self10, sigma2_w, tag):
    rng = np.random.default_rng(np.random.SeedSequence(
        MASTER_SEED, spawn_key=(88, int(round(sigma2_w * 1e4)))))
    W = np.column_stack([Z, w_self10])
    me_cols = [Z.shape[1]]
    out = {}
    b_naive = L.poisson_irls(Y, W)
    out["naive"] = (b_naive, np.sqrt(np.diag(
        L.poisson_sandwich(Y, W, b_naive))))
    b_rc, v_rc = L.regression_calibration(Y, W, [sigma2_w], me_cols)
    out["rc"] = (b_rc, np.sqrt(np.diag(v_rc)))
    b_cs, v_cs = L.nakamura_corrected_score(Y, W, [sigma2_w], me_cols,
                                           beta_init=b_naive)
    out["cs"] = (b_cs, np.sqrt(np.diag(v_cs)))
    curve, info, beta_all = L.simex_curve(Y, W, me_cols, [sigma2_w],
                                          L.STD_LAM, 100, rng,
                                          beta_init=b_naive)
    d = L.combined_weights(L.STD_LAM, 2)
    out["std_simex"] = (d @ curve,
                        np.sqrt(np.diag(L.simex_sandwich(
                            Y, W, me_cols, [sigma2_w], L.STD_LAM, curve,
                            info, beta_all, 2, weights=d)[0]) / len(Y)))
    lm1, dg1, lma, dga, _, _ = L.cv_select(Y, W, me_cols, [sigma2_w], rng,
                                           B_cv=25)
    for tag2, (lm, dg) in [("adaptive", (lm1, dg1)),
                           ("adaptive_am", (lma, dga))]:
        grid = np.arange(0.0, lm + 1e-9, 0.5)
        curve, info, beta_all = L.simex_curve(Y, W, me_cols, [sigma2_w],
                                              grid, 100, rng,
                                              beta_init=b_naive)
        dd = L.combined_weights(grid, dg)
        vv, _ = L.simex_sandwich(Y, W, me_cols, [sigma2_w], grid, curve,
                                 info, beta_all, dg, weights=dd)
        out[tag2] = (dd @ curve, np.sqrt(np.diag(vv) / len(Y)))
        out[f"{tag2}_pick"] = (lm, dg)
    out["tag"] = tag
    return out


def main():
    df = load_cycle("H")
    n = len(df)
    print(f"2015-2016 analysis sample: n = {n}")
    d = df["w_self"] - df["w_meas"]
    sigma2_1516 = float(d.var(ddof=1))
    bias_1516 = float(d.mean())
    print(f"within-cycle error: mean {bias_1516:+.2f} kg, "
          f"variance {sigma2_1516:.2f} kg^2 (2017-18 calibration: "
          f"{SIGMA2_1718:.2f})")

    Y = df["nmeds"].to_numpy(float)
    w_self10 = (df["w_self"] / 10.0).to_numpy()
    w_meas10 = (df["w_meas"] / 10.0).to_numpy()
    Z = np.column_stack([
        np.ones(n), df["age"], (df["sex"] == 2.0).astype(float),
        df["height"] / 100.0, df["pir"].fillna(df["pir"].median())])
    X_or = np.column_stack([Z, w_meas10])
    b_or, se_or = poisson_fit(Y, X_or)

    res = {}
    res["transferred"] = run_all(Y, Z, w_self10, SIGMA2_1718 / 100.0,
                                 "transferred")
    res["within"] = run_all(Y, Z, w_self10, sigma2_1516 / 100.0, "within")

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
    rows = []
    for m in order:
        if m == "oracle":
            rows.append((label[m], b_or[-1] * 10.0, se_or[-1] * 10.0,
                         b_or[-1] * 10.0, se_or[-1] * 10.0, "--"))
            continue
        bt, st = res["transferred"][m]
        bw, sw = res["within"][m]
        rows.append((label[m], bt[-1] * 10.0, st[-1] * 10.0,
                     bw[-1] * 10.0, sw[-1] * 10.0,
                     str(res["transferred"].get(f"{m}_pick", "--"))))
        print(f"{label[m]:34s} transferred {bt[-1]*10:+.3f} ({st[-1]*10:.3f})"
              f"  within {bw[-1]*10:+.3f} ({sw[-1]*10:.3f})")

    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Temporal transfer, NHANES 2015--2016 ($n=%d$):" % n,
        "Poisson regression of the number of prescription medications on",
        "self-reported weight (coefficient per 10\\,kg, robust standard",
        "errors). The transferred column feeds the methods the error",
        "variance calibrated on the 2017--2018 cycle",
        "($\\hat\\sigma^2_\\epsilon = " + f"{SIGMA2_1718:.1f}" + "\\,\\mathrm{kg}^2$);",
        "the within column re-estimates it from the 2015--2016",
        "self-report minus measured differences",
        "($\\hat\\sigma^2_\\epsilon = " + f"{sigma2_1516:.1f}" + "\\,\\mathrm{kg}^2$).}",
        "\\label{tab:transfer}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & \\multicolumn{2}{c}{Transferred $\\hat\\sigma^2_\\epsilon$} & "
        "\\multicolumn{2}{c}{Within-cycle $\\hat\\sigma^2_\\epsilon$} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Method & $\\hat\\beta$ & SE & $\\hat\\beta$ & SE \\\\",
        "\\midrule"]
    for lab, bt, st, bw, sww, pick in rows:
        lines.append(f"{lab} & {bt:+.3f} & ({st:.3f}) & {bw:+.3f} & "
                     f"({sww:.3f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "transfer_table.tex").write_text("\n".join(lines))

    # json dump for the forest figure
    dump = {"n": n, "sigma2_1516": sigma2_1516, "sigma2_1718": SIGMA2_1718,
            "bias_1516": bias_1516,
            "methods": {m: {"b_t": float(res["transferred"][m][0][-1] * 10)
                            if m in res["transferred"] else float(b_or[-1] * 10),
                            "s_t": float(res["transferred"][m][1][-1] * 10)
                            if m in res["transferred"] else float(se_or[-1] * 10),
                            "b_w": float(res["within"][m][0][-1] * 10)
                            if m in res["within"] else float(b_or[-1] * 10),
                            "s_w": float(res["within"][m][1][-1] * 10)
                            if m in res["within"] else float(se_or[-1] * 10)}
                        for m in order}}
    (OUT / "transfer_results.json").write_text(json.dumps(dump, indent=1))

    # ---- figure: grouped dot-and-whisker ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ypos = np.arange(len(order))
    for off, key, col, mk in [(-0.17, "t", "#1a9850", "o"),
                              (0.17, "w", "#762a83", "s")]:
        bs = [dump["methods"][m][f"b_{key}"] for m in order]
        ses = [dump["methods"][m][f"s_{key}"] for m in order]
        ax.errorbar(np.array(ypos) + off, bs, yerr=1.96 * np.array(ses),
                    fmt=mk, ms=4.5, color=col, capsize=2.5, lw=1.1,
                    label=("transferred " r"$\hat\sigma^2_\epsilon$"
                           if key == "t" else
                           "within-cycle " r"$\hat\sigma^2_\epsilon$"))
    ax.axhline(dump["methods"]["oracle"]["b_t"], color="k", ls="--", lw=1,
               label="measured weight (anchor)")
    ax.axhline(dump["methods"]["naive"]["b_t"], color="gray", ls=":",
               lw=1, label="naive")
    ax.set_xticks(ypos)
    ax.set_xticklabels(["Naive", "RC", "Corr. score", "Std. SIMEX",
                        "A-SIMEX", "A-SIMEX (arg)", "Anchor"],
                       fontsize=8)
    ax.set_ylabel(r"$\hat\beta$ per 10 kg")
    ax.set_title("NHANES 2015-2016: transfer of the calibrated error "
                 "variance", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_transfer.pdf")
    fig.savefig(OUT / "fig8_transfer.png", dpi=150)
    print("saved transfer_table.tex, fig8_transfer.*, transfer_results.json")


if __name__ == "__main__":
    main()
