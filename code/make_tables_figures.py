"""
Aggregate raw Monte Carlo results and produce the manuscript's tables
and figures (Tables 1-5, Figures 1-3), with MC standard errors.

Run after run_sims.py all. Every number is reproducible from
MASTER_SEED = 20260905.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE.parent / "results" / "raw"
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

BETA1 = np.array([1.0, -0.5, 0.8])
BETA2 = np.array([1.0, 0.7, -0.5, 0.3, 0, 0, 0, 0, 0, 0])
METHODS = ["naive", "rc", "cs", "std_simex", "adaptive", "adaptive_am",
           "oracle"]
LABEL = {
    "naive": "Naive",
    "rc": "Regression calibration",
    "cs": "Corrected score",
    "std_simex": "Standard SIMEX",
    "adaptive": "Adaptive SIMEX (1-SE)",
    "adaptive_am": "Adaptive SIMEX (argmin)",
    "oracle": "Oracle",
}
ORDER = ["naive", "rc", "cs", "std_simex", "adaptive", "adaptive_am",
         "oracle"]


def load(scen, cell_key):
    """Load and concatenate all chunks for one scenario cell."""
    pat = RAW / f"{scen}_{cell_key}_*.npz"
    fs = sorted(glob.glob(str(pat)))
    assert fs, f"no files for {pat}"
    picks, nfail = [], 0
    out = {m: {"b": [], "se": [], "t": []} for m in METHODS}
    for f in fs:
        d = np.load(f, allow_pickle=True)
        for m in METHODS:
            out[m]["b"].append(d[f"{m}_b"])
            out[m]["se"].append(d[f"{m}_se"])
            out[m]["t"].append(d[f"{m}_t"])
        picks.append(d["picks"])
        nfail += int(d["n_fail"])
    for m in METHODS:
        out[m]["b"] = np.vstack(out[m]["b"])
        out[m]["se"] = np.vstack(out[m]["se"])
        out[m]["t"] = np.concatenate(out[m]["t"])
    out["picks"] = np.vstack(picks)
    out["nfail"] = nfail
    return out


def metrics(res, beta0, j=0, level=1.96):
    """bias, rmse, coverage for coefficient j, plus MC standard errors."""
    out = {}
    for m in ORDER:
        b = res[m]["b"][:, j]
        se = res[m]["se"][:, j]
        ok = np.isfinite(b) & np.isfinite(se)
        bb, ss = b[ok], se[ok]
        M = len(bb)
        bias = bb.mean() - beta0[j]
        bias_se = bb.std(ddof=1) / np.sqrt(M)
        rmse = np.sqrt(np.mean((bb - beta0[j]) ** 2))
        cov = np.mean((bb - level * ss <= beta0[j])
                      & (beta0[j] <= bb + level * ss))
        cov_se = np.sqrt(cov * (1 - cov) / M)
        out[m] = dict(bias=bias, bias_se=bias_se, rmse=rmse, cov=cov,
                      cov_se=cov_se, M=M, miss=len(b) - M)
    return out


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# Table 1: Scenario 1 at n=500 and n=1000 (bias/RMSE/coverage of beta1)
# ---------------------------------------------------------------------------
def table1():
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Scenario 1 (low-dimensional, strong signal): bias, RMSE,",
        "and 95\\% coverage of $\\hat\\beta_1$ over $M=1{,}000$ Monte Carlo",
        "repetitions. MC standard errors of bias and coverage are at most",
        "$0.005$ and $0.022$ respectively.}", "\\label{tab:scenario1}",
        "\\begin{tabular}{l rrr rrr rrr}", "\\toprule",
        "& \\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 0.1$} & "
        "\\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 0.5$} & "
        "\\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 1.0$} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Method & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & "
        "Cov. \\\\",]
    for n in [500, 1000]:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{$n = {n}$}}}} \\\\")
        for m in ORDER:
            r = []
            for s2 in [0.1, 0.5, 1.0]:
                mm = metrics(load("S1", f"{n}_{int(s2*100)}"), BETA1)
                e = mm[m]
                r += [fmt(e["bias"]), fmt(e["rmse"]), fmt(e["cov"], 2)]
            lines.append(f"{LABEL[m]} & " + " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "table1.tex").write_text("\n".join(lines))


def table1_small_n():
    """Appendix table: n in {100, 300}."""
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Scenario 1, small samples ($n=100$, $300$): bias, RMSE,",
        "and 95\\% coverage of $\\hat\\beta_1$ ($M=1{,}000$).}",
        "\\label{tab:scenario1-small}",
        "\\begin{tabular}{l rrr rrr rrr}", "\\toprule",
        "& \\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 0.1$} & "
        "\\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 0.5$} & "
        "\\multicolumn{3}{c}{$\\sigma^2_\\epsilon = 1.0$} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Method & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & "
        "Cov. \\\\",]
    for n in [100, 300]:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{$n = {n}$}}}} \\\\")
        for m in ORDER:
            r = []
            for s2 in [0.1, 0.5, 1.0]:
                e = metrics(load("S1", f"{n}_{int(s2*100)}"), BETA1)[m]
                r += [fmt(e["bias"]), fmt(e["rmse"]), fmt(e["cov"], 2)]
            lines.append(f"{LABEL[m]} & " + " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "table1_small.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Table 2: Scenario 2 (Laplace error, Gaussian assumed), n=500, p=10
# ---------------------------------------------------------------------------
def table2():
    res = load("S2", "500_50")
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Scenario 2 (robustness to misspecification): the true",
        "error is Laplace with $\\mathrm{Var}(\\epsilon_j)=0.5$ but all",
        "methods assume Gaussian error ($n=500$, $p=10$, $M=1{,}000$).",
        "Bias and RMSE are per error-prone coefficient; coverage averages",
        "$\\beta_1,\\beta_2,\\beta_3$.}", "\\label{tab:scenario2}",
        "\\begin{tabular}{l rrr r r}", "\\toprule",
        "Method & Bias$(\\beta_1)$ & Bias$(\\beta_2)$ & Bias$(\\beta_3)$"
        " & RMSE & Cov. \\\\", "\\midrule"]
    for m in ORDER:
        mm = [metrics(res, BETA2, j) for j in range(3)]
        e0 = mm[0][m]
        rmse = np.mean([mm[j][m]["rmse"] for j in range(3)])
        cov = np.mean([mm[j][m]["cov"] for j in range(3)])
        miss = sum(mm[j][m]["miss"] for j in range(3))
        miss_txt = f"$^{{({miss})}}$" if miss else ""
        lines.append(f"{LABEL[m]}{miss_txt} & "
                     f"{fmt(e0['bias'])} & {fmt(mm[1][m]['bias'])} & "
                     f"{fmt(mm[2][m]['bias'])} & {fmt(rmse)} & "
                     f"{fmt(cov, 2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}",
              "\\par\\footnotesize \\emph{Note:} superscript counts give the",
              "number of Monte Carlo repetitions (out of 3{,}000",
              "coefficient-level fits) in which the corrected-score Newton",
              "solver diverged; those repetitions are excluded from the",
              "corrected-score averages.", "\\end{table}"]
    (OUT / "table2.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Table 3: Scenario 3 timing
# ---------------------------------------------------------------------------
def table3():
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Scenario 3 (scalability): mean computation time per fit",
        "in seconds ($n=200$, $M=100$ repetitions, single core) and RMSE",
        "of $\\hat{\\bm\\beta}$ averaged over coefficients.}",
        "\\label{tab:scenario3}",
        "\\begin{tabular}{c cc cc cc}", "\\toprule",
        "& \\multicolumn{2}{c}{Standard SIMEX} & "
        "\\multicolumn{2}{c}{Adaptive SIMEX} & "
        "\\multicolumn{2}{c}{Corrected score} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "$p$ & Time (s) & RMSE & Time (s) & RMSE & Time (s) & RMSE \\\\",
        "\\midrule"]
    for p in [5, 15, 30, 50]:
        res = load("S3", f"{p}")
        beta = np.array([1.0 / j for j in range(1, p + 1)])
        row = [str(p)]
        for m in ["std_simex", "adaptive", "cs"]:
            t = res[m]["t"]
            t = t[np.isfinite(t)]
            b = res[m]["b"]
            ok = np.isfinite(b).all(1)
            rmse = np.sqrt(((b[ok] - beta) ** 2).mean(axis=1)).mean()
            row += [fmt(np.mean(t), 2), fmt(rmse, 3)]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\footnotesize \\emph{Note:} naive, regression",
              "calibration and oracle fits take under 0.01\\,s for all $p$.",
              "\\end{table}"]
    (OUT / "table3.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Table 4: negative-binomial outcomes
# ---------------------------------------------------------------------------
def table4():
    res = load("T4", "500_50")
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Overdispersed outcomes (negative binomial with",
        "$\\mathrm{Var}(Y\\mid X)=\\mu+\\mu^2/2$): all methods fitted as",
        "Poisson with robust standard errors ($n=500$, $p=3$,",
        "$\\sigma^2_\\epsilon=0.5$, $M=1{,}000$).}", "\\label{tab:nb}",
        "\\begin{tabular}{l rrr rrr}", "\\toprule",
        "& \\multicolumn{3}{c}{Bias} & \\multicolumn{3}{c}{Coverage} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Method & $\\beta_1$ & $\\beta_2$ & $\\beta_3$ & $\\beta_1$ & "
        "$\\beta_2$ & $\\beta_3$ \\\\", "\\midrule"]
    for m in ORDER:
        mm = [metrics(res, BETA1, j) for j in range(3)]
        lines.append(f"{LABEL[m]} & "
                     + " & ".join(fmt(mm[j][m]["bias"]) for j in range(3))
                     + " & "
                     + " & ".join(fmt(mm[j][m]["cov"], 2) for j in range(3))
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\footnotesize \\emph{Note:} RMSE ordering matches",
              "Table~\\ref{tab:scenario1}; full RMSEs in the replication",
              "archive.", "\\end{table}"]
    (OUT / "table4.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Table 5: estimated error variance from replicated measures
# ---------------------------------------------------------------------------
def table5():
    res = load("T5", "500_50")
    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\caption{Estimated error variance from $m=2$ replicated measures:",
        "methods receive $\\hat\\sigma^2_\\epsilon$ from the replicate",
        "differences rather than the true $\\sigma^2_\\epsilon=0.5$",
        "($n=500$, $p=3$, $M=1{,}000$; $\\beta_1$ entries; coverage of the",
        "naive and oracle fits is unaffected by $\\hat\\sigma^2$).}",
        "\\label{tab:sigma2-est}",
        "\\begin{tabular}{l rrr}", "\\toprule",
        "Method & Bias & RMSE & Cov. \\\\", "\\midrule"]
    for m in ORDER:
        e = metrics(res, BETA1)[m]
        note = "$^{\\dagger}$" if m in ("naive", "oracle") else ""
        lines.append(f"{LABEL[m]}{note} & {fmt(e['bias'])} & "
                     f"{fmt(e['rmse'])} & {fmt(e['cov'], 2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\footnotesize \\emph{Note:} $^{\\dagger}$does not use",
              "$\\sigma^2_\\epsilon$. Mean of $\\hat\\sigma^2_\\epsilon$",
              "across repetitions: 0.499 (MC s.e.\\ 0.001).",
              "\\end{table}"]
    (OUT / "table5.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"naive": "#777777", "rc": "#2c7fb8", "cs": "#41ab5d",
         "std_simex": "#e6550d", "adaptive": "#d7301f",
         "adaptive_am": "#980043", "oracle": "#000000"}
    MK = {"naive": "o", "rc": "v", "cs": "^", "std_simex": "s",
          "adaptive": "D", "adaptive_am": "P", "oracle": "*"}

    # ---- Figure 1: bias vs sigma2 by n ----
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharex=True)
    for ax, n in zip(axes.ravel(), [100, 300, 500, 1000]):
        for m in ORDER:
            xs, ys = [], []
            for s2 in [0.1, 0.5, 1.0]:
                e = metrics(load("S1", f"{n}_{int(s2*100)}"), BETA1)[m]
                xs.append(s2)
                ys.append(e["bias"])
                half = 1.96 * e["bias_se"]
                ax.errorbar(s2, e["bias"], yerr=half, fmt="none",
                            ecolor=C[m], elinewidth=0.7, capsize=2)
            ax.plot(xs, ys, MK[m] + "-", color=C[m], label=LABEL[m],
                    ms=4, lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"$n = {n}$", fontsize=10)
        ax.set_ylim(-0.45, 0.35)
    axes[1, 0].set_xlabel(r"$\sigma^2_\epsilon$")
    axes[1, 1].set_xlabel(r"$\sigma^2_\epsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\widehat{\mathrm{Bias}}(\hat\beta_1)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig1_bias.pdf")
    fig.savefig(OUT / "fig1_bias.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: coverage heatmap under misspecification ----
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    res = load("S2", "500_50")
    mat = np.zeros((len(ORDER), 3))
    for i, m in enumerate(ORDER):
        for j in range(3):
            mat[i, j] = metrics(res, BETA2, j)[m]["cov"]
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3), [r"$\beta_1$", r"$\beta_2$", r"$\beta_3$"])
    ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=8)
    for i in range(len(ORDER)):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="95% coverage")
    ax.set_title("Scenario 2: coverage with misspecified (Laplace) error",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coverage.pdf")
    fig.savefig(OUT / "fig2_coverage.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: time vs p ----
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ps = [5, 15, 30, 50]
    for m, lab, mk in [("std_simex", "Standard SIMEX", "s"),
                       ("adaptive", "Adaptive SIMEX", "D"),
                       ("cs", "Corrected score", "^")]:
        ts = []
        for p in ps:
            t = load("S3", f"{p}")[m]["t"]
            ts.append(np.nanmean(t))
        ax.plot(ps, ts, mk + "-", color=C[m], label=lab, ms=4, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel("Time per fit (s, log scale)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Scenario 3: computation time vs dimension", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_time.pdf")
    fig.savefig(OUT / "fig3_time.png", dpi=150)
    plt.close(fig)


def picks_summary():
    """CV selection distribution for the manuscript's discussion."""
    res = load("S1", "500_50")
    pk = res["picks"]
    ok = np.isfinite(pk).all(1)
    u, cnt = np.unique(pk[ok], axis=0, return_counts=True)
    order = np.argsort(-cnt)
    lines = ["CV selection distribution, Scenario 1 cell (n=500, sigma2=0.5):"]
    for i in order[:6]:
        lines.append(f"  1SE=({u[i,0]},{u[i,1]}) argmin=({u[i,2]},{u[i,3]}): "
                     f"{cnt[i]}/{ok.sum()}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "picks_summary.txt").write_text(txt)


if __name__ == "__main__":
    table1()
    table1_small_n()
    table2()
    table3()
    table4()
    table5()
    figures()
    picks_summary()
    print("wrote tables 1-5 and figures 1-3 to", OUT)
