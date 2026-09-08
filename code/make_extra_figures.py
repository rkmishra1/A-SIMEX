"""
Additional figures for the manuscript:
  fig5_simex_curve.pdf  SIMEX curve diagnostic on one simulated dataset
  fig6_rmse.pdf         RMSE vs sigma2 panels (companion to Fig 1)
  fig7_nhanes_forest.pdf  dot-and-whisker of the NHANES weight coefficients
All reproducible; seeds fixed (master 20260905).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import asimex_lib as L
from make_tables_figures import load, metrics, BETA1, ORDER, LABEL

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"naive": "#777777", "rc": "#2c7fb8", "cs": "#41ab5d",
     "std_simex": "#e6550d", "adaptive": "#d7301f",
     "adaptive_am": "#980043", "oracle": "#000000"}


def fig5_curve():
    """SIMEX curve diagnostic: one dataset, n=500, sigma2=0.5, beta0=(1,-.5,.8)."""
    beta0 = np.array([1.0, -0.5, 0.8])
    sigma2 = 0.5
    rng = np.random.default_rng(np.random.SeedSequence(
        20260905, spawn_key=(555,)))
    n, p = 500, 3
    X = rng.standard_normal((n, p))
    Y = rng.poisson(np.exp(np.clip(X @ beta0, -30, 30)))
    W = X + rng.standard_normal((n, p)) * np.sqrt(sigma2)
    me = [0, 1, 2]
    b_naive = L.poisson_irls(Y, W)
    grid = np.arange(0.0, 2.0 + 1e-9, 0.5)
    curve, info, beta_all = L.simex_curve(Y, W, me, [sigma2] * p, grid,
                                          200, rng, beta_init=b_naive)
    d = L.combined_weights(grid, 2)
    b_ex = d @ curve

    # primary (quadratic) fit for the smooth line
    Phi = np.vander(grid, 3, increasing=True)
    coef_p, *_ = np.linalg.lstsq(Phi, curve[:, 0], rcond=None)
    lam_fine = np.linspace(-1.0, 2.0, 200)
    fine = np.vander(lam_fine, 3, increasing=True) @ coef_p

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(lam_fine, fine, "-", color="#e6550d", lw=1.4,
            label="primary quadratic fit")
    ax.plot(grid, curve[:, 0], "s", ms=5, color="#e6550d",
            label=r"empirical curve $\bar\beta(\lambda_j)$")
    ax.plot([-1.0], [b_ex[0]], "D", ms=7, color="#d7301f",
            label="two-stage extrapolant (A-SIMEX)")
    ax.axhline(beta0[0], color="k", lw=0.9, ls="--", label=r"true $\beta_1$")
    ax.axhline(b_naive[0], color="#777777", lw=0.9, ls=":",
               label="naive estimate")
    ax.axvline(-1.0, color="#bbbbbb", lw=0.8)
    ax.annotate(r"$\lambda = -1$", xy=(-1.0, ax.get_ylim()[0]),
                xytext=(-0.93, ax.get_ylim()[0] + 0.02), fontsize=8,
                color="#666666")
    ax.set_xlabel(r"augmentation level $\lambda$")
    ax.set_ylabel(r"$\bar\beta_1(\lambda)$")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("SIMEX curve diagnostic (one dataset, "
                 r"$n=500$, $\sigma^2_\epsilon=0.5$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_simex_curve.pdf")
    fig.savefig(OUT / "fig5_simex_curve.png", dpi=150)
    plt.close(fig)


def fig6_rmse():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharex=True)
    for ax, n in zip(axes.ravel(), [100, 300, 500, 1000]):
        for m in ORDER:
            xs, ys = [], []
            for s2 in [0.1, 0.5, 1.0]:
                e = metrics(load("S1", f"{n}_{int(s2*100)}"), BETA1)[m]
                xs.append(s2)
                ys.append(e["rmse"])
            ax.plot(xs, ys, ".-", color=C[m], label=LABEL[m], ms=5, lw=1.2)
        ax.set_title(f"$n = {n}$", fontsize=10)
        ax.set_ylim(0, 0.45)
    axes[1, 0].set_xlabel(r"$\sigma^2_\epsilon$")
    axes[1, 1].set_xlabel(r"$\sigma^2_\epsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"RMSE$(\hat\beta_1)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig6_rmse.pdf")
    fig.savefig(OUT / "fig6_rmse.png", dpi=150)
    plt.close(fig)


def fig7_forest():
    j = json.loads((OUT / "nhanes_results.json").read_text())
    rows = j["rows"]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ypos = np.arange(len(rows))[::-1]
    bs = [r["b"] for r in rows]
    ses = [r["se"] for r in rows]
    cols = ["#777777", "#2c7fb8", "#41ab5d", "#e6550d", "#d7301f",
            "#980043", "#000000"]
    ax.errorbar(bs, ypos, xerr=1.96 * np.array(ses), fmt="none",
                ecolor="#999999", elinewidth=1.1, capsize=2.5)
    ax.scatter(bs, ypos, c=cols, s=34, zorder=3)
    anchor = rows[-1]["b"]
    ax.axvline(anchor, color="k", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_xlabel(r"$\hat\beta$ per 10 kg (95\% CI)")
    ax.set_title(f"NHANES 2017-2018 application (n = {j['n']:,})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_nhanes_forest.pdf")
    fig.savefig(OUT / "fig7_nhanes_forest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig5_curve()
    fig6_rmse()
    fig7_forest()
    print("wrote fig5, fig6, fig7 to", OUT)
