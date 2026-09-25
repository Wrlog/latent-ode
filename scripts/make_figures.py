"""Draw the figures in figures/ from the result tables in results/.

    python scripts/make_figures.py [--results results] [--out figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK = "#0b0b0b"
MUTED = "#52514e"
COLORS = {
    "latent ODE": "#2a78d6",
    "popPK, no covariates": "#eb6834",
    "popPK with covariates": "#1baf7a",
    "popPK, time-varying CL": "#eda100",
    "true model (ceiling)": "#4a3aa7",
    "popPK": "#1baf7a",
    "truth": INK,
}
MARKERS = {
    "latent ODE": "o",
    "popPK, no covariates": "s",
    "popPK with covariates": "D",
    "popPK, time-varying CL": "^",
    "true model (ceiling)": "v",
}
TITLES = {
    "clean": "Clean (5 mg/kg, no assay error)",
    "doses": "Four dose levels, assay error, IOV",
    "ada": "As before, plus ADA onset",
}


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.edgecolor": "#c9c8c2",
            "grid.color": "#e7e6e1",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def method_lines(ax, df, x, y, methods=None):
    methods = methods or [m for m in COLORS if m in df["method"].unique()]
    for m in methods:
        d = df[df["method"] == m].sort_values(x)
        if d.empty:
            continue
        ls = "--" if m.startswith("true model") else "-"
        ax.plot(d[x], d[y], ls, color=COLORS[m], marker=MARKERS.get(m, "o"), markersize=6, label=m)


def fig_fold_error(res: Path, out: Path):
    fe = pd.read_csv(res / "fold_error.csv")
    kinds = [k for k in TITLES if k in fe["dataset"].unique()]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.6 * len(kinds), 7.2), sharex=True)
    axes = np.atleast_2d(axes)
    for j, kind in enumerate(kinds):
        d = fe[fe["dataset"] == kind]
        method_lines(axes[0, j], d, "k", "mfe")
        method_lines(axes[1, j], d, "k", "r2")
        axes[0, j].set_title(TITLES[kind], fontsize=11)
        axes[1, j].set_xlabel("levels given (k)")
        axes[0, j].set_ylim(bottom=1.0)
    axes[0, 0].set_ylabel("median fold error (forecast)")
    axes[1, 0].set_ylabel("R² on log concentration")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Forecast error after the k-th level, scored against the exact solution", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / "fold_error_vs_levels.png")
    plt.close(fig)


def fig_covariates(res: Path, out: Path):
    cr = pd.read_csv(res / "covariate_recovery.csv")
    units = {"WT": "weight (kg)", "ALB": "albumin (g/dL)", "INF": "INF (log scale)", "PLT": "PLT"}
    covs = list(units)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, c in zip(axes, covs):
        d = cr[cr["covariate"] == c]
        for m, style in (("truth", "-"), ("popPK with covariates", "-"), ("latent ODE", "-")):
            dd = d[d["method"] == m].sort_values("value")
            ax.plot(dd["value"], dd["cavg"], style, color=COLORS[m], marker="o" if m != "truth" else None, markersize=5, label=m, lw=2.5 if m == "truth" else 2)
        ax.set_xlabel(units[c])
        if c == "INF":
            ax.set_xscale("log")
    axes[0].set_ylabel("mean concentration, days 42-98 (mg/L)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Partial dependence at 5 mg/kg, averaged over the test patients (no levels given)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "covariate_recovery.png")
    plt.close(fig)


def fig_profiles(res: Path, out: Path):
    pr = pd.read_csv(res / "example_profiles.csv")
    pats = list(dict.fromkeys(pr["patient"]))
    fig, axes = plt.subplots(1, len(pats), figsize=(5 * len(pats), 4), sharey=True)
    axes = np.atleast_1d(axes)
    k_used = int(pr[pr["method"] == "latent ODE"]["k"].max())
    for ax, p in zip(axes, pats):
        d = pr[pr["patient"] == p]
        t = d[d["method"] == "truth"]
        ax.plot(t["time"], t["conc"], color=INK, lw=2.5, label="truth")
        for m, k, ls in (("latent ODE", 0, ":"), ("latent ODE", k_used, "-"), ("popPK", k_used, "-")):
            dd = d[(d["method"] == m) & (d["k"] == k)]
            lab = f"{m}, k={k}" if m != "popPK" else f"popPK with covariates, k={k}"
            ax.plot(dd["time"], dd["conc"], ls, color=COLORS[m], lw=1.8, label=lab)
        o = d[d["method"] == "observed"]
        used = o[o["k"] == 1]
        rest = o[o["k"] == 0]
        ax.scatter(rest["time"], rest["conc"], s=24, facecolor="white", edgecolor=MUTED, zorder=3, label="other levels")
        ax.scatter(used["time"], used["conc"], s=30, color=INK, zorder=4, label=f"first {k_used} levels (given)")
        onset = d["ada_onset"].iloc[0]
        if np.isfinite(onset):
            ax.axvline(onset, color=MUTED, lw=1, ls="--")
            ax.text(onset + 2, 0.7, "ADA onset", color=MUTED, fontsize=9)
            ax.set_title(f"test patient {p}, ADA from day {onset:.0f}", fontsize=11)
        else:
            ax.set_title(f"test patient {p}, ADA negative", fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.set_xlabel("day")
    axes[0].set_ylabel("Drug X concentration (mg/L)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "example_profiles.png")
    plt.close(fig)


def fig_ada(res: Path, out: Path):
    f = res / "ada_forecast_by_status.csv"
    if not f.exists():
        return
    ada = pd.read_csv(f)
    groups = ["ADA negative", "onset before last level", "onset after last level"]
    groups = [g for g in groups if g in ada["group"].unique()]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        d = ada[ada["group"] == g]
        method_lines(ax, d, "k", "mfe")
        n = d.groupby("k")["n_patients"].first()
        ax.set_title(f"{g}\n(patients per k: {n.min()}-{n.max()})", fontsize=11)
        ax.set_xlabel("levels given (k)")
    axes[0].set_ylabel("median fold error (forecast)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("ADA dataset: forecast error by antibody status relative to the last level given", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "ada_forecast_error.png")
    plt.close(fig)


def fig_training(res: Path, out: Path):
    files = sorted(res.glob("training_*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for f, color in zip(files, ["#2a78d6", "#eb6834", "#1baf7a"]):
        h = pd.read_csv(f)
        kind = f.stem.replace("training_", "")
        ax.plot(h["epoch"], h["val_mse"], color=color, label=kind)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE, log concentration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    fig_fold_error(args.results, args.out)
    fig_covariates(args.results, args.out)
    fig_profiles(args.results, args.out)
    fig_ada(args.results, args.out)
    fig_training(args.results, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
