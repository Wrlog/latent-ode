"""Interactive figures for the dashboard, built with plotly from results/.

Each builder returns a chart spec for app.js:

    {"variant_by": key or None, "variants": {value: {"light": fig, "dark": fig}}}

A chart with `variant_by` swaps the whole figure when that control changes
(used when the axes change, e.g. one patient or covariate per figure). Within a
figure, traces carry `meta` tags such as {"dataset": "clean"} and app.js hides
the traces whose tags don't match the current controls.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .theme import DASH, FONT, METHOD_ORDER, SYMBOLS, THEMES, Theme

LABELS = {
    "latent ODE": "Latent ODE",
    "popPK, no covariates": "popPK, no covariates",
    "popPK with covariates": "popPK with covariates",
    "popPK, time-varying CL": "popPK, time-varying CL",
    "true model (ceiling)": "True model (ceiling)",
}
DATASETS = {"clean": "Clean", "doses": "Dose levels + assay error", "ada": "ADA"}
COVARIATES = {
    "WT": ("Weight", "weight (kg)", "linear"),
    "ALB": ("Albumin", "albumin (g/dL)", "linear"),
    "INF": ("INF", "INF (log scale)", "log"),
    "PLT": ("PLT", "PLT", "linear"),
}
ADA_GROUPS = ["ADA negative", "onset before last level", "onset after last level"]


# ---------------------------------------------------------------------------
# shared layout
# ---------------------------------------------------------------------------
def _axis(th: Theme, title: str, **kw) -> dict:
    return {
        "title": {"text": title, "font": {"size": 12, "color": th.ink_2}, "standoff": 8},
        "gridcolor": th.grid,
        "zeroline": False,
        "showline": True,
        "linecolor": th.axis,
        "ticks": "",
        "tickfont": {"size": 11, "color": th.ink_2},
        "automargin": True,
        **kw,
    }


def _layout(th: Theme, xtitle: str, ytitle: str, x: dict | None = None, y: dict | None = None) -> dict:
    return {
        "template": {},
        "paper_bgcolor": th.surface,
        "plot_bgcolor": th.surface,
        "font": {"family": FONT, "size": 12, "color": th.ink},
        "xaxis": _axis(th, xtitle, **(x or {})),
        "yaxis": _axis(th, ytitle, **(y or {})),
        "margin": {"l": 8, "r": 12, "t": 8, "b": 8},
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": th.surface, "bordercolor": th.grid, "font": {"family": FONT, "size": 12, "color": th.ink}},
        "legend": {
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11.5, "color": th.ink_2},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "showlegend": True,
        "dragmode": "zoom",
    }


def _log_ticks(lo: float, hi: float) -> dict:
    """1-2-5 tick values inside [lo, hi] for a log axis, so minor ticks aren't labelled."""
    vals = [m * 10.0**e for e in range(-4, 5) for m in (1, 2, 5)]
    vals = [v for v in vals if lo <= v <= hi]
    return {"tickvals": vals, "ticktext": [f"{v:g}" for v in vals]}


def _method_trace(th: Theme, method: str, x, y, meta: dict, hover: str, customdata=None) -> go.Scatter:
    return go.Scatter(
        x=list(x),
        y=list(y),
        name=LABELS.get(method, method),
        legendgroup=method,
        mode="lines+markers",
        line={"color": th.methods[method], "width": 2, "dash": DASH.get(method, "solid")},
        marker={"symbol": SYMBOLS.get(method, "circle"), "size": 8, "color": th.methods[method], "line": {"color": th.surface, "width": 1.5}},
        meta=meta,
        customdata=customdata,
        hovertemplate=hover,
    )


def _dump(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig, validate=True, pretty=False))


def _spec(build, variants=None, variant_by=None) -> dict:
    keys = variants if variants is not None else ["_"]
    return {
        "variant_by": variant_by,
        "variants": {str(v): {th.name: _dump(build(th, v)) for th in THEMES} for v in keys},
    }


def _methods_in(df: pd.DataFrame) -> list[str]:
    present = set(df["method"])
    return [m for m in METHOD_ORDER if m in present]


# ---------------------------------------------------------------------------
# forecast error against the number of levels
# ---------------------------------------------------------------------------
def fold_error(fe: pd.DataFrame, metric: str) -> dict:
    ytitle = {"mfe": "median fold error", "r2": "R² on log concentration"}[metric]
    fmt = {"mfe": ".3f", "r2": ".3f"}[metric]
    ks = sorted(fe["k"].unique())

    def build(th: Theme, _):
        fig = go.Figure(layout=_layout(th, "levels given (k)", ytitle, x={"tickvals": ks, "unifiedhovertitle": {"text": "%{x} levels given"}}))
        for ds in [d for d in DATASETS if d in set(fe["dataset"])]:
            d = fe[fe["dataset"] == ds]
            for m in _methods_in(d):
                dm = d[d["method"] == m].sort_values("k")
                fig.add_trace(_method_trace(th, m, dm["k"], dm[metric], {"dataset": ds}, f"%{{fullData.name}}: %{{y:{fmt}}}<extra></extra>"))
        return fig

    return _spec(build)


def ada_status(ada: pd.DataFrame) -> dict:
    ks = sorted(ada["k"].unique())

    def build(th: Theme, _):
        fig = go.Figure(layout=_layout(th, "levels given (k)", "median fold error", x={"tickvals": ks, "unifiedhovertitle": {"text": "%{x} levels given"}}))
        for g in [g for g in ADA_GROUPS if g in set(ada["group"])]:
            d = ada[ada["group"] == g]
            for i, m in enumerate(_methods_in(d)):
                dm = d[d["method"] == m].sort_values("k")
                hover = "%{fullData.name}: %{y:.3f}"
                hover += " (%{customdata} patients)<extra></extra>" if i == 0 else "<extra></extra>"
                fig.add_trace(_method_trace(th, m, dm["k"], dm["mfe"], {"group": g}, hover, customdata=dm["n_patients"].tolist()))
        return fig

    return _spec(build)


# ---------------------------------------------------------------------------
# example profiles
# ---------------------------------------------------------------------------
def profiles(pr: pd.DataFrame) -> dict:
    patients = list(dict.fromkeys(pr["patient"]))
    ks = sorted(pr.loc[pr["method"] == "latent ODE", "k"].unique())

    def build(th: Theme, p):
        d = pr[pr["patient"] == int(p)]
        truth = d[d["method"] == "truth"]
        obs = d[d["method"] == "observed"]
        pos = pd.concat([truth["conc"], obs["conc"]])
        lo, hi = float(pos[pos > 0].min()), float(pos.max())
        y = {"type": "log", "range": [np.log10(lo * 0.6), np.log10(hi * 1.4)], **_log_ticks(lo * 0.6, hi * 1.4)}
        fig = go.Figure(layout=_layout(th, "day", "Drug X concentration (mg/L)", x={"unifiedhovertitle": {"text": "day %{x:.1f}"}}, y=y))
        fig.add_trace(
            go.Scatter(
                x=truth["time"].round(3).tolist(),
                y=truth["conc"].round(4).tolist(),
                name="Truth",
                legendgroup="truth",
                mode="lines",
                line={"color": th.ink, "width": 2.5},
                hovertemplate="Truth: %{y:.2f} mg/L<extra></extra>",
            )
        )
        for k in ks:
            meta = {"k": str(k)}
            for method, label, color in (
                ("latent ODE", "Latent ODE", th.methods["latent ODE"]),
                ("popPK", "popPK with covariates", th.methods["popPK with covariates"]),
            ):
                dm = d[(d["method"] == method) & (d["k"] == k)]
                fig.add_trace(
                    go.Scatter(
                        x=dm["time"].round(3).tolist(),
                        y=dm["conc"].round(4).tolist(),
                        name=label,
                        legendgroup=method,
                        mode="lines",
                        line={"color": color, "width": 2},
                        meta=meta,
                        hovertemplate=f"{label}: %{{y:.2f}} mg/L<extra></extra>",
                    )
                )
            given = obs[obs["k"] == 1] if k > 0 else obs.iloc[0:0]
            rest = obs[obs["k"] == 0] if k > 0 else obs
            if len(given):
                fig.add_trace(
                    go.Scatter(
                        x=given["time"].round(3).tolist(),
                        y=given["conc"].round(4).tolist(),
                        name=f"levels given (first {len(given)})",
                        legendgroup="given",
                        mode="markers",
                        marker={"size": 9, "color": th.ink, "line": {"color": th.surface, "width": 1.5}},
                        meta=meta,
                        hovertemplate="Level given: %{y:.2f} mg/L<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=rest["time"].round(3).tolist(),
                    y=rest["conc"].round(4).tolist(),
                    name="levels not given",
                    legendgroup="rest",
                    mode="markers",
                    marker={"size": 8, "color": th.surface, "line": {"color": th.ink_2, "width": 1.5}},
                    meta=meta,
                    hovertemplate="Level not given: %{y:.2f} mg/L<extra></extra>",
                )
            )
        onset = float(d["ada_onset"].iloc[0])
        if np.isfinite(onset):
            fig.add_vline(x=onset, line={"color": th.muted, "width": 1.2, "dash": "dot"})
            fig.add_annotation(
                x=onset, y=1, yref="paper", yanchor="top", xanchor="left", xshift=4, showarrow=False,
                text=f"ADA onset, day {onset:.0f}", font={"size": 11, "color": th.ink_2},
            )
        return fig

    return _spec(build, variants=patients, variant_by="patient")


def patient_options(pr: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for p, d in pr.groupby("patient", sort=False):
        onset = float(d["ada_onset"].iloc[0])
        desc = f"ADA from day {onset:.0f}" if np.isfinite(onset) else "ADA negative"
        out.append((str(p), f"Test patient {p}, {desc}"))
    return out


# ---------------------------------------------------------------------------
# covariate recovery
# ---------------------------------------------------------------------------
def covariates(cr: pd.DataFrame) -> dict:
    styles = {"truth": ("Truth", None, 3), "popPK with covariates": ("popPK with covariates", "diamond", 2), "latent ODE": ("Latent ODE", "circle", 2)}

    def build(th: Theme, c):
        _, xtitle, xtype = COVARIATES[c]
        fig = go.Figure(layout=_layout(th, xtitle, "mean concentration after dose 3 (mg/L)", x={"type": xtype}))
        d = cr[cr["covariate"] == c]
        for m, (label, symbol, width) in styles.items():
            dm = d[d["method"] == m].sort_values("value")
            color = th.ink if m == "truth" else th.methods[m]
            fig.add_trace(
                go.Scatter(
                    x=dm["value"].round(4).tolist(),
                    y=dm["cavg"].round(4).tolist(),
                    name=label,
                    mode="lines" if symbol is None else "lines+markers",
                    line={"color": color, "width": width},
                    marker={"symbol": symbol or "circle", "size": 8, "color": color, "line": {"color": th.surface, "width": 1.5}},
                    hovertemplate=f"{label}: %{{y:.2f}} mg/L<extra></extra>",
                )
            )
        fig.update_layout(xaxis_unifiedhovertitle_text=f"{COVARIATES[c][0]} %{{x:.3g}}")
        return fig

    return _spec(build, variants=[c for c in COVARIATES if c in set(cr["covariate"])], variant_by="covariate")


def covariate_ratios(cr: pd.DataFrame) -> list[dict]:
    """Mean concentration at the highest grid value over the lowest."""
    rows = []
    for c in [c for c in COVARIATES if c in set(cr["covariate"])]:
        d = cr[cr["covariate"] == c]
        lo, hi = d["value"].min(), d["value"].max()
        row = {"covariate": c, "name": COVARIATES[c][0], "lo": lo, "hi": hi}
        for m in ("truth", "popPK with covariates", "latent ODE"):
            dm = d[d["method"] == m].set_index("value")["cavg"]
            row[m] = float(dm[hi] / dm[lo])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# training curves
# ---------------------------------------------------------------------------
def training(curves: dict[str, pd.DataFrame], column: str) -> dict:
    ytitle = {"train_loss": "training loss (per observed level)", "val_mse": "validation MSE, log concentration"}[column]
    name = {"train_loss": "Training loss", "val_mse": "Validation MSE"}[column]

    def build(th: Theme, _):
        y = {}
        if column == "val_mse":
            vals = pd.concat([h[column] for h in curves.values()])
            y = {"type": "log", **_log_ticks(vals.min() * 0.9, vals.max() * 1.1)}
        fig = go.Figure(layout=_layout(th, "epoch", ytitle, x={"unifiedhovertitle": {"text": "epoch %{x}"}}, y=y))
        color = th.methods["latent ODE"]
        for ds, h in curves.items():
            meta = {"dataset": ds}
            fig.add_trace(
                go.Scatter(
                    x=h["epoch"].tolist(),
                    y=h[column].round(5).tolist(),
                    name=name,
                    mode="lines",
                    line={"color": color, "width": 2},
                    customdata=np.c_[h["beta"], h["lr"]].tolist(),
                    meta=meta,
                    hovertemplate="%{y:.4f}<br>KL weight %{customdata[0]:.2f}, learning rate %{customdata[1]:.2g}<extra></extra>",
                )
            )
            best = h.loc[h["val_mse"].idxmin()]
            fig.add_trace(
                go.Scatter(
                    x=[int(best["epoch"])],
                    y=[round(float(best[column]), 5)],
                    name="best validation epoch",
                    mode="markers",
                    marker={"size": 10, "symbol": "diamond", "color": th.ink, "line": {"color": th.surface, "width": 1.5}},
                    meta=meta,
                    hovertemplate="Best validation epoch (%{x})<extra></extra>",
                )
            )
        return fig

    return _spec(build)
