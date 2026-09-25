"""Build the results dashboard from the committed result files.

    python -m dashboard.build            # writes site/index.html and assets
    python dashboard/build.py --out site

Reads results/*.csv and results/*.json, builds the interactive charts with
plotly, and renders one HTML page with the charts' JSON inlined. plotly.js is
loaded from jsDelivr. Every number on the page comes from a result file (or,
for the network sizes in the architecture note, from the config defaults in
src/, read without importing torch). Nothing is retrained.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dashboard import figures as F  # noqa: E402
else:
    from . import figures as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Wrlog/latent-ode"

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js"
PLOTLY_SRI = "sha384-N2HZsG+IG/3J8CwhGGYz/kmzZ0sprpPWwjhor9ZI4lxuf47i9DXK2/NDGxMaJiEb"

PNGS = {
    "fold": "fold_error_vs_levels.png",
    "profiles": "example_profiles.png",
    "ada": "ada_forecast_error.png",
    "cov": "covariate_recovery.png",
    "training": "training_curves.png",
}

esc = html.escape


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def load(res: Path) -> dict:
    return {
        "fe": pd.read_csv(res / "fold_error.csv"),
        "ada": pd.read_csv(res / "ada_forecast_by_status.csv"),
        "cr": pd.read_csv(res / "covariate_recovery.csv"),
        "pr": pd.read_csv(res / "example_profiles.csv"),
        "sc": pd.read_csv(res / "structural_checks.csv"),
        "train": {k: pd.read_csv(res / f"training_{k}.csv") for k in F.DATASETS if (res / f"training_{k}.csv").exists()},
        "pop": json.loads((res / "poppk_estimates.json").read_text()),
        "info": json.loads((res / "run_info.json").read_text()),
    }


def dataclass_defaults(path: Path, name: str) -> dict:
    """Default values of a dataclass in a source file, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except ValueError:
                        pass
            return out
    raise KeyError(name)


def function_defaults(path: Path, name: str) -> dict:
    """Keyword defaults of a function in a source file, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args.args[len(node.args.args) - len(node.args.defaults):]
            return {a.arg: ast.literal_eval(v) for a, v in zip(args, node.args.defaults)}
    raise KeyError(name)


# ---------------------------------------------------------------------------
# small HTML helpers
# ---------------------------------------------------------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"


def mfe(fe: pd.DataFrame, dataset: str, method: str, k: int) -> float:
    r = fe[(fe["dataset"] == dataset) & (fe["method"] == method) & (fe["k"] == k)]
    return float(r["mfe"].iloc[0])


def segmented(key: str, options: list[tuple[str, str]], label: str) -> str:
    buttons = "".join(
        f'<button type="button" role="radio" data-value="{esc(v)}" aria-checked="{"true" if i == 0 else "false"}">{esc(t)}</button>'
        for i, (v, t) in enumerate(options)
    )
    return f'<div class="segmented" role="radiogroup" aria-label="{esc(label)}" data-key="{esc(key)}">{buttons}</div>'


def select(key: str, options: list[tuple[str, str]], label: str) -> str:
    opts = "".join(f'<option value="{esc(v)}">{esc(t)}</option>' for v, t in options)
    return f'<label class="control">{esc(label)}<select data-key="{esc(key)}">{opts}</select></label>'


def plot(name: str, png: str, alt: str, tall: bool = False) -> str:
    cls = "plot plot-tall" if tall else "plot"
    return (
        f'<div class="{cls}" data-fig="{esc(name)}" role="img" aria-label="{esc(alt)}"></div>'
        f'<img class="fallback" src="figures/{esc(png)}" alt="{esc(alt)}" loading="lazy" hidden>'
    )


def png_link(png: str) -> str:
    return f'<p class="links"><a href="figures/{esc(png)}">Static figure (PNG)</a></p>'


def card(title: str, sub: str, body: str, controls: str = "", cls: str = "card") -> str:
    head = f'<div><h2>{esc(title)}</h2><p class="sub">{sub}</p></div>'
    if controls:
        head = f'<div class="card-head">{head}<div class="filters">{controls}</div></div>'
    return f'<article class="{cls}">{head}{body}</article>'


def table(headers: list[tuple[str, str]], rows: list[str], sortable: bool = True) -> str:
    th = "".join(
        f'<th scope="col"{" data-sort=" + chr(34) + kind + chr(34) if sortable and kind else ""}>{esc(h)}</th>'
        for h, kind in headers
    )
    cls = ' class="sortable"' if sortable else ""
    return f'<div class="table-wrap"><table{cls}><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def td_num(x: float, fmt: str = ".3f") -> str:
    return f'<td data-v="{x:.10g}">{x:{fmt}}</td>'


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def tiles(d: dict) -> str:
    fe, ada, sc, info = d["fe"], d["ada"], d["sc"], d["info"]
    lode, cov, ceil = "latent ODE", "popPK with covariates", "true model (ceiling)"
    k_late = 16
    before = ada[ada["group"] == "onset before last level"]
    k_ada = int(before["k"].max())
    b = before[before["k"] == k_ada].set_index("method")["mfe"]
    n_ada = int(before[before["k"] == k_ada]["n_patients"].iloc[0])
    failed = sc[~sc["passed"]]
    fail_txt = ", ".join(f"{F.DATASETS[r.dataset].lower()} network, {r.check}" for r in failed.itertuples()) or "none"
    epochs = info["epochs"]
    items = [
        ("Clean data, no levels given", f3(mfe(fe, "clean", lode, 0)),
         f"Latent ODE median fold error. popPK with covariates {f3(mfe(fe, 'clean', cov, 0))}, true model {f3(mfe(fe, 'clean', ceil, 0))}."),
        (f"Clean data, {k_late} levels given", f3(mfe(fe, "clean", lode, k_late)),
         f"Latent ODE. popPK MAP gets {f3(mfe(fe, 'clean', cov, k_late))}, the true model {f3(mfe(fe, 'clean', ceil, k_late))}."),
        (f"ADA onset before level {k_ada}", f3(float(b[lode])),
         f"Latent ODE, {n_ada} patients. Constant-CL popPK {f3(float(b[cov]))}, time-varying CL {f3(float(b['popPK, time-varying CL']))}."),
        ("Structural checks passed", f"{int(sc['passed'].sum())} of {len(sc)}", f"Failed: {fail_txt}."),
        ("Full pipeline", f"{info['total_seconds']:.0f} s",
         f"Laptop CPU. Networks stopped early after {min(epochs.values())} to {max(epochs.values())} epochs."),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{esc(a)}</div><div class="value">{esc(v)}</div><div class="detail">{esc(t)}</div></div>'
        for a, v, t in items
    ) + "</div>"


def fold_section(d: dict) -> str:
    fe = d["fe"]
    ds_opts = [(k, v) for k, v in F.DATASETS.items() if k in set(fe["dataset"])]
    n_test = d["info"]["sizes"]["clean"]["test"]
    controls = segmented("dataset", ds_opts, "Dataset")
    body = (
        '<div class="grid-2">'
        f'<figure class="fig">{plot("fe_mfe", PNGS["fold"], "Median fold error against levels given")}</figure>'
        f'<figure class="fig">{plot("fe_r2", PNGS["fold"], "R squared against levels given")}</figure>'
        "</div>" + png_link(PNGS["fold"])
    )
    sub = (
        f"Forecast after each patient's k-th level, scored against the exact concentration at every later sampling time "
        f"({n_test} test patients per dataset). Median fold error is 1 for a perfect forecast. The true model is the ceiling: "
        "it knows the structural model, the covariate effects and, in the ADA data, each patient's onset time."
    )
    main = card("Forecast error as levels come in", sub, body, controls)

    rows = []
    for r in fe.itertuples():
        rows.append(
            f'<tr data-filter="dataset:{esc(r.dataset)}"><td data-v="{F.METHOD_ORDER.index(r.method)}">{esc(F.LABELS[r.method])}</td>'
            f"<td data-v=\"{r.k}\">{r.k}</td>{td_num(r.mfe)}{td_num(r.r2)}<td data-v=\"{r.n_points}\">{r.n_points:,}</td></tr>"
        )
    tab = table([("Method", "num"), ("k", "num"), ("Median fold error", "num"), ("R²", "num"), ("Points scored", "num")], rows)
    tcard = card("All forecast errors", "For the dataset picked above. Click a column header to sort.", tab)
    return main + tcard


def profile_section(d: dict) -> str:
    pr = d["pr"]
    ks = sorted(pr.loc[pr["method"] == "latent ODE", "k"].unique())
    k_opts = [(str(k), "No levels" if k == 0 else f"First {k} levels") for k in ks]
    controls = select("patient", F.patient_options(pr), "Patient") + segmented("k", k_opts[::-1], "Levels given")
    body = f'<figure class="fig">{plot("profiles", PNGS["profiles"], "Concentration profiles for one test patient", tall=True)}</figure>' + png_link(PNGS["profiles"])
    sub = (
        "Three test patients from the ADA dataset. Pick a patient and how many of their levels the models see. "
        "The popPK forecast is MAP with the covariate model. Filled points are the levels given, open ones are held out. "
        "Neither model knows about ADA."
    )
    return card("Example profiles", sub, body, controls)


def ada_section(d: dict) -> str:
    ada = d["ada"]
    groups = [g for g in F.ADA_GROUPS if g in set(ada["group"])]
    opts = [(g, g[0].upper() + g[1:]) for g in groups]
    n = ada.groupby("group")["n_patients"]
    ranges = "; ".join(
        f"{g}: {n.min()[g]}" + (f" to {n.max()[g]}" if n.min()[g] != n.max()[g] else "") + " patients" for g in groups
    )
    body = f'<figure class="fig">{plot("ada", PNGS["ada"], "Forecast error by ADA status")}</figure>' + png_link(PNGS["ada"])
    sub = (
        "ADA test patients split by whether onset happened before the last level given. The groups change with k, so the "
        f"patient counts do too ({esc(ranges)}). Hover for the count at each k. Nobody can forecast an onset that hasn't "
        "happened yet; the ceiling is only low there because it knows the onset time."
    )
    return card("Forecast error by ADA status", sub, body, segmented("group", opts, "ADA group"))


def covariate_section(d: dict) -> str:
    cr = d["cr"]
    covs = [c for c in F.COVARIATES if c in set(cr["covariate"])]
    dose = function_defaults(ROOT / "src" / "latent_ode_pk" / "evaluate.py", "covariate_recovery")["mgkg"]
    body = f'<figure class="fig">{plot("cov", PNGS["cov"], "Partial dependence of mean concentration on one covariate")}</figure>' + png_link(PNGS["cov"])
    sub = (
        f"Every test patient from the dose-level dataset gets each value of one covariate in turn, at {dose:g} mg/kg with no "
        "levels given. The curve is the mean concentration over the dosing interval after the third dose, averaged over "
        "patients. PLT has no effect in the true model."
    )
    c1 = card("Covariate recovery", sub, body, select("covariate", [(c, F.COVARIATES[c][0]) for c in covs], "Covariate"))
    rows = []
    for r in F.covariate_ratios(cr):
        rng = f"{r['lo']:.3g} to {r['hi']:.3g}"
        rows.append(
            f"<tr><td>{esc(r['name'])}, {esc(rng)}</td>{td_num(r['truth'], '.2f')}{td_num(r['popPK with covariates'], '.2f')}{td_num(r['latent ODE'], '.2f')}</tr>"
        )
    tab = table([("Covariate range", ""), ("Truth", "num"), ("popPK with covariates", "num"), ("Latent ODE", "num")], rows)
    c2 = card(
        "Effect size, highest over lowest value",
        "Ratio of the mean concentrations at the two ends of each grid. The network gets the direction of the weight, "
        "albumin and INF effects and most of their size, but it also picks up a PLT effect that isn't there.",
        tab,
    )
    return f'<div class="grid-2">{c1}{c2}</div>'


def checks_section(d: dict) -> str:
    sc = d["sc"]
    datasets = [k for k in F.DATASETS if k in set(sc["dataset"])]
    checks = list(dict.fromkeys(sc["check"]))

    def fmt(check: str, v: float) -> str:
        if check == "positivity":
            return f"{v:.2f} mg/L"
        if check == "determinism":
            return f"{v:.1e}"
        if check == "one peak per dose":
            return f"{v:.0%}"
        return f"{v:.3f}"

    rows = []
    for c in checks:
        sub = sc[sc["check"] == c].set_index("dataset")
        detail = sub["detail"].iloc[0]
        cells = []
        for ds in datasets:
            if ds not in sub.index:
                cells.append('<td class="na">not run</td>')
                continue
            r = sub.loc[ds]
            ok = bool(r["passed"])
            badge = f'<span class="status {"pass" if ok else "fail"}"><span aria-hidden="true">{"&#10003;" if ok else "&#10005;"}</span> {"pass" if ok else "fail"}</span>'
            cells.append(f"<td>{esc(fmt(c, float(r['value'])))} {badge}</td>")
        rows.append(f'<tr><td>{esc(c[0].upper() + c[1:])}</td><td class="detail">{esc(detail)}</td>{"".join(cells)}</tr>')
    headers = [("Check", ""), ("What is measured", "")] + [(F.DATASETS[ds], "") for ds in datasets]
    tab = table(headers, rows, sortable=False)
    sub = (
        "Run on each network's test patients. The clean network fails dose proportionality: everyone in that dataset got "
        "the same mg/kg dose, and dose-rescaling augmentation alone doesn't fix it. The two networks trained on several dose "
        "levels pass. The unseen-interval check gives the dose-level network a maintenance interval it never saw in training."
    )
    return card("Structural checks", sub, tab, cls="card checks")


def training_section(d: dict) -> str:
    opts = [(k, v) for k, v in F.DATASETS.items() if k in d["train"]]
    info = d["info"]
    per = ", ".join(f"{F.DATASETS[k].lower()} {info['epochs'][k]} epochs in {info['train_seconds'][k]:.0f} s" for k in d["train"])
    body = (
        '<div class="grid-2">'
        f'<figure class="fig">{plot("train_loss", PNGS["training"], "Training loss by epoch")}</figure>'
        f'<figure class="fig">{plot("val_mse", PNGS["training"], "Validation error by epoch")}</figure>'
        "</div>" + png_link(PNGS["training"])
    )
    sub = (
        "Loss is the negative log-likelihood of the observed log levels plus the KL term, so it can go below zero. "
        "Validation error is the mean squared error on log levels from the posterior mean. Hover for the KL weight and "
        f"learning rate. Training stopped early: {esc(per)}."
    )
    return card("Training curves", sub, body, segmented("dataset", opts, "Dataset"))


def about_section(d: dict) -> str:
    m = dataclass_defaults(ROOT / "src" / "latent_ode_pk" / "model.py", "ModelConfig")
    t = dataclass_defaults(ROOT / "src" / "latent_ode_pk" / "train.py", "TrainConfig")
    covs = ", ".join(v[0].lower() if k in ("WT", "ALB") else k for k, v in F.COVARIATES.items())
    arch = f"""
      <ol class="steps">
        <li>The covariates ({covs}) go through a small MLP to a context vector c ({m['d_ctx']} numbers).</li>
        <li>Each patient has a latent vector u ({m['d_u']} numbers), a learned stand-in for the random effects. Its prior depends on c.
          A GRU reads the first k measured levels and gives the posterior of u. That is the network's version of Bayesian updating;
          with no levels the posterior is the prior.</li>
        <li>A state x ({m['d_x']} numbers) starts at x0(c, u) and follows dx/dt = f(x, u, c), solved with a fixed-step RK4 written in PyTorch.</li>
        <li>Each dose is a learned jump in x at the dose time.</li>
        <li>A small head maps (x, u) to log concentration, so predictions are always positive.</li>
      </ol>
      <p class="note">Trained with Adam (learning rate {t['lr']:g}, batches of {t['batch_size']}), a KL warm-up over {t['kl_warmup_epochs']} epochs,
        early stopping after {t['patience']} epochs without improvement, and dose-rescaling augmentation, which is valid because Drug X has linear PK.</p>
    """
    c1 = card("The network", "A latent neural ODE with a GRU encoder for the levels seen so far.", arch)

    sizes = d["info"]["sizes"]
    rows = []
    for k, s in sizes.items():
        ada = f"{s['ada_share']:.0%}" if "ada_share" in s else "none"
        rows.append(
            f"<tr><td>{esc(F.DATASETS.get(k, k))}</td><td>{s['train']}</td><td>{s['val']}</td><td>{s['test']}</td>"
            f"<td>{s['blq_share']:.1%}</td><td>{ada}</td></tr>"
        )
    tab = table([("Dataset", ""), ("Train", ""), ("Validation", ""), ("Test", ""), ("Levels below LLOQ", ""), ("Turned ADA positive", "")], rows, sortable=False)
    text = (
        '<p class="note">Clean: one dose level, no assay error, no IOV. Dose levels: several mg/kg dose levels with IOV, assay error '
        "and censoring below the LLOQ. ADA: the same, plus anti-drug antibodies that switch on partway through and raise clearance. "
        "popPK comparators are two-compartment models fitted by iterative two-stage estimation and forecast by MAP.</p>"
    )
    c2 = card("The simulated data", "Patient counts per dataset, from the run log.", tab + text)

    pop = d["pop"]
    pars = ["CL", "V1", "Q", "V2", "omega_CL", "omega_V1", "sigma", "omega_rw", "CL~WT", "CL~ALB", "CL~INF", "CL~PLT", "V1~WT"]
    tables = []
    for ds, models in pop.items():
        rows = []
        for p in pars:
            vals = [models[n].get(p) for n in models]
            if all(v is None for v in vals):
                continue
            cells = "".join("<td></td>" if v is None else f"<td>{v:.3f}</td>" for v in vals)
            rows.append(f"<tr><td><code>{esc(p)}</code></td>{cells}</tr>")
        heads = [("Parameter", "")] + [(n, "") for n in models]
        tables.append(f'<div data-filter="dataset:{esc(ds)}">{table(heads, rows, sortable=False)}</div>')
    c3 = card(
        "popPK estimates",
        "Population estimates from the training patients (CL and Q in L/day, V1 and V2 in L, omegas and sigma as log-scale SDs, "
        "covariate effects as power exponents). The true values are in the README.",
        "".join(tables),
        segmented("dataset", [(k, v) for k, v in F.DATASETS.items() if k in pop], "Dataset"),
    )
    return f'<div class="grid-2">{c1}{c2}</div>{c3}'


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
def page(d: dict, figs: dict, built: str) -> str:
    data = json.dumps(figs, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    n_test = d["info"]["sizes"]["clean"]["test"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Latent ODE for Drug X PK</title>
  <meta name="description" content="Results from a latent neural ODE trained on simulated Drug X concentrations, compared with popPK MAP forecasting.">
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="style.css">
  <script src="{PLOTLY_JS}" integrity="{PLOTLY_SRI}" crossorigin="anonymous"></script>
  <noscript><style>.plot{{display:none}} .fallback{{display:block !important}}</style></noscript>
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <div class="header-row">
        <p class="eyebrow">Latent neural ODE, simulated Drug X</p>
        <button type="button" id="theme" class="theme-toggle" aria-pressed="false">Dark mode</button>
      </div>
      <h1>Can a neural ODE learn a drug's PK as well as popPK?</h1>
      <p class="lede">A latent neural ODE learns Drug X concentrations from data alone and is scored against the exact solution,
        next to two-compartment popPK models forecast by MAP, on three simulated datasets with {n_test} test patients each.</p>
      <p class="banner">All data on this page are simulated from a made-up drug. It's a research and teaching example, not validated, and not for patient care.</p>
    </div>
  </header>

  <main class="wrap">
    {tiles(d)}
    <article class="card">
      <h2>What the results say</h2>
      <ul class="plain">
        <li>With no levels, the network forecasts as well as the popPK model with covariates on all three datasets, and better than popPK without covariates.</li>
        <li>Once levels come in, MAP with the popPK model pulls ahead. On clean data it's almost exact, while the network stays several percent off.</li>
        <li>When ADA raises clearance partway through, the network beats the constant-clearance popPK model for patients whose onset shows in their levels.
          A popPK model with a random walk on clearance does better than both.</li>
      </ul>
    </article>
    {fold_section(d)}
    {profile_section(d)}
    {ada_section(d)}
    {covariate_section(d)}
    {checks_section(d)}
    {training_section(d)}
    {about_section(d)}
  </main>

  <footer class="wrap footer">
    <p>Code, methods and result files: <a href="{REPO_URL}">github.com/Wrlog/latent-ode</a>.
      Simulated data only. Not for patient care. Charts drawn with plotly from the files in <code>results/</code>. Built {esc(built)}.</p>
  </footer>

  <script id="figs" type="application/json">{data}</script>
  <script src="app.js"></script>
</body>
</html>
"""


def build(results: Path, out: Path) -> Path:
    d = load(results)
    figs = {
        "fe_mfe": F.fold_error(d["fe"], "mfe"),
        "fe_r2": F.fold_error(d["fe"], "r2"),
        "profiles": F.profiles(d["pr"]),
        "ada": F.ada_status(d["ada"]),
        "cov": F.covariates(d["cr"]),
        "train_loss": F.training(d["train"], "train_loss"),
        "val_mse": F.training(d["train"], "val_mse"),
    }
    out.mkdir(parents=True, exist_ok=True)
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    (out / "index.html").write_text(page(d, figs, built), encoding="utf-8")
    for name in ("style.css", "app.js"):
        shutil.copyfile(HERE / name, out / name)
    (out / "figures").mkdir(exist_ok=True)
    for png in PNGS.values():
        src = ROOT / "figures" / png
        if src.exists():
            shutil.copyfile(src, out / "figures" / png)
    (out / ".nojekyll").write_text("")
    return out / "index.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    a = ap.parse_args()
    path = build(a.results, a.out)
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
