"""Scoring against the exact solution, structural checks and follow-up analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import pk
from .data import make_batch
from .model import LatentODE
from .poppk import PopModel, _predict, first_k_mask, forecast, kth_level_time, map_fit, true_model_forecast
from .simulate import HORIZON, Dataset

K_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def forecast_mask(ds: Dataset, k: int) -> np.ndarray:
    """Sample times after each patient's k-th level where the true
    concentration is at or above the LLOQ."""
    tk = kth_level_time(ds.obs, ds.obs_times, k)
    return (ds.obs_times[None, :] > tk[:, None]) & (ds.true_conc >= pk.LLOQ)


def score(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    e = np.log(pred[mask]) - np.log(truth[mask])
    lt = np.log(truth[mask])
    return {
        "mfe": float(np.exp(np.median(np.abs(e)))),
        "r2": float(1 - np.sum(e**2) / np.sum((lt - lt.mean()) ** 2)),
        "n_points": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# network predictions
# ---------------------------------------------------------------------------
def network_batch(model: LatentODE, ds: Dataset, times=None, regimen: pk.Regimen | None = None, max_gap=None):
    times = ds.obs_times if times is None else np.asarray(times, float)
    reg = regimen or ds.regimen
    kw = {} if max_gap is None else {"max_gap": max_gap}
    return make_batch(
        ds.cov,
        reg,
        times,
        level_times=np.broadcast_to(ds.obs_times, ds.obs.shape),
        levels=ds.obs,
        y_mean=float(model.y_mean),
        y_std=float(model.y_std),
        **kw,
    )


def network_forecast(model: LatentODE, ds: Dataset, k: int, times=None, regimen=None, max_gap=None) -> np.ndarray:
    return model.predict(network_batch(model, ds, times, regimen, max_gap), k).numpy().astype(float)


def fold_error_table(ds: Dataset, methods: dict, ks=K_GRID) -> pd.DataFrame:
    """methods: name -> callable(k) returning predictions at ds.obs_times."""
    rows = []
    for k in ks:
        mask = forecast_mask(ds, k)
        for name, fn in methods.items():
            rows.append({"dataset": ds.name, "method": name, "k": k, **score(fn(k), ds.true_conc, mask)})
    return pd.DataFrame(rows)


def standard_methods(ds: Dataset, net: LatentODE | None, pop: dict[str, PopModel]) -> dict:
    m = {}
    if net is not None:
        m["latent ODE"] = lambda k: network_forecast(net, ds, k)
    for name, model in pop.items():
        m[name] = (lambda mod: (lambda k: forecast(mod, ds, k)))(model)
    m["true model (ceiling)"] = lambda k: true_model_forecast(ds, k)
    return m


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------
def dense_times(regimen: pk.Regimen, step: float = 0.25, horizon: float = HORIZON) -> np.ndarray:
    t = np.arange(step, horizon + 1e-9, step)
    return np.unique(np.concatenate([t, regimen.times + pk.INFUSION_DAYS]))


def count_peaks(times: np.ndarray, conc: np.ndarray, dose_times: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Number of local maxima per dosing interval, (n, n_doses).

    Each interval's series starts with the last point before the dose (zero
    for the first dose), so the rise after the dose is seen. Changes in log concentration smaller than `tol`
    count as flat, and a series that is still rising at the end of the interval
    counts that end as a peak."""
    n = conc.shape[0]
    bounds = list(dose_times[1:]) + [np.inf]
    out = np.zeros((n, len(dose_times)), dtype=int)
    logc = np.log(conc)
    for j, (t0, t1) in enumerate(zip(dose_times, bounds)):
        inside = np.where((times > t0) & (times < t1))[0]
        before = np.where(times <= t0)[0]
        idx = np.concatenate([before[-1:], inside]) if len(before) else inside
        for i in range(n):
            d = np.diff(logc[i, idx])
            if not len(before):
                d = np.concatenate([[1.0], d])  # the first dose starts from zero
            d = d[np.abs(d) > tol]
            if len(d) == 0:
                continue
            s = np.sign(d)
            peaks = int(np.sum((s[:-1] > 0) & (s[1:] < 0)) + (s[-1] > 0))
            out[i, j] = peaks
    return out


def structural_checks(model: LatentODE, ds: Dataset, unseen: Dataset | None = None) -> pd.DataFrame:
    rows = []
    dt_ = dense_times(ds.regimen)

    # 1. positivity on a dense grid, with and without levels
    vals = [network_forecast(model, ds, k, times=dt_) for k in (0, 4)]
    allv = np.concatenate([v.ravel() for v in vals])
    ok = bool(np.all(np.isfinite(allv)) and np.all(allv > 0))
    rows.append({"check": "positivity", "value": float(allv.min()), "detail": "smallest prediction on a 6-hour grid (mg/L)", "passed": ok})

    # 2. determinism: repeat calls, and a patient's prediction does not depend on the batch
    b = network_batch(model, ds)
    p1 = model.predict(b, 4).numpy()
    p2 = model.predict(b, 4).numpy()
    sub = ds.subset(np.arange(min(5, ds.n)))
    p3 = network_forecast(model, sub, 4)
    d_rep = float(np.max(np.abs(p1 - p2)))
    d_batch = float(np.max(np.abs(np.log(p3) - np.log(p1[: sub.n]))))
    rows.append({"check": "determinism", "value": max(d_rep, d_batch), "detail": "max difference between repeat calls / batch sizes", "passed": d_rep == 0.0 and d_batch < 1e-5})

    # 3. one peak per dose
    peaks = count_peaks(dt_, vals[0], ds.regimen.times)
    frac = float(np.mean(peaks == 1))
    rows.append({"check": "one peak per dose", "value": frac, "detail": "share of patient-intervals with exactly one peak", "passed": frac >= 0.95})

    # 4. dose proportionality: doubling every dose should double every concentration
    base = network_forecast(model, ds, 0)
    reg2 = pk.Regimen(ds.regimen.times, ds.regimen.amounts * 2)
    dbl = network_forecast(model, ds, 0, regimen=reg2)
    dev = np.abs(np.log(dbl / base) - np.log(2))
    rows.append({"check": "dose proportionality", "value": float(np.exp(np.median(dev))), "detail": "median fold deviation of C(2D)/C(D) from 2", "passed": float(np.median(dev)) < np.log(1.05)})

    # 5. grid invariance: adding query points (a finer solver grid) should not move predictions
    extra = np.unique(np.concatenate([ds.obs_times, np.arange(0.5, HORIZON, 0.5)]))
    fine = network_forecast(model, ds, 4, times=extra)
    pos = np.searchsorted(extra, ds.obs_times)
    coarse = network_forecast(model, ds, 4)
    gdev = np.abs(np.log(fine[:, pos]) - np.log(coarse))
    rows.append({"check": "grid invariance", "value": float(np.quantile(gdev, 0.99)), "detail": "99th percentile |change in log C| with a 12-hour grid added", "passed": float(np.quantile(gdev, 0.99)) < 0.02})

    # 6. a dosing interval not seen in training
    if unseen is not None:
        for k in (0, 4):
            e_seen = score(network_forecast(model, ds, k), ds.true_conc, forecast_mask(ds, k))
            e_new = score(network_forecast(model, unseen, k), unseen.true_conc, forecast_mask(unseen, k))
            ratio = np.log(e_new["mfe"]) / np.log(e_seen["mfe"])
            rows.append(
                {
                    "check": f"unseen interval (k={k})",
                    "value": float(ratio),
                    "detail": f"median fold error {e_new['mfe']:.3f} on q{int(unseen.meta['maintenance_interval'])}d vs {e_seen['mfe']:.3f} on q56d (ratio of log errors)",
                    "passed": ratio <= 1.25,
                }
            )
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", ds.name)
    return out


# ---------------------------------------------------------------------------
# follow-ups
# ---------------------------------------------------------------------------
def covariate_recovery(model: LatentODE, pop_cov: PopModel, ds: Dataset, n_grid: int = 9, mgkg: float = 5.0) -> pd.DataFrame:
    """Partial dependence of the average concentration over days 42-98 on each
    covariate, for the network, the covariate popPK model and the truth (random
    effects at zero). Every test patient is given the swept value in turn and
    the curve is the mean over patients."""
    grids = {
        "WT": np.linspace(25, 95, n_grid),
        "ALB": np.linspace(3.0, 4.8, n_grid),
        "INF": np.exp(np.linspace(np.log(1.0), np.log(40.0), n_grid)),
        "PLT": np.linspace(180, 520, n_grid),
    }
    t = np.arange(42.5, 98.0, 0.5)
    rows = []
    for cname, grid in grids.items():
        for v in grid:
            cov = ds.cov.copy()
            cov[cname] = v
            amounts = mgkg * cov["WT"].to_numpy()[:, None] * np.ones((1, ds.regimen.n_doses))
            reg = pk.Regimen(ds.regimen.times, amounts)
            b = make_batch(cov, reg, t, y_mean=float(model.y_mean), y_std=float(model.y_std))
            net = model.predict(b, 0).numpy().mean(1)
            covd = {c: cov[c].to_numpy() for c in cov.columns}
            truth = pk.simulate(pk.typical_params(covd), reg, t).mean(1)
            pop = pk.simulate(pop_cov.typical(covd), reg, t).mean(1)
            for name, arr in (("latent ODE", net), ("popPK with covariates", pop), ("truth", truth)):
                rows.append({"covariate": cname, "value": float(v), "method": name, "cavg": float(np.mean(arr))})
    return pd.DataFrame(rows)


def ada_status_table(ds: Dataset, methods: dict, ks=K_GRID) -> pd.DataFrame:
    """Forecast error split by ADA status relative to the last level supplied."""
    rows = []
    onset = ds.ada_onset
    for k in ks:
        mask = forecast_mask(ds, k)
        tk = kth_level_time(ds.obs, ds.obs_times, k)
        groups = {
            "ADA negative": ~np.isfinite(onset),
            "onset before last level": np.isfinite(onset) & (onset < tk),
            "onset after last level": np.isfinite(onset) & (onset >= tk),
        }
        preds = {name: fn(k) for name, fn in methods.items()}
        for g, sel in groups.items():
            m = mask & sel[:, None]
            if sel.sum() < 3 or m.sum() == 0:
                continue
            for name, p in preds.items():
                rows.append({"k": k, "group": g, "n_patients": int(sel.sum()), "method": name, **score(p, ds.true_conc, m)})
    return pd.DataFrame(rows)


def example_profiles(model: LatentODE, pop: PopModel, ds: Dataset, ids, ks=(0, 8)) -> pd.DataFrame:
    sub = ds.subset(np.asarray(ids))
    t = dense_times(sub.regimen, step=0.5)
    truth = sub.truth(t)
    rows = []
    for j, pid in enumerate(ids):
        for tt, c in zip(t, truth[j]):
            rows.append({"patient": int(pid), "method": "truth", "k": -1, "time": tt, "conc": c})
    for k in ks:
        net = network_forecast(model, sub, k, times=t)
        if k == 0:
            theta = np.zeros((sub.n, 2 + (sub.regimen.n_doses - 1 if pop.omega_rw is not None else 0)))
        else:
            theta, _ = map_fit(pop, sub.cov_dict(), sub.regimen, sub.obs_times, sub.obs, first_k_mask(sub.obs, k))
        pp = _predict(pop, sub.cov_dict(), sub.regimen, t, theta)
        for j, pid in enumerate(ids):
            for tt, a, b in zip(t, net[j], pp[j]):
                rows.append({"patient": int(pid), "method": "latent ODE", "k": k, "time": tt, "conc": a})
                rows.append({"patient": int(pid), "method": "popPK", "k": k, "time": tt, "conc": b})
    for j, pid in enumerate(ids):
        used = first_k_mask(sub.obs[j : j + 1], max(ks))[0]
        for tt, o, u in zip(sub.obs_times, sub.obs[j], used):
            rows.append({"patient": int(pid), "method": "observed", "k": int(u), "time": tt, "conc": o})
    return pd.DataFrame(rows)

