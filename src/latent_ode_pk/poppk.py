"""Comparators: a two-compartment popPK model with MAP Bayesian forecasting,
and the true-model reference ("ceiling").

The popPK model is estimated from the training patients by iterative two-stage
(ITS) estimation: individual MAP fits, then population means, covariate
coefficients, IIV and residual error from those fits, repeated a few times.
Three variants:

- no covariates (IIV on CL and V1 only)
- power covariate model on CL (WT, ALB, INF, PLT) and V1 (WT), fixed allometry on Q and V2
- the covariate model plus a random walk on log CL from one dosing interval to
  the next, for the ADA dataset. Forecasts carry the last estimated CL forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

from . import pk
from .simulate import ADA_HALF_RISE, COVARIATES, Dataset

CL_COVS = ["WT", "ALB", "INF", "PLT"]
REF = {"WT": 70.0, "ALB": 4.0, "INF": 5.0, "PLT": 300.0}


# ---------------------------------------------------------------------------
# batched Levenberg-Marquardt: many small independent least-squares problems
# ---------------------------------------------------------------------------
def batched_lm(resid_fn, theta0: np.ndarray, n_iter: int = 40, eps: float = 1e-4):
    """Minimise sum(r_i(theta_i)^2) separately for every row i.

    resid_fn(theta (n, p)) -> residuals (n, m). Returns theta and the
    Gauss-Newton covariance (J'J)^-1 at the solution, shape (n, p, p).
    """
    theta = theta0.astype(float).copy()
    n, p = theta.shape
    lam = np.full(n, 1e-2)
    r = resid_fn(theta)
    cost = (r**2).sum(1)

    def jac(th, r0):
        cols = []
        for j in range(p):
            tp = th.copy()
            tp[:, j] += eps
            cols.append((resid_fn(tp) - r0) / eps)
        return np.stack(cols, -1)  # (n, m, p)

    for _ in range(n_iter):
        J = jac(theta, r)
        JtJ = np.einsum("nmp,nmq->npq", J, J)
        Jtr = np.einsum("nmp,nm->np", J, r)
        A = JtJ + lam[:, None, None] * (np.eye(p)[None] * (np.diagonal(JtJ, axis1=1, axis2=2)[:, :, None] + 1e-9))
        step = -np.linalg.solve(A, Jtr[..., None])[..., 0]
        cand = theta + step
        r_c = resid_fn(cand)
        cost_c = (r_c**2).sum(1)
        better = cost_c < cost
        theta[better] = cand[better]
        r[better] = r_c[better]
        cost[better] = cost_c[better]
        lam = np.where(better, lam * 0.3, lam * 5.0)
        lam = np.clip(lam, 1e-7, 1e7)
        if np.all(np.abs(step).max(1) < 1e-6):
            break
    J = jac(theta, r)
    JtJ = np.einsum("nmp,nmq->npq", J, J)
    cov = np.linalg.inv(JtJ + 1e-9 * np.eye(p)[None])
    return theta, cov


def first_k_mask(obs: np.ndarray, k: int) -> np.ndarray:
    """Mask of the first k quantifiable levels per patient."""
    ok = np.isfinite(obs) & (obs > 0)
    rank = np.cumsum(ok, axis=1)
    return ok & (rank <= k)


def kth_level_time(obs: np.ndarray, times: np.ndarray, k: int) -> np.ndarray:
    """Time of each patient's k-th quantifiable level (-inf for k=0)."""
    if k == 0:
        return np.full(obs.shape[0], -np.inf)
    m = first_k_mask(obs, k)
    t = np.where(m, times[None, :], -np.inf)
    return t.max(1)


# ---------------------------------------------------------------------------
# popPK model
# ---------------------------------------------------------------------------
@dataclass
class PopModel:
    use_cov: bool
    log_cl: float = np.log(0.3)
    log_v1: float = np.log(3.0)
    log_q: float = np.log(0.5)
    log_v2: float = np.log(2.0)
    beta_cl: dict = field(default_factory=lambda: {c: 0.0 for c in CL_COVS})
    beta_v1_wt: float = 0.0
    omega_cl: float = 0.5
    omega_v1: float = 0.5
    sigma: float = 0.2
    omega_rw: float | None = None  # random walk SD on log CL per dosing interval

    def typical(self, cov: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        n = len(cov["WT"])
        lcl = np.full(n, self.log_cl)
        lv1 = np.full(n, self.log_v1)
        lq = np.full(n, self.log_q)
        lv2 = np.full(n, self.log_v2)
        if self.use_cov:
            for c in CL_COVS:
                lcl = lcl + self.beta_cl[c] * np.log(cov[c] / REF[c])
            lw = np.log(cov["WT"] / 70.0)
            lv1 = lv1 + self.beta_v1_wt * lw
            lq = lq + 0.75 * lw
            lv2 = lv2 + 1.0 * lw
        return {"CL": np.exp(lcl), "V1": np.exp(lv1), "Q": np.exp(lq), "V2": np.exp(lv2)}

    @property
    def n_eta(self) -> int:
        return 2

    def summary(self) -> dict:
        d = {
            "use_cov": self.use_cov,
            "CL": float(np.exp(self.log_cl)),
            "V1": float(np.exp(self.log_v1)),
            "Q": float(np.exp(self.log_q)),
            "V2": float(np.exp(self.log_v2)),
            "omega_CL": self.omega_cl,
            "omega_V1": self.omega_v1,
            "sigma": self.sigma,
            "omega_rw": self.omega_rw,
        }
        if self.use_cov:
            d.update({f"CL~{c}": self.beta_cl[c] for c in CL_COVS})
            d["V1~WT"] = self.beta_v1_wt
        return d


def _predict(model: PopModel, cov, regimen: pk.Regimen, times, theta):
    base = model.typical(cov)
    p = dict(base)
    p["CL"] = base["CL"] * np.exp(theta[:, 0])
    p["V1"] = base["V1"] * np.exp(theta[:, 1])
    mult = None
    if model.omega_rw is not None and theta.shape[1] > 2:
        n_occ = regimen.n_doses
        steps = np.zeros((theta.shape[0], n_occ))
        steps[:, 1:] = theta[:, 2 : 2 + n_occ - 1]
        mult = np.exp(np.cumsum(steps, 1))
    return pk.simulate(p, regimen, times, cl_occasion_mult=mult)


def map_fit(model: PopModel, cov, regimen, times, obs, use_mask):
    """MAP estimates of the random effects from the levels where use_mask is True."""
    n = obs.shape[0]
    n_occ = regimen.n_doses
    tv = model.omega_rw is not None
    p = 2 + (n_occ - 1 if tv else 0)
    logy = np.where(use_mask, np.log(np.where(use_mask, obs, 1.0)), 0.0)
    w = use_mask.astype(float)
    prior_sd = np.array([model.omega_cl, model.omega_v1] + ([model.omega_rw] * (n_occ - 1) if tv else []))

    def resid(theta):
        c = _predict(model, cov, regimen, times, theta)
        r = w * (logy - np.log(np.maximum(c, 1e-12))) / model.sigma
        return np.concatenate([r, theta / prior_sd], 1)

    theta, pcov = batched_lm(resid, np.zeros((n, p)))
    return theta, pcov


def forecast(model: PopModel, ds: Dataset, k: int, times=None) -> np.ndarray:
    """MAP forecast from each patient's first k levels."""
    times = ds.obs_times if times is None else times
    mask = first_k_mask(ds.obs, k)
    cov = ds.cov_dict()
    if k == 0:
        theta = np.zeros((ds.n, 2 + (ds.regimen.n_doses - 1 if model.omega_rw is not None else 0)))
    else:
        theta, _ = map_fit(model, cov, ds.regimen, ds.obs_times, ds.obs, mask)
    return _predict(model, cov, ds.regimen, times, theta)


def fit_population(train: Dataset, use_cov: bool, n_iter: int = 5, time_varying: bool = False, verbose: bool = False) -> PopModel:
    """Iterative two-stage estimation on the training patients."""
    cov = train.cov_dict()
    ok = np.isfinite(train.obs) & (train.obs > 0)
    logy = np.log(np.where(ok, train.obs, 1.0))
    model = PopModel(use_cov=use_cov)

    # naive pooled start for the four structural parameters
    def pooled_resid(x):
        m = replace(model, log_cl=x[0], log_v1=x[1], log_q=x[2], log_v2=x[3])
        c = pk.simulate(m.typical(cov), train.regimen, train.obs_times)
        return (np.where(ok, logy - np.log(np.maximum(c, 1e-12)), 0.0)).ravel()

    x = least_squares(pooled_resid, [model.log_cl, model.log_v1, model.log_q, model.log_v2], method="trf").x
    model = replace(model, log_cl=x[0], log_v1=x[1], log_q=x[2], log_v2=x[3])

    lw = np.log(cov["WT"] / 70.0)
    X_cl = np.column_stack([np.ones(train.n)] + [np.log(cov[c] / REF[c]) for c in CL_COVS])
    X_v1 = np.column_stack([np.ones(train.n), lw])
    for it in range(n_iter):
        theta, pcov = map_fit(model, cov, train.regimen, train.obs_times, train.obs, ok)
        base = model.typical(cov)
        ind_lcl = np.log(base["CL"]) + theta[:, 0]
        ind_lv1 = np.log(base["V1"]) + theta[:, 1]
        c = _predict(model, cov, train.regimen, train.obs_times, theta)
        res = np.where(ok, logy - np.log(np.maximum(c, 1e-12)), 0.0)
        sigma = float(np.sqrt((res**2).sum() / ok.sum()))
        if use_cov:
            b_cl, *_ = np.linalg.lstsq(X_cl, ind_lcl, rcond=None)
            b_v1, *_ = np.linalg.lstsq(X_v1, ind_lv1, rcond=None)
            r_cl = ind_lcl - X_cl @ b_cl
            r_v1 = ind_lv1 - X_v1 @ b_v1
            new = dict(log_cl=b_cl[0], beta_cl={cc: float(b) for cc, b in zip(CL_COVS, b_cl[1:])}, log_v1=b_v1[0], beta_v1_wt=float(b_v1[1]))
        else:
            r_cl = ind_lcl - ind_lcl.mean()
            r_v1 = ind_lv1 - ind_lv1.mean()
            new = dict(log_cl=ind_lcl.mean(), log_v1=ind_lv1.mean())
        om_cl = float(np.sqrt(np.mean(r_cl**2) + np.mean(pcov[:, 0, 0]) * sigma**2 / model.sigma**2))
        om_v1 = float(np.sqrt(np.mean(r_v1**2) + np.mean(pcov[:, 1, 1]) * sigma**2 / model.sigma**2))
        model = replace(model, **new, omega_cl=max(om_cl, 0.02), omega_v1=max(om_v1, 0.02), sigma=max(sigma, 0.01))

        # Q and V2 by pooled least squares, holding the individual CL and V1
        def qv2_resid(x):
            m = replace(model, log_q=x[0], log_v2=x[1])
            b = m.typical(cov)
            p_ = dict(b)
            p_["CL"] = np.exp(ind_lcl)
            p_["V1"] = np.exp(ind_lv1)
            cc = pk.simulate(p_, train.regimen, train.obs_times)
            return (np.where(ok, logy - np.log(np.maximum(cc, 1e-12)), 0.0)).ravel()

        x = least_squares(qv2_resid, [model.log_q, model.log_v2], method="trf").x
        model = replace(model, log_q=x[0], log_v2=x[1])
        if verbose:
            print(f"    ITS {it}: {model.summary()}")

    if time_varying:
        model = add_random_walk(model, train, verbose=verbose)
    return model


def add_random_walk(model: PopModel, train: Dataset, n_iter: int = 3, verbose: bool = False) -> PopModel:
    """Give a fitted model a random walk on log CL between dosing intervals and
    estimate its SD (and the residual error) by a few more two-stage iterations."""
    cov = train.cov_dict()
    ok = np.isfinite(train.obs) & (train.obs > 0)
    logy = np.log(np.where(ok, train.obs, 1.0))
    model = replace(model, omega_rw=0.2)
    for it in range(n_iter):
        theta, pcov = map_fit(model, cov, train.regimen, train.obs_times, train.obs, ok)
        d = theta[:, 2:]
        dv = np.einsum("npp->np", pcov)[:, 2:]
        c = _predict(model, cov, train.regimen, train.obs_times, theta)
        res = np.where(ok, logy - np.log(np.maximum(c, 1e-12)), 0.0)
        sigma = float(np.sqrt((res**2).sum() / ok.sum()))
        om_rw = float(np.sqrt(np.mean(d**2 + dv * sigma**2 / model.sigma**2)))
        model = replace(model, omega_rw=max(om_rw, 0.01), sigma=max(sigma, 0.01))
        if verbose:
            print(f"    random walk {it}: omega_rw={model.omega_rw:.3f} sigma={model.sigma:.3f}")
    return model


# ---------------------------------------------------------------------------
# true-model reference
# ---------------------------------------------------------------------------
def true_model_forecast(ds: Dataset, k: int, times=None, sigma: float | None = None) -> np.ndarray:
    """The true model with CL and V1 at their covariate-predicted values (k=0), or
    MAP-fitted to the first k levels with the true priors. It knows each patient's
    ADA onset but not the IOV, which is left at zero."""
    times = ds.obs_times if times is None else times
    cov = ds.cov_dict()
    base = pk.typical_params(cov)
    sigma = sigma if sigma is not None else (pk.SIGMA_PROP if ds.name != "clean" else 0.01)
    prior_sd = np.array([pk.OMEGA_CL, pk.OMEGA_V1])

    def pred(theta, t):
        p = dict(base)
        p["CL"] = base["CL"] * np.exp(theta[:, 0])
        p["V1"] = base["V1"] * np.exp(theta[:, 1])
        return pk.simulate(p, ds.regimen, t, ada_onset=ds.ada_onset, ada_half_rise=ADA_HALF_RISE)

    theta = np.zeros((ds.n, 2))
    if k > 0:
        mask = first_k_mask(ds.obs, k)
        # only the times up to the last used level are needed for the fit
        t_fit = ds.obs_times
        logy = np.where(mask, np.log(np.where(mask, ds.obs, 1.0)), 0.0)
        w = mask.astype(float)
        last = int(np.where(mask.any(0))[0].max()) + 1
        t_fit = ds.obs_times[:last]

        def resid(th):
            c = pred(th, t_fit)
            r = w[:, :last] * (logy[:, :last] - np.log(np.maximum(c, 1e-12))) / sigma
            return np.concatenate([r, th / prior_sd], 1)

        theta, _ = batched_lm(resid, theta)
    return pred(theta, times)


__all__ = [
    "PopModel",
    "batched_lm",
    "fit_population",
    "add_random_walk",
    "forecast",
    "true_model_forecast",
    "first_k_mask",
    "kth_level_time",
    "COVARIATES",
]
