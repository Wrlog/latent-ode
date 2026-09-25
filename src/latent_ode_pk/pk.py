"""The Drug X two-compartment PK model and an exact piecewise solver.

Parameters follow the shared Drug X model used across my demo repos:

    CL = 0.30 L/day * (WT/70)^0.75 * (ALB/4)^-1 * (INF/5)^0.10 * 1.8^ADA * exp(eta_CL + kappa_CL)
    V1 = 3.2 L * (WT/70) * exp(eta_V1)
    Q  = 0.50 L/day * (WT/70)^0.75
    V2 = 2.0 L * (WT/70)

Doses are 2-hour IV infusions. Between breakpoints every parameter is constant,
so the state is carried forward with the closed-form matrix exponential of the
2x2 system. The only approximation is for the ADA rise, whose clearance
multiplier is evaluated at the midpoint of 6-hour pieces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INFUSION_DAYS = 2.0 / 24.0
LLOQ = 0.5  # mg/L

TV_CL = 0.30
TV_V1 = 3.2
TV_Q = 0.50
TV_V2 = 2.0
ADA_FACTOR = 1.8

OMEGA_CL = float(np.sqrt(np.log(1 + 0.30**2)))
OMEGA_V1 = float(np.sqrt(np.log(1 + 0.20**2)))
OMEGA_IOV = float(np.sqrt(np.log(1 + 0.15**2)))
SIGMA_PROP = 0.15


@dataclass
class Regimen:
    """Dose start times (days, shared by all patients) and amounts in mg per patient."""

    times: np.ndarray  # (n_doses,)
    amounts: np.ndarray  # (n_patients, n_doses)

    @property
    def n_doses(self) -> int:
        return len(self.times)


def standard_dose_times(maintenance_interval: float = 56.0, horizon: float = 182.0) -> np.ndarray:
    """Induction at days 0, 21 and 42, then maintenance every `maintenance_interval` days."""
    times = [0.0, 21.0, 42.0]
    t = 42.0 + maintenance_interval
    while t < horizon - 1e-9:
        times.append(t)
        t += maintenance_interval
    return np.array(times)


def typical_params(cov: dict[str, np.ndarray], ada: np.ndarray | float = 0.0) -> dict[str, np.ndarray]:
    """Covariate-predicted parameters with all random effects at zero."""
    wt = np.asarray(cov["WT"], float)
    cl = (
        TV_CL
        * (wt / 70) ** 0.75
        * (np.asarray(cov["ALB"], float) / 4.0) ** -1.0
        * (np.asarray(cov["INF"], float) / 5.0) ** 0.10
        * ADA_FACTOR ** np.asarray(ada, float)
    )
    return {
        "CL": cl,
        "V1": TV_V1 * wt / 70,
        "Q": TV_Q * (wt / 70) ** 0.75,
        "V2": TV_V2 * wt / 70,
    }


def individual_params(cov: dict[str, np.ndarray], eta_cl: np.ndarray, eta_v1: np.ndarray) -> dict[str, np.ndarray]:
    p = typical_params(cov)
    p["CL"] = p["CL"] * np.exp(eta_cl)
    p["V1"] = p["V1"] * np.exp(eta_v1)
    return p


def ada_multiplier(t: np.ndarray, onset: np.ndarray, half_rise: float) -> np.ndarray:
    """Clearance multiplier after ADA onset; rises smoothly from 1 towards ADA_FACTOR.

    `t` broadcasts against `onset` (use np.inf for patients who never turn positive).
    """
    dt = np.clip(t - onset, 0.0, None)
    frac = 1.0 - np.exp(-np.log(2) * dt / half_rise)
    return 1.0 + (ADA_FACTOR - 1.0) * frac


def _expm_step(k10, k12, k21, h, a1, a2, rate):
    """Advance the 2-compartment amounts by h days with constant infusion rate.

    All arguments are arrays of shape (n,). Uses the eigen-decomposition of the
    2x2 rate matrix, which always has two distinct negative real eigenvalues.
    """
    m11 = -(k10 + k12)
    m12 = k21
    m21 = k12
    m22 = -k21
    half_tr = 0.5 * (m11 + m22)
    det = m11 * m22 - m12 * m21
    q = np.sqrt(np.maximum(half_tr**2 - det, 1e-300))
    l1 = half_tr + q
    l2 = half_tr - q
    e1 = np.exp(l1 * h)
    e2 = np.exp(l2 * h)
    denom = l1 - l2

    # E = (e1 (M - l2 I) - e2 (M - l1 I)) / (l1 - l2)
    def e_times(v1, v2):
        mv1 = m11 * v1 + m12 * v2
        mv2 = m21 * v1 + m22 * v2
        o1 = (e1 * (mv1 - l2 * v1) - e2 * (mv1 - l1 * v1)) / denom
        o2 = (e1 * (mv2 - l2 * v2) - e2 * (mv2 - l1 * v2)) / denom
        return o1, o2

    n1, n2 = e_times(a1, a2)
    # particular part: M^-1 (E - I) b with b = (rate, 0)
    eb1, eb2 = e_times(rate, np.zeros_like(rate))
    d1 = eb1 - rate
    d2 = eb2
    inv11 = m22 / det
    inv12 = -m12 / det
    inv21 = -m21 / det
    inv22 = m11 / det
    n1 = n1 + inv11 * d1 + inv12 * d2
    n2 = n2 + inv21 * d1 + inv22 * d2
    return n1, n2


def simulate(
    params: dict[str, np.ndarray],
    regimen: Regimen,
    query_times: np.ndarray,
    kappa: np.ndarray | None = None,
    ada_onset: np.ndarray | None = None,
    ada_half_rise: float = 4.0,
    cl_occasion_mult: np.ndarray | None = None,
    fine_step: float = 0.25,
) -> np.ndarray:
    """Exact concentrations (mg/L) at `query_times` for every patient.

    params: CL, V1, Q, V2 arrays of shape (n,).
    kappa: optional (n, n_doses) log-scale IOV on CL, one occasion per dosing interval.
    ada_onset: optional (n,) onset day (np.inf if never).
    cl_occasion_mult: optional (n, n_doses) extra multiplier on CL per occasion
        (used by the time-varying popPK comparator).
    """
    query_times = np.asarray(query_times, float)
    n = len(params["CL"])
    dose_t = regimen.times
    starts = dose_t
    ends = dose_t + INFUSION_DAYS
    pts = [np.array([0.0]), starts, ends, query_times]
    if ada_onset is not None and np.any(np.isfinite(ada_onset)):
        pts.append(np.arange(0.0, query_times.max() + fine_step, fine_step))
    grid = np.unique(np.concatenate(pts))
    grid = grid[grid <= query_times.max() + 1e-12]

    cl0 = np.asarray(params["CL"], float)
    v1 = np.asarray(params["V1"], float)
    k12 = np.asarray(params["Q"], float) / v1
    k21 = np.asarray(params["Q"], float) / np.asarray(params["V2"], float)

    a1 = np.zeros(n)
    a2 = np.zeros(n)
    out = np.full((n, len(query_times)), np.nan)
    q_lookup: dict[float, list[int]] = {}
    for i, t in enumerate(query_times):
        q_lookup.setdefault(float(t), []).append(i)

    rate_per_dose = regimen.amounts / INFUSION_DAYS  # (n, n_doses)
    for j in range(len(grid)):
        t0 = grid[j]
        if float(t0) in q_lookup:
            for i in q_lookup[float(t0)]:
                out[:, i] = a1 / v1
        if j == len(grid) - 1:
            break
        t1 = grid[j + 1]
        h = t1 - t0
        mid = 0.5 * (t0 + t1)
        active = (starts <= mid) & (mid < ends)
        rate = rate_per_dose[:, active].sum(axis=1) if active.any() else np.zeros(n)
        occ = max(int(np.searchsorted(starts, mid, side="right")) - 1, 0)
        cl = cl0.copy()
        if kappa is not None:
            cl = cl * np.exp(kappa[:, occ])
        if cl_occasion_mult is not None:
            cl = cl * cl_occasion_mult[:, occ]
        if ada_onset is not None:
            cl = cl * ada_multiplier(mid, ada_onset, ada_half_rise)
        a1, a2 = _expm_step(cl / v1, k12, k21, h, a1, a2, rate)
    return out
