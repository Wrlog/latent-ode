"""Virtual patients and the three simulated datasets.

- "clean": everyone gets 5 mg/kg on the standard regimen, no assay error, no IOV.
- "doses": dose arms of 3, 5, 8 or 12 mg/kg, IOV on CL, 15% proportional assay
  error and values below the LLOQ reported as BLQ.
- "ada": like "doses", but some patients develop anti-drug antibodies (ADA) at a
  random time, after which clearance rises towards 1.8x. Onset is more likely
  when exposure is low.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import pk

HORIZON = 182.0
COVARIATES = ["WT", "ALB", "INF", "PLT"]
DOSE_ARMS = np.array([3.0, 5.0, 8.0, 12.0])  # mg/kg

# ADA onset model for the "ada" dataset (my own choices)
ADA_WINDOW = (10.0, 120.0)  # onset day, uniform
ADA_HALF_RISE = 4.0  # days for clearance to get half way to 1.8x
ADA_B0 = -0.3
ADA_B1 = 0.8  # per log-unit of the pre-dose-3 level below the reference
ADA_CREF = 20.0  # mg/L

# the benchmark used in the README: seeds and sizes (train / val / test)
SEEDS = {"clean": 101, "doses": 202, "ada": 303, "unseen": 404}
SIZES = {"n": 480, "n_val": 60, "n_test": 120}
UNSEEN_INTERVAL = 35.0  # days, maintenance interval never seen in training


def benchmark_dataset(kind: str) -> "Dataset":
    """The exact datasets used by scripts/run_all.py."""
    if kind == "unseen":
        n = SIZES["n_test"]
        return make_dataset("doses", n, SEEDS["unseen"], maintenance_interval=UNSEEN_INTERVAL, n_val=0, n_test=n)
    return make_dataset(kind, SIZES["n"], SEEDS[kind], n_val=SIZES["n_val"], n_test=SIZES["n_test"])


def sample_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Weight, albumin, inflammation marker and platelets.

    INF and ALB share a latent inflammation score, so they correlate negatively.
    """
    wt = np.clip(np.exp(np.log(55.0) + 0.28 * rng.standard_normal(n)), 20, 100)
    score = rng.standard_normal(n)
    inf = np.clip(np.exp(np.log(5.0) + 0.85 * score), 0.5, 60)
    alb = np.clip(3.9 - 0.4 * (0.6 * score + 0.8 * rng.standard_normal(n)), 2.5, 5.0)
    plt_ = np.clip(np.exp(np.log(300.0) + 0.25 * rng.standard_normal(n)), 150, 600)
    return pd.DataFrame({"WT": wt, "ALB": alb, "INF": inf, "PLT": plt_})


def observation_times(dose_times: np.ndarray, horizon: float = HORIZON) -> np.ndarray:
    """Dense sampling: end of infusion, days 1, 3, 7, 14, 28, 42 after each dose
    (while they fall inside the interval), a pre-dose level one hour before the
    next dose, and a last sample at the horizon."""
    offsets = np.array([pk.INFUSION_DAYS, 1, 3, 7, 14, 28, 42])
    times = []
    bounds = list(dose_times[1:]) + [horizon]
    for t0, t1 in zip(dose_times, bounds):
        for off in offsets:
            if t0 + off < t1 - 1.0:
                times.append(t0 + off)
        times.append(t1 - 1.0 / 24 if t1 < horizon else horizon)
    return np.array(times)


@dataclass
class Dataset:
    name: str
    cov: pd.DataFrame
    dose_mgkg: np.ndarray  # (n,)
    regimen: pk.Regimen
    obs_times: np.ndarray  # (n_obs,)
    eta_cl: np.ndarray
    eta_v1: np.ndarray
    kappa: np.ndarray | None  # (n, n_doses)
    ada_onset: np.ndarray | None  # (n,), inf if never
    true_conc: np.ndarray  # (n, n_obs) exact
    obs: np.ndarray  # (n, n_obs), NaN where BLQ
    split: np.ndarray  # "train" / "val" / "test"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.cov)

    def params(self) -> dict[str, np.ndarray]:
        return pk.individual_params(self.cov_dict(), self.eta_cl, self.eta_v1)

    def cov_dict(self) -> dict[str, np.ndarray]:
        return {c: self.cov[c].to_numpy() for c in COVARIATES}

    def truth(self, query_times: np.ndarray, idx: np.ndarray | None = None, regimen: pk.Regimen | None = None) -> np.ndarray:
        """Exact concentration at arbitrary times (optionally for a subset)."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        p = {k: v[idx] for k, v in self.params().items()}
        reg = regimen or self.regimen
        reg = pk.Regimen(reg.times, reg.amounts[idx])
        return pk.simulate(
            p,
            reg,
            query_times,
            kappa=None if self.kappa is None else self.kappa[idx][:, : reg.n_doses],
            ada_onset=None if self.ada_onset is None else self.ada_onset[idx],
            ada_half_rise=ADA_HALF_RISE,
        )

    def subset(self, idx: np.ndarray) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset(
            name=self.name,
            cov=self.cov.iloc[idx].reset_index(drop=True),
            dose_mgkg=self.dose_mgkg[idx],
            regimen=pk.Regimen(self.regimen.times, self.regimen.amounts[idx]),
            obs_times=self.obs_times,
            eta_cl=self.eta_cl[idx],
            eta_v1=self.eta_v1[idx],
            kappa=None if self.kappa is None else self.kappa[idx],
            ada_onset=None if self.ada_onset is None else self.ada_onset[idx],
            true_conc=self.true_conc[idx],
            obs=self.obs[idx],
            split=self.split[idx],
            meta=dict(self.meta),
        )

    def split_part(self, part: str) -> "Dataset":
        return self.subset(np.where(self.split == part)[0])


def make_dataset(
    kind: str,
    n: int,
    seed: int,
    maintenance_interval: float = 56.0,
    n_val: int | None = None,
    n_test: int | None = None,
) -> Dataset:
    """Simulate one of the three datasets ("clean", "doses", "ada")."""
    if kind not in {"clean", "doses", "ada"}:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    cov = sample_covariates(n, rng)
    covd = {c: cov[c].to_numpy() for c in COVARIATES}
    eta_cl = pk.OMEGA_CL * rng.standard_normal(n)
    eta_v1 = pk.OMEGA_V1 * rng.standard_normal(n)
    times = standard_dose_times_for(maintenance_interval)
    if kind == "clean":
        mgkg = np.full(n, 5.0)
        kappa = None
    else:
        mgkg = rng.choice(DOSE_ARMS, size=n)
        kappa = pk.OMEGA_IOV * rng.standard_normal((n, len(times)))
    amounts = mgkg[:, None] * covd["WT"][:, None] * np.ones((1, len(times)))
    regimen = pk.Regimen(times, amounts)
    params = pk.individual_params(covd, eta_cl, eta_v1)
    obs_t = observation_times(times)

    ada_onset = None
    meta: dict = {"maintenance_interval": maintenance_interval}
    if kind == "ada":
        # exposure before the third dose, without ADA, drives the onset risk
        c_pre3 = pk.simulate(params, regimen, np.array([42.0 - 1.0 / 24]), kappa=kappa)[:, 0]
        logit = ADA_B0 - ADA_B1 * np.log(c_pre3 / ADA_CREF)
        p_ada = 1.0 / (1.0 + np.exp(-logit))
        has = rng.random(n) < p_ada
        onset_t = rng.uniform(*ADA_WINDOW, size=n)
        ada_onset = np.where(has, onset_t, np.inf)
        meta["p_ada"] = p_ada

    true_conc = pk.simulate(params, regimen, obs_t, kappa=kappa, ada_onset=ada_onset, ada_half_rise=ADA_HALF_RISE)
    if kind == "clean":
        obs = true_conc.copy()
    else:
        obs = true_conc * (1.0 + pk.SIGMA_PROP * rng.standard_normal(true_conc.shape))
        obs = np.where(obs < pk.LLOQ, np.nan, obs)

    n_val = n_val if n_val is not None else max(n // 8, 4)
    n_test = n_test if n_test is not None else max(n // 4, 4)
    split = np.array(["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test)
    return Dataset(kind, cov, mgkg, regimen, obs_t, eta_cl, eta_v1, kappa, ada_onset, true_conc, obs, split, meta)


def standard_dose_times_for(maintenance_interval: float) -> np.ndarray:
    return pk.standard_dose_times(maintenance_interval, HORIZON)
