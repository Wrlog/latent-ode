"""Turn simulated patients into padded tensors for the latent ODE.

Each patient gets an event grid: the union of dose times, query times and
filler points so that no gap is longer than `max_gap` days. The solver steps
from one grid point to the next in normalized time, so padding a short
patient with zero-length steps leaves its state unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from .pk import Regimen

TIME_SCALE = 28.0  # days per unit of model time
DOSE_SCALE = 350.0  # mg (5 mg/kg for a 70 kg patient)
MAX_GAP = 4.0  # days


def cov_features(cov: pd.DataFrame) -> np.ndarray:
    """Standardized covariates on roughly unit scale (fixed transform, no fitting)."""
    return np.column_stack(
        [
            np.log(cov["WT"].to_numpy() / 55.0) / 0.3,
            (cov["ALB"].to_numpy() - 3.9) / 0.4,
            np.log(cov["INF"].to_numpy() / 5.0) / 0.85,
            np.log(cov["PLT"].to_numpy() / 300.0) / 0.25,
        ]
    )


@dataclass
class Batch:
    cov: torch.Tensor  # (B, n_cov)
    dt: torch.Tensor  # (B, S) normalized step lengths, 0 for padding
    dose: torch.Tensor  # (B, S) dose amount / DOSE_SCALE applied after reaching the point
    q_idx: torch.Tensor  # (B, Q) long, index into the grid
    q_time: torch.Tensor  # (B, Q) days
    y: torch.Tensor  # (B, Q) log observed concentration (0 where missing)
    y_mask: torch.Tensor  # (B, Q) 1 where observed and quantifiable
    lev: torch.Tensor  # (B, L, 4) GRU features of quantifiable levels in time order
    lev_time: torch.Tensor  # (B, L) days
    lev_len: torch.Tensor  # (B,) number of quantifiable levels
    lev_idx: torch.Tensor  # (B, L) long, grid index of each level

    def __len__(self) -> int:
        return self.cov.shape[0]

    def index(self, idx) -> "Batch":
        return Batch(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})

    def scale_doses(self, factor: torch.Tensor, y_std: float) -> "Batch":
        """Multiply every dose by `factor` (B,) and every concentration with it.

        Only valid if the system is linear in dose (see README)."""
        lf = torch.log(factor)
        lev = self.lev.clone()
        lev[..., 0] = lev[..., 0] + lf[:, None] / y_std
        lev[..., 3] = lev[..., 3] + lf[:, None]
        return replace(self, dose=self.dose * factor[:, None], y=self.y + lf[:, None] * self.y_mask, lev=lev)


def _grid(dose_times: np.ndarray, query_times: np.ndarray, max_gap: float) -> np.ndarray:
    pts = np.unique(np.concatenate([[0.0], dose_times, query_times]))
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n_fill = int(np.ceil((b - a) / max_gap - 1e-9)) - 1
        for j in range(1, n_fill + 1):
            out.append(a + j * (b - a) / (n_fill + 1))
        out.append(b)
    return np.array(out)


def make_batch(
    cov: pd.DataFrame,
    regimen: Regimen,
    query_times: np.ndarray | list[np.ndarray],
    obs: np.ndarray | None = None,
    level_times: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    max_gap: float = MAX_GAP,
) -> Batch:
    """Build a batch.

    query_times: shared array (Q,) or a list of per-patient arrays.
    obs: (n, Q) observed concentrations at the query times (NaN = missing/BLQ), used as targets.
    level_times, levels: (n, L) measured levels that the GRU may read (NaN = BLQ, dropped).
        Defaults to (query_times, obs).
    """
    n = len(cov)
    if isinstance(query_times, np.ndarray) and query_times.ndim == 1:
        q_list = [query_times] * n
    else:
        q_list = [np.asarray(q, float) for q in query_times]
    if levels is None and obs is not None:
        level_times, levels = np.broadcast_to(q_list[0], obs.shape), obs

    if levels is not None:
        extra = [np.asarray(level_times[i], float)[np.isfinite(np.asarray(levels[i], float))] for i in range(n)]
    else:
        extra = [np.zeros(0)] * n
    grids = [_grid(regimen.times, np.concatenate([q, e]), max_gap) for q, e in zip(q_list, extra)]
    s_max = max(len(g) for g in grids)
    q_max = max(len(q) for q in q_list)
    dt = np.zeros((n, s_max))
    dose = np.zeros((n, s_max))
    q_idx = np.zeros((n, q_max), dtype=np.int64)
    q_time = np.zeros((n, q_max))
    y = np.zeros((n, q_max))
    y_mask = np.zeros((n, q_max))
    for i, (g, q) in enumerate(zip(grids, q_list)):
        dt[i, 1 : len(g)] = np.diff(g) / TIME_SCALE
        pos = np.searchsorted(g, regimen.times)
        dose[i, pos] = regimen.amounts[i] / DOSE_SCALE
        qi = np.searchsorted(g, q)
        q_idx[i, : len(q)] = qi
        q_time[i, : len(q)] = q
        if obs is not None:
            o = obs[i, : len(q)]
            ok = np.isfinite(o) & (o > 0)
            y[i, : len(q)][ok] = np.log(o[ok])
            y_mask[i, : len(q)][ok] = 1.0

    l_list = []
    if levels is not None:
        for i in range(n):
            lt = np.asarray(level_times[i], float)
            lv = np.asarray(levels[i], float)
            ok = np.isfinite(lv) & (lv > 0)
            lt, lv = lt[ok], lv[ok]
            order = np.argsort(lt, kind="stable")
            l_list.append((lt[order], lv[order], i))
    l_max = max([len(x[0]) for x in l_list], default=0)
    l_max = max(l_max, 1)
    lev = np.zeros((n, l_max, 4))
    lev_time = np.zeros((n, l_max))
    lev_len = np.zeros(n, dtype=np.int64)
    lev_idx = np.zeros((n, l_max), dtype=np.int64)
    for lt, lv, i in l_list:
        m = len(lt)
        lev_len[i] = m
        if m == 0:
            continue
        di = np.searchsorted(regimen.times, lt, side="right") - 1
        di = np.clip(di, 0, None)
        last_amt = regimen.amounts[i][di] / DOSE_SCALE
        lev[i, :m, 0] = (np.log(lv) - y_mean) / y_std
        lev[i, :m, 1] = lt / TIME_SCALE / 4.0
        lev[i, :m, 2] = (lt - regimen.times[di]) / TIME_SCALE
        lev[i, :m, 3] = np.log(last_amt)
        lev_time[i, :m] = lt
        lev_idx[i, :m] = np.searchsorted(grids[i], lt)

    f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    return Batch(
        cov=f(cov_features(cov)),
        dt=f(dt),
        dose=f(dose),
        q_idx=torch.as_tensor(q_idx),
        q_time=f(q_time),
        y=f(y),
        y_mask=f(y_mask),
        lev=f(lev),
        lev_time=f(lev_time),
        lev_len=torch.as_tensor(lev_len),
        lev_idx=torch.as_tensor(lev_idx),
    )
