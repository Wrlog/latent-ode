"""Latent neural ODE for concentration-time data with doses.

Structure:

- covariates -> context vector c
- a per-patient latent u (a learned analogue of the random effects) with prior
  p(u | c) and, optionally, a posterior q(u | c, first k levels) from a causal GRU
- the dynamic latent state x starts at x0(c, u) and follows dx/dt = f(x, u, c)
- each dose makes x jump: x -> x + g(x, u, c, dose)
- log concentration = h(x, u), so predictions are always positive

Each step between grid points is solved in normalized time s in [0, 1] with
dx/ds = dt * f, using a fixed number of RK4 sub-steps. Zero-length steps
(padding) leave the state unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .data import Batch


@dataclass
class ModelConfig:
    n_cov: int = 4
    d_ctx: int = 8
    d_u: int = 4
    d_x: int = 6
    hidden: int = 64
    gru_hidden: int = 32
    rk4_steps: int = 1
    use_gru: bool = True
    init_post_logsd: float = -3.0
    innovations: bool = False
    sigma_by_k: bool = False


class LatentODE(nn.Module):
    def __init__(self, cfg: ModelConfig, y_mean: float = 3.5, y_std: float = 1.2):
        super().__init__()
        self.cfg = cfg
        h, dx, du, dc = cfg.hidden, cfg.d_x, cfg.d_u, cfg.d_ctx
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

        self.ctx = nn.Sequential(nn.Linear(cfg.n_cov, 32), nn.Tanh(), nn.Linear(32, dc), nn.Tanh())
        self.prior = nn.Linear(dc, 2 * du)
        # GRU input: 4 level features + prior prediction + innovation (level minus prior prediction)
        self.gru = nn.GRU(6 if cfg.innovations else 4, cfg.gru_hidden, batch_first=True)
        self.post = nn.Sequential(nn.Linear(cfg.gru_hidden + dc, 32), nn.Tanh(), nn.Linear(32, 2 * du))
        self.x0 = nn.Sequential(nn.Linear(dc + du, 32), nn.Tanh(), nn.Linear(32, dx))
        # f(x, u, c): the (u, c) part of the first layer is computed once per solve
        self.f_x = nn.Linear(dx, h, bias=False)
        self.f_uc = nn.Linear(du + dc, h)
        self.f_out = nn.Sequential(nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, dx))
        self.jump = nn.Sequential(nn.Linear(dx + du + dc + 2, h), nn.Tanh(), nn.Linear(h, dx))
        self.head = nn.Sequential(nn.Linear(dx + du, 32), nn.Tanh(), nn.Linear(32, 1))
        # residual SD on the log scale; optionally larger when few levels were given
        self.log_sigma = nn.Parameter(torch.tensor(-1.5))
        self.log_sigma_extra = nn.Parameter(torch.tensor(0.0))
        # start with a narrow posterior so the decoder learns to use u before the
        # KL term widens it (helps against posterior collapse)
        with torch.no_grad():
            self.post[-1].bias[du:] = cfg.init_post_logsd
        # small initial dynamics
        nn.init.zeros_(self.f_out[-1].bias)
        self.f_out[-1].weight.data.mul_(0.1)

    # ----- latent distribution -------------------------------------------------
    def latent(self, batch: Batch, k: torch.Tensor | None):
        """Prior and posterior over u. k (B,) is how many levels each patient's
        posterior may read; k=0 (or no GRU) gives the prior."""
        c = self.ctx(batch.cov)
        mp, lsp = self.prior(c).chunk(2, dim=-1)
        lsp = lsp.clamp(-4, 2)
        if k is None or not self.cfg.use_gru or int(k.max()) == 0:
            return c, (mp, lsp), (mp, lsp)
        k = torch.minimum(k, batch.lev_len)
        # what the prior mean predicts at each level time, like the prediction
        # step of a filter; the GRU then reads the innovations
        feats = batch.lev
        if self.cfg.innovations:
            with torch.no_grad():
                st = self.solve(batch, mp, c)
                prior_lev = (self.decode(st, mp, batch.lev_idx) - self.y_mean) / self.y_std
            feats = torch.cat([feats, prior_lev[..., None], (batch.lev[..., :1] - prior_lev[..., None])], -1)
        hs, _ = self.gru(feats)  # causal: h_j only sees levels 1..j
        idx = (k - 1).clamp(min=0)
        hk = hs[torch.arange(len(k)), idx]
        dm, lsq = self.post(torch.cat([hk, c], -1)).chunk(2, dim=-1)
        lsq = lsq.clamp(-5, 2)
        has = (k > 0).float()[:, None]
        mq = mp + has * dm
        lsq = has * lsq + (1 - has) * lsp
        return c, (mp, lsp), (mq, lsq)

    # ----- dynamics ------------------------------------------------------------
    def _f(self, x, uc_proj):
        return self.f_out(self.f_x(x) + uc_proj)

    def _rk4(self, x, dt, uc_proj):
        n = self.cfg.rk4_steps
        hstep = (dt / n)[:, None]
        for _ in range(n):
            k1 = self._f(x, uc_proj)
            k2 = self._f(x + 0.5 * hstep * k1, uc_proj)
            k3 = self._f(x + 0.5 * hstep * k2, uc_proj)
            k4 = self._f(x + hstep * k3, uc_proj)
            x = x + hstep * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def solve(self, batch: Batch, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Latent states at every grid point, before that point's dose. (B, S, d_x)"""
        uc = torch.cat([u, c], -1)
        uc_proj = self.f_uc(uc)
        x = self.x0(uc)
        states = []
        n_steps = batch.dt.shape[1]
        for s in range(n_steps):
            dt = batch.dt[:, s]
            if s > 0 and bool((dt > 0).any()):
                x = self._rk4(x, dt, uc_proj)
            states.append(x)
            d = batch.dose[:, s]
            if bool((d > 0).any()):
                dd = d.clamp(min=1e-6)[:, None]
                g = self.jump(torch.cat([x, uc, dd, torch.log(dd)], -1))
                x = x + (d > 0).float()[:, None] * g
        return torch.stack(states, 1)

    def decode(self, states, u, q_idx):
        xq = torch.gather(states, 1, q_idx[..., None].expand(-1, -1, states.shape[-1]))
        uq = u[:, None, :].expand(-1, xq.shape[1], -1)
        return self.y_mean + self.y_std * self.head(torch.cat([xq, uq], -1)).squeeze(-1)

    def forward(self, batch: Batch, k: torch.Tensor | None = None, sample: bool = False):
        """Returns (log concentration at the query points, KL per patient)."""
        c, (mp, lsp), (mq, lsq) = self.latent(batch, k)
        u = mq + torch.exp(lsq) * torch.randn_like(mq) if sample else mq
        states = self.solve(batch, u, c)
        logc = self.decode(states, u, batch.q_idx)
        kl = (lsp - lsq + (torch.exp(2 * lsq) + (mq - mp) ** 2) / (2 * torch.exp(2 * lsp)) - 0.5).sum(-1)
        return logc, kl

    @torch.no_grad()
    def predict(self, batch: Batch, k: int | torch.Tensor = 0) -> torch.Tensor:
        """Deterministic prediction (posterior mean of u) in mg/L."""
        self.eval()
        if isinstance(k, int):
            k = torch.full((len(batch),), k, dtype=torch.long)
        logc, _ = self.forward(batch, k, sample=False)
        return torch.exp(logc)

    def sigma(self, k: torch.Tensor) -> torch.Tensor:
        """Residual SD per patient (B, 1) for a posterior that read k levels."""
        ls = self.log_sigma.expand(len(k))
        if self.cfg.sigma_by_k:
            ls = ls + torch.nn.functional.softplus(self.log_sigma_extra) * torch.exp(-k.float() / 4.0)
        return torch.exp(ls).clamp(min=0.01)[:, None]

    def config_dict(self):
        return asdict(self.cfg)
