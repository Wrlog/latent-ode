"""Training loop for the latent ODE."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import Batch, make_batch
from .model import LatentODE, ModelConfig
from .simulate import Dataset


@dataclass
class TrainConfig:
    epochs: int = 250
    batch_size: int = 50
    lr: float = 3e-3
    k_max: int = 34  # training draws k uniformly from 0..k_max levels
    kl_warmup_epochs: int = 25
    grad_clip: float = 1.0
    patience: int = 30  # early stopping on validation error
    lr_patience: int = 10
    schedule: str = "plateau"  # or "cosine" (decays to 3% of lr over `epochs`)
    augment: bool = True  # dose-rescaling augmentation
    augment_prob: float = 0.5
    augment_range: tuple[float, float] = (0.5, 2.0)
    val_ks: tuple[int, ...] = (0, 2, 6, 12)
    seed: int = 7
    max_minutes: float = 10.0  # safety net only; normally early stopping ends training


def dataset_batch(ds: Dataset, y_mean: float, y_std: float) -> Batch:
    return make_batch(ds.cov, ds.regimen, ds.obs_times, obs=ds.obs, y_mean=y_mean, y_std=y_std)


def log_stats(ds: Dataset) -> tuple[float, float]:
    v = np.log(ds.obs[np.isfinite(ds.obs) & (ds.obs > 0)])
    return float(v.mean()), float(v.std())


@torch.no_grad()
def val_error(model: LatentODE, batch: Batch, ks) -> float:
    model.eval()
    errs = []
    for k in ks:
        kk = torch.full((len(batch),), k, dtype=torch.long)
        logc, _ = model(batch, kk)
        m = batch.y_mask
        errs.append(float((((logc - batch.y) ** 2) * m).sum() / m.sum()))
    return float(np.mean(errs))


def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    model_cfg: ModelConfig | None = None,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[LatentODE, list[dict]]:
    model_cfg = model_cfg or ModelConfig()
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    y_mean, y_std = log_stats(train_ds)
    model = LatentODE(model_cfg, y_mean, y_std)
    tr = dataset_batch(train_ds, y_mean, y_std)
    va = dataset_batch(val_ds, y_mean, y_std)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.03 * cfg.lr)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.lr_patience, min_lr=1e-4)
    best, best_state, bad = np.inf, None, 0
    history = []
    t_start = time.time()
    n = len(tr)
    lo, hi = np.log(cfg.augment_range[0]), np.log(cfg.augment_range[1])
    for epoch in range(cfg.epochs):
        model.train()
        beta = min(1.0, (epoch + 1) / max(cfg.kl_warmup_epochs, 1))
        perm = rng.permutation(n)
        tot, tot_obs = 0.0, 0.0
        for start in range(0, n, cfg.batch_size):
            idx = torch.as_tensor(perm[start : start + cfg.batch_size])
            b = tr.index(idx)
            if cfg.augment:
                f = np.where(rng.random(len(idx)) < cfg.augment_prob, np.exp(rng.uniform(lo, hi, len(idx))), 1.0)
                b = b.scale_doses(torch.as_tensor(f, dtype=torch.float32), y_std)
            k = torch.as_tensor(rng.integers(0, cfg.k_max + 1, len(idx)))
            logc, kl = model(b, k, sample=True)
            sigma = model.sigma(torch.minimum(k, b.lev_len))
            nll = 0.5 * ((b.y - logc) / sigma) ** 2 + torch.log(sigma)
            n_obs = b.y_mask.sum()
            # loss per observed sample; KL on the same scale
            loss = ((nll * b.y_mask).sum() + beta * kl.sum()) / n_obs
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach()) * float(n_obs)
            tot_obs += float(n_obs)
        v = val_error(model, va, cfg.val_ks)
        sched.step() if cfg.schedule == "cosine" else sched.step(v)
        history.append({"epoch": epoch, "train_loss": tot / tot_obs, "val_mse": v, "beta": beta, "lr": opt.param_groups[0]["lr"]})
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  loss {tot / tot_obs:7.4f}  val mse {v:.4f}  lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t_start:5.0f}s", flush=True)
        # only start early stopping once the KL weight is fully on
        if epoch + 1 >= cfg.kl_warmup_epochs:
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch}", flush=True)
                    break
        if (time.time() - t_start) / 60 > cfg.max_minutes:
            if verbose:
                print(f"  time budget reached at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_model(model: LatentODE, path) -> None:
    torch.save({"config": model.config_dict(), "state": model.state_dict()}, path)


def load_model(path) -> LatentODE:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = LatentODE(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return model
