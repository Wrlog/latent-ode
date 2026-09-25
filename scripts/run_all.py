"""Simulate the datasets, train the latent ODE on each, fit the popPK comparators
and write every result table used by the figures and the README.

    python scripts/run_all.py            # full run (about 10 minutes on a laptop CPU)
    python scripts/run_all.py --quick    # small smoke test for CI
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_ode_pk import evaluate as ev  # noqa: E402
from latent_ode_pk import poppk  # noqa: E402
from latent_ode_pk.model import ModelConfig  # noqa: E402
from latent_ode_pk.simulate import SEEDS, SIZES, UNSEEN_INTERVAL, make_dataset  # noqa: E402
from latent_ode_pk.train import TrainConfig, load_model, save_model, train_model  # noqa: E402

QUICK_TRAIN = dict(epochs=4, kl_warmup_epochs=2, patience=5)


def sizes(quick: bool):
    if quick:
        return 60, 12, 24
    return SIZES["n"], SIZES["n_val"], SIZES["n_test"]


def train_job(kind: str, quick: bool, models_dir: Path, out: Path):
    """Simulate one dataset and train its network (run in a worker process)."""
    torch.set_num_threads(1)
    n, n_val, n_test = sizes(quick)
    ds = make_dataset(kind, n, SEEDS[kind], n_val=n_val, n_test=n_test)
    t = time.time()
    net, hist = train_model(ds.split_part("train"), ds.split_part("val"), ModelConfig(), TrainConfig(**(QUICK_TRAIN if quick else {})), verbose=False)
    save_model(net, models_dir / f"latent_ode_{kind}.pt")
    pd.DataFrame(hist).to_csv(out / f"training_{kind}.csv", index=False)
    return kind, round(time.time() - t, 1), len(hist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny datasets and a few epochs (CI smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="results folder (default results/, or results/quick with --quick)")
    ap.add_argument("--models", type=Path, default=None, help="where to save the trained networks (default models/)")
    ap.add_argument("--serial", action="store_true", help="train the three networks one after another instead of in parallel")
    args = ap.parse_args()

    torch.set_num_threads(1)
    out = args.out or (ROOT / "results" / "quick" if args.quick else ROOT / "results")
    models_dir = args.models or (out / "models" if args.quick else ROOT / "models")
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    n, n_val, n_test = sizes(args.quick)
    ks = (0, 2, 8) if args.quick else ev.K_GRID
    kinds = ("clean", "doses", "ada")

    t0 = time.time()
    info: dict = {"quick": args.quick, "python": platform.python_version(), "torch": torch.__version__, "sizes": {}}
    fe_tables, checks, pop_params = [], [], {}
    nets, pops, datasets = {}, {}, {}

    # 1. train the three networks (in parallel: each is single-threaded)
    print("training the latent ODEs ...", flush=True)
    if args.serial:
        jobs = [train_job(k, args.quick, models_dir, out) for k in kinds]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs = list(pool.map(train_job, kinds, [args.quick] * 3, [models_dir] * 3, [out] * 3))
    for kind, secs, n_ep in jobs:
        info.setdefault("train_seconds", {})[kind] = secs
        info.setdefault("epochs", {})[kind] = n_ep
        print(f"  {kind}: {n_ep} epochs in {secs:.0f} s", flush=True)
    info["train_wall_seconds"] = round(time.time() - t0, 1)

    unseen = make_dataset("doses", n_test, SEEDS["unseen"], maintenance_interval=UNSEEN_INTERVAL, n_val=0, n_test=n_test)
    for kind in kinds:
        print(f"\n=== dataset: {kind}", flush=True)
        ds = make_dataset(kind, n, SEEDS[kind], n_val=n_val, n_test=n_test)
        tr, va, te = ds.split_part("train"), ds.split_part("val"), ds.split_part("test")
        datasets[kind] = te
        info["sizes"][kind] = {"train": tr.n, "val": va.n, "test": te.n, "blq_share": float(np.isnan(ds.obs).mean())}
        if kind == "ada":
            info["sizes"][kind]["ada_share"] = float(np.isfinite(ds.ada_onset).mean())

        net = load_model(models_dir / f"latent_ode_{kind}.pt")
        nets[kind] = net

        t = time.time()
        pop = {
            "popPK, no covariates": poppk.fit_population(tr, use_cov=False),
            "popPK with covariates": poppk.fit_population(tr, use_cov=True),
        }
        if kind == "ada":
            pop["popPK, time-varying CL"] = poppk.add_random_walk(pop["popPK with covariates"], tr)
        info.setdefault("poppk_seconds", {})[kind] = round(time.time() - t, 1)
        pops[kind] = pop
        pop_params[kind] = {name: m.summary() for name, m in pop.items()}

        t = time.time()
        methods = ev.standard_methods(te, net, pop)
        fe = ev.fold_error_table(te, methods, ks)
        fe_tables.append(fe)
        print(fe.pivot(index="k", columns="method", values="mfe").round(3).to_string(), flush=True)
        checks.append(ev.structural_checks(net, te, unseen if kind == "doses" else None))
        if kind == "ada":
            ada = ev.ada_status_table(te, methods, ks)
            ada.to_csv(out / "ada_forecast_by_status.csv", index=False)
        info.setdefault("eval_seconds", {})[kind] = round(time.time() - t, 1)

    fe_all = pd.concat(fe_tables, ignore_index=True)
    fe_all.to_csv(out / "fold_error.csv", index=False)
    chk = pd.concat(checks, ignore_index=True)
    chk.to_csv(out / "structural_checks.csv", index=False)
    print("\n" + chk.to_string(index=False), flush=True)

    # covariate recovery on the multi-dose dataset
    cr = ev.covariate_recovery(nets["doses"], pops["doses"]["popPK with covariates"], datasets["doses"])
    cr.to_csv(out / "covariate_recovery.csv", index=False)

    # example profiles from the ADA dataset: one negative, one early and one late onset
    te = datasets["ada"]
    onset = te.ada_onset
    neg = np.where(~np.isfinite(onset))[0]
    early = np.where(onset < 40)[0]
    late = np.where(np.isfinite(onset) & (onset > 90))[0]
    ids = [int(g[0]) for g in (neg, early, late) if len(g)]
    prof = ev.example_profiles(nets["ada"], pops["ada"]["popPK with covariates"], te, ids)
    prof["ada_onset"] = prof["patient"].map({i: float(onset[i]) for i in ids})
    prof.to_csv(out / "example_profiles.csv", index=False)

    info["total_seconds"] = round(time.time() - t0, 1)
    (out / "poppk_estimates.json").write_text(json.dumps(pop_params, indent=2, default=float))
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\ndone in {info['total_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
