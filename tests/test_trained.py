"""Structural checks on the trained networks in models/.

These use the checkpoints written by scripts/run_all.py and the same simulated
test patients. They are skipped if the checkpoints are missing.
"""

from pathlib import Path

import numpy as np
import pytest

from latent_ode_pk import pk
from latent_ode_pk.evaluate import count_peaks, dense_times, forecast_mask, network_forecast, score
from latent_ode_pk.simulate import benchmark_dataset
from latent_ode_pk.train import load_model

MODELS = Path(__file__).resolve().parents[1] / "models"
KINDS = ["clean", "doses", "ada"]


def _load(kind):
    path = MODELS / f"latent_ode_{kind}.pt"
    if not path.exists():
        pytest.skip(f"{path.name} not found; run scripts/run_all.py first")
    return load_model(path)


@pytest.fixture(scope="module")
def test_sets():
    return {k: benchmark_dataset(k).split_part("test") for k in KINDS}


@pytest.mark.parametrize("kind", KINDS)
def test_positive_and_finite(kind, test_sets):
    net, ds = _load(kind), test_sets[kind]
    pred = network_forecast(net, ds, 4, times=dense_times(ds.regimen))
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic(kind, test_sets):
    net, ds = _load(kind), test_sets[kind]
    np.testing.assert_array_equal(network_forecast(net, ds, 2), network_forecast(net, ds, 2))


@pytest.mark.parametrize("kind", KINDS)
def test_one_peak_per_dose(kind, test_sets):
    net, ds = _load(kind), test_sets[kind]
    t = dense_times(ds.regimen)
    peaks = count_peaks(t, network_forecast(net, ds, 0, times=t), ds.regimen.times)
    assert np.mean(peaks == 1) >= 0.95


CLEAN_XFAIL = pytest.param(
    "clean",
    marks=pytest.mark.xfail(
        strict=True,
        reason="the clean dataset has a single dose level; dose rescaling augmentation alone "
        "leaves C(2D)/C(D) about 10% off (see README)",
    ),
)


@pytest.mark.parametrize("kind", [CLEAN_XFAIL, "doses", "ada"])
def test_dose_proportionality(kind, test_sets):
    net, ds = _load(kind), test_sets[kind]
    base = network_forecast(net, ds, 0)
    double = network_forecast(net, ds, 0, regimen=pk.Regimen(ds.regimen.times, 2 * ds.regimen.amounts))
    dev = np.abs(np.log(double / base) - np.log(2))
    assert np.median(dev) < np.log(1.05)


@pytest.mark.parametrize("kind", KINDS)
def test_grid_invariance(kind, test_sets):
    net, ds = _load(kind), test_sets[kind]
    extra = np.unique(np.concatenate([ds.obs_times, np.arange(0.5, 182, 0.5)]))
    fine = network_forecast(net, ds, 4, times=extra)[:, np.searchsorted(extra, ds.obs_times)]
    coarse = network_forecast(net, ds, 4)
    assert np.quantile(np.abs(np.log(fine / coarse)), 0.99) < 0.02


def test_unseen_dosing_interval(test_sets):
    net = _load("doses")
    seen = test_sets["doses"]
    new = benchmark_dataset("unseen")
    for k in (0, 4):
        e_seen = score(network_forecast(net, seen, k), seen.true_conc, forecast_mask(seen, k))["mfe"]
        e_new = score(network_forecast(net, new, k), new.true_conc, forecast_mask(new, k))["mfe"]
        assert np.log(e_new) <= 1.25 * np.log(e_seen)
