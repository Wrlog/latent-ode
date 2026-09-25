"""Properties of the architecture that hold for any weights."""

import numpy as np
import pytest
import torch

from latent_ode_pk import pk
from latent_ode_pk.data import make_batch
from latent_ode_pk.evaluate import network_forecast
from latent_ode_pk.model import LatentODE, ModelConfig
from latent_ode_pk.simulate import make_dataset


@pytest.fixture(scope="module")
def ds():
    return make_dataset("doses", 20, seed=11, n_val=0, n_test=20)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LatentODE(ModelConfig()).eval()


def test_predictions_are_positive_even_with_wild_weights(ds):
    torch.manual_seed(1)
    m = LatentODE(ModelConfig()).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(5.0)
    pred = network_forecast(m, ds, 3, times=np.arange(0.5, 182, 0.5))
    assert np.all(pred > 0)


def test_deterministic_and_batch_independent(model, ds):
    a = network_forecast(model, ds, 4)
    b = network_forecast(model, ds, 4)
    np.testing.assert_array_equal(a, b)
    one = network_forecast(model, ds.subset([3]), 4)
    np.testing.assert_allclose(one[0], a[3], rtol=1e-5)


def test_padding_leaves_state_unchanged(model, ds):
    """Two patients with different query grids in one padded batch give the same
    answer as each on its own."""
    sub = ds.subset([0, 1])
    q_short = ds.obs_times[:10]
    q_long = ds.obs_times
    mixed = make_batch(sub.cov, sub.regimen, [q_short, q_long], y_mean=3.5, y_std=1.2)
    alone = make_batch(sub.subset([0]).cov, pk.Regimen(sub.regimen.times, sub.regimen.amounts[:1]), q_short, y_mean=3.5, y_std=1.2)
    p_mixed = model.predict(mixed, 0).numpy()
    p_alone = model.predict(alone, 0).numpy()
    np.testing.assert_allclose(p_mixed[0, :10], p_alone[0], rtol=1e-5)


def test_gru_is_causal(model, ds):
    """The posterior after k levels cannot see level k+1 or later."""
    k = 3
    base = network_forecast(model, ds, k)
    changed = ds.subset(np.arange(ds.n))
    changed.obs = ds.obs.copy()
    later = np.cumsum(np.isfinite(changed.obs), 1) > k
    changed.obs[later] = changed.obs[later] * 3.0
    np.testing.assert_allclose(network_forecast(model, changed, k), base, rtol=1e-5)
    # but the levels it may read do matter
    changed.obs = ds.obs * 2.0
    assert np.max(np.abs(network_forecast(model, changed, k) / base - 1)) > 1e-4


def test_k_zero_is_the_prior(model, ds):
    b = make_batch(ds.cov, ds.regimen, ds.obs_times, obs=ds.obs, y_mean=3.5, y_std=1.2)
    _, (mp, lsp), (mq, lsq) = model.latent(b, torch.zeros(len(b), dtype=torch.long))
    torch.testing.assert_close(mp, mq)
    torch.testing.assert_close(lsp, lsq)


def test_rk4_on_a_linear_system(model):
    m = LatentODE(ModelConfig(rk4_steps=4))
    m._f = lambda x, uc: -2.0 * x  # dx/ds = dt * f with f = -2x
    x0 = torch.ones(3, 5)
    dt = torch.tensor([0.1, 0.5, 0.0])
    x1 = m._rk4(x0, dt, None)
    expected = torch.exp(-2.0 * dt)[:, None] * x0
    torch.testing.assert_close(x1, expected, atol=1e-4, rtol=1e-4)


def test_extra_grid_points_barely_move_an_untrained_model(model, ds):
    fine_t = np.unique(np.concatenate([ds.obs_times, np.arange(0.5, 182, 0.5)]))
    fine = network_forecast(model, ds, 0, times=fine_t)
    coarse = network_forecast(model, ds, 0)
    pos = np.searchsorted(fine_t, ds.obs_times)
    assert np.max(np.abs(np.log(fine[:, pos] / coarse))) < 1e-3


def test_dose_scaling_augmentation(ds):
    b = make_batch(ds.cov, ds.regimen, ds.obs_times, obs=ds.obs, y_mean=3.5, y_std=1.2)
    f = torch.full((len(b),), 2.0)
    s = b.scale_doses(f, 1.2)
    torch.testing.assert_close(s.dose, 2 * b.dose)
    torch.testing.assert_close(s.y, b.y + np.log(2.0) * b.y_mask)
    torch.testing.assert_close(s.lev[..., 3], b.lev[..., 3] + np.log(2.0))


def test_peak_counter_on_the_exact_solution():
    from latent_ode_pk.evaluate import count_peaks, dense_times

    ds = make_dataset("doses", 30, seed=12)
    t = dense_times(ds.regimen)
    peaks = count_peaks(t, ds.truth(t), ds.regimen.times)
    assert np.all(peaks == 1)
    # a profile with a second bump in one interval is caught
    bumpy = ds.truth(t)[:1].copy()
    j = np.searchsorted(t, 60.0)
    bumpy[0, j : j + 8] *= 1.5
    assert count_peaks(t, bumpy, ds.regimen.times)[0, 2] == 2
