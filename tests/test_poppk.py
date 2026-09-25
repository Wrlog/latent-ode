import numpy as np

from latent_ode_pk import pk, poppk
from latent_ode_pk.simulate import make_dataset


def test_batched_lm_fits_independent_problems():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(50, 2))
    x = np.linspace(0, 1, 8)

    def resid(th):
        return th[:, :1] + th[:, 1:] * x[None] - (target[:, :1] + target[:, 1:] * x[None])

    th, _ = poppk.batched_lm(resid, np.zeros((50, 2)))
    np.testing.assert_allclose(th, target, atol=1e-6)


def test_first_k_mask_skips_blq():
    obs = np.array([[1.0, np.nan, 2.0, 3.0, 4.0]])
    m = poppk.first_k_mask(obs, 2)
    np.testing.assert_array_equal(m[0], [True, False, True, False, False])
    t = np.arange(5.0)
    assert poppk.kth_level_time(obs, t, 2)[0] == 2.0
    assert poppk.kth_level_time(obs, t, 0)[0] == -np.inf


def _true_popmodel():
    return poppk.PopModel(
        use_cov=True,
        log_cl=np.log(pk.TV_CL),
        log_v1=np.log(pk.TV_V1),
        log_q=np.log(pk.TV_Q),
        log_v2=np.log(pk.TV_V2),
        beta_cl={"WT": 0.75, "ALB": -1.0, "INF": 0.10, "PLT": 0.0},
        beta_v1_wt=1.0,
        omega_cl=pk.OMEGA_CL,
        omega_v1=pk.OMEGA_V1,
        sigma=0.01,
    )


def test_popmodel_with_true_values_matches_simulator():
    ds = make_dataset("clean", 30, seed=1)
    m = _true_popmodel()
    typ = m.typical(ds.cov_dict())
    ref = pk.typical_params(ds.cov_dict())
    for k in ref:
        np.testing.assert_allclose(typ[k], ref[k], rtol=1e-10)


def test_map_recovers_random_effects_from_clean_data():
    ds = make_dataset("clean", 30, seed=2)
    theta = poppk.forecast(_true_popmodel(), ds, 8)
    # forecast from 8 noise-free levels reproduces the truth
    np.testing.assert_allclose(theta, ds.true_conc, rtol=5e-3)


def test_its_recovers_population_values():
    ds = make_dataset("clean", 300, seed=3)
    m = poppk.fit_population(ds, use_cov=True, n_iter=4)
    assert abs(np.exp(m.log_cl) / pk.TV_CL - 1) < 0.1
    assert abs(np.exp(m.log_v1) / pk.TV_V1 - 1) < 0.1
    # albumin has a small spread and correlates with INF, so its SE is about 0.2
    assert -1.5 < m.beta_cl["ALB"] < -0.5
    assert abs(m.beta_cl["WT"] - 0.75) < 0.2
    assert abs(m.omega_cl - pk.OMEGA_CL) < 0.08


def test_true_model_reference_is_exact_on_clean_data():
    ds = make_dataset("clean", 20, seed=4)
    pred = poppk.true_model_forecast(ds, 8)
    np.testing.assert_allclose(pred, ds.true_conc, rtol=2e-2)
    # with no levels it is the population prediction
    pop = poppk.true_model_forecast(ds, 0)
    np.testing.assert_allclose(pop, pk.simulate(pk.typical_params(ds.cov_dict()), ds.regimen, ds.obs_times))


def test_random_walk_model_tracks_rising_clearance():
    ds = make_dataset("ada", 240, seed=5)
    tr = ds.split_part("train")
    base = poppk.fit_population(tr, use_cov=True, n_iter=3)
    tv = poppk.add_random_walk(base, tr, n_iter=2)
    assert tv.omega_rw > 0.05
