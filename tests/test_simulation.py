import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from latent_ode_pk import pk
from latent_ode_pk.simulate import ADA_HALF_RISE, DOSE_ARMS, make_dataset, observation_times


def _numerical(ds, i, times):
    p = {k: v[i] for k, v in ds.params().items()}
    starts = ds.regimen.times

    def rhs(t, a):
        occ = max(np.searchsorted(starts, t, side="right") - 1, 0)
        cl = p["CL"]
        if ds.kappa is not None:
            cl = cl * np.exp(ds.kappa[i, occ])
        if ds.ada_onset is not None:
            cl = cl * pk.ada_multiplier(t, ds.ada_onset[i], ADA_HALF_RISE)
        rate = sum(ds.regimen.amounts[i, j] / pk.INFUSION_DAYS for j, t0 in enumerate(starts) if t0 <= t < t0 + pk.INFUSION_DAYS)
        k12, k21 = p["Q"] / p["V1"], p["Q"] / p["V2"]
        return [rate - (cl / p["V1"] + k12) * a[0] + k21 * a[1], k12 * a[0] - k21 * a[1]]

    sol = solve_ivp(rhs, [0, times.max()], [0.0, 0.0], t_eval=times, rtol=1e-10, atol=1e-10, max_step=0.02)
    return sol.y[0] / p["V1"]


def test_exact_solver_matches_numerical_integration():
    ds = make_dataset("ada", 40, seed=1)
    ids = [0, 1] + list(np.where(np.isfinite(ds.ada_onset))[0][:2])
    for i in ids:
        num = _numerical(ds, i, ds.obs_times)
        # exact for piecewise-constant parameters; the ADA rise uses 6-hour pieces
        assert np.max(np.abs(num / ds.true_conc[i] - 1)) < 5e-4


def test_covariates_enter_the_parameters():
    ds = make_dataset("doses", 200, seed=2)
    cov = ds.cov
    cl = ds.params()["CL"]
    expected = (
        0.30
        * (cov.WT / 70) ** 0.75
        * (cov.ALB / 4.0) ** -1.0
        * (cov.INF / 5) ** 0.10
        * np.exp(ds.eta_cl)
    )
    np.testing.assert_allclose(cl, expected.to_numpy(), rtol=1e-12)
    # weight-based dosing: amount in mg is mg/kg times weight
    np.testing.assert_allclose(ds.regimen.amounts[:, 0], ds.dose_mgkg * cov.WT.to_numpy())


def test_covariates_change_the_simulated_concentrations():
    """Same patient, one covariate changed at a time, random effects at zero."""
    base = {"WT": 60.0, "ALB": 4.0, "INF": 5.0, "PLT": 300.0}
    times = pk.standard_dose_times()
    obs_t = observation_times(times)
    trough = int(np.argmin(np.abs(obs_t - (98 - 1 / 24))))

    def run(**changes):
        c = {k: np.array([v]) for k, v in {**base, **changes}.items()}
        amounts = 5.0 * c["WT"][:, None] * np.ones((1, len(times)))
        return pk.simulate(pk.typical_params(c), pk.Regimen(times, amounts), obs_t)[0]

    ref = run()
    assert run(ALB=3.0)[trough] < ref[trough]  # low albumin, faster clearance
    assert run(INF=40.0)[trough] < ref[trough]  # more inflammation, faster clearance
    assert run(WT=90.0)[trough] > ref[trough]  # mg/kg dosing over-compensates allometric CL
    np.testing.assert_allclose(run(PLT=550.0), ref)  # platelets do not affect PK


def test_covariate_effects_are_visible_in_the_dataset():
    """Regressing simulated log troughs on the covariates recovers the right signs."""
    ds = make_dataset("clean", 600, seed=3)
    j = int(np.argmin(np.abs(ds.obs_times - (98 - 1 / 24))))
    y = np.log(ds.obs[:, j])
    X = np.column_stack([np.ones(ds.n)] + [np.log(ds.cov[c]) for c in ["WT", "ALB", "INF", "PLT"]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert b[1] > 0.1  # weight (via mg/kg dose and V)
    assert b[2] > 0.5  # albumin
    assert b[3] < 0  # inflammation marker
    assert abs(b[4]) < 0.15  # platelets: none
    # INF and ALB come from a shared score
    assert np.corrcoef(np.log(ds.cov.INF), ds.cov.ALB)[0, 1] < -0.3


def test_ada_raises_clearance_only_after_onset():
    ds = make_dataset("ada", 300, seed=4)
    pos = np.where(np.isfinite(ds.ada_onset))[0]
    assert 0.2 < len(pos) / ds.n < 0.55
    no_ada = pk.simulate(ds.params(), ds.regimen, ds.obs_times, kappa=ds.kappa)
    for i in pos[:20]:
        before = ds.obs_times <= ds.ada_onset[i]
        np.testing.assert_allclose(ds.true_conc[i, before], no_ada[i, before], rtol=1e-12)
        after = ds.obs_times > ds.ada_onset[i] + 7
        if after.any():
            assert np.all(ds.true_conc[i, after] < no_ada[i, after])
    neg = np.where(~np.isfinite(ds.ada_onset))[0]
    np.testing.assert_allclose(ds.true_conc[neg], no_ada[neg], rtol=1e-12)
    # onset risk is higher when exposure is low
    p = ds.meta["p_ada"]
    assert p[ds.dose_mgkg == DOSE_ARMS.min()].mean() > p[ds.dose_mgkg == DOSE_ARMS.max()].mean()


def test_assay_error_and_blq():
    clean = make_dataset("clean", 50, seed=5)
    np.testing.assert_array_equal(clean.obs, clean.true_conc)
    noisy = make_dataset("doses", 300, seed=5)
    blq = np.isnan(noisy.obs)
    assert np.all(noisy.obs[~blq] >= pk.LLOQ)
    r = np.log(noisy.obs[~blq] / noisy.true_conc[~blq])
    assert 0.12 < r.std() < 0.18


def test_dense_schedule():
    t = observation_times(pk.standard_dose_times())
    assert np.all(np.diff(t) > 0)
    assert len(t) == 34
    assert isinstance(make_dataset("clean", 10, seed=0).cov, pd.DataFrame)
