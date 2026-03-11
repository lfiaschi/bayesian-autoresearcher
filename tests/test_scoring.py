"""Tests for scoring functions."""
import numpy as np
import pytest
from scoring import (
    compute_crps,
    compute_mae,
    compute_rmse,
    check_convergence,
    compute_ate_bias,
    compute_elpd,
)


def test_crps_perfect_prediction():
    observed = np.array([1.0, 2.0, 3.0])
    samples = np.column_stack([
        np.random.normal(loc=observed, scale=0.01, size=len(observed))
        for _ in range(1000)
    ]).T
    crps = compute_crps(observed, samples)
    assert crps < 0.05


def test_crps_bad_prediction():
    observed = np.array([1.0, 2.0, 3.0])
    samples = np.random.normal(loc=100.0, scale=1.0, size=(1000, 3))
    crps = compute_crps(observed, samples)
    assert crps > 10.0


def test_crps_returns_float():
    observed = np.array([1.0, 2.0])
    samples = np.random.normal(size=(100, 2))
    assert isinstance(compute_crps(observed, samples), float)


def test_mae():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 2.2, 2.7])
    assert abs(compute_mae(observed, predicted) - 0.2) < 1e-10


def test_rmse():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(observed, predicted) == 0.0


def test_ate_bias():
    assert abs(compute_ate_bias(3.5, 4.0) - 0.5) < 1e-10


def test_ate_bias_none_true():
    assert compute_ate_bias(3.5, None) is None


def test_convergence_good(good_idata):
    result = check_convergence(good_idata)
    assert result["ok"] is True
    assert result["r_hat_max"] < 1.01
    assert result["ess_min"] > 400
    assert result["divergences"] == 0


def test_convergence_bad_rhat(bad_rhat_idata):
    result = check_convergence(bad_rhat_idata)
    assert result["ok"] is False


def test_elpd_returns_none_without_log_likelihood():
    import arviz as az
    rng = np.random.default_rng(42)
    data = {"mu": rng.normal(size=(4, 1000))}
    idata = az.from_dict(posterior=data)
    assert compute_elpd(idata) is None


@pytest.fixture
def good_idata():
    import arviz as az
    rng = np.random.default_rng(42)
    data = {"mu": rng.normal(size=(4, 1000)), "sigma": rng.exponential(size=(4, 1000))}
    return az.from_dict(posterior=data)


@pytest.fixture
def bad_rhat_idata():
    import arviz as az
    rng = np.random.default_rng(42)
    data = {
        "mu": np.array([
            rng.normal(loc=0, size=1000),
            rng.normal(loc=10, size=1000),
            rng.normal(loc=20, size=1000),
            rng.normal(loc=30, size=1000),
        ]),
    }
    return az.from_dict(posterior=data)
