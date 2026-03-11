"""Shared scoring functions for Bayesian model evaluation."""
from typing import Optional
import arviz as az
import numpy as np
from properscoring import crps_ensemble


def compute_crps(observed: np.ndarray, samples: np.ndarray) -> float:
    """Compute mean CRPS across observations.
    Args:
        observed: Shape (n_obs,). Actual values.
        samples: Shape (n_samples, n_obs). Posterior predictive draws.
    Returns:
        Mean CRPS (lower is better).
    """
    scores = np.array([
        crps_ensemble(observed[i], samples[:, i])
        for i in range(len(observed))
    ])
    return float(np.mean(scores))


def compute_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(observed - predicted)))


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def compute_ate_bias(estimated_ate: float, true_ate: Optional[float]) -> Optional[float]:
    """Absolute bias of ATE estimate vs ground truth. Returns None if true_ate is not available."""
    if true_ate is None:
        return None
    return float(abs(estimated_ate - true_ate))


def check_convergence(idata: az.InferenceData) -> dict:
    """Check MCMC convergence diagnostics.
    Returns dict with: ok, r_hat_max, ess_min, divergences.
    """
    summary = az.summary(idata)
    r_hat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())
    ess_min = min(ess_bulk_min, ess_tail_min)

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())

    total_samples = 0
    if hasattr(idata, "posterior"):
        shape = list(idata.posterior.data_vars.values())[0].shape
        total_samples = shape[0] * shape[1]

    div_rate = divergences / max(total_samples, 1)
    ok = (r_hat_max < 1.01) and (ess_min > 400) and (div_rate < 0.001)

    return {
        "ok": ok,
        "r_hat_max": r_hat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }


def compute_elpd(idata: az.InferenceData) -> Optional[float]:
    """Compute ELPD via LOO-CV. Returns None if log_likelihood not available."""
    if not hasattr(idata, "log_likelihood"):
        return None
    loo = az.loo(idata, pointwise=False)
    return float(loo.elpd_loo)
