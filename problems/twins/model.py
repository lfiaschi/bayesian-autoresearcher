"""Twins Bayesian causal model — tight priors.
Simple linear regression with treatment and confounders.
Outcome is binary mortality (0/1) with ~3% prevalence.
Priors tightened to reflect low base rate and small effects.
"""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model for Twins with tight priors."""
    coords = train_data["coords"]

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=0.5)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=0.3)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.3, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    new_coords = {"obs": np.arange(n_obs)}
    with model:
        pm.set_data(
            {"X": new_data["X"], "treatment": new_data["treatment"]},
            coords=new_coords,
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    samples = ppc.predictions["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via structural parameter approach.
    ATE = beta_treatment (coefficient on binary treatment).
    """
    beta_t_samples = idata.posterior["beta_treatment"].values.flatten()
    ate = float(beta_t_samples.mean())

    return {"ate": ate, "ate_samples": beta_t_samples}
