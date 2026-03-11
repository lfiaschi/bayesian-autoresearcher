"""NHEFS Bayesian causal model — Student-t likelihood.
Robust linear regression with treatment and confounders.
Uses Student-t likelihood to handle heavy tails in weight change data.
Outcome is weight change in kg (wt82_71), prior scales adjusted accordingly.
"""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    """Build a robust linear Bayesian model for NHEFS with Student-t likelihood."""
    coords = train_data["coords"]

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=10)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata: object, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    with model:
        pm.set_data(
            {"X": new_data["X"], "treatment": new_data["treatment"]},
            coords={"obs": np.arange(n_obs)},
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    samples = ppc.predictions["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata: object, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via posterior predictive intervention."""
    n_obs = len(train_data["outcome"])
    train_coords = {"obs": np.arange(n_obs)}

    with model:
        pm.set_data(
            {"X": train_data["X"], "treatment": np.ones(n_obs)},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X": train_data["X"], "treatment": np.zeros(n_obs)},
            coords=train_coords,
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.predictions["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
