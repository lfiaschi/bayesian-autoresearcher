"""IHDP Bayesian causal model — centered outcome with regularizing priors.
Centers Y to help the sampler converge, then un-centers predictions.
"""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model for IHDP with centered outcome."""
    coords = train_data["coords"]

    # Center the outcome
    y_raw = train_data["outcome"]
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw))
    y_centered = (y_raw - y_mean) / y_std

    with pm.Model(coords=coords) as model:
        # Store centering constants as pm.Data for access in predict/estimate
        y_mean_data = pm.Data("y_mean", y_mean)
        y_std_data = pm.Data("y_std", y_std)

        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        # Regularizing priors (outcome is centered, so unit-scale priors work)
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=1)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.5, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_centered, dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs) on the ORIGINAL scale.
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

    # Get centered predictions and un-center
    samples_centered = ppc.predictions["y"].values
    n_chains, n_draws, n_pts = samples_centered.shape
    samples_centered = samples_centered.reshape(n_chains * n_draws, n_pts)

    # Retrieve centering constants from the model's data
    y_mean = float(model["y_mean"].get_value())
    y_std = float(model["y_std"].get_value())

    return samples_centered * y_std + y_mean


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via posterior predictive intervention.
    Returns ATE on the ORIGINAL scale.
    """
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

    # ATE on centered scale; multiply by y_std to get original scale
    # (the y_mean cancels in the difference)
    y_std = float(model["y_std"].get_value())
    ate_samples_centered = (y1 - y0).mean(axis=1)
    ate_samples = ate_samples_centered * y_std

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
