"""IHDP Bayesian causal model — quadratic terms for continuous covariates.
Adds x1^2..x6^2 to capture non-linear confounder-outcome relationships.
"""
import numpy as np
import pymc as pm


def _augment_with_quadratics(X: np.ndarray) -> np.ndarray:
    """Append squared continuous covariates (first 6 cols) to feature matrix."""
    X_quad = X[:, :6] ** 2
    return np.hstack([X, X_quad])


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model with quadratic terms for IHDP."""
    coords = train_data["coords"].copy()
    quad_names = [f"x{i}_sq" for i in range(1, 7)]
    coords["features_aug"] = list(coords["features"]) + quad_names

    # Center the outcome
    y_raw = train_data["outcome"]
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw))
    y_centered = (y_raw - y_mean) / y_std

    # Augment X with quadratic terms
    X_aug = _augment_with_quadratics(train_data["X"])

    with pm.Model(coords=coords) as model:
        y_mean_data = pm.Data("y_mean", y_mean)
        y_std_data = pm.Data("y_std", y_std)

        X = pm.Data("X_aug", X_aug, dims=("obs", "features_aug"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=1)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.5, dims="features_aug")
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
    X_aug = _augment_with_quadratics(new_data["X"])
    with model:
        pm.set_data(
            {"X_aug": X_aug, "treatment": new_data["treatment"]},
            coords=new_coords,
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    samples_centered = ppc.predictions["y"].values
    n_chains, n_draws, n_pts = samples_centered.shape
    samples_centered = samples_centered.reshape(n_chains * n_draws, n_pts)

    y_mean = float(model["y_mean"].get_value())
    y_std = float(model["y_std"].get_value())

    return samples_centered * y_std + y_mean


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via posterior predictive intervention.
    Returns ATE on the ORIGINAL scale.
    """
    n_obs = len(train_data["outcome"])
    train_coords = {"obs": np.arange(n_obs)}
    X_aug = _augment_with_quadratics(train_data["X"])

    with model:
        pm.set_data(
            {"X_aug": X_aug, "treatment": np.ones(n_obs)},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X_aug": X_aug, "treatment": np.zeros(n_obs)},
            coords=train_coords,
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.predictions["y"].values.reshape(-1, n_obs)

    y_std = float(model["y_std"].get_value())
    ate_samples_centered = (y1 - y0).mean(axis=1)
    ate_samples = ate_samples_centered * y_std

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
