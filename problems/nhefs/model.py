"""NHEFS Bayesian causal model — Student-t likelihood with quadratic terms.
Robust linear regression with treatment, confounders, and quadratic terms
for continuous confounders to capture non-linear relationships.
Uses Student-t likelihood to handle heavy tails in weight change data.
Outcome is weight change in kg (wt82_71), prior scales adjusted accordingly.
"""
import numpy as np
import pymc as pm

CONTINUOUS_IDX = [2, 3, 4, 5, 8]  # age, school, smokeintensity, smokeyrs, wt71
CONTINUOUS_NAMES = ["age_sq", "school_sq", "smokeintensity_sq", "smokeyrs_sq", "wt71_sq"]


def _quad_features(X: np.ndarray) -> np.ndarray:
    """Compute squared terms for continuous confounders."""
    return X[:, CONTINUOUS_IDX] ** 2


def build_model(train_data: dict) -> pm.Model:
    """Build a robust linear Bayesian model for NHEFS with Student-t likelihood
    and quadratic terms for continuous confounders."""
    coords = dict(train_data["coords"])
    coords["features_quad"] = CONTINUOUS_NAMES

    X_raw = train_data["X"]
    X_quad_raw = _quad_features(X_raw)

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_raw, dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        X_quad = pm.Data("X_quad", X_quad_raw, dims=("obs", "features_quad"))

        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        beta_quad = pm.Normal("beta_quad", mu=0, sigma=1, dims="features_quad")
        sigma = pm.HalfNormal("sigma", sigma=10)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_quad, beta_quad)
        )
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata: object, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    X_new = new_data["X"]
    X_quad_new = _quad_features(X_new)

    with model:
        pm.set_data(
            {"X": X_new, "treatment": new_data["treatment"], "X_quad": X_quad_new},
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
    X_raw = train_data["X"]
    X_quad_raw = _quad_features(X_raw)

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.ones(n_obs), "X_quad": X_quad_raw},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.zeros(n_obs), "X_quad": X_quad_raw},
            coords=train_coords,
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.predictions["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
