"""IHDP Bayesian causal model — treatment-confounder interactions.
Heterogeneous treatment effects (CATE) via interaction terms between
treatment and all 25 confounders.
"""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model with treatment-confounder interactions."""
    coords = train_data["coords"]

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        # Priors — regularizing
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=3)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=1, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=3)

        # Linear predictor with treatment-confounder interactions
        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + treatment * pm.math.dot(X, beta_tx)
        )
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
    n_chains, n_draws, _ = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE using posterior samples of structural parameters.

    Under this model, the individual treatment effect for observation i is:
        tau_i = beta_t + X_i @ beta_tx
    The ATE is the average of tau_i over all training observations.
    This avoids two expensive posterior-predictive passes.

    Falls back gracefully if idata comes from an older model without beta_tx
    (e.g., during runner's comparison with best.nc from a previous model).
    """
    # Extract posterior samples: shape (n_chains, n_draws)
    beta_t_samples = idata.posterior["beta_treatment"].values
    n_chains, n_draws = beta_t_samples.shape
    beta_t_flat = beta_t_samples.reshape(-1)  # (n_samples,)

    # Check if interaction terms exist in this idata
    if "beta_tx" in idata.posterior:
        beta_tx_samples = idata.posterior["beta_tx"].values
        n_features = beta_tx_samples.shape[2]
        beta_tx_flat = beta_tx_samples.reshape(-1, n_features)  # (n_samples, n_features)

        X_train = train_data["X"]  # (n_obs, n_features)

        # Individual treatment effects: tau_i = beta_t + X_i @ beta_tx
        # X_train @ beta_tx^T gives (n_obs, n_samples), then mean over obs axis
        interaction_term = X_train @ beta_tx_flat.T  # (n_obs, n_samples)
        # ATE per sample = beta_t + mean_i(X_i @ beta_tx)
        ate_samples = beta_t_flat + interaction_term.mean(axis=0)
    else:
        # Old model without interactions: ATE = beta_t
        ate_samples = beta_t_flat

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
