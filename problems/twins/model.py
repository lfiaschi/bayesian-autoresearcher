"""Twins Bayesian causal model — tight priors + quadratic confounders.
Linear regression with treatment, confounders, and quadratic terms for
continuous confounders (mager8, mrace, gestat10).
Outcome is binary mortality (0/1) with ~3% prevalence.
Priors tightened to reflect low base rate and small effects.
"""
import numpy as np
import pymc as pm

# Indices of continuous confounders in the confounder matrix X
# CONFOUNDER_COLS = ["pldel", "birattnd", "mager8", "ormoth", "mrace",
#                    "meduc6", "dmar", "adequacy", "gestat10", "csex"]
N_CONTINUOUS_INDICES = [2, 4, 8]


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model for Twins with tight priors and quadratic terms."""
    coords = dict(train_data["coords"])
    coords["cont_features"] = ["mager8", "mrace", "gestat10"]

    X_raw = train_data["X"]
    X_cont = X_raw[:, N_CONTINUOUS_INDICES]
    X_sq = X_cont ** 2

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_raw, dims=("obs", "features"))
        X_cont_data = pm.Data("X_cont", X_cont, dims=("obs", "cont_features"))
        X_sq_data = pm.Data("X_sq", X_sq, dims=("obs", "cont_features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=0.5)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=0.3)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.3, dims="features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=0.2, dims="cont_features")
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_sq_data, beta_sq)
        )
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    new_coords = {"obs": np.arange(n_obs)}

    X_new = new_data["X"]
    X_cont_new = X_new[:, N_CONTINUOUS_INDICES]
    X_sq_new = X_cont_new ** 2

    with model:
        pm.set_data(
            {
                "X": X_new,
                "X_cont": X_cont_new,
                "X_sq": X_sq_new,
                "treatment": new_data["treatment"],
            },
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
    Quadratic terms don't interact with treatment, so ATE is still just beta_treatment.
    """
    beta_t_samples = idata.posterior["beta_treatment"].values.flatten()
    ate = float(beta_t_samples.mean())

    return {"ate": ate, "ate_samples": beta_t_samples}
