"""NHEFS Bayesian causal model — quadratic + interactions.

Extends the baseline linear model with:
- Quadratic terms for continuous confounders (age, school, smokeintensity,
  smokeyrs, wt71) to capture non-linear confounder-outcome relationships.
- Treatment-confounder interactions for continuous confounders to capture
  heterogeneous treatment effects.

Outcome is weight change in kg (wt82_71), prior scales adjusted accordingly.
"""
import numpy as np
import pymc as pm

# Indices of continuous confounders within the confounder column order:
# ["sex", "race", "age", "school", "smokeintensity", "smokeyrs",
#  "exercise", "active", "wt71"]
CONTINUOUS_INDICES: list[int] = [2, 3, 4, 5, 8]

CONTINUOUS_NAMES: list[str] = [
    "age", "school", "smokeintensity", "smokeyrs", "wt71",
]


def build_model(train_data: dict) -> pm.Model:
    """Build a Bayesian linear model with quadratic and interaction terms.

    Parameters
    ----------
    train_data : dict
        Keys: X (n_obs, n_features), treatment (n_obs,), outcome (n_obs,),
        coords (dict with 'obs' and 'features').

    Returns
    -------
    pm.Model
    """
    coords = {
        **train_data["coords"],
        "continuous": CONTINUOUS_NAMES,
    }

    with pm.Model(coords=coords) as model:
        # --- Data containers ---
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        # --- Derived quantities (pytensor ops inside the model) ---
        X_cont = X[:, CONTINUOUS_INDICES]          # (n_obs, 5)
        X_sq = X_cont ** 2                         # (n_obs, 5)
        X_interact = treatment[:, None] * X_cont   # (n_obs, 5)

        # --- Priors ---
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        beta_x2 = pm.Normal("beta_x2", mu=0, sigma=1, dims="continuous")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=1, dims="continuous")
        sigma = pm.HalfNormal("sigma", sigma=10)

        # --- Linear predictor ---
        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_sq, beta_x2)
            + pm.math.dot(X_interact, beta_tx)
        )

        # --- Likelihood ---
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted inference data with posterior samples.
    model : pm.Model
        The PyMC model (must match idata).
    new_data : dict
        Same structure as train_data.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_obs).
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
    """Estimate ATE via posterior predictive intervention.

    Sets treatment to 1 for all observations, then to 0, and computes
    the difference in predicted outcomes.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted inference data.
    model : pm.Model
        The PyMC model.
    train_data : dict
        Training data dict.

    Returns
    -------
    dict
        Keys: ate (float), ate_samples (np.ndarray of per-draw ATEs).
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
    ate_samples = (y1 - y0).mean(axis=1)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
