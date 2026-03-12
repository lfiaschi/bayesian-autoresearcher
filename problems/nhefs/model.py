"""NHEFS Bayesian causal model — linear confounders only.

Student-t model with:
- Linear confounder effects (no quadratic terms).
- Student-t likelihood for robustness against outlier weight changes
  (range roughly -40 to +50 kg).
- Confounder-dependent heteroscedasticity: log(sigma) depends on
  smokeintensity and wt71, allowing heavier smokers and individuals with
  different baseline weights to have more variable weight change.

Outcome is weight change in kg (wt82_71), prior scales adjusted accordingly.
"""
import numpy as np
import pymc as pm

# Indices of confounders used for heteroscedastic sigma:
# smokeintensity (index 4) and wt71 (index 8)
SIGMA_INDICES: list[int] = [4, 8]

SIGMA_FEATURE_NAMES: list[str] = ["smokeintensity", "wt71"]


def build_model(train_data: dict) -> pm.Model:
    """Build a Bayesian model with linear confounders and Student-t likelihood.

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
        "sigma_features": SIGMA_FEATURE_NAMES,
    }

    with pm.Model(coords=coords) as model:
        # --- Data containers ---
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        # --- Derived quantities for heteroscedastic sigma ---
        X_sigma = X[:, SIGMA_INDICES]              # (n_obs, 2)

        # --- Priors ---
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # --- Confounder-dependent heteroscedastic sigma ---
        log_sigma_intercept = pm.Normal("log_sigma_intercept", mu=2, sigma=0.5)
        log_sigma_coeffs = pm.Normal(
            "log_sigma_coeffs", mu=0, sigma=0.3, dims="sigma_features"
        )
        log_sigma = log_sigma_intercept + pm.math.dot(X_sigma, log_sigma_coeffs)
        sigma = pm.math.exp(log_sigma)

        # --- Linear predictor ---
        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)

        # --- Likelihood (Student-t for robustness against outliers) ---
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

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
