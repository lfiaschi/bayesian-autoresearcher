"""LaLonde Bayesian causal model — quadratic + treatment interactions.

Extends baseline with:
- Quadratic terms for continuous confounders (age^2, education^2, re75^2)
- Treatment x continuous interactions (treatment * age, treatment * education, treatment * re75)
- Treatment x quadratic interactions (treatment * age^2, treatment * education^2, treatment * re75^2)

Priors tightened for dollar-scale outcome with standardized confounders.
"""
import numpy as np
import pymc as pm

# Continuous confounder column names and their indices in the full X matrix
CONT_FEATURE_NAMES: list[str] = ["age", "education", "re75"]
CONT_FEATURE_INDICES: list[int] = [0, 1, 6]


def _extract_continuous(X: np.ndarray) -> np.ndarray:
    """Extract continuous confounder columns from the full feature matrix.

    Args:
        X: Shape (n_obs, 7) — all confounders.

    Returns:
        Shape (n_obs, 3) — continuous confounders only.
    """
    return X[:, CONT_FEATURE_INDICES]


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model with quadratic and interaction terms."""
    coords = {
        **train_data["coords"],
        "cont_features": CONT_FEATURE_NAMES,
    }

    X_all = train_data["X"]
    X_cont = _extract_continuous(X_all)
    X_sq = X_cont ** 2
    treatment = train_data["treatment"]

    with pm.Model(coords=coords) as model:
        # Data containers
        X_data = pm.Data("X", X_all, dims=("obs", "features"))
        X_cont_data = pm.Data("X_cont", X_cont, dims=("obs", "cont_features"))
        X_sq_data = pm.Data("X_sq", X_sq, dims=("obs", "cont_features"))
        t_data = pm.Data("treatment", treatment, dims="obs")

        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=3000)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=2000)
        beta_x = pm.Normal("beta_x", mu=0, sigma=1500, dims="features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=1000, dims="cont_features")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=1000, dims="cont_features")
        beta_tx_sq = pm.Normal("beta_tx_sq", mu=0, sigma=500, dims="cont_features")
        sigma = pm.HalfNormal("sigma", sigma=3000)

        # Linear predictor
        mu = (
            alpha
            + beta_t * t_data
            + pm.math.dot(X_data, beta_x)
            + pm.math.dot(X_sq_data, beta_sq)
            + t_data * pm.math.dot(X_cont_data, beta_tx)
            + t_data * pm.math.dot(X_sq_data, beta_tx_sq)
        )

        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.

    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    X_cont = _extract_continuous(new_data["X"])
    X_sq = X_cont ** 2

    with model:
        pm.set_data(
            {
                "X": new_data["X"],
                "X_cont": X_cont,
                "X_sq": X_sq,
                "treatment": new_data["treatment"],
            },
            coords={"obs": np.arange(n_obs)},
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    samples = ppc.predictions["y"].values
    n_chains, n_draws, _ = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE analytically from posterior samples.

    ATE = beta_t + mean(X_cont) @ beta_tx + mean(X_sq) @ beta_tx_sq.

    Uses posterior parameter samples directly — no need for two posterior
    predictive passes. Falls back gracefully if interaction parameters are
    absent (e.g. when called with a baseline model's idata for comparison).
    """
    posterior = idata.posterior
    beta_t_samples = posterior["beta_treatment"].values  # (chains, draws)
    n_chains, n_draws = beta_t_samples.shape
    n_samples = n_chains * n_draws
    beta_t_flat = beta_t_samples.reshape(-1)  # (n_samples,)

    ate_samples = beta_t_flat.copy()

    # Add interaction contributions if present in the posterior
    if "beta_tx" in posterior:
        X_cont = _extract_continuous(train_data["X"])  # (n_obs, 3)
        mean_X_cont = X_cont.mean(axis=0)  # (3,)
        beta_tx_flat = posterior["beta_tx"].values.reshape(n_samples, -1)
        ate_samples = ate_samples + beta_tx_flat @ mean_X_cont

    if "beta_tx_sq" in posterior:
        X_cont = _extract_continuous(train_data["X"])
        X_sq = X_cont ** 2
        mean_X_sq = X_sq.mean(axis=0)  # (3,)
        beta_tx_sq_flat = posterior["beta_tx_sq"].values.reshape(n_samples, -1)
        ate_samples = ate_samples + beta_tx_sq_flat @ mean_X_sq

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
