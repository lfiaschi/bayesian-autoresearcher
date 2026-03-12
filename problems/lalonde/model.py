"""LaLonde Bayesian causal model — Gamma likelihood with log link.

Switches from Student-t to Gamma likelihood to better match the
non-negative, right-skewed earnings distribution. Since some
observations have re78=0, we shift the outcome by +1.0 so that all
values are strictly positive (required by the Gamma support).

The log-link GLM parameterises log(mu) as a linear function of
covariates, which naturally ensures mu > 0 without clipping.

Features:
- Gamma likelihood (non-negative, right-skewed)
- Log link for the mean
- Quadratic terms for continuous confounders
- Outcome shifted by +1 for Gamma support

Simplified from previous version: removed treatment x covariate
interactions (beta_tx, beta_tx_sq) and X_cont data container.
With only 722 observations and a log-link, these added noise and
widened the ATE HDI without clear benefit.
"""
import numpy as np
import pymc as pm

# Continuous confounder column names and their indices in the full X matrix
CONT_FEATURE_NAMES: list[str] = ["age", "education", "re75"]
CONT_FEATURE_INDICES: list[int] = [0, 1, 6]

# Shift constant added to outcome so that zeros become positive (Gamma requires > 0)
OUTCOME_SHIFT: float = 1.0


def _extract_continuous(X: np.ndarray) -> np.ndarray:
    """Extract continuous confounder columns from the full feature matrix.

    Args:
        X: Shape (n_obs, 7) — all confounders.

    Returns:
        Shape (n_obs, 3) — continuous confounders only.
    """
    return X[:, CONT_FEATURE_INDICES]


def build_model(train_data: dict) -> pm.Model:
    """Build a Gamma GLM with log link and quadratic terms.

    Args:
        train_data: Dict with keys X, treatment, outcome, coords.

    Returns:
        A compiled PyMC model.
    """
    coords = {
        **train_data["coords"],
        "cont_features": CONT_FEATURE_NAMES,
    }

    X_all = train_data["X"]
    X_sq = _extract_continuous(X_all) ** 2
    treatment = train_data["treatment"]
    outcome_shifted = train_data["outcome"] + OUTCOME_SHIFT

    with pm.Model(coords=coords) as model:
        # Data containers
        X_data = pm.Data("X", X_all, dims=("obs", "features"))
        X_sq_data = pm.Data("X_sq", X_sq, dims=("obs", "cont_features"))
        t_data = pm.Data("treatment", treatment, dims="obs")

        # --- Log-link linear predictor ---
        # Intercept on log scale: log(5000) ≈ 8.5
        alpha = pm.Normal("alpha", mu=np.log(5000), sigma=2)

        # Treatment effect on log scale
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=0.3)

        # Main effects (confounders already standardised)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.5, dims="features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=0.3, dims="cont_features")

        log_mu = (
            alpha
            + beta_t * t_data
            + pm.math.dot(X_data, beta_x)
            + pm.math.dot(X_sq_data, beta_sq)
        )
        mu = pm.math.exp(log_mu)

        # Gamma shape parameter (higher = less variance relative to mean)
        phi = pm.HalfNormal("phi", sigma=5)

        # Gamma likelihood: mean = mu, variance = mu^2 / phi
        # Parameterised as Gamma(alpha=phi, beta=phi/mu)
        pm.Gamma("y", alpha=phi, beta=phi / mu, observed=outcome_shifted, dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.

    Predictions are returned on the original dollar scale (shift subtracted).

    Args:
        idata: InferenceData from sampling.
        model: The compiled PyMC model.
        new_data: Dict with keys X, treatment, outcome, coords.

    Returns:
        Array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    X_sq = _extract_continuous(new_data["X"]) ** 2

    with model:
        pm.set_data(
            {
                "X": new_data["X"],
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
    # Subtract the shift to return predictions on the original dollar scale
    return samples.reshape(n_chains * n_draws, n_obs) - OUTCOME_SHIFT


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via two posterior predictive passes (do-calculus style).

    Because the model uses a log link, ATE is NOT a simple coefficient.
    Instead, we compute:
        ATE = E[Y | do(T=1), X] - E[Y | do(T=0), X]
    by averaging over the covariate distribution.

    Args:
        idata: InferenceData from sampling.
        model: The compiled PyMC model.
        train_data: Dict with keys X, treatment, outcome, coords.

    Returns:
        Dict with 'ate' (float) and 'ate_samples' (ndarray).
    """
    n_obs = len(train_data["outcome"])
    X_sq = _extract_continuous(train_data["X"]) ** 2

    # Pass 1: predict under treatment = 1 for everyone
    with model:
        pm.set_data(
            {
                "X": train_data["X"],
                "X_sq": X_sq,
                "treatment": np.ones(n_obs),
            },
            coords={"obs": np.arange(n_obs)},
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    # Pass 2: predict under treatment = 0 for everyone
    with model:
        pm.set_data(
            {
                "X": train_data["X"],
                "X_sq": X_sq,
                "treatment": np.zeros(n_obs),
            },
            coords={"obs": np.arange(n_obs)},
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values  # (chains, draws, obs)
    y0 = ppc_t0.predictions["y"].values

    # Average over observations for each sample to get ATE per sample
    n_chains, n_draws, _ = y1.shape
    y1_flat = y1.reshape(n_chains * n_draws, n_obs)
    y0_flat = y0.reshape(n_chains * n_draws, n_obs)

    # ITE for each (sample, obs), then average over obs
    ate_samples = (y1_flat - y0_flat).mean(axis=1)  # (n_samples,)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
