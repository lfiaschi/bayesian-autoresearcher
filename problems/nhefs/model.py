"""NHEFS Bayesian causal model — Student-t likelihood with heteroscedastic sigma.
Robust linear regression with treatment, confounders, and piecewise linear
basis functions for wt71 and smokeintensity to capture non-linearity in the
two most important continuous confounders.
Uses Student-t likelihood to handle heavy tails in weight change data.
Sigma is modeled as a log-linear function of treatment and linear confounders.
A pre-computed propensity score (probability of treatment given confounders)
is included as an additional balancing covariate in both the mean and sigma
models.  Outcome is weight change in kg (wt82_71).
"""
import numpy as np
import pymc as pm
from sklearn.linear_model import LogisticRegression

# Indices into the confounder matrix X (standardized)
# 0=sex, 1=race, 2=age, 3=school, 4=smokeintensity, 5=smokeyrs,
# 6=exercise, 7=active, 8=wt71
SPLINE_IDX_SMOKEINTENSITY = 4
SPLINE_IDX_WT71 = 8

# Knots placed at quantile-like positions of standardized values
SPLINE_KNOTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

SPLINE_FEATURE_NAMES = [
    f"smokeint_pw{i}" for i in range(len(SPLINE_KNOTS))
] + [
    f"wt71_pw{i}" for i in range(len(SPLINE_KNOTS))
]


def _piecewise_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Compute piecewise linear (truncated power) basis functions.

    For each knot k, computes max(x - k, 0).  This creates a hinge at each
    knot, allowing the model to learn a piecewise linear function.

    Parameters
    ----------
    x : array of shape (n,)
        Input variable (standardized).
    knots : array of shape (K,)
        Knot locations.

    Returns
    -------
    basis : array of shape (n, K)
        Basis matrix.
    """
    return np.maximum(x[:, None] - knots[None, :], 0.0)


def _compute_spline_features(X: np.ndarray) -> np.ndarray:
    """Build the full spline basis matrix from the confounder matrix.

    Returns array of shape (n, 2*K) where K = len(SPLINE_KNOTS).
    First K columns are for smokeintensity, last K for wt71.
    """
    basis_smoke = _piecewise_basis(X[:, SPLINE_IDX_SMOKEINTENSITY], SPLINE_KNOTS)
    basis_wt71 = _piecewise_basis(X[:, SPLINE_IDX_WT71], SPLINE_KNOTS)
    return np.hstack([basis_smoke, basis_wt71])


def _fit_propensity(X: np.ndarray, treatment: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression model to estimate propensity scores."""
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X, treatment)
    return lr


def _propensity_scores(lr: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Predict propensity scores (P(treatment=1|X)) from a fitted model."""
    return lr.predict_proba(X)[:, 1].astype(np.float64)


def build_model(train_data: dict) -> pm.Model:
    """Build a robust Bayesian model for NHEFS with Student-t likelihood,
    piecewise linear basis for smokeintensity and wt71, heteroscedastic
    sigma, and a propensity score covariate."""
    coords = dict(train_data["coords"])
    coords["spline_features"] = SPLINE_FEATURE_NAMES

    X_raw = train_data["X"]
    X_spline_raw = _compute_spline_features(X_raw)

    # Fit propensity score model on training data
    ps_model = _fit_propensity(X_raw, train_data["treatment"])
    ps_raw = _propensity_scores(ps_model, X_raw)

    with pm.Model(coords=coords) as model:
        model.ps_model = ps_model  # store for predict / estimate_causal_effect

        X = pm.Data("X", X_raw, dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        X_spline = pm.Data("X_spline", X_spline_raw, dims=("obs", "spline_features"))
        propensity = pm.Data("propensity", ps_raw, dims="obs")

        # --- Mean model ---
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        beta_spline = pm.Normal("beta_spline", mu=0, sigma=1, dims="spline_features")
        beta_ps = pm.Normal("beta_ps", mu=0, sigma=2)

        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_spline, beta_spline)
            + beta_ps * propensity
        )

        # --- Heteroscedastic sigma model (linear confounders only) ---
        sigma_alpha = pm.Normal("sigma_alpha", mu=2, sigma=1)
        sigma_beta_t = pm.Normal("sigma_beta_t", mu=0, sigma=0.5)
        sigma_beta_x = pm.Normal("sigma_beta_x", mu=0, sigma=0.5, dims="features")
        sigma_beta_ps = pm.Normal("sigma_beta_ps", mu=0, sigma=0.5)

        log_sigma = (
            sigma_alpha
            + sigma_beta_t * treatment
            + pm.math.dot(X, sigma_beta_x)
            + sigma_beta_ps * propensity
        )
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma), dims="obs")

        # --- Degrees of freedom ---
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata: object, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    X_new = new_data["X"]
    X_spline_new = _compute_spline_features(X_new)
    ps_new = _propensity_scores(model.ps_model, X_new)

    with model:
        pm.set_data(
            {"X": X_new, "treatment": new_data["treatment"],
             "X_spline": X_spline_new,
             "propensity": ps_new},
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
    X_spline_raw = _compute_spline_features(X_raw)
    ps_raw = _propensity_scores(model.ps_model, X_raw)

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.ones(n_obs),
             "X_spline": X_spline_raw,
             "propensity": ps_raw},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.zeros(n_obs),
             "X_spline": X_spline_raw,
             "propensity": ps_raw},
            coords=train_coords,
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.predictions["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
