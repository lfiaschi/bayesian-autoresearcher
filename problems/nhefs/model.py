"""NHEFS Bayesian causal model — Student-t likelihood with heteroscedastic sigma.
Robust linear regression with treatment, confounders, quadratic AND cubic terms
for continuous confounders to capture richer non-linear relationships.
Uses Student-t likelihood to handle heavy tails in weight change data.
Sigma is modeled as a log-linear function of treatment and confounders
(linear + quadratic), allowing different patient subgroups to have different
variability.  A pre-computed propensity score (probability of treatment given
confounders) is included as an additional balancing covariate in both the mean
and sigma models.  Outcome is weight change in kg (wt82_71).
"""
import numpy as np
import pymc as pm
from sklearn.linear_model import LogisticRegression

CONTINUOUS_IDX = [2, 3, 4, 5, 8]  # age, school, smokeintensity, smokeyrs, wt71
CONTINUOUS_NAMES_QUAD = ["age_sq", "school_sq", "smokeintensity_sq", "smokeyrs_sq", "wt71_sq"]
CONTINUOUS_NAMES_CUBIC = ["age_cu", "school_cu", "smokeintensity_cu", "smokeyrs_cu", "wt71_cu"]


def _quad_features(X: np.ndarray) -> np.ndarray:
    """Compute squared terms for continuous confounders."""
    return X[:, CONTINUOUS_IDX] ** 2


def _cubic_features(X: np.ndarray) -> np.ndarray:
    """Compute cubic terms for continuous confounders."""
    return X[:, CONTINUOUS_IDX] ** 3


def _fit_propensity(X: np.ndarray, treatment: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression model to estimate propensity scores."""
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X, treatment)
    return lr


def _propensity_scores(lr: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Predict propensity scores (P(treatment=1|X)) from a fitted model."""
    return lr.predict_proba(X)[:, 1].astype(np.float64)


def build_model(train_data: dict) -> pm.Model:
    """Build a robust linear Bayesian model for NHEFS with Student-t likelihood,
    quadratic terms for continuous confounders, heteroscedastic sigma, and
    a propensity score covariate for improved confounding control."""
    coords = dict(train_data["coords"])
    coords["features_quad"] = CONTINUOUS_NAMES_QUAD
    coords["features_cubic"] = CONTINUOUS_NAMES_CUBIC

    X_raw = train_data["X"]
    X_quad_raw = _quad_features(X_raw)
    X_cubic_raw = _cubic_features(X_raw)

    # Fit propensity score model on training data
    ps_model = _fit_propensity(X_raw, train_data["treatment"])
    ps_raw = _propensity_scores(ps_model, X_raw)

    with pm.Model(coords=coords) as model:
        model.ps_model = ps_model  # store for predict / estimate_causal_effect

        X = pm.Data("X", X_raw, dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        X_quad = pm.Data("X_quad", X_quad_raw, dims=("obs", "features_quad"))
        X_cubic = pm.Data("X_cubic", X_cubic_raw, dims=("obs", "features_cubic"))
        propensity = pm.Data("propensity", ps_raw, dims="obs")

        # --- Mean model ---
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        beta_quad = pm.Normal("beta_quad", mu=0, sigma=1, dims="features_quad")
        beta_cubic = pm.Normal("beta_cubic", mu=0, sigma=0.5, dims="features_cubic")
        beta_ps = pm.Normal("beta_ps", mu=0, sigma=2)

        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_quad, beta_quad)
            + pm.math.dot(X_cubic, beta_cubic)
            + beta_ps * propensity
        )

        # --- Heteroscedastic sigma model ---
        sigma_alpha = pm.Normal("sigma_alpha", mu=2, sigma=1)
        sigma_beta_t = pm.Normal("sigma_beta_t", mu=0, sigma=0.5)
        sigma_beta_x = pm.Normal("sigma_beta_x", mu=0, sigma=0.5, dims="features")
        sigma_beta_quad = pm.Normal("sigma_beta_quad", mu=0, sigma=0.5, dims="features_quad")
        sigma_beta_ps = pm.Normal("sigma_beta_ps", mu=0, sigma=0.5)

        log_sigma = (
            sigma_alpha
            + sigma_beta_t * treatment
            + pm.math.dot(X, sigma_beta_x)
            + pm.math.dot(X_quad, sigma_beta_quad)
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
    X_quad_new = _quad_features(X_new)
    X_cubic_new = _cubic_features(X_new)
    ps_new = _propensity_scores(model.ps_model, X_new)

    with model:
        pm.set_data(
            {"X": X_new, "treatment": new_data["treatment"],
             "X_quad": X_quad_new, "X_cubic": X_cubic_new,
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
    X_quad_raw = _quad_features(X_raw)
    X_cubic_raw = _cubic_features(X_raw)
    ps_raw = _propensity_scores(model.ps_model, X_raw)

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.ones(n_obs),
             "X_quad": X_quad_raw, "X_cubic": X_cubic_raw,
             "propensity": ps_raw},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X": X_raw, "treatment": np.zeros(n_obs),
             "X_quad": X_quad_raw, "X_cubic": X_cubic_raw,
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
