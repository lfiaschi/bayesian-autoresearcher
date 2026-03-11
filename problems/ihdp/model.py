"""IHDP Bayesian causal model — doubly robust with propensity score.
Treatment-confounder interactions for heterogeneous effects (CATE), plus
squared terms for the 6 continuous confounders (x1-x6) to capture
non-linear confounder-outcome relationships, and an externally estimated
propensity score (logistic regression) as an additional covariate for
doubly robust confounding adjustment.
"""
import numpy as np
import pymc as pm
from sklearn.linear_model import LogisticRegression

N_CONTINUOUS = 6  # x1-x6 are continuous (already standardized by prepare.py)


def _estimate_propensity_score(X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Estimate propensity score via logistic regression.

    Returns raw probability P(T=1|X).
    """
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, treatment)
    return lr.predict_proba(X)[:, 1]


def build_model(train_data: dict) -> pm.Model:
    """Build a doubly robust model with interactions, quadratic terms, and propensity score."""
    coords = dict(train_data["coords"])
    coords["cont_features"] = [f"x{i}" for i in range(1, N_CONTINUOUS + 1)]

    X_raw = train_data["X"]
    X_squared = X_raw[:, :N_CONTINUOUS] ** 2
    ps = _estimate_propensity_score(X_raw, train_data["treatment"])

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_raw, dims=("obs", "features"))
        X_sq = pm.Data("X_sq", X_squared, dims=("obs", "cont_features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        ps_data = pm.Data("ps", ps, dims="obs")

        # Priors — tightened for standardized confounders (598 training obs)
        alpha = pm.Normal("alpha", mu=0, sigma=3)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=2)
        beta_x = pm.Normal("beta_x", mu=0, sigma=1, dims="features")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=0.7, dims="features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=0.5, dims="cont_features")
        beta_tps = pm.Normal("beta_tps", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=2)

        # Linear predictor with interactions + quadratic terms + PS-treatment interaction
        # PS * treatment captures residual confounding that varies with propensity;
        # main confounding is already handled by X and X^2.
        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + treatment * pm.math.dot(X, beta_tx)
            + pm.math.dot(X_sq, beta_sq)
            + beta_tps * treatment * ps_data
        )
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.
    Returns array of shape (n_samples, n_obs).
    """
    n_obs = len(new_data["outcome"])
    new_coords = {"obs": np.arange(n_obs)}
    X_squared = new_data["X"][:, :N_CONTINUOUS] ** 2
    ps_new = _estimate_propensity_score(new_data["X"], new_data["treatment"])
    with model:
        pm.set_data(
            {
                "X": new_data["X"],
                "treatment": new_data["treatment"],
                "X_sq": X_squared,
                "ps": ps_new,
            },
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
        tau_i = beta_t + X_i @ beta_tx + beta_tps * ps_i
    The quadratic terms (beta_sq) cancel out because they do not interact
    with treatment.

    Falls back gracefully if idata comes from an older model without beta_tx
    or beta_tps (e.g., during runner's comparison with best.nc from a previous
    model).
    """
    # Extract posterior samples: shape (n_chains, n_draws)
    beta_t_samples = idata.posterior["beta_treatment"].values
    n_chains, n_draws = beta_t_samples.shape
    beta_t_flat = beta_t_samples.reshape(-1)  # (n_samples,)

    ate_samples = beta_t_flat.copy()

    # Add interaction terms if present
    if "beta_tx" in idata.posterior:
        beta_tx_samples = idata.posterior["beta_tx"].values
        n_features = beta_tx_samples.shape[2]
        beta_tx_flat = beta_tx_samples.reshape(-1, n_features)  # (n_samples, n_features)

        X_train = train_data["X"]  # (n_obs, n_features)

        # Individual treatment effects: tau_i includes beta_t + X_i @ beta_tx
        interaction_term = X_train @ beta_tx_flat.T  # (n_obs, n_samples)
        # ATE contribution = mean_i(X_i @ beta_tx)
        ate_samples = ate_samples + interaction_term.mean(axis=0)

    # Add PS-treatment interaction if present
    if "beta_tps" in idata.posterior:
        beta_tps_samples = idata.posterior["beta_tps"].values.reshape(-1)
        ps_train = _estimate_propensity_score(train_data["X"], train_data["treatment"])
        # ATE contribution = beta_tps * mean(ps)
        ate_samples = ate_samples + beta_tps_samples * ps_train.mean()

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
