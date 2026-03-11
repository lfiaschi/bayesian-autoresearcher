"""IHDP Bayesian causal model — doubly robust with propensity score.
Treatment interactions restricted to continuous confounders (x1-x6) only,
plus squared terms for those same confounders to capture non-linear
confounder-outcome relationships, treatment x quadratic confounder
interactions for non-linear heterogeneity in treatment effects, and an
externally estimated propensity score (logistic regression) as an additional
covariate for doubly robust confounding adjustment.  Binary confounders
(x7-x25) still enter the main effects (beta_x) but do NOT get interaction
terms, eliminating 19 noisy parameters for better regularization.
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
    """Build a doubly robust model with continuous-only interactions, quadratic terms, and propensity score."""
    coords = dict(train_data["coords"])
    coords["cont_features"] = [f"x{i}" for i in range(1, N_CONTINUOUS + 1)]

    X_raw = train_data["X"]
    X_cont_raw = X_raw[:, :N_CONTINUOUS]
    X_squared = X_cont_raw ** 2
    ps = _estimate_propensity_score(X_raw, train_data["treatment"])

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_raw, dims=("obs", "features"))
        X_cont = pm.Data("X_cont", X_cont_raw, dims=("obs", "cont_features"))
        X_sq = pm.Data("X_sq", X_squared, dims=("obs", "cont_features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        ps_data = pm.Data("ps", ps, dims="obs")

        # Priors — tightened for standardized confounders (598 training obs)
        alpha = pm.Normal("alpha", mu=0, sigma=3)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=2)
        beta_x = pm.Normal("beta_x", mu=0, sigma=1, dims="features")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=0.7, dims="cont_features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=0.5, dims="cont_features")
        beta_tx_sq = pm.Normal("beta_tx_sq", mu=0, sigma=0.3, dims="cont_features")
        beta_tps = pm.Normal("beta_tps", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=2)

        # Linear predictor: all 25 confounders in main effects, but only
        # continuous confounders (x1-x6) get treatment interactions.
        # beta_tx_sq captures non-linear heterogeneity in treatment effects
        # via treatment x X^2 interactions.
        # PS * treatment captures residual confounding that varies with
        # propensity; main confounding is already handled by X and X^2.
        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + treatment * pm.math.dot(X_cont, beta_tx)
            + pm.math.dot(X_sq, beta_sq)
            + treatment * pm.math.dot(X_sq, beta_tx_sq)
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
    X_cont_new = new_data["X"][:, :N_CONTINUOUS]
    X_squared = X_cont_new ** 2
    ps_new = _estimate_propensity_score(new_data["X"], new_data["treatment"])
    with model:
        pm.set_data(
            {
                "X": new_data["X"],
                "X_cont": X_cont_new,
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
        tau_i = beta_t + X_cont_i @ beta_tx + X_sq_i @ beta_tx_sq + beta_tps * ps_i
    Only continuous confounders (x1-x6) participate in beta_tx and beta_tx_sq
    interactions. The main quadratic terms (beta_sq) cancel out because they
    do not interact with treatment.

    Falls back gracefully if idata comes from an older model without beta_tx,
    beta_tx_sq, or beta_tps (e.g., during runner's comparison with best.nc
    from a previous model).
    """
    # Extract posterior samples: shape (n_chains, n_draws)
    beta_t_samples = idata.posterior["beta_treatment"].values
    n_chains, n_draws = beta_t_samples.shape
    beta_t_flat = beta_t_samples.reshape(-1)  # (n_samples,)

    ate_samples = beta_t_flat.copy()

    # Add linear interaction terms if present (only continuous confounders)
    if "beta_tx" in idata.posterior:
        beta_tx_samples = idata.posterior["beta_tx"].values
        n_features = beta_tx_samples.shape[2]
        beta_tx_flat = beta_tx_samples.reshape(-1, n_features)  # (n_samples, n_features)

        # Use only the first n_features columns of X (continuous confounders)
        X_cont_train = train_data["X"][:, :n_features]

        # Individual treatment effects: tau_i includes beta_t + X_cont_i @ beta_tx
        interaction_term = X_cont_train @ beta_tx_flat.T  # (n_obs, n_samples)
        # ATE contribution = mean_i(X_cont_i @ beta_tx)
        ate_samples = ate_samples + interaction_term.mean(axis=0)

    # Add quadratic interaction terms if present (treatment x X^2)
    if "beta_tx_sq" in idata.posterior:
        beta_tx_sq_samples = idata.posterior["beta_tx_sq"].values
        n_features_sq = beta_tx_sq_samples.shape[2]
        beta_tx_sq_flat = beta_tx_sq_samples.reshape(-1, n_features_sq)  # (n_samples, n_features)

        X_sq_train = train_data["X"][:, :n_features_sq] ** 2

        # Individual treatment effects: tau_i includes X_sq_i @ beta_tx_sq
        interaction_sq = X_sq_train @ beta_tx_sq_flat.T  # (n_obs, n_samples)
        # ATE contribution = mean_i(X_sq_i @ beta_tx_sq)
        ate_samples = ate_samples + interaction_sq.mean(axis=0)

    # Add PS-treatment interaction if present
    if "beta_tps" in idata.posterior:
        beta_tps_samples = idata.posterior["beta_tps"].values.reshape(-1)
        ps_train = _estimate_propensity_score(train_data["X"], train_data["treatment"])
        # ATE contribution = beta_tps * mean(ps)
        ate_samples = ate_samples + beta_tps_samples * ps_train.mean()

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
