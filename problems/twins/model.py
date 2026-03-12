"""Twins Bayesian causal model — confounder-dependent heteroscedasticity + quadratic, cubic & pairwise interactions + gestat10 quartic/quintic.
Linear regression with treatment, confounders, quadratic and cubic terms for
continuous confounders (mager8, mrace, gestat10), plus pairwise interactions
between the 3 continuous confounders, plus 4th and 5th degree polynomial terms
for gestat10 only (the dominant confounder).
Heteroscedastic noise: log(sigma) depends on gestat10, so premature babies
(low gestat10) can have different outcome variance than full-term babies.
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
    """Build a linear Bayesian model for Twins with tight priors, quadratic, cubic and pairwise terms."""
    coords = dict(train_data["coords"])
    coords["cont_features"] = ["mager8", "mrace", "gestat10"]
    coords["pair_features"] = ["mager8_x_mrace", "mager8_x_gestat10", "mrace_x_gestat10"]

    X_raw = train_data["X"]
    X_cont = X_raw[:, N_CONTINUOUS_INDICES]
    X_sq = X_cont ** 2
    X_cubed = X_cont ** 3
    X_pairs = np.column_stack([
        X_cont[:, 0] * X_cont[:, 1],
        X_cont[:, 0] * X_cont[:, 2],
        X_cont[:, 1] * X_cont[:, 2],
    ])

    # Higher-order polynomial terms for gestat10 only (index 2 in cont_features)
    gestat10 = X_cont[:, 2]
    gestat_4 = gestat10 ** 4
    gestat_5 = gestat10 ** 5

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_raw, dims=("obs", "features"))
        X_cont_data = pm.Data("X_cont", X_cont, dims=("obs", "cont_features"))
        X_sq_data = pm.Data("X_sq", X_sq, dims=("obs", "cont_features"))
        X_cubed_data = pm.Data("X_cubed", X_cubed, dims=("obs", "cont_features"))
        X_pairs_data = pm.Data("X_pairs", X_pairs, dims=("obs", "pair_features"))
        gestat_4_data = pm.Data("gestat_4", gestat_4, dims=("obs",))
        gestat_5_data = pm.Data("gestat_5", gestat_5, dims=("obs",))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=0.5)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=0.3)
        beta_x = pm.Normal("beta_x", mu=0, sigma=0.3, dims="features")
        beta_sq = pm.Normal("beta_sq", mu=0, sigma=0.2, dims="cont_features")
        beta_cb = pm.Normal("beta_cb", mu=0, sigma=0.1, dims="cont_features")
        beta_pair = pm.Normal("beta_pair", mu=0, sigma=0.15, dims="pair_features")
        beta_g4 = pm.Normal("beta_g4", mu=0, sigma=0.05)
        beta_g5 = pm.Normal("beta_g5", mu=0, sigma=0.02)
        log_sigma_0 = pm.Normal("log_sigma_0", mu=-1.5, sigma=0.5)  # base log-sigma (exp(-1.5) ≈ 0.22)
        log_sigma_gestat = pm.Normal("log_sigma_gestat", mu=0, sigma=0.3)  # gestat10 effect on log-sigma
        sigma = pm.math.exp(log_sigma_0 + log_sigma_gestat * X_cont_data[:, 2])

        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(X_sq_data, beta_sq)
            + pm.math.dot(X_cubed_data, beta_cb)
            + pm.math.dot(X_pairs_data, beta_pair)
            + beta_g4 * gestat_4_data
            + beta_g5 * gestat_5_data
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
    X_cubed_new = X_cont_new ** 3
    X_pairs_new = np.column_stack([
        X_cont_new[:, 0] * X_cont_new[:, 1],
        X_cont_new[:, 0] * X_cont_new[:, 2],
        X_cont_new[:, 1] * X_cont_new[:, 2],
    ])
    gestat10_new = X_cont_new[:, 2]
    gestat_4_new = gestat10_new ** 4
    gestat_5_new = gestat10_new ** 5

    with model:
        pm.set_data(
            {
                "X": X_new,
                "X_cont": X_cont_new,
                "X_sq": X_sq_new,
                "X_cubed": X_cubed_new,
                "X_pairs": X_pairs_new,
                "gestat_4": gestat_4_new,
                "gestat_5": gestat_5_new,
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
    Quadratic/cubic terms don't interact with treatment, so ATE is still just beta_treatment.
    """
    beta_t_samples = idata.posterior["beta_treatment"].values.flatten()
    ate = float(beta_t_samples.mean())

    return {"ate": ate, "ate_samples": beta_t_samples}
