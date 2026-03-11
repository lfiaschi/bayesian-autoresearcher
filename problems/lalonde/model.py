"""LaLonde Bayesian causal model — v8.
Censored Student-T with quadratic terms for continuous confounders.
Augments the feature set with squared age, education, re75 to capture
nonlinear covariate effects while keeping the censored likelihood.
"""
import numpy as np
import pymc as pm


CONTINUOUS_NAMES = ("age", "education", "re75")


def _augment_features(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Add squared terms for continuous confounders."""
    continuous_idx = [i for i, n in enumerate(feature_names) if n in CONTINUOUS_NAMES]
    squared = X[:, continuous_idx] ** 2
    aug_names = feature_names + [f"{feature_names[i]}_sq" for i in continuous_idx]
    return np.hstack([X, squared]), aug_names


def build_model(train_data: dict) -> pm.Model:
    """Build censored Student-T model with quadratic features."""
    coords = train_data["coords"]
    feature_names = list(coords["features"])
    X_aug, aug_names = _augment_features(train_data["X"], feature_names)
    n_aug = len(aug_names)

    coords["aug_features"] = aug_names

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_aug, dims=("obs", "aug_features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=5000, sigma=3000)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=2000)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2000, dims="aug_features")
        beta_tx = pm.Normal("beta_tx", mu=0, sigma=1000, dims="aug_features")
        sigma = pm.HalfNormal("sigma", sigma=5000)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        tx_interaction = X * treatment[:, None]
        mu = (
            alpha
            + beta_t * treatment
            + pm.math.dot(X, beta_x)
            + pm.math.dot(tx_interaction, beta_tx)
        )

        latent = pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma)
        pm.Censored("y", latent, lower=0, upper=None, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data."""
    aug_names = list(model.coords["aug_features"])
    base_names = [n for n in aug_names if not n.endswith("_sq")]
    X_aug, _ = _augment_features(new_data["X"], base_names)

    n_obs = len(new_data["outcome"])
    new_coords = {"obs": np.arange(n_obs)}
    with model:
        pm.set_data(
            {"X": X_aug, "treatment": new_data["treatment"]},
            coords=new_coords,
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    samples = ppc.predictions["y"].values
    n_chains, n_draws, _ = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via posterior predictive intervention."""
    aug_names = list(model.coords["aug_features"])
    base_names = [n for n in aug_names if not n.endswith("_sq")]
    X_aug, _ = _augment_features(train_data["X"], base_names)

    n_obs = len(train_data["outcome"])
    train_coords = {"obs": np.arange(n_obs)}

    with model:
        pm.set_data(
            {"X": X_aug, "treatment": np.ones(n_obs)},
            coords=train_coords,
        )
        ppc_t1 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    with model:
        pm.set_data(
            {"X": X_aug, "treatment": np.zeros(n_obs)},
            coords=train_coords,
        )
        ppc_t0 = pm.sample_posterior_predictive(
            idata, var_names=["y"], predictions=True
        )

    y1 = ppc_t1.predictions["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.predictions["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)

    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
