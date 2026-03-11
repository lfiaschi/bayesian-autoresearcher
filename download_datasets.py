"""Download all causal inference datasets.
Usage: uv run python download_datasets.py
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent / "problems"


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination path."""
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"  Saved to {dest}")


def download_ihdp() -> None:
    """Download IHDP dataset from CEVAE repository."""
    print("\n=== IHDP ===")
    data_dir = BASE_DIR / "ihdp" / "data"
    dest = data_dir / "ihdp.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    col_names = (
        ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
        + [f"x{i}" for i in range(1, 26)]
    )
    df = pd.read_csv(io.StringIO(response.text), header=None, names=col_names)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  IHDP: {len(df)} rows, {len(df.columns)} columns")
    print(f"  True ATE: {(df['mu1'] - df['mu0']).mean():.4f}")


def download_twins() -> None:
    """Download Twins dataset from CEVAE repository.

    Treatment is assigned as the heavier twin (dbirwt_1 > dbirwt_0).
    Outcome is one-year mortality (mort_0 / mort_1 from the Y file).
    Covariates come from twin_pairs_X_3years_samesex.csv.
    """
    print("\n=== Twins ===")
    data_dir = BASE_DIR / "twins" / "data"
    dest = data_dir / "twins.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    base = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS"
    bw_url = f"{base}/twin_pairs_T_3years_samesex.csv"
    mort_url = f"{base}/twin_pairs_Y_3years_samesex.csv"
    covar_url = f"{base}/twin_pairs_X_3years_samesex.csv"

    bw_resp = requests.get(bw_url, timeout=60)
    bw_resp.raise_for_status()
    mort_resp = requests.get(mort_url, timeout=60)
    mort_resp.raise_for_status()
    covar_resp = requests.get(covar_url, timeout=60)
    covar_resp.raise_for_status()

    bw_df = pd.read_csv(io.StringIO(bw_resp.text), index_col=0)
    mort_df = pd.read_csv(io.StringIO(mort_resp.text), index_col=0)
    covar_df = pd.read_csv(io.StringIO(covar_resp.text), index_col=0)

    # Treatment = 1 if twin 1 is heavier (the "treated" lighter-birth-weight twin)
    # Following standard CEVAE setup: treatment = (dbirwt_0 < dbirwt_1).astype(int)
    treatment = (bw_df["dbirwt_0"] < bw_df["dbirwt_1"]).astype(int).values
    mort_0 = mort_df["mort_0"].values
    mort_1 = mort_df["mort_1"].values
    y_factual = np.where(treatment == 0, mort_0, mort_1)
    y_cfactual = np.where(treatment == 0, mort_1, mort_0)

    covar_cols = [c for c in covar_df.columns if c not in ["Unnamed: 0"]]
    n = len(bw_df)

    result_df = pd.DataFrame({
        "treatment": treatment,
        "y_factual": y_factual,
        "y_cfactual": y_cfactual,
    })
    for col in covar_cols:
        result_df[col] = covar_df[col].values[:n]

    result_df = result_df.dropna()
    data_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(dest, index=False)
    print(f"  Twins: {len(result_df)} rows, {len(result_df.columns)} columns")


def download_lalonde() -> None:
    """Download LaLonde dataset (NSW experimental data)."""
    print("\n=== LaLonde ===")
    data_dir = BASE_DIR / "lalonde" / "data"
    dest = data_dir / "lalonde.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    treat_url = "https://users.nber.org/~rdehejia/data/nsw_treated.txt"
    control_url = "https://users.nber.org/~rdehejia/data/nsw_control.txt"

    # NBER NSW files have 9 columns (no pre-treatment earnings re74/re75).
    col_names = [
        "treatment", "age", "education", "black", "hispanic",
        "married", "nodegree", "re75", "re78"
    ]

    treat_resp = requests.get(treat_url, timeout=60)
    treat_resp.raise_for_status()
    control_resp = requests.get(control_url, timeout=60)
    control_resp.raise_for_status()

    treat_df = pd.read_csv(
        io.StringIO(treat_resp.text), sep=r"\s+", header=None, names=col_names
    )
    control_df = pd.read_csv(
        io.StringIO(control_resp.text), sep=r"\s+", header=None, names=col_names
    )

    df = pd.concat([treat_df, control_df], ignore_index=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  LaLonde: {len(df)} rows")
    ate = treat_df["re78"].mean() - control_df["re78"].mean()
    print(f"  Experimental ATE (re78): ${ate:.2f}")


def download_nhefs() -> None:
    """Download NHEFS dataset for smoking cessation study."""
    print("\n=== NHEFS ===")
    data_dir = BASE_DIR / "nhefs" / "data"
    dest = data_dir / "nhefs.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    url = "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/nhefs/NHEFS.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    key_cols = [
        "qsmk", "sex", "race", "age", "school", "smokeintensity",
        "smokeyrs", "exercise", "active", "wt71", "wt82", "wt82_71"
    ]
    df = df.dropna(subset=key_cols)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  NHEFS: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    print("Downloading causal inference datasets...")
    download_ihdp()
    download_twins()
    download_lalonde()
    download_nhefs()
    print("\nDone! All datasets ready.")
