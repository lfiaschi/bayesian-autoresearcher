"""Tests for experiment runner utilities."""
import numpy as np
import pandas as pd
import pytest
from runner import (
    parse_problem_config,
    split_data_random,
    split_data_temporal,
    make_data_dict,
    print_results,
    _format_line,
)


def test_parse_problem_config(tmp_path):
    problem_md = tmp_path / "problem.md"
    problem_md.write_text(
        "---\n"
        "primary_metric: crps\n"
        "secondary_metrics: [elpd, mae]\n"
        "split_strategy: random\n"
        "split_ratios: [0.6, 0.2, 0.2]\n"
        "---\n"
        "# Test Problem\n"
    )
    config = parse_problem_config(problem_md)
    assert config["primary_metric"] == "crps"
    assert config["split_strategy"] == "random"
    assert config["split_ratios"] == [0.6, 0.2, 0.2]


def test_split_data_random():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "treatment": rng.choice([0, 1], size=200),
        "outcome": rng.normal(size=200),
        "x1": rng.normal(size=200),
    })
    train, val, test = split_data_random(df, treatment_col="treatment", ratios=[0.6, 0.2, 0.2], seed=42)
    assert len(train) + len(val) + len(test) == 200
    assert abs(len(train) - 120) < 10
    assert abs(len(val) - 40) < 10
    orig_rate = df["treatment"].mean()
    train_rate = train["treatment"].mean()
    assert abs(train_rate - orig_rate) < 0.1


def test_split_data_temporal():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "treatment": np.random.choice([0, 1], size=100),
        "outcome": np.random.normal(size=100),
    })
    train, val, test = split_data_temporal(df, temporal_col="date", ratios=[0.6, 0.2, 0.2])
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_make_data_dict():
    df = pd.DataFrame({
        "treatment": [0, 1, 0, 1],
        "outcome": [1.0, 2.0, 1.5, 2.5],
        "x1": [0.1, 0.2, 0.3, 0.4],
        "x2": [0.5, 0.6, 0.7, 0.8],
    })
    result = make_data_dict(df, treatment_col="treatment", outcome_col="outcome", confounder_cols=["x1", "x2"])
    assert result["X"].shape == (4, 2)
    assert result["treatment"].shape == (4,)
    assert result["outcome"].shape == (4,)
    assert "obs" in result["coords"]
    assert result["coords"]["features"] == ["x1", "x2"]


def test_print_results_format(capsys):
    scores = {"val_crps": 0.45, "val_mae": 1.23}
    convergence = {"ok": True, "r_hat_max": 1.002, "ess_min": 856, "divergences": 0}
    causal = {"ate_estimate": 3.45, "ate_hdi_3": 2.1, "ate_hdi_97": 4.8}
    timing = {"sampling_seconds": 142.3, "total_seconds": 180.1}
    print_results(scores, convergence, causal, timing, n_params=5)
    captured = capsys.readouterr()
    assert "---" in captured.out
    assert "val_crps:" in captured.out
    assert "convergence_ok:" in captured.out
    assert "True" in captured.out


def test_format_line_bool():
    assert _format_line("convergence_ok", True) == "convergence_ok:     True"


def test_format_line_float():
    line = _format_line("val_crps", 0.4523)
    assert line.startswith("val_crps:")
    assert "0.4523" in line


def test_format_line_int():
    line = _format_line("divergences", 0)
    assert "divergences:" in line
    assert "0" in line
