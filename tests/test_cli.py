"""
test_cli.py  -  END-TO-END CHECKS

ROLE OF THIS FILE
-----------------
Instead of testing one small piece, these checks run the real commands from start to
finish (train, predict, demo) and look at the files they produce - just like a user.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from linkfail.cli import main
from linkfail.config import DEFAULT_CONFIG, deep_merge
from linkfail.sample_data import SAMPLE_CONFIG, generate_sample_data

# The top folder of the project (this file lives in <root>/tests/).
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_creates_all_outputs(tmp_path):
    """CHECK: the demo command produces every promised file and every model scores well."""
    out = tmp_path / "run"          # tmp_path = a throw-away folder pytest deletes afterwards
    main(["demo", "--rows", "600", "--data-out", str(tmp_path / "d.csv"), "--out", str(out)])
    for name in ("model.joblib", "metrics.json", "report.md", "config_used.yaml"):
        assert (out / name).exists(), name
    for fig in ("confusion_matrices", "metric_comparison", "roc_curves", "decision_boundaries"):
        assert (out / "figures" / f"{fig}.png").exists(), fig
    metrics = json.loads((out / "metrics.json").read_text())
    # The made-up data is nearly separable, so every model must beat 90% accuracy.
    for name, res in metrics["results"].items():
        assert res["test"]["accuracy"] > 0.9, name


def test_train_then_predict_roundtrip(tmp_path):
    """
    CHECK: the most common real use. A plain CSV with two feature columns and a 0/1
    label (no YAML file) -> train -> save -> use the saved model on data WITHOUT labels.
    """
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"X1": np.r_[rng.normal(0, 1, n), rng.normal(4, 1, n)],       # healthy cloud, then failing cloud
                       "X2": np.r_[rng.normal(0, 1, n), rng.normal(-4, 1, n)],
                       "label": np.r_[np.zeros(n), np.ones(n)].astype(int)})
    csv = tmp_path / "links.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "run"
    main(["train", "--data", str(csv), "--features", "X1", "X2", "--label", "label", "--out", str(out)])

    # Predict on the same data WITHOUT its label column (new data has no labels).
    new_csv = tmp_path / "new.csv"
    df.drop(columns="label").to_csv(new_csv, index=False)
    preds = tmp_path / "preds.csv"
    main(["predict", "--model", str(out / "model.joblib"),
          "--data", str(new_csv), "--output", str(preds)])
    result = pd.read_csv(preds)
    assert len(result) == len(df)                                                  # one verdict per link
    accuracy = ((result["prediction"] == "not healthy").astype(int) == df["label"]).mean()
    assert accuracy > 0.97                                                         # and they are nearly all right


def test_predict_localize_adds_span_diagnostic(tmp_path):
    """CHECK: raw span data can return an actionable location alongside a prediction."""
    df = generate_sample_data(n_rows=240, missing_fraction=0)
    source = tmp_path / "telemetry.csv"
    df.to_csv(source, index=False)
    out = tmp_path / "run"
    main(["train", "--data", str(source), "--config", str(REPO_ROOT / "configs" / "sample.yaml"),
          "--out", str(out)])
    predictions = tmp_path / "predictions.csv"
    main(["predict", "--model", str(out / "model.joblib"), "--data", str(source),
          "--output", str(predictions), "--localize"])
    result = pd.read_csv(predictions)
    assert {"suspected_span", "localization_score"} <= set(result.columns)
    assert result.loc[result["prediction"] == "not healthy", "suspected_span"].notna().all()


def test_bad_input_exits_with_code_2(tmp_path, capsys):
    """CHECK: a mistake like a wrong column name ends with a short 'error:' message and exit code 2."""
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit) as exc:
        main(["train", "--data", str(csv), "--features", "missing_col", "--label", "a"])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err        # capsys captures what was printed as an error


def test_sample_yaml_matches_builtin_sample_config():
    """
    CHECK: configs/sample.yaml (the readable copy) and the demo's built-in settings must
    be identical, so the documentation never lies about what the demo does.
    """
    shipped = yaml.safe_load((REPO_ROOT / "configs" / "sample.yaml").read_text())
    assert deep_merge(DEFAULT_CONFIG, shipped) == deep_merge(DEFAULT_CONFIG, SAMPLE_CONFIG)


def test_sample_generator_is_reproducible():
    """CHECK: the same seed gives exactly the same made-up table twice."""
    a, b = generate_sample_data(n_rows=50, seed=1), generate_sample_data(n_rows=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
