import json

import numpy as np
import pytest

from linkfail.rf_extension import (
    calculate_metrics,
    load_baseline_results,
    prepare_span_features,
    train_random_forest,
)
from linkfail.sample_data import generate_sample_data


def test_random_forest_span_features():
    df = generate_sample_data(
        n_rows=200,
        missing_fraction=0.0,
        seed=7,
    )

    work, X, y = prepare_span_features(df)

    assert len(work) == 200
    assert X.shape == (200, 38)
    assert len(y) == 200

    assert "gain_diff_01" in X.columns
    assert "loss_diff_01" in X.columns
    assert "gain_diff_19" in X.columns
    assert "loss_diff_19" in X.columns

    assert set(y.unique()).issubset({0, 1})


def test_random_forest_metrics_allow_missing_roc_auc():
    y_true = np.zeros(4, dtype=int)
    y_pred = np.zeros(4, dtype=int)
    probability = np.array([0.1, 0.2, 0.15, 0.05])

    metrics = calculate_metrics(y_true, y_pred, probability)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None


def test_baseline_results_allow_null_roc_auc(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "results": {
                    "logistic_regression": {
                        "test": {
                            "accuracy": 1.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "roc_auc": None,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rows = load_baseline_results(metrics_path)

    assert rows[0]["roc_auc"] is None


def test_random_forest_training_rejects_one_class(tmp_path):
    df = generate_sample_data(
        n_rows=50,
        missing_fraction=0.0,
        seed=11,
    )
    df["osnr_db"] = 30.0

    data_path = tmp_path / "healthy_only.csv"
    config_path = tmp_path / "config.yaml"
    baseline_path = tmp_path / "missing_baseline.json"

    df.to_csv(data_path, index=False)
    config_path.write_text(
        """
data:
  derived_features:
    - name: X1_amp_gain_diff
      minuend: "amp_gain_*"
      subtrahend: "target_gain_*"
    - name: X2_span_loss_diff
      minuend: "target_span_loss_*"
      subtrahend: "span_loss_*"
  missing: interpolate
  sort_by: timestamp
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only one class"):
        train_random_forest(
            data_path=data_path,
            output_directory=tmp_path / "run",
            baseline_metrics_path=baseline_path,
            config_path=config_path,
            n_estimators=5,
        )
