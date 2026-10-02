"""
test_data.py  -  CHECKS FOR THE PREPARATION ROOM (data.py and config.py)

ROLE OF THIS FILE
-----------------
Confirms that tables are cleaned, features are calculated, and labels are created
correctly - using tiny tables small enough to verify by hand.
"""

import numpy as np
import pandas as pd
import pytest

from linkfail.config import ConfigError, DEFAULT_CONFIG, deep_merge, load_config
from linkfail.data import DataError, Standardizer, expand_columns, localize_spans, prepare_dataset


def data_cfg(**changes):
    """HELPER: the default data settings with a few keys changed, to keep each test short."""
    return deep_merge(DEFAULT_CONFIG, {"data": changes})["data"]


def test_wildcards_use_natural_order():
    """CHECK: 's_*' lists s_1, s_2, s_10 in human order (not s_1, s_10, s_2)."""
    cols = ["s_10", "s_2", "s_1", "other"]
    assert expand_columns(cols, "s_*") == ["s_1", "s_2", "s_10"]


def test_unknown_column_message_lists_available():
    """CHECK: a typo in a column name produces an error that shows the real column names."""
    with pytest.raises(DataError, match="Available columns"):
        expand_columns(["a", "b"], "zzz")


def test_derived_feature_is_sum_of_pairwise_diffs():
    """CHECK: the feature equals the sum of (a - b) over spans: the paper's Eq. 1 + Eq. 2."""
    df = pd.DataFrame({"a_1": [5.0, 1.0], "a_2": [4.0, 1.0], "b_1": [1.0, 0.0], "b_2": [1.0, 3.0],
                       "label": [0, 1]})
    cfg = data_cfg(derived_features=[{"name": "X", "minuend": "a_*", "subtrahend": "b_*"}])
    out = prepare_dataset(df, cfg)
    # Row 0 by hand: (5-1) + (4-1) = 7.   Row 1 by hand: (1-0) + (1-3) = -1.
    assert out.X["X"].tolist() == [7.0, -1.0]
    assert out.y.tolist() == [0, 1]


def test_mismatched_pairs_raise():
    """CHECK: 2 'a' columns but only 1 'b' column can't be paired up, so we must get a clear error."""
    df = pd.DataFrame({"a_1": [1.0], "a_2": [1.0], "b_1": [1.0], "label": [0]})
    cfg = data_cfg(derived_features=[{"name": "X", "minuend": "a_*", "subtrahend": "b_*"}])
    with pytest.raises(DataError, match="pair up"):
        prepare_dataset(df, cfg)


def test_interpolation_fills_gap_linearly():
    """CHECK: the gap between 1 and 3 is filled with 2 (the straight-line middle)."""
    df = pd.DataFrame({"f": [1.0, np.nan, 3.0], "label": [0, 1, 0]})
    out = prepare_dataset(df, data_cfg(feature_columns=["f"], missing="interpolate"))
    assert out.X["f"].tolist() == [1.0, 2.0, 3.0]


def test_drop_strategy_removes_rows():
    """CHECK: with 'drop', the row that has an empty cell is removed and counted."""
    df = pd.DataFrame({"f": [1.0, np.nan, 3.0], "label": [0, 1, 0]})
    out = prepare_dataset(df, data_cfg(feature_columns=["f"], missing="drop"))
    assert len(out.X) == 2 and out.n_dropped == 1


def test_text_labels_with_healthy_labels():
    """CHECK: if 'no-failure' means healthy, then EDFA and NLI both become 'not healthy' (1)."""
    df = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0], "label": ["no-failure", "EDFA", "NLI", "no-failure"]})
    out = prepare_dataset(df, data_cfg(feature_columns=["f"], healthy_labels=["no-failure"]))
    assert out.y.tolist() == [0, 1, 1, 0]


def test_threshold_labels():
    """CHECK: OSNR below 20 means 'not healthy': 25 -> 0, 19 -> 1, 10 -> 1."""
    df = pd.DataFrame({"f": [0.0, 1.0, 2.0], "osnr": [25.0, 19.0, 10.0]})
    out = prepare_dataset(df, data_cfg(feature_columns=["f"],
                                       label_from_threshold={"column": "osnr", "below": 20.0}))
    assert out.y.tolist() == [0, 1, 1]


def test_non_binary_label_without_mapping_is_explained():
    """CHECK: a word-valued label without instructions gives an error that says how to fix it."""
    df = pd.DataFrame({"f": [0.0, 1.0], "label": ["ok", "bad"]})
    with pytest.raises(DataError, match="positive_labels"):
        prepare_dataset(df, data_cfg(feature_columns=["f"]))


def test_prediction_mode_needs_no_label_column():
    """CHECK: when predicting on new data (with_labels=False) a label column is not required."""
    df = pd.DataFrame({"f": [0.0, 1.0]})
    out = prepare_dataset(df, data_cfg(feature_columns=["f"]), with_labels=False)
    assert out.y is None and len(out.X) == 2


def test_localization_returns_span_with_largest_adverse_deviation():
    """CHECK: the diagnostic names the span that is furthest below its baseline."""
    df = pd.DataFrame({"gain_01": [9.5], "gain_02": [7.0],
                       "target_01": [10.0], "target_02": [10.0]})
    cfg = data_cfg(derived_features=[{"name": "gain_diff", "minuend": "gain_*",
                                      "subtrahend": "target_*"}])
    result = localize_spans(df, cfg)
    assert result.loc[0, "suspected_span"] == "span_02"
    assert result.loc[0, "localization_score"] == pytest.approx(3.0)


def test_standardizer_uses_train_statistics():
    """CHECK: after rescaling, the training numbers average 0, and the training average maps to exactly 0."""
    train = np.array([[0.0], [2.0], [4.0]])
    sc = Standardizer().fit(train)
    np.testing.assert_allclose(sc.transform(train).mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(sc.transform(np.array([[2.0]])), [[0.0]])    # 2 is the training average


def test_standardizer_handles_constant_column():
    """CHECK: a feature that never changes must not cause a divide-by-zero crash."""
    sc = Standardizer().fit(np.ones((5, 1)))
    assert np.isfinite(sc.transform(np.ones((5, 1)))).all()


# Each dictionary below is a deliberately BAD setting; the same check runs for all four.
@pytest.mark.parametrize("bad", [
    {"data": {"missing": "guess"}},                 # unknown cleaning strategy
    {"data": {"feature_columns": []}},              # no features at all
    {"split": {"test_size": 1.5}},                  # 150% test data is impossible
    {"models": ["random_forest"]},                  # a model we don't have
])
def test_invalid_configs_are_rejected(bad):
    """CHECK: every bad setting above is caught with a ConfigError."""
    base = {"data": {"feature_columns": ["f"]}}
    with pytest.raises(ConfigError):
        load_config(None, deep_merge(base, bad))
