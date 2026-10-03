from linkfail.rf_extension import prepare_span_features
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
