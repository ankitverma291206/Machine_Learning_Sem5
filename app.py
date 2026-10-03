from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import yaml

from linkfail.data import localize_spans
from linkfail.rf_extension import prepare_span_features


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "sample" / "sample_link_data.csv"
MODEL_PATH = ROOT / "runs" / "random_forest" / "rf_model.joblib"
COMPARISON_PATH = ROOT / "runs" / "random_forest" / "comparison_metrics.csv"
CONFIG_PATH = ROOT / "runs" / "demo" / "config_used.yaml"


st.set_page_config(
    page_title="Link Failure Prediction and Localization",
    layout="wide",
)


st.title("Link Failure Prediction and Localization")

st.caption(
    "Cloud-scale optical network monitoring using Random Forest "
    "with 38 span-level telemetry features"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
bundle = load_model()
model = bundle["model"]

config = yaml.safe_load(
    CONFIG_PATH.read_text(encoding="utf-8")
)

work, X, y = prepare_span_features(df)

locations = localize_spans(
    df,
    config["data"],
    kept_index=X.index,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Demo Sample Selection")

sample_type = st.sidebar.radio(
    "Filter by ground-truth class (demo only)",
    [
        "All samples",
        "Healthy samples",
        "Failure samples",
    ],
)


if sample_type == "Healthy samples":
    available_rows = y[y == 0].index.tolist()

elif sample_type == "Failure samples":
    available_rows = y[y == 1].index.tolist()

else:
    available_rows = X.index.tolist()


selected_row = st.sidebar.selectbox(
    "Dataset row",
    available_rows,
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

selected_features = X.loc[[selected_row]]

failure_probability = float(
    model.predict_proba(selected_features)[0, 1]
)

predicted_failure = int(
    failure_probability >= 0.5
)

actual_failure = int(
    y.loc[selected_row]
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted Status",
    "FAILURE" if predicted_failure else "HEALTHY",
)

col2.metric(
    "Failure Probability",
    f"{failure_probability * 100:.2f}%",
)

col3.metric(
    "Actual Status",
    "FAILURE" if actual_failure else "HEALTHY",
)

col4.metric(
    "OSNR",
    f"{float(work.loc[selected_row, 'osnr_db']):.2f} dB",
)


st.caption(
    "Actual Status is shown only for evaluation; "
    "it is not provided to the Random Forest as an input feature."
)


# ---------------------------------------------------------
# LOCALIZATION
# ---------------------------------------------------------

if predicted_failure:

    st.subheader("Failure Localization")

    suspected_span = locations.loc[
        selected_row,
        "suspected_span",
    ]

    localization_score = float(
        locations.loc[
            selected_row,
            "localization_score",
        ]
    )

    loc1, loc2 = st.columns(2)

    loc1.metric(
        "Most Suspicious Span",
        suspected_span,
    )

    loc2.metric(
        "Localization Deviation Score",
        f"{localization_score:.3f}",
    )

    st.info(
        "Localization identifies the span with the strongest adverse "
        "gain/loss deviation after a failure is detected."
    )

else:

    st.success(
        "The link is predicted healthy, so localization is not invoked."
    )


# ---------------------------------------------------------
# SPAN-LEVEL DEVIATIONS
# ---------------------------------------------------------

st.subheader("Span-Level Deviations")

span_rows = []

for span in range(1, 20):

    suffix = f"{span:02d}"

    gain_diff = float(
        selected_features.iloc[0][
            f"gain_diff_{suffix}"
        ]
    )

    loss_diff = float(
        selected_features.iloc[0][
            f"loss_diff_{suffix}"
        ]
    )

    span_rows.append(
        {
            "Span": f"Span {suffix}",
            "Gain adverse deviation": max(-gain_diff, 0.0),
            "Loss adverse deviation": max(-loss_diff, 0.0),
        }
    )


span_df = pd.DataFrame(
    span_rows
).set_index("Span")

st.bar_chart(span_df)


# ---------------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------

st.subheader("Random Forest Feature Importance")

importance = (
    pd.Series(
        model.feature_importances_,
        index=bundle["feature_names"],
        name="Importance",
    )
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(
    importance.to_frame(),
    use_container_width=True,
)


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

if COMPARISON_PATH.exists():

    st.subheader("Model Comparison")

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    display_comparison = comparison.copy()

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        display_comparison[column] = (
            display_comparison[column] * 100
        ).round(2)

    st.dataframe(
        display_comparison,
        use_container_width=True,
    )

    chart_df = comparison[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ].melt(
        id_vars="model",
        var_name="metric",
        value_name="score",
    )

    comparison_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "model:N",
                title="Model",
                axis=alt.Axis(
                    labelAngle=-20
                ),
            ),
            xOffset="metric:N",
            y=alt.Y(
                "score:Q",
                title="Score",
                scale=alt.Scale(
                    domain=[0, 1]
                ),
            ),
            color=alt.Color(
                "metric:N",
                title="Metric",
            ),
            tooltip=[
                alt.Tooltip(
                    "model:N",
                    title="Model",
                ),
                alt.Tooltip(
                    "metric:N",
                    title="Metric",
                ),
                alt.Tooltip(
                    "score:Q",
                    title="Score",
                    format=".4f",
                ),
            ],
        )
        .properties(
            title="Model Performance Comparison"
        )
    )

    st.altair_chart(
        comparison_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------

with st.expander(
    "View raw telemetry for selected sample"
):

    st.dataframe(
        work.loc[[selected_row]],
        use_container_width=True,
    )
