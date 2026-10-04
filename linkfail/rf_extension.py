"""
Random Forest extension for Link Failure Prediction.

Baseline:
    The original implementation aggregates all 19 spans into two features:
        X1 = sum(amplifier_gain - target_gain)
        X2 = sum(target_span_loss - span_loss)

Extension:
    Preserve each span separately:
        gain_diff_01 ... gain_diff_19
        loss_diff_01 ... loss_diff_19

    This produces 38 span-level features.

    A Random Forest classifier is trained for failure prediction.
    Existing span-deviation logic is reused for diagnostic localization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import load_table, localize_spans


N_SPANS = 19
OSNR_THRESHOLD_DB = 20.0
RANDOM_STATE = 42


def prepare_span_features(df: pd.DataFrame):
    """
    Convert raw 19-span telemetry into 38 individual deviation features.

    For every span:

        gain_diff = measured amplifier gain - target amplifier gain

        loss_diff = target span loss - measured span loss

    The target is identical to the baseline project:

        not healthy = OSNR < 20 dB
    """

    work = df.copy()

    if "timestamp" in work.columns:
        work = work.sort_values(
            "timestamp",
            kind="stable",
        )

    required_columns = []

    for span in range(1, N_SPANS + 1):

        suffix = f"{span:02d}"

        required_columns.extend(
            [
                f"amp_gain_{suffix}",
                f"target_gain_{suffix}",
                f"span_loss_{suffix}",
                f"target_span_loss_{suffix}",
            ]
        )

    required_columns.append("osnr_db")

    missing = [
        column
        for column in required_columns
        if column not in work.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        column
        for column in required_columns
        if column != "osnr_db"
    ]

    work[numeric_columns] = work[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    work["osnr_db"] = pd.to_numeric(
        work["osnr_db"],
        errors="coerce",
    )

    # Same overall missing-value strategy as the baseline:
    # interpolate telemetry in time order.
    work[numeric_columns] = work[
        numeric_columns
    ].interpolate(
        method="linear",
        limit_direction="both",
    )

    # Median fallback in case an edge value survives interpolation.
    work[numeric_columns] = work[
        numeric_columns
    ].fillna(
        work[numeric_columns].median()
    )

    # Rows without OSNR cannot be labelled.
    work = work[
        work["osnr_db"].notna()
    ].copy()

    features = {}

    for span in range(1, N_SPANS + 1):

        suffix = f"{span:02d}"

        features[
            f"gain_diff_{suffix}"
        ] = (
            work[f"amp_gain_{suffix}"]
            - work[f"target_gain_{suffix}"]
        )

        features[
            f"loss_diff_{suffix}"
        ] = (
            work[f"target_span_loss_{suffix}"]
            - work[f"span_loss_{suffix}"]
        )

    X = pd.DataFrame(
        features,
        index=work.index,
    )

    y = (
        work["osnr_db"]
        < OSNR_THRESHOLD_DB
    ).astype(int)

    return work, X, y


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):
    """Return standard binary-classification metrics."""

    both_classes = len(
        np.unique(
            y_true,
        )
    ) == 2

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": (
            float(
                roc_auc_score(
                    y_true,
                    y_probability,
                )
            )
            if both_classes
            else None
        ),
    }


def _format_optional_metric(value):
    """Format a metric that may be undefined for one-class data."""

    return "n/a" if value is None else f"{value:.4f}"


def save_confusion_matrix(
    y_true,
    y_pred,
    path: Path,
):
    """Save Random Forest confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5)
    )

    image = ax.imshow(matrix)

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_xticks(
        [0, 1],
        ["Healthy", "Not Healthy"],
    )

    ax.set_yticks(
        [0, 1],
        ["Healthy", "Not Healthy"],
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        "Random Forest Confusion Matrix"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
    )

    plt.close(fig)


def save_feature_importance(
    model,
    feature_names,
    csv_path: Path,
    figure_path: Path,
):
    """Save all RF feature importances and plot the top 20."""

    importance = (
        pd.Series(
            model.feature_importances_,
            index=feature_names,
            name="importance",
        )
        .sort_values(
            ascending=False
        )
    )

    importance.to_csv(
        csv_path,
        header=True,
    )

    top = (
        importance
        .head(20)
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        top.index,
        top.values,
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_title(
        "Top 20 Random Forest Features"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=180,
    )

    plt.close(fig)

    return importance


def load_baseline_results(
    baseline_metrics_path: Path,
):
    """
    Extract TEST metrics produced by the existing baseline implementation.
    """

    rows = []

    if not baseline_metrics_path.exists():
        return rows

    payload = json.loads(
        baseline_metrics_path.read_text(
            encoding="utf-8"
        )
    )

    results = payload.get(
        "results",
        {}
    )

    for model_name, model_result in results.items():

        test = model_result.get(
            "test",
            {}
        )

        if not test:
            continue

        rows.append(
            {
                "model": model_name,
                "accuracy": float(
                    test["accuracy"]
                ),
                "precision": float(
                    test["precision"]
                ),
                "recall": float(
                    test["recall"]
                ),
                "f1": float(
                    test["f1"]
                ),
                "roc_auc": float(
                    test["roc_auc"]
                )
                if test.get("roc_auc") is not None
                else None,
            }
        )

    return rows


def save_comparison_plot(
    comparison_df: pd.DataFrame,
    path: Path,
):
    """Compare baseline test metrics against Random Forest."""

    plot_df = (
        comparison_df
        .set_index("model")
        [
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ]
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_ylabel("Score")

    ax.set_title(
        "Baseline Models vs Random Forest"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


def write_report(
    output_path: Path,
    metrics,
    comparison_df,
    importance,
    total_rows,
    train_rows,
    test_rows,
):
    """Create a concise Markdown report."""

    comparison_markdown = (
        comparison_df.to_markdown(
            index=False
        )
    )

    importance_markdown = (
        importance
        .head(10)
        .to_frame()
        .to_markdown()
    )

    report = f"""# Random Forest Extension Report

## Dataset

- Source: `data/sample/sample_link_data.csv`
- Dataset type: synthetic optical-network telemetry
- Total rows used: {total_rows}
- Training rows: {train_rows}
- Test rows: {test_rows}
- Optical spans: 19
- Failure rule: `osnr_db < 20 dB`

## Baseline methodology

The existing implementation follows the Stanford-inspired methodology and
aggregates measurements from all 19 spans into two features:

- `X1_amp_gain_diff`
- `X2_span_loss_diff`

The baseline classifiers are:

- Logistic Regression
- Gaussian Discriminant Analysis
- Linear SVM

## Random Forest extension

Instead of aggregating all span information into two values, this extension
retains the two deviations for each of the 19 spans:

- 19 amplifier-gain deviation features
- 19 span-loss deviation features

Total Random Forest input features: **38**

The Random Forest can therefore model nonlinear interactions while retaining
span-level information.

## Random Forest test performance

- Accuracy: {metrics["accuracy"]:.4f}
- Precision: {metrics["precision"]:.4f}
- Recall: {metrics["recall"]:.4f}
- F1-score: {metrics["f1"]:.4f}
- ROC-AUC: {_format_optional_metric(metrics["roc_auc"])}

## Model comparison

{comparison_markdown}

## Top 10 Random Forest features

{importance_markdown}

## Localization

For samples predicted as not healthy, the project reuses the existing
span-deviation localization algorithm.

The localization method identifies the span with the strongest adverse
deviation from its expected amplifier-gain and span-loss values.

This is a diagnostic localization heuristic rather than a separately trained
multiclass localization model.
"""

    output_path.write_text(
        report,
        encoding="utf-8",
    )


def train_random_forest(
    data_path,
    output_directory,
    baseline_metrics_path,
    config_path,
    n_estimators=400,
):
    """Train and evaluate the Random Forest extension."""

    data_path = Path(data_path)

    output_directory = Path(
        output_directory
    )

    baseline_metrics_path = Path(
        baseline_metrics_path
    )

    config_path = Path(
        config_path
    )

    figures_directory = (
        output_directory
        / "figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading dataset: {data_path}"
    )

    df = load_table(
        data_path
    )

    work, X, y = (
        prepare_span_features(
            df
        )
    )

    print(
        f"Rows available: {len(X)}"
    )

    print(
        f"Random Forest features: "
        f"{X.shape[1]}"
    )

    print(
        f"Healthy rows: "
        f"{int((y == 0).sum())}"
    )

    print(
        f"Not healthy rows: "
        f"{int((y == 1).sum())}"
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "After labelling, only one class remains. "
            "Check the OSNR threshold or input data."
        )

    # Same split parameters as the baseline:
    # 75% train, 25% test, seed 42, stratified.
    train_indices, test_indices = (
        train_test_split(
            X.index.to_numpy(),
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y.to_numpy(),
        )
    )

    X_train = X.loc[
        train_indices
    ]

    X_test = X.loc[
        test_indices
    ]

    y_train = y.loc[
        train_indices
    ]

    y_test = y.loc[
        test_indices
    ]

    print(
        f"Train rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train,
        y_train,
    )

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        prediction,
        probability,
    )

    print(
        "\n=== RANDOM FOREST TEST RESULTS ==="
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{_format_optional_metric(metrics['roc_auc'])}"
    )

    # Save model bundle
    model_bundle = {
        "model": model,
        "feature_names": list(
            X.columns
        ),
        "osnr_threshold_db":
            OSNR_THRESHOLD_DB,
        "random_state":
            RANDOM_STATE,
        "n_spans":
            N_SPANS,
    }

    joblib.dump(
        model_bundle,
        output_directory
        / "rf_model.joblib",
    )

    # Confusion matrix
    save_confusion_matrix(
        y_test,
        prediction,
        figures_directory
        / "rf_confusion_matrix.png",
    )

    # Feature importance
    importance = (
        save_feature_importance(
            model,
            list(X.columns),
            output_directory
            / "feature_importance.csv",
            figures_directory
            / "feature_importance.png",
        )
    )

    # Compare with baseline
    comparison_rows = (
        load_baseline_results(
            baseline_metrics_path
        )
    )

    comparison_rows.append(
        {
            "model":
                "random_forest",
            **metrics,
        }
    )

    comparison_df = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison_df.to_csv(
        output_directory
        / "comparison_metrics.csv",
        index=False,
    )

    save_comparison_plot(
        comparison_df,
        figures_directory
        / "model_comparison.png",
    )

    # Existing project configuration is reused for localization.
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    locations = localize_spans(
        df,
        config["data"],
        kept_index=pd.Index(
            test_indices
        ),
    )

    # Ensure locations align exactly with the test-row order.
    locations = locations.loc[
        test_indices
    ]

    prediction_df = pd.DataFrame(
        {
            "row_index":
                test_indices,

            "actual_label":
                np.where(
                    y_test.to_numpy() == 1,
                    "not healthy",
                    "healthy",
                ),

            "failure_probability":
                probability,

            "predicted_label":
                np.where(
                    prediction == 1,
                    "not healthy",
                    "healthy",
                ),

            "suspected_span":
                np.where(
                    prediction == 1,
                    locations[
                        "suspected_span"
                    ].to_numpy(),
                    "",
                ),

            "localization_score":
                np.where(
                    prediction == 1,
                    locations[
                        "localization_score"
                    ].to_numpy(),
                    0.0,
                ),
        }
    )

    prediction_df.to_csv(
        output_directory
        / "predictions.csv",
        index=False,
    )

    metrics_payload = {
        "summary": {
            "source":
                str(data_path),
            "rows_used":
                len(X),
            "train_rows":
                len(X_train),
            "test_rows":
                len(X_test),
            "healthy":
                int((y == 0).sum()),
            "not_healthy":
                int((y == 1).sum()),
            "number_of_features":
                X.shape[1],
        },
        "model":
            "random_forest",
        "test":
            metrics,
    }

    (
        output_directory
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        output_directory
        / "report.md",
        metrics,
        comparison_df,
        importance,
        len(X),
        len(X_train),
        len(X_test),
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  {output_directory / 'rf_model.joblib'}"
    )

    print(
        f"  {output_directory / 'metrics.json'}"
    )

    print(
        f"  {output_directory / 'comparison_metrics.csv'}"
    )

    print(
        f"  {output_directory / 'feature_importance.csv'}"
    )

    print(
        f"  {output_directory / 'predictions.csv'}"
    )

    print(
        f"  {output_directory / 'report.md'}"
    )

    print(
        f"  {figures_directory}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest using "
            "38 span-level optical features."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "data/sample/"
            "sample_link_data.csv"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "runs/random_forest"
        ),
    )

    parser.add_argument(
        "--baseline",
        default=(
            "runs/demo/"
            "metrics.json"
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "runs/demo/"
            "config_used.yaml"
        ),
    )

    parser.add_argument(
        "--trees",
        type=int,
        default=400,
    )

    args = parser.parse_args()

    train_random_forest(
        data_path=args.data,
        output_directory=args.out,
        baseline_metrics_path=
            args.baseline,
        config_path=args.config,
        n_estimators=args.trees,
    )


if __name__ == "__main__":
    main()
