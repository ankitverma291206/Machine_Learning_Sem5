# Implementation and Experimental Results

## System Overview

This project implements supervised machine learning for optical link failure
prediction and span-level failure localization.

The implementation contains two approaches:

1. A Stanford-inspired baseline using two aggregated end-to-end features.
2. A Random Forest extension using 38 individual span-level features.

Both approaches use the same synthetic 19-span optical-network telemetry.

## Dataset

The final experiment uses 5,000 telemetry samples.

Class distribution:

- 3,400 healthy samples
- 1,600 not-healthy samples

Each optical span contains:

- Actual amplifier gain
- Target amplifier gain
- Actual span loss
- Target span loss

For 19 spans, this produces 76 raw span measurements.

The dataset also contains:

- Timestamp
- OSNR

Therefore the raw dataset contains 78 columns.

A link is labelled not healthy when:

OSNR < 20 dB

OSNR is used to generate the ground-truth label and is not supplied as a
Random Forest input feature.

## Feature Engineering

For each span, two deviation features are calculated.

Gain deviation:

Actual Gain - Target Gain

Loss deviation:

Target Span Loss - Actual Span Loss

A negative value therefore represents an adverse deviation.

### Baseline Representation

The Stanford-inspired baseline sums all span-level deviations.

X1 is the total amplifier-gain deviation across all 19 spans.

X2 is the total span-loss deviation across all 19 spans.

This reduces the network to two features.

### Random Forest Representation

The Random Forest extension keeps every span separately.

The model receives:

- 19 gain-deviation features
- 19 loss-deviation features

Total input features: 38.

This preserves information about where abnormal behavior occurs in the
network instead of only preserving the end-to-end aggregate.

## Baseline Models

The baseline evaluates:

- Logistic Regression
- Gaussian Discriminant Analysis
- Linear SVM

These models operate on the two aggregated features.

## Random Forest Model

The Random Forest extension uses:

- 400 decision trees
- balanced class weighting
- square-root feature sampling at each split
- random state 42
- all available CPU cores during training

Random Forest was selected because it can model nonlinear relationships,
works effectively with numerical tabular data, does not require feature
standardization, and provides feature-importance information.

## Training and Evaluation

The dataset is divided using:

- 75% training data
- 25% test data
- stratified splitting
- random state 42

This results in:

- 3,750 training samples
- 1,250 test samples

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Experimental Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.32% | 97.73% | 97.00% | 97.37% | 99.78% |
| GDA | 95.28% | 99.71% | 85.50% | 92.06% | 99.78% |
| Linear SVM | 98.56% | 97.51% | 98.00% | 97.76% | 99.77% |
| Random Forest | 97.92% | 93.90% | 100.00% | 96.85% | 99.65% |

Linear SVM achieves the highest overall accuracy and F1-score.

Random Forest achieves the highest recall of 100%, meaning that every
not-healthy sample in this particular test split was detected.

The lower Random Forest precision shows that this comes with additional
false-positive alarms.

## Failure Localization

Random Forest performs binary link-health prediction.

When a sample is predicted as not healthy, the system separately evaluates
the adverse gain and loss deviation of every span.

The span with the largest abnormal deviation is reported as the most
suspicious span.

The localization deviation score represents abnormality magnitude and is
not a probability.

## Interactive Dashboard

A Streamlit dashboard is provided for live demonstration.

It displays:

- Predicted health status
- Failure probability
- Ground-truth status for evaluation
- OSNR
- Most suspicious span
- Localization deviation score
- Span-level deviation graph
- Random Forest feature importance
- Comparison between all four models
- Raw telemetry for the selected sample

The ground-truth filter in the dashboard is used only to select convenient
examples during demonstration and is not used as a model input.

## Main Observation

The Random Forest does not outperform the linear baseline on every metric.

Its main contribution is instead that it:

- preserves span-level information
- supports nonlinear feature interactions
- provides feature importance
- achieves perfect failure recall on this test split
- supports more informative span-level diagnostics

The project therefore demonstrates a trade-off between compact aggregated
linear classification and richer span-level nonlinear modelling.

## Limitations

Current limitations include:

- synthetic rather than production network telemetry
- binary healthy/not-healthy classification
- localization based on a diagnostic heuristic
- no supervised multiclass span-localization model
- no time-series future-failure forecasting
- fixed OSNR threshold of 20 dB

## Future Work

Possible improvements include:

- evaluation on real optical-network telemetry
- supervised multiclass failure localization
- temporal failure forecasting
- failure-type classification
- hyperparameter optimization
- SHAP-based explainability
- comparison with XGBoost and nonlinear SVM
- integration with automatic rerouting or preventive-maintenance systems
