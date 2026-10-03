# Baseline vs Random Forest Extension

## Overview

The project compares a Stanford-inspired baseline approach with a
span-level Random Forest extension for optical link failure prediction.

## Common Network Setup

Both approaches operate on the same synthetic optical-network telemetry.

- 19 optical spans
- Actual amplifier gain
- Target amplifier gain
- Actual span loss
- Target span loss
- OSNR
- Healthy / not-healthy classification
- Failure condition: OSNR < 20 dB

The final experiment uses:

- 5,000 samples
- 3,400 healthy samples
- 1,600 not-healthy samples
- 75% training split
- 25% testing split
- Random seed 42

## Baseline Feature Representation

For every span:

Gain Difference:

Actual Gain - Target Gain

Loss Difference:

Target Span Loss - Actual Span Loss

The baseline then aggregates all spans:

X1 = sum of gain deviations across all 19 spans

X2 = sum of loss deviations across all 19 spans

This reduces an entire 19-span optical link to two features.

## Baseline Models

The baseline evaluates:

- Logistic Regression
- Gaussian Discriminant Analysis
- Linear SVM

These models classify the link as healthy or not healthy.

## Limitation of Aggregation

Aggregation provides a compact and effective representation for link-level
classification but removes information about which individual span caused a
large deviation.

Different span-level failure patterns can potentially produce similar
aggregate values.

## Random Forest Extension

The Random Forest extension does not aggregate span measurements.

It retains:

- 19 amplifier-gain deviations
- 19 span-loss deviations

Total input features:

38

This preserves individual span information and allows the model to learn
nonlinear interactions between telemetry measurements.

## Why Random Forest

Random Forest was selected because it:

- models nonlinear relationships
- works well with numerical tabular data
- does not require feature scaling
- reduces overfitting compared with a single decision tree
- supports class weighting
- provides feature importance
- can handle the 38-dimensional span-level representation efficiently

The implementation uses 400 trees and balanced class weighting.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.32% | 97.73% | 97.00% | 97.37% | 99.78% |
| GDA | 95.28% | 99.71% | 85.50% | 92.06% | 99.78% |
| Linear SVM | 98.56% | 97.51% | 98.00% | 97.76% | 99.77% |
| Random Forest | 97.92% | 93.90% | 100.00% | 96.85% | 99.65% |

Linear SVM achieves the highest accuracy and F1-score.

Random Forest achieves the highest recall, detecting every unhealthy sample
in the test split, although with a larger number of false-positive alarms.

## Localization

Random Forest performs binary failure prediction.

When a failure is predicted, localization is handled separately using the
existing span-deviation localization mechanism.

The span with the strongest adverse gain/loss deviation is reported as the
most suspicious span.

The localization deviation score is an abnormality score, not a probability.

## Main Difference

Baseline:

19 spans -> aggregate to 2 features -> linear classifiers

Random Forest extension:

19 spans -> preserve 38 span-level features -> nonlinear ensemble classifier

The extension therefore focuses on retaining diagnostic information and
supporting nonlinear relationships rather than simply trying to maximize
binary classification accuracy.
