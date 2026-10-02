"""
evaluate.py  -  THE EXAM MARKER

ROLE OF THIS FILE
-----------------
After a model has learned, we give it rows it has NEVER seen (the "test set") and
compare its answers with the truth. This file calculates the marks and writes a
readable report.

WHY SEVERAL MARKS INSTEAD OF JUST "ACCURACY"?
    Suppose only 2 links in 100 fail. A lazy model that ALWAYS says "healthy" is right
    98 times - 98% accuracy - yet it catches zero failures and is useless. So we also
    measure how well the model finds failures and how many false alarms it raises.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, proba):
    """
    ROLE: mark one model on one set of rows.

    y_true = the real answers, proba = the model's score for each row (0 to 1).
    We first turn scores into yes/no answers (score >= 0.5 means "not healthy").
    """
    y_pred = (proba >= 0.5).astype(int)
    both_classes = len(np.unique(y_true)) == 2     # ROC-AUC needs both kinds of rows to exist

    return {
        # ACCURACY: out of ALL rows, what share did we get right?
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # PRECISION: when we shout "failing!", how often are we right?
        #            (low precision = lots of false alarms)
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),

        # RECALL: out of the links that REALLY are failing, what share did we catch?
        #         (low recall = missed failures, the costly kind of mistake)
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),

        # F1: a single number that balances precision and recall (harsh if either is low).
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),

        # ROC-AUC: pick one failing and one healthy link at random - how often does the
        # model give the failing one the higher score? 1.0 = always, 0.5 = coin flip.
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,

        # CONFUSION MATRIX: a 2x2 table of counts.
        #   [[healthy called healthy,   healthy called failing ],
        #    [failing called healthy,   failing called failing ]]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _fmt(value):
    """ROLE: show a number with 4 decimals, or 'n/a' if it could not be calculated."""
    return "n/a" if value is None else f"{value:.4f}"


def write_report(path, summary, results, feature_names, models):
    """
    ROLE: write report.md, a human-readable summary of the run:
        * facts about the dataset
        * a table comparing the models on the test rows
        * the "weights" each model learned
    """
    lines = ["# Training report", ""]

    # --- section 1: dataset facts (how many rows, how many of each class, ...) ---
    lines += ["## Dataset", ""]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")

    # --- section 2: the comparison table ---
    lines += ["", "## Test-set performance", "",
              "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
              "|---|---|---|---|---|---|"]
    for name, res in results.items():
        t = res["test"]
        lines.append(f"| {name} | {_fmt(t['accuracy'])} | {_fmt(t['precision'])} | "
                     f"{_fmt(t['recall'])} | {_fmt(t['f1'])} | {_fmt(t['roc_auc'])} |")

    # --- section 3: what each model learned ---
    # A big positive weight means "when this feature goes up, the link looks more unhealthy".
    lines += ["", "## Learned linear boundaries (standardised feature space)", "",
              "Class 1 = not healthy. Score = intercept + sum(coef * standardised feature).", "",
              "| Model | Intercept | " + " | ".join(feature_names) + " |",
              "|---|---|" + "---|" * len(feature_names)]
    for name, model in models.items():
        coefs = " | ".join(f"{c:.4f}" for c in model.coef_)
        lines.append(f"| {name} | {model.intercept_:.4f} | {coefs} |")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
