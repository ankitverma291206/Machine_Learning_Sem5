"""
plotting.py  -  THE ARTIST

ROLE OF THIS FILE
-----------------
Turns the results into pictures (PNG files) that are easy to understand and to put
in a presentation or README. Pictures are saved to files and never opened in a
window, so the tool also works on servers that have no screen.
"""

import matplotlib
matplotlib.use("Agg")       # "Agg" = draw straight into image files, no window needed (must come first)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# One fixed colour per model so a model looks the same in every figure.
# (These colours stay distinguishable for people with colour-blindness.)
MODEL_COLORS = {"logistic_regression": "#0072B2", "gda": "#D55E00", "linear_svm": "#009E73"}
CLASS_COLORS = {0: "#56B4E9", 1: "#D55E00"}                 # blue = healthy, orange = not healthy
CLASS_NAMES = {0: "healthy", 1: "not healthy"}

# Global look-and-feel: sharper images, readable text, light grid, no clutter on the frame.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, path):
    """ROLE: tidy the layout, save the picture to a file, then release its memory."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)       # closing frees memory, which matters if many figures are made


def plot_confusion_matrices(results, path):
    """
    ROLE: draw one small 2x2 table (as a coloured grid) per model, showing how many
    healthy and failing links it classified correctly or incorrectly. The diagonal
    (top-left, bottom-right) holds the correct answers - darker is more.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)   # one panel per model
    for ax, (name, res) in zip(axes[0], results.items()):
        cm = np.array(res["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")                          # colour each cell by its count
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                # Write the count in each cell; white text on dark cells, black on light cells.
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["healthy", "not healthy"])
        ax.set_yticks([0, 1], ["healthy", "not healthy"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(name)
    _save(fig, path)


def plot_roc_curves(models, X_test, y_test, path):
    """
    ROLE: draw ROC curves. Each curve shows the trade-off between catching more real
    failures (up) and raising more false alarms (right). A curve hugging the top-left
    corner is excellent; the dashed diagonal is the performance of pure guessing.
    """
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test))   # false-alarm rate vs catch rate
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), lw=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_metric_comparison(results, path):
    """
    ROLE: draw grouped bars so you can compare the models on accuracy, precision,
    recall and F1 at a single glance. Each bar has its value printed above it.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = list(results)
    width = 0.8 / len(names)                  # each model's bar gets an equal slice of the group
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(names):
        vals = [results[name]["test"][m] for m in metrics]
        bars = ax.bar(np.arange(len(metrics)) + i * width, vals, width,
                      label=name, color=MODEL_COLORS.get(name))
        for bar, v in zip(bars, vals):        # print the exact number above each bar
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(metrics)) + width * (len(names) - 1) / 2, metrics)
    # Start the vertical axis near the lowest score so small differences are visible.
    lowest = min(results[n]["test"][m] for n in names for m in metrics)
    ax.set_ylim(max(0.0, lowest - 0.05), 1.02)
    ax.set_title("Test-set metrics by model")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_decision_boundaries(models, standardizer, X_raw, y, feature_names, path):
    """
    ROLE: draw the "map" the model learned. Only possible with exactly TWO features
    (one for each axis).

    Each dot is a link. The black line is where the model switches from "healthy" to
    "not healthy" (its score is exactly 0.5). The tinted area is the region the model
    calls "not healthy". Dots on the wrong side of the line are its mistakes.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)

    # Build a fine grid of points covering the whole chart (with a 5% margin) ...
    pad = 0.05 * (X_raw.max(axis=0) - X_raw.min(axis=0))
    lo, hi = X_raw.min(axis=0) - pad, X_raw.max(axis=0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    for ax, (name, model) in zip(axes[0], models.items()):
        # ... ask the model to score every grid point, then draw the 0.5 contour line.
        score = model.predict_proba(standardizer.transform(grid)).reshape(gx.shape)
        ax.contourf(gx, gy, score, levels=[0.5, 1.0], colors=[CLASS_COLORS[1]], alpha=0.12)  # tinted "not healthy" zone
        ax.contour(gx, gy, score, levels=[0.5], colors="black", linewidths=1.5)               # the boundary line
        for cls in (0, 1):                                   # plot the real links on top
            pts = X_raw[y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.65, color=CLASS_COLORS[cls],
                       marker="o" if cls == 0 else "x", label=CLASS_NAMES[cls])
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(name)
        ax.legend(fontsize=7, loc="best")
    _save(fig, path)
