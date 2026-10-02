"""
cli.py  -  THE CONTROL PANEL

ROLE OF THIS FILE
-----------------
"CLI" = Command Line Interface: the commands you type in a terminal. This file
connects those commands to the rest of the program. It contains no maths itself; it
just calls the other files in the right order.

THE FOUR COMMANDS
    inspect   look at a data file (column names, empty cells) before you start
    train     learn from a data file, test the models, save everything
    predict   use a saved model to label NEW links
    demo      do a complete run on made-up data, to see the tool working
"""

import argparse        # reads what you typed after the program name
import json            # saves the results as a .json file
import sys             # lets us print to the error channel and set the exit code
import time            # stopwatch for "how long did training take"
from pathlib import Path

import joblib          # saves/loads Python objects (our trained models) to/from a file
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split    # splits rows into "study" and "exam" groups

from . import __version__
from .config import ConfigError, deep_merge, load_config
from .data import DataError, Standardizer, load_table, localize_spans, prepare_dataset
from .evaluate import compute_metrics, write_report
from .models import build_model
from .plotting import (plot_confusion_matrices, plot_decision_boundaries,
                       plot_metric_comparison, plot_roc_curves)
from .sample_data import SAMPLE_CONFIG, generate_sample_data


# ===========================================================================
# COMMAND 1: inspect
# ===========================================================================
def cmd_inspect(args):
    """
    ROLE: show the user what is inside a data file, so they know which column names
    to write in their settings. Prints the size, each column's type, how many cells
    are empty, an example value, and which columns look like they could be labels.
    """
    df = load_table(args.data)
    print(f"{args.data}: {len(df)} rows x {df.shape[1]} columns\n")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str),                 # kind of data in the column
                         "missing": df.isna().sum(),                      # number of empty cells
                         "example": df.iloc[0].astype(str) if len(df) else ""})   # first row's value
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(info.to_string())
    # A label column usually has only a few distinct values (like 0/1 or a handful of failure names).
    print("\nTip: low-cardinality columns are label candidates:")
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 8:
            print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")


# ===========================================================================
# COMMAND 2: train
# ===========================================================================
def run_training(df, cfg, out_dir, source_name="in-memory data"):
    """
    ROLE: THE MAIN EVENT. Does the whole learning process and saves the results.
    It is a separate function (not buried inside cmd_train) so that the demo and the
    automated tests can call it directly.

    The steps are numbered below; they match the "What happens during train" section
    of docs/HOW_IT_WORKS.md.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)     # create the output folders

    # ---- STEP 1: clean the table, build the features and the 0/1 answers ----
    prepared = prepare_dataset(df, cfg["data"], with_labels=True)
    X_df, y = prepared.X, prepared.y
    if len(np.unique(y)) < 2:                                    # need both healthy and failing examples
        raise DataError("After labelling, only one class remains. Check the label settings.")
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)                               # plain number grid for the models
    print(f"Prepared {len(X)} rows ({prepared.n_dropped} dropped), features: {feature_names}")
    print(f"Class balance: {int((y == 0).sum())} healthy / {int((y == 1).sum())} not healthy")

    # ---- STEP 2: split into a "study" part (train) and a hidden "exam" part (test) ----
    sp = cfg["split"]
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=sp["test_size"], random_state=sp["seed"],
            stratify=y if sp["stratify"] else None)              # stratify = same healthy:failing ratio in both parts
    except ValueError as exc:      # happens e.g. when a class has too few rows to share between the parts
        raise DataError(f"Could not split the data: {exc}") from exc

    # ---- STEP 3: rescale the numbers (learned from the study part only) ----
    scaler = Standardizer(enabled=cfg["standardize"]).fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

    # ---- STEP 4: teach every requested model, then mark it on both parts ----
    models, results = {}, {}
    for name in cfg["models"]:
        model = build_model(name, cfg["model_params"].get(name))
        t0 = time.perf_counter()                                 # start the stopwatch
        model.fit(Z_tr, y_tr)                                    # <-- the actual learning happens here
        results[name] = {
            "train": compute_metrics(y_tr, model.predict_proba(Z_tr)),   # marks on the data it studied
            "test": compute_metrics(y_te, model.predict_proba(Z_te)),    # marks on the hidden exam (the honest ones)
            "fit_seconds": round(time.perf_counter() - t0, 4),
        }
        models[name] = model
        t = results[name]["test"]
        print(f"  {name:<20} test acc={t['accuracy']:.4f}  precision={t['precision']:.4f}  "
              f"recall={t['recall']:.4f}  f1={t['f1']:.4f}")

    # ---- STEP 5: save everything to disk ----
    # model.joblib is a single "package" containing all that is needed to label new
    # data later: the models, the rescaler, the feature names and the settings.
    joblib.dump({"models": models, "scaler": scaler, "feature_names": feature_names,
                 "config": cfg, "version": __version__}, out_dir / "model.joblib")
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as fh:     # a record of the exact settings
        yaml.safe_dump(cfg, fh, sort_keys=False)

    summary = {"source": source_name, "rows used": len(X), "rows dropped": prepared.n_dropped,
               "train / test rows": f"{len(X_tr)} / {len(X_te)}",
               "healthy / not healthy": f"{int((y == 0).sum())} / {int((y == 1).sum())}",
               "features": ", ".join(feature_names)}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:        # all the marks, machine-readable
        json.dump({"summary": summary, "results": results}, fh, indent=2)
    write_report(out_dir / "report.md", summary, results, feature_names, models)   # the human-readable report

    # ---- STEP 6: draw the pictures ----
    fig_dir = out_dir / "figures"
    plot_confusion_matrices(results, fig_dir / "confusion_matrices.png")
    plot_metric_comparison(results, fig_dir / "metric_comparison.png")
    plot_roc_curves(models, Z_te, y_te, fig_dir / "roc_curves.png")
    if len(feature_names) == 2:        # a flat chart can only show two features
        plot_decision_boundaries(models, scaler, X_te, y_te, feature_names,
                                 fig_dir / "decision_boundaries.png")
    print(f"\nSaved model, metrics, report and figures to: {out_dir}/")
    return results


def _overrides_from_args(args):
    """
    ROLE: convert the flags you typed (like --features X1 X2) into a small settings
    dictionary, so they can override whatever the YAML file says.
    """
    ov = {"data": {}, "split": {}}
    if args.features:
        ov["data"]["feature_columns"] = args.features
    if args.label:
        ov["data"]["label_column"] = args.label
    if args.positive_labels:
        ov["data"]["positive_labels"] = args.positive_labels
    if args.healthy_labels:
        ov["data"]["healthy_labels"] = args.healthy_labels
    if args.test_size is not None:
        ov["split"]["test_size"] = args.test_size
    if args.seed is not None:
        ov["split"]["seed"] = args.seed
    if args.models:
        ov["models"] = args.models
    return ov


def cmd_train(args):
    """ROLE: the `train` command: read settings, read the file, then run the training."""
    cfg = load_config(args.config, _overrides_from_args(args))
    df = load_table(args.data)
    run_training(df, cfg, args.out, source_name=str(args.data))


# ===========================================================================
# COMMAND 3: predict
# ===========================================================================
def cmd_predict(args):
    """
    ROLE: use a model that was trained earlier to label brand-new links.

    It re-opens the saved package, prepares the new data EXACTLY the way the training
    data was prepared (same settings, same rescaling), and writes a CSV with a score
    and a verdict for each row.
    """
    bundle = joblib.load(args.model)                 # open the saved package
    cfg = bundle["config"]                           # the settings used during training
    available = bundle["models"]
    # Unless you choose one, use logistic regression (or the first model available).
    name = args.model_name or ("logistic_regression" if "logistic_regression" in available
                               else next(iter(available)))
    if name not in available:
        raise ConfigError(f"Model '{name}' not in bundle. Available: {sorted(available)}")

    df = load_table(args.data)
    # with_labels=False: new links have no known answer, so no label column is needed.
    prepared = prepare_dataset(df, cfg["data"], with_labels=False)
    if prepared.n_dropped:
        print(f"Warning: {prepared.n_dropped} rows skipped (missing values); "
              f"see the row_index column to match results to your file.", file=sys.stderr)

    Z = bundle["scaler"].transform(prepared.X.to_numpy(dtype=float))     # same rescaling as training
    score = available[name].predict_proba(Z)                             # probability of "not healthy"
    out = pd.DataFrame({"row_index": prepared.kept_index,                # which row of YOUR file this is
                        "unhealthy_score": score.round(5),
                        "prediction": np.where(score >= 0.5, "not healthy", "healthy")})
    if args.localize:
        locations = localize_spans(df, cfg["data"], prepared.kept_index)
        # A location is useful only for an alerted link; leave healthy rows unlabelled.
        unhealthy = score >= 0.5
        out["suspected_span"] = np.where(unhealthy, locations["suspected_span"].to_numpy(), "")
        out["localization_score"] = np.where(unhealthy, locations["localization_score"].to_numpy(), 0.0).round(5)
    out.to_csv(args.output, index=False)
    flagged = int((score >= 0.5).sum())
    print(f"Model '{name}': {flagged} of {len(out)} rows flagged as not healthy -> {args.output}")


# ===========================================================================
# COMMAND 4: demo
# ===========================================================================
def cmd_demo(args):
    """
    ROLE: a "try it now" button. Makes up measurement data, saves it as a CSV file,
    then trains on it exactly as a real user would train on their own file.
    """
    data_path = Path(args.data_out)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_data(n_rows=args.rows)                  # invent the measurements
    df.to_csv(data_path, index=False)                            # save them so you can open the file
    print(f"Wrote synthetic telemetry ({len(df)} rows) to {data_path}\n")
    cfg = deep_merge(load_config(None, SAMPLE_CONFIG), {})       # defaults + the demo's settings
    run_training(load_table(data_path), cfg, args.out, source_name=f"{data_path} (synthetic)")


# ===========================================================================
# THE COMMAND-LINE PARSER
# ===========================================================================
def build_parser():
    """
    ROLE: describe which commands and flags exist, so argparse can understand what the
    user typed (and generate the --help text automatically).
    """
    p = argparse.ArgumentParser(prog="linkfail",
                                description="Link-failure prediction with linear classifiers.")
    p.add_argument("--version", action="version", version=f"linkfail {__version__}")
    sub = p.add_subparsers(dest="command", required=True)      # one sub-command is mandatory

    # --- inspect ---
    sp = sub.add_parser("inspect", help="show columns, dtypes and label candidates of a dataset")
    sp.add_argument("--data", required=True)
    sp.set_defaults(func=cmd_inspect)                          # "when this command is chosen, run cmd_inspect"

    # --- train ---
    sp = sub.add_parser("train", help="train, evaluate and save models")
    sp.add_argument("--data", required=True, help="CSV / TSV / Parquet file")
    sp.add_argument("--config", help="YAML settings (see configs/)")
    sp.add_argument("--out", default="runs/latest", help="output folder (default runs/latest)")
    sp.add_argument("--features", nargs="+", help="feature column names (shortcut, no YAML needed)")
    sp.add_argument("--label", help="label column name")
    sp.add_argument("--positive-labels", nargs="+", help="label values that mean 'not healthy'")
    sp.add_argument("--healthy-labels", nargs="+", help="label values that mean 'healthy'")
    sp.add_argument("--test-size", type=float)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--models", nargs="+", help="subset of: logistic_regression gda linear_svm")
    sp.set_defaults(func=cmd_train)

    # --- predict ---
    sp = sub.add_parser("predict", help="score new data with a saved model")
    sp.add_argument("--model", required=True, help="model.joblib from a training run")
    sp.add_argument("--data", required=True)
    sp.add_argument("--output", default="predictions.csv")
    sp.add_argument("--model-name", help="which of the trained models to use")
    sp.add_argument("--localize", action="store_true",
                    help="add the most adverse derived-feature span for each alerted row")
    sp.set_defaults(func=cmd_predict)

    # --- demo ---
    sp = sub.add_parser("demo", help="run the whole pipeline on synthetic data")
    sp.add_argument("--rows", type=int, default=1200)
    sp.add_argument("--data-out", default="data/sample/sample_link_data.csv")
    sp.add_argument("--out", default="runs/demo")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    """
    ROLE: the program's front door. Reads what you typed, runs the chosen command, and
    turns EXPECTED problems (bad settings, bad data) into a short message plus exit
    code 2, instead of a long, scary technical error.
    """
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
