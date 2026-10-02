"""
config.py  -  THE INSTRUCTION SHEET

ROLE OF THIS FILE
-----------------
Every training run needs settings: which columns are the features? which column
is the label? how much data to keep for testing? This file defines:
  * DEFAULT_CONFIG  - sensible default for every setting
  * load_config()   - combines defaults + your YAML file + command-line flags
  * validate_config() - checks the settings make sense and explains mistakes

Settings live in one nested dictionary (a "dictionary" is Python's labelled box:
{"name": value}). Anything you don't set falls back to the default, so a YAML file
only has to mention the things you want to change.
"""

import copy                  # copy.deepcopy makes a fully independent copy of a dictionary
from pathlib import Path     # Path is a friendly way to handle file paths on any computer

import yaml                  # reads .yaml settings files


class ConfigError(ValueError):
    """
    ROLE: a special kind of error used ONLY for "your settings don't make sense".
    The command-line tool catches it and prints a short, friendly message instead of
    a scary technical crash.
    """


# ---------------------------------------------------------------------------
# DEFAULT SETTINGS
# Every key is explained in README.md under "Configuration reference".
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "data": {
        # ===== LABEL: "what answer are we trying to predict?" =====
        # Name of the column holding the answer (0 = healthy, 1 = not healthy).
        "label_column": "label",
        # If the label column holds words (like "failure"), list the words that mean
        # "NOT healthy". Every other word is treated as healthy.
        "positive_labels": None,
        # The opposite way round: list the words that mean "healthy"; every other
        # word is treated as "not healthy". Great for datasets with many failure types.
        "healthy_labels": None,
        # A third way: make the label from a number. Example: OSNR (signal quality)
        # below 20 means "not healthy".  Written as {column: osnr_db, below: 20.0}.
        "label_from_threshold": None,

        # ===== FEATURES: "what information does the model look at?" =====
        # Columns that are already the final numbers you want to use.
        "feature_columns": [],
        # Features the tool should CALCULATE from raw measurements. Each one looks like
        #   {name: X1, minuend: "amp_gain_*", subtrahend: "target_gain_*"}
        # and means: for every span, take (measured - target), then add them all up.
        # The "*" is a wildcard meaning "any ending", e.g. amp_gain_01, amp_gain_02...
        "derived_features": [],

        # ===== CLEANING =====
        # What to do about empty cells:
        #   "interpolate" = fill using the neighbouring rows (draw a straight line between them)
        #   "median"      = fill with the column's middle value
        #   "drop"        = throw the row away
        "missing": "interpolate",
        # Optional column to sort by first (like a timestamp), so "neighbouring rows"
        # really are neighbours in time.
        "sort_by": None,
    },
    "split": {
        "test_size": 0.25,   # keep 25% of rows hidden for the final exam (75% to learn from)
        "seed": 42,          # fixes the "random" shuffle so you get the same split every time
        "stratify": True,    # keep the healthy:unhealthy ratio the same in both parts
    },
    # Rescale each feature so it is centred on 0 and has a typical size of 1.
    # Models learn faster and more reliably this way.
    "standardize": True,
    # Which brains to train. Allowed: logistic_regression, gda, linear_svm.
    "models": ["logistic_regression", "gda", "linear_svm"],
    # Extra knobs for individual models, e.g. {"linear_svm": {"C": 10}}. Usually empty.
    "model_params": {},
}


def deep_merge(base, override):
    """
    ROLE: combine two settings dictionaries, where `override` wins.

    "Deep" means it also looks INSIDE nested dictionaries. Example:
        base     = {"split": {"test_size": 0.25, "seed": 42}}
        override = {"split": {"seed": 7}}
        result   = {"split": {"test_size": 0.25, "seed": 7}}   <- only seed changed
    A shallow merge would have thrown away test_size, which we don't want.
    """
    merged = copy.deepcopy(base)                         # start from a copy so `base` stays untouched
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)  # both are boxes -> merge the boxes
        else:
            merged[key] = copy.deepcopy(value)            # otherwise simply replace the value
    return merged


def load_config(path=None, overrides=None):
    """
    ROLE: build the FINAL settings for a run.

    Priority, lowest to highest:   defaults  <  YAML file  <  command-line flags.
    A higher level always wins over a lower one.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)                   # 1) start with the defaults
    if path is not None:                                  # 2) layer the YAML file on top, if given
        with open(Path(path), "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg = deep_merge(cfg, overrides or {})                # 3) layer the command-line flags on top
    validate_config(cfg)                                  # 4) make sure the result is sensible
    return cfg


def validate_config(cfg):
    """
    ROLE: the safety check. Stops the run EARLY with a clear message if the settings
    are contradictory or incomplete, instead of crashing in a confusing way later.
    """
    d = cfg["data"]

    # "missing" must be one of the three known strategies.
    if d["missing"] not in {"interpolate", "median", "drop"}:
        raise ConfigError("data.missing must be one of: interpolate, median, drop")

    # The model needs SOMETHING to look at.
    if not d["feature_columns"] and not d["derived_features"]:
        raise ConfigError(
            "No features defined. Set data.feature_columns (or pass --features) "
            "and/or data.derived_features. Run `python -m linkfail inspect --data FILE` "
            "to see the available columns."
        )

    # Every calculated feature needs a name and the two column patterns to subtract.
    for item in d["derived_features"]:
        missing = {"name", "minuend", "subtrahend"} - set(item)
        if missing:
            raise ConfigError(f"derived_features entry {item} lacks keys: {sorted(missing)}")

    # Only ONE way of building labels may be active, otherwise it's unclear which to use.
    strategies = [k for k in ("positive_labels", "healthy_labels", "label_from_threshold") if d[k]]
    if len(strategies) > 1:
        raise ConfigError(f"Use only one of positive_labels / healthy_labels / "
                          f"label_from_threshold (got {strategies}).")

    # A threshold rule needs a column and exactly one direction (below OR above).
    thr = d["label_from_threshold"]
    if thr:
        if "column" not in thr or not ({"below", "above"} & set(thr)) or {"below", "above"} <= set(thr):
            raise ConfigError("label_from_threshold needs `column` and exactly one of `below` / `above`.")

    # The test share must be a proper fraction (not 0, not 1, not 150%).
    if not 0.0 < cfg["split"]["test_size"] < 1.0:
        raise ConfigError("split.test_size must be between 0 and 1 (e.g. 0.25).")

    # Only models we actually have can be requested.
    valid_models = {"logistic_regression", "gda", "linear_svm"}
    unknown = set(cfg["models"]) - valid_models
    if unknown or not cfg["models"]:
        raise ConfigError(f"models must be a non-empty subset of {sorted(valid_models)}; got {cfg['models']}")
