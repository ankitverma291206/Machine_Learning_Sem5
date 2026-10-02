"""
data.py  -  THE PREPARATION ROOM

ROLE OF THIS FILE
-----------------
Real-world tables are messy: empty cells, text where numbers should be, hundreds of
columns. A model only understands a small, tidy grid of numbers. This file turns
your messy table into exactly that:

    your table  -->  [ X: a few feature numbers per row ]  +  [ y: 0/1 answer per row ]

The steps, in order (all done by prepare_dataset at the bottom):
    1. sort the rows (e.g. by time)
    2. make sure the columns we need are numbers
    3. fill in or remove empty cells
    4. calculate the features   (the paper's "Eq. 1" and "Eq. 2")
    5. work out the 0/1 labels
    6. throw away rows that are still unusable
"""

import fnmatch                      # matches names against wildcard patterns like "amp_gain_*"
import re                           # "regular expressions": used here only to split names into text/number parts
from dataclasses import dataclass   # a quick way to make a small "container" class
from pathlib import Path

import numpy as np                  # fast number arrays
import pandas as pd                 # tables (called DataFrames)


class DataError(ValueError):
    """ROLE: a special error meaning "your DATA doesn't match your settings" (shown as a friendly message)."""


# ===========================================================================
# PART 1 - READING A FILE
# ===========================================================================
def load_table(path):
    """
    ROLE: open a data file and return it as a table.
    Works for CSV (comma separated), TSV (tab separated) and Parquet files.
    """
    path = Path(path)
    if not path.exists():                                   # friendly message for a typo in the path
        raise DataError(f"Data file not found: {path}")
    suffix = path.suffix.lower()                            # file ending, e.g. ".csv"
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)                        # needs the optional "pyarrow" package
    raise DataError(f"Unsupported file type '{suffix}'. Use .csv, .tsv or .parquet.")


# ===========================================================================
# PART 2 - PICKING COLUMNS BY NAME OR BY PATTERN
# ===========================================================================
def _natural_key(text):
    """
    ROLE: a helper that sorts names the way a HUMAN would.

    Computers sort text letter by letter, so "span_10" lands before "span_2".
    This splits a name into text and number chunks and compares the numbers as
    numbers, so we get span_1, span_2, ..., span_10. The order matters because we
    later pair "measured" columns with "target" columns by their position.
    """
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def expand_columns(columns, pattern):
    """
    ROLE: turn a column name OR a wildcard pattern into a list of real column names.

        "osnr_db"      ->  ["osnr_db"]                          (exact name)
        "amp_gain_*"   ->  ["amp_gain_01", "amp_gain_02", ...]  (every name that fits)
    """
    columns = list(columns)
    if pattern in columns:                 # an exact name always wins
        return [pattern]
    # Keep every column whose name fits the pattern, in human order.
    matched = sorted((c for c in columns if fnmatch.fnmatchcase(c, pattern)), key=_natural_key)
    if not matched:                        # nothing fits -> tell the user what IS available
        preview = ", ".join(columns[:15]) + (" ..." if len(columns) > 15 else "")
        raise DataError(f"No column matches '{pattern}'. Available columns: {preview}")
    return matched


def resolve_feature_plan(columns, data_cfg):
    """
    ROLE: translate the settings into a concrete "shopping list" of features.

    Each item says: this feature is called NAME, and it is computed from the "plus"
    columns and (optionally) the "minus" columns.
        * no "minus"  -> the feature is simply that one column, unchanged
        * with "minus" -> feature = sum over pairs of (plus_column - minus_column)
    """
    plan = []
    # --- features that are already final columns ---
    for col in data_cfg["feature_columns"]:
        found = expand_columns(columns, col)
        if len(found) != 1:      # a plain feature must be exactly one column
            raise DataError(f"feature_columns entry '{col}' must be ONE column; "
                            f"use derived_features for wildcard patterns.")
        plan.append({"name": col, "plus": found, "minus": None})
    # --- features we calculate from raw measurements ---
    for item in data_cfg["derived_features"]:
        plus = expand_columns(columns, item["minuend"])        # e.g. all the measured amp_gain columns
        minus = expand_columns(columns, item["subtrahend"])    # e.g. all the matching target_gain columns
        if len(plus) != len(minus):                            # they must pair up one-to-one
            raise DataError(
                f"Feature '{item['name']}': '{item['minuend']}' matches {len(plus)} columns but "
                f"'{item['subtrahend']}' matches {len(minus)}. They must pair up one-to-one."
            )
        plan.append({"name": item["name"], "plus": plus, "minus": minus})
    return plan


# ===========================================================================
# PART 3 - LABELS (the "right answers" the model learns from)
# ===========================================================================
def _as_text(series):
    """
    ROLE: a helper that turns a column into text so values can be compared safely.

    Settings files and CSV files can disagree about types (the number 1 versus the
    word "1" versus 1.0). Converting everything to text, and writing whole numbers
    without ".0", makes the comparison dependable.
    """
    if pd.api.types.is_numeric_dtype(series):
        as_float = series.astype(float)
        all_whole = as_float.dropna().apply(float.is_integer).all()   # are there no decimals at all?
        if all_whole:
            return as_float.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)   # 1.0 -> "1"
    return series.map(lambda v: str(v).strip() if pd.notna(v) else np.nan)


def build_labels(df, data_cfg):
    """
    ROLE: decide, for every row, whether the link is healthy (0) or not healthy (1).

    Returns TWO things:
        y      True where the link is NOT healthy
        valid  False where the label itself is missing (those rows can't be used to learn)
    """
    thr = data_cfg["label_from_threshold"]

    # ---- Way 1: compare a number against a limit (e.g. OSNR below 20 dB) ----
    if thr:
        col = expand_columns(df.columns, thr["column"])[0]
        values = pd.to_numeric(df[col], errors="coerce")           # unreadable cells become "missing"
        valid = values.notna()
        y = values < thr["below"] if "below" in thr else values > thr["above"]
        return y.fillna(False), valid

    # ---- Ways 2-4 need an actual label column ----
    col = data_cfg["label_column"]
    if col not in df.columns:
        raise DataError(f"Label column '{col}' not found. Set data.label_column "
                        f"(or pass --label). Columns: {list(df.columns)[:15]}")
    series = df[col]
    valid = series.notna()
    text = _as_text(series)

    if data_cfg["positive_labels"]:
        # Way 2: you told us which words mean "not healthy".
        wanted = {str(v) for v in data_cfg["positive_labels"]}
        y = text.isin(wanted)
    elif data_cfg["healthy_labels"]:
        # Way 3: you told us which words mean "healthy"; everything else is "not healthy".
        healthy = {str(v) for v in data_cfg["healthy_labels"]}
        y = ~text.isin(healthy)
    else:
        # Way 4: no hints given, so the column must ALREADY be just 0s and 1s.
        observed = set(text.dropna().unique())
        if not observed <= {"0", "1"}:
            raise DataError(
                f"Label column '{col}' has values {sorted(observed)[:10]}, not just 0/1. "
                f"Tell me which mean 'unhealthy' with data.positive_labels, or which mean "
                f"'healthy' with data.healthy_labels."
            )
        y = text == "1"
    return y.fillna(False), valid


# ===========================================================================
# PART 4 - THE FULL PREPARATION PIPELINE
# ===========================================================================
@dataclass
class Prepared:
    """
    ROLE: a small container that carries the results of preparation together.
    (A @dataclass is Python shorthand for a class that just holds named values.)
    """
    X: pd.DataFrame          # the feature table: one column per feature, one row per sample
    y: "np.ndarray | None"   # the 0/1 answers (None when preparing new data without labels)
    kept_index: pd.Index     # the ORIGINAL row numbers that survived cleaning
    n_dropped: int           # how many rows had to be removed


def prepare_dataset(df, data_cfg, with_labels=True):
    """
    ROLE: the whole preparation line in one function.

    `with_labels=True`  -> training: we need the right answers too.
    `with_labels=False` -> predicting on new data: no answers exist, only features.
    Using this ONE function for both guarantees features are built identically.
    """
    df = df.copy()                        # work on a copy so the caller's table is never changed
    n_rows_in = len(df)

    # ---- Step 1: sort (so "neighbouring rows" are neighbours in time) ----
    if data_cfg["sort_by"]:
        if data_cfg["sort_by"] not in df.columns:
            raise DataError(f"sort_by column '{data_cfg['sort_by']}' not found.")
        df = df.sort_values(data_cfg["sort_by"], kind="stable")

    # ---- Step 2: find exactly which raw columns the features need ----
    plan = resolve_feature_plan(df.columns, data_cfg)
    raw_cols = sorted({c for p in plan for c in p["plus"] + (p["minus"] or [])})

    # ---- Step 3: force those columns to be numbers; junk like "N/A" becomes "missing" ----
    df[raw_cols] = df[raw_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    # ---- Step 4: deal with missing cells ----
    keep = pd.Series(True, index=df.index)        # a yes/no flag per row: "do we keep it?"
    strategy = data_cfg["missing"]
    if strategy == "interpolate":
        # Fill a gap with a straight line between the values above and below it.
        # limit_direction="both" also fills gaps at the very top or bottom.
        df[raw_cols] = df[raw_cols].interpolate(method="linear", limit_direction="both")
    elif strategy == "median":
        df[raw_cols] = df[raw_cols].fillna(df[raw_cols].median())   # fill with each column's middle value
    else:  # "drop"
        keep &= df[raw_cols].notna().all(axis=1)                    # mark rows with any gap for removal

    # A column with NO numbers at all can't be repaired by any strategy -> report it.
    still_bad = [c for c in raw_cols if keep.any() and df.loc[keep, c].isna().all()]
    if still_bad:
        raise DataError(f"Columns contain no usable numbers: {still_bad[:5]}")

    # ---- Step 5: calculate the features (paper Eq. 1 and Eq. 2) ----
    features = {}
    for p in plan:
        if p["minus"] is None:
            features[p["name"]] = df[p["plus"][0]]            # plain feature: just copy the column
        else:
            # Eq. 1: per-span difference (measured minus target), for all spans at once.
            diffs = df[p["plus"]].to_numpy() - df[p["minus"]].to_numpy()
            # Eq. 2: add the differences of all spans together -> ONE number per row.
            features[p["name"]] = pd.Series(diffs.sum(axis=1), index=df.index)
    X = pd.DataFrame(features, index=df.index)

    # ---- Step 6: labels (only when training) ----
    y = None
    if with_labels:
        y_bool, valid = build_labels(df, data_cfg)
        keep &= valid                      # rows without a label can't teach the model anything
        y = y_bool

    # ---- Step 7: apply the keep/throw-away decision ONCE, at the very end ----
    X = X[keep]
    if y is not None:
        y = y[keep].astype(int).to_numpy()   # True/False -> 1/0
    return Prepared(X=X, y=y, kept_index=X.index, n_dropped=n_rows_in - len(X))


# ===========================================================================
# PART 5 - FINDING THE MOST LIKELY TROUBLESOME SPAN
# ===========================================================================
def localize_spans(df, data_cfg, kept_index=None):
    """Rank spans by their negative deviation from their configured baseline.

    The classifier intentionally sums span deviations into link-level features.  This
    helper keeps those same per-span differences long enough to answer the operational
    follow-up: *which span contributed most to the bad link?*  A positive score means
    a larger adverse deviation.  It is a diagnostic ranking, not a separately trained
    fault-location classifier.
    """
    if not data_cfg["derived_features"]:
        raise DataError("Localization requires at least one data.derived_features entry.")

    df = df.copy()
    if data_cfg["sort_by"]:
        df = df.sort_values(data_cfg["sort_by"], kind="stable")
    plan = resolve_feature_plan(df.columns, data_cfg)
    derived = [p for p in plan if p["minus"] is not None]
    raw_cols = sorted({c for p in derived for c in p["plus"] + p["minus"]})
    df[raw_cols] = df[raw_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    if data_cfg["missing"] == "interpolate":
        df[raw_cols] = df[raw_cols].interpolate(method="linear", limit_direction="both")
    elif data_cfg["missing"] == "median":
        df[raw_cols] = df[raw_cols].fillna(df[raw_cols].median())

    if kept_index is not None:
        df = df.loc[df.index.intersection(kept_index)]
        # Restore the prediction output's row order, even when input was time-sorted.
        df = df.reindex(kept_index)

    scores = np.zeros(len(df), dtype=float)
    labels = np.full(len(df), "", dtype=object)
    for p in derived:
        diffs = df[p["plus"]].to_numpy() - df[p["minus"]].to_numpy()
        adverse = np.maximum(-diffs, 0.0)  # both paper features use negative = worse
        for position, (plus, minus) in enumerate(zip(p["plus"], p["minus"])):
            suffix = plus.rsplit("_", 1)[-1]
            span = f"span_{suffix}" if minus.rsplit("_", 1)[-1] == suffix else f"pair_{position + 1}"
            wins = adverse[:, position] > scores
            labels[wins] = span
            scores = np.maximum(scores, adverse[:, position])
    return pd.DataFrame({"suspected_span": labels, "localization_score": scores}, index=df.index)


# ===========================================================================
# PART 6 - RESCALING THE NUMBERS
# ===========================================================================
class Standardizer:
    """
    ROLE: rescale each feature so it is centred on 0 and typically about 1 wide.

        new_value = (value - average) / typical_spread

    WHY: one feature might be in the thousands and another in decimals. Models learn
    faster, and more fairly, when all features have a similar size.

    IMPORTANT: the average and spread are learned from the TRAINING rows only. Using
    the test rows too would let information about the "exam" leak into the "study"
    phase and make results look better than they really are.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled     # if False, the numbers pass through unchanged
        self.mean_ = None          # the learned average of each feature
        self.std_ = None           # the learned spread of each feature

    def fit(self, X):
        """ROLE: learn the average and spread from the training numbers."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.enabled else np.zeros(X.shape[1])
        std = X.std(axis=0) if self.enabled else np.ones(X.shape[1])
        # A feature that never changes has zero spread; dividing by 0 would crash, so use 1.
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X):
        """ROLE: apply the rescaling (using the numbers learned in fit) to any data."""
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_
