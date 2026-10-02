"""
sample_data.py  -  THE PRACTICE-DATA MAKER

ROLE OF THIS FILE
-----------------
The paper's real measurements are private, so to let anyone try the tool (and to let
the automated tests run) this file INVENTS measurements that look like the real thing.
Results on this fake data say nothing about a real network - it only proves the
program works from start to finish.

WHAT THE FAKE TABLE LOOKS LIKE
    * one row per 2-minute time slot
    * for each of 19 spans: amp_gain, target_gain, span_loss, target_span_loss
    * one overall signal-quality reading called OSNR (higher = better)

HOW THE MAKE-BELIEVE WORKS
    * Each span has fixed "target" values (what an engineer expects).
    * HEALTHY rows: real readings wobble a tiny bit around the targets.
    * FAILING rows: in 1 to 3 random spans the amplifier gives less boost and the
      fibre loses more light. OSNR drops in proportion to the total damage.
    * A row is labelled "not healthy" when its OSNR is below 20 dB.
    * OSNR has its own random wobble, so a few rows near the limit are genuinely
      hard to call - just like real life.
"""

import numpy as np
import pandas as pd

N_SPANS = 19                      # a link is made of 19 spans (same as the paper)
OSNR_THRESHOLD_DB = 20.0          # OSNR below this value means "not healthy"

# The settings the demo uses. configs/sample.yaml is a copy for people to read and
# edit; a test checks the two never drift apart.
SAMPLE_CONFIG = {
    "data": {
        "label_from_threshold": {"column": "osnr_db", "below": OSNR_THRESHOLD_DB},
        "derived_features": [
            # Feature 1: add up (measured gain - target gain) over all spans.
            {"name": "X1_amp_gain_diff", "minuend": "amp_gain_*", "subtrahend": "target_gain_*"},
            # Feature 2: add up (target loss - measured loss) over all spans.
            {"name": "X2_span_loss_diff", "minuend": "target_span_loss_*", "subtrahend": "span_loss_*"},
        ],
        "missing": "interpolate",
        "sort_by": "timestamp",
    },
}


def generate_sample_data(n_rows=1200, unhealthy_fraction=0.35, missing_fraction=0.002, seed=7):
    """
    ROLE: build and return the fake measurement table.

    n_rows            how many time slots (rows) to create
    unhealthy_fraction roughly what share of rows should have damaged spans
    missing_fraction  share of cells to blank out (so the cleaning code gets exercised)
    seed              fixes the randomness: the same seed always gives the same table
    """
    rng = np.random.default_rng(seed)         # the random number generator

    # The "expected" values for each span, repeated for every row (they never change).
    target_gain = np.tile(rng.uniform(15, 22, N_SPANS), (n_rows, 1))
    target_loss = np.tile(rng.uniform(8, 22, N_SPANS), (n_rows, 1))

    # Tiny random wobble that exists in every measurement, even on a perfect link.
    gain_noise = rng.normal(0, 0.12, (n_rows, N_SPANS))
    loss_noise = rng.normal(0, 0.15, (n_rows, N_SPANS))

    # Decide which rows are damaged, and by how much in which spans.
    is_bad = rng.random(n_rows) < unhealthy_fraction
    gain_drop = np.zeros((n_rows, N_SPANS))    # how much amplifier gain is lost (0 = none)
    loss_rise = np.zeros((n_rows, N_SPANS))    # how much extra fibre loss appears (0 = none)
    for i in np.flatnonzero(is_bad):           # go through the damaged rows only
        spans = rng.choice(N_SPANS, size=rng.integers(1, 4), replace=False)   # pick 1-3 spans
        gain_drop[i, spans] = rng.uniform(0.6, 2.5, len(spans))
        loss_rise[i, spans] = rng.uniform(0.8, 3.0, len(spans))

    # What the instruments "read": target, minus/plus the damage, plus the wobble.
    amp_gain = target_gain - gain_drop + gain_noise
    span_loss = target_loss + loss_rise + loss_noise

    # Overall signal quality: starts at 24 dB and falls with the total damage.
    osnr = 24.0 - 1.6 * loss_rise.sum(axis=1) - 1.3 * gain_drop.sum(axis=1) \
        + rng.normal(0, 0.5, n_rows)

    # Assemble the table. Span numbers are padded (01, 02, ... 19) so the columns
    # sort in the natural order.
    cols = {"timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="2min")}
    for prefix, block in (("amp_gain", amp_gain), ("target_gain", target_gain),
                          ("span_loss", span_loss), ("target_span_loss", target_loss)):
        for s in range(N_SPANS):
            cols[f"{prefix}_{s + 1:02d}"] = block[:, s].round(3)
    cols["osnr_db"] = osnr.round(2)
    df = pd.DataFrame(cols)

    # Blank out a few random cells, like a real instrument that sometimes drops a reading.
    raw = [c for c in df.columns if c not in ("timestamp", "osnr_db")]
    holes = rng.random((n_rows, len(raw))) < missing_fraction
    df[raw] = df[raw].mask(holes)
    return df
