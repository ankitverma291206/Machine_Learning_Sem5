"""
conftest.py  -  SHARED TEST HELPERS

ROLE OF THIS FILE
-----------------
pytest automatically loads this file. Anything marked @pytest.fixture is a reusable
"ingredient" that any test can ask for just by naming it as a parameter.
"""

import numpy as np
import pytest


@pytest.fixture
def separable_blobs():
    """
    ROLE: provide an EASY practice problem: two clouds of dots far apart from each other.
    Any sensible classifier should get (almost) every dot right, so if one fails here,
    something is genuinely broken.
    """
    rng = np.random.default_rng(0)                       # fixed seed = the same dots every run
    healthy = rng.normal([0, 0], 1.0, size=(300, 2))     # 300 dots centred at (0, 0)
    bad = rng.normal([5, 5], 1.0, size=(300, 2))         # 300 dots centred at (5, 5), far away
    X = np.vstack([healthy, bad])                        # stack them into one table
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)   # answers: 300 zeros then 300 ones
    order = rng.permutation(len(y))                      # shuffle so the two groups are mixed up
    return X[order], y[order]
