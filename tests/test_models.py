"""
test_models.py  -  CHECKS FOR THE THREE CLASSIFIERS

ROLE OF THIS FILE
-----------------
Each function starting with `test_` is one automatic check. If any assertion fails,
the check fails and you know the models are broken.
"""

import numpy as np
import pytest

from linkfail.models import GDA, LinearSVM, LogisticRegressionNewton, build_model


# "parametrize" runs the same check three times, once per model name.
@pytest.mark.parametrize("name", ["logistic_regression", "gda", "linear_svm"])
def test_models_separate_easy_data(name, separable_blobs):
    """CHECK: on two far-apart clouds every model gets at least 98% right."""
    X, y = separable_blobs
    model = build_model(name).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.98
    # CHECK: scores must stay between 0 and 1 (valid probability-like numbers).
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_newton_loss_decreases(separable_blobs):
    """
    CHECK: with Newton's method the "how wrong are we" number should keep going down
    (or stay level) each round, and the method should finish before the 50-round limit.
    """
    X, y = separable_blobs
    model = LogisticRegressionNewton().fit(X, y)
    losses = np.array(model.loss_history)
    assert np.all(np.diff(losses) <= 1e-9)     # each round is not worse than the previous (tiny rounding allowed)
    assert len(losses) < 50                    # it converged instead of running out of rounds


def test_logreg_matches_sklearn_decision(separable_blobs):
    """
    CHECK: our hand-written logistic regression should agree with scikit-learn's
    trusted version on messier, overlapping data. This guards against maths mistakes.
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    # Two clouds that overlap a bit, so the answer is not trivially obvious.
    X = np.vstack([rng.normal(0, 1, (400, 2)), rng.normal(1.5, 1, (400, 2))])
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    ours = LogisticRegressionNewton(l2=1e-8).fit(X, y)              # ours, with almost no safety penalty
    ref = LogisticRegression(C=1e8, max_iter=1000).fit(X, y)         # reference, also almost no penalty
    agreement = (ours.predict(X) == ref.predict(X)).mean()
    assert agreement > 0.99                                          # they give the same answer for >99% of rows
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], rtol=0.05, atol=0.02)   # and learned nearly the same weights


def test_gda_recovers_class_means():
    """CHECK: GDA's cloud centres must equal the plain averages of each group (easy to do by hand)."""
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    y = np.array([0, 0, 1, 1])
    gda = GDA().fit(X, y)
    np.testing.assert_allclose(gda.mu0_, [1.0, 0.0])      # average of the first two rows
    np.testing.assert_allclose(gda.mu1_, [11.0, 10.0])    # average of the last two rows
    assert gda.phi_ == pytest.approx(0.5)                 # half of the rows are class 1


def test_single_class_is_rejected():
    """CHECK: training with only one kind of answer must raise an error, not return nonsense."""
    X = np.zeros((10, 2))
    for model in (LogisticRegressionNewton(), GDA(), LinearSVM()):
        with pytest.raises(ValueError):
            model.fit(X, np.zeros(10, dtype=int))          # ten answers, all "healthy"


def test_unknown_model_name():
    """CHECK: asking for a model that doesn't exist gives a clear error."""
    with pytest.raises(ValueError):
        build_model("random_forest")
