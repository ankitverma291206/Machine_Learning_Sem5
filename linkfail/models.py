"""
models.py  -  THE THREE "BRAINS"

ROLE OF THIS FILE
-----------------
A "classifier" is a program that looks at a few numbers and answers a yes/no
question. Ours answers: "is this link NOT healthy?"  It learns the answer from
past examples (rows where we already know the truth).

We provide three classifiers so they can be compared:
    1. Logistic regression   - draws a line, turns distance-from-line into a probability
    2. GDA                   - models each class as a cloud of points, asks "which cloud fits better?"
    3. Linear SVM            - draws the line with the widest empty gap on each side

All three are "linear": the dividing boundary between healthy and not healthy is a
straight line (in two features) or a flat plane (in more).

THE SHARED RULES (so the rest of the program treats them all the same)
    model.fit(X, y)          learn from examples. X = table of numbers, y = 0/1 answers
    model.predict_proba(X)   a score between 0 and 1: closer to 1 = more likely NOT healthy
    model.predict(X)         the final yes/no answer (1 if score is 0.5 or more)
"""

import numpy as np
from sklearn.svm import SVC     # a ready-made, well-tested SVM from scikit-learn


def sigmoid(z):
    """
    ROLE: squash any number into the range 0 to 1 so it can be read as a probability.

        very negative number -> close to 0     (e.g. -5 -> 0.007)
        zero                 -> exactly 0.5    (completely unsure)
        very positive number -> close to 1     (e.g.  5 -> 0.993)
    The formula is 1 / (1 + e^-z).
    """
    # np.clip stops absurdly large numbers from overflowing the maths.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _check_binary(y):
    """
    ROLE: refuse to train when the answers contain only ONE kind (all healthy or all
    unhealthy). You can't learn to tell two things apart if you've only seen one.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels contain only one class; need both healthy and unhealthy rows.")


class BaseClassifier:
    """
    ROLE: the shared parent of all three models. It holds the one piece of
    behaviour they have in common: turning a score into a final yes/no answer.
    """

    name = "base"

    def predict(self, X):
        """Final answer: 1 (not healthy) if the score is at least 0.5, otherwise 0 (healthy)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # After training, every model keeps its dividing line as  score_line = coef * features + intercept.
    # coef_ says how much each feature pushes towards "not healthy"; intercept_ is the starting offset.
    coef_ = None
    intercept_ = None


# ===========================================================================
# MODEL 1 - LOGISTIC REGRESSION (trained with Newton's method)   [paper Eq. 3]
# ===========================================================================
class LogisticRegressionNewton(BaseClassifier):
    """
    HOW IT THINKS
        1. Compute a "line score":  z = w1*X1 + w2*X2 + b
        2. Squash it with the sigmoid to get a probability of "not healthy".
        3. Learning = finding the weights w1, w2 and offset b that make the
           probabilities match the known answers as closely as possible.

    "How wrong are we?" is measured by the LOSS (cross-entropy): it is small when we
    gave high probability to true failures and low probability to healthy links.

    HOW IT LEARNS: NEWTON'S METHOD
        Imagine standing on a hillside in fog and wanting the lowest point of the valley.
        Gradient descent takes many small careful steps downhill.
        Newton's method also looks at how the slope itself is CURVING, so it can jump
        almost straight to the bottom. It usually finishes in about 5-15 rounds.

    THE SMALL SAFETY PENALTY (l2)
        If the two groups are perfectly separable, the maths wants the weights to grow
        forever. A tiny penalty on large weights keeps them reasonable and the
        calculation stable. It does not noticeably change the results.
    """

    name = "logistic_regression"

    def __init__(self, max_iter=50, tol=1e-8, l2=1e-4):
        self.max_iter = max_iter     # never do more than this many Newton rounds (safety limit)
        self.tol = tol               # stop early once an update is smaller than this (we've arrived)
        self.l2 = l2                 # strength of the small safety penalty
        self.loss_history = []       # the loss after each round, so you can check it keeps falling

    def fit(self, X, y):
        """ROLE: learn the weights from examples (X = numbers, y = known 0/1 answers)."""
        _check_binary(y)
        m = X.shape[0]                                   # m = number of example rows

        # Put a column of 1s in front of the data. That way the offset b is learned
        # exactly like any other weight (it is simply the weight of the always-1 column).
        Xb = np.hstack([np.ones((m, 1)), X])
        d = Xb.shape[1]                                  # d = number of things to learn (b plus one weight per feature)
        theta = np.zeros(d)                              # start knowing nothing: all weights 0 -> probability 0.5 everywhere

        # The safety penalty applies to every weight except the offset b (index 0).
        reg = self.l2 * np.eye(d)
        reg[0, 0] = 0.0

        self.loss_history = []
        for _ in range(self.max_iter):
            h = sigmoid(Xb @ theta)                      # current guesses: probability of "not healthy" per row

            # GRADIENT = the direction in which the loss increases fastest ("the slope").
            grad = Xb.T @ (h - y) / m + reg @ theta

            # HESSIAN = how the slope itself changes ("the curvature").
            # Rows where the model is unsure (h near 0.5) count more than rows it is sure about.
            hess = (Xb * (h * (1 - h))[:, None]).T @ Xb / m + reg

            # The Newton step = slope divided by curvature. `solve` is the safe way to
            # do that division for matrices; the tiny 1e-12 term avoids a divide-by-zero crash.
            step = np.linalg.solve(hess + 1e-12 * np.eye(d), grad)
            theta = theta - step                         # move to the improved weights

            # Record the new loss so we can confirm it is going down.
            h = sigmoid(Xb @ theta)
            eps = 1e-12                                  # tiny number: log(0) would be infinite
            ce = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
            self.loss_history.append(float(ce + 0.5 * self.l2 * np.sum(theta[1:] ** 2)))

            if np.linalg.norm(step) < self.tol:          # the update was tiny -> we have arrived
                break

        # Store the result: first number is the offset b, the rest are one weight per feature.
        self.intercept_, self.coef_ = float(theta[0]), theta[1:].copy()
        return self

    def predict_proba(self, X):
        """ROLE: score new rows: line score -> sigmoid -> probability of "not healthy"."""
        return sigmoid(X @ self.coef_ + self.intercept_)


# ===========================================================================
# MODEL 2 - GAUSSIAN DISCRIMINANT ANALYSIS (GDA)   [paper Eq. 4]
# ===========================================================================
class GDA(BaseClassifier):
    """
    HOW IT THINKS
        Imagine every healthy link as a dot on a chart. The dots form a round-ish cloud.
        Failing links form a second cloud somewhere else. GDA simply measures where each
        cloud sits (its centre) and how spread out it is. To judge a NEW dot it asks:
        "which cloud would more likely have produced a dot right here?"

    "Gaussian" just means each cloud has the classic bell-curve shape: dense in the
    middle, thinning out towards the edges. We assume both clouds share the same shape
    (same spread), which makes the dividing boundary a straight line.

    NO REPEATED LEARNING ROUNDS: the centres and spread are just averages, calculated
    in one step with simple formulas.

    WHEN IT WORKS WELL: when the data really does look like bell-shaped clouds.
    WHEN IT STRUGGLES: when a class is oddly shaped or mixes several sub-groups
    (which is why it came second in the paper).
    """

    name = "gda"

    def __init__(self, ridge=1e-6):
        self.ridge = ridge     # a tiny amount added for numerical safety if two features carry the same information

    def fit(self, X, y):
        """ROLE: measure the two clouds (centres, spread, and how common each class is)."""
        _check_binary(y)
        m, d = X.shape                                   # m rows, d features
        self.phi_ = float(np.mean(y))                    # fraction of rows that are "not healthy"
        self.mu0_ = X[y == 0].mean(axis=0)               # centre of the HEALTHY cloud
        self.mu1_ = X[y == 1].mean(axis=0)               # centre of the NOT-HEALTHY cloud

        # Spread (covariance): how far points sit from THEIR OWN cloud's centre,
        # and how the features move together. Both clouds share this one shape.
        centred = X - np.where(y[:, None] == 1, self.mu1_, self.mu0_)
        self.sigma_ = centred.T @ centred / m + self.ridge * np.eye(d)

        # Algebra trick: "which cloud fits better?" can be rewritten as a plain
        # line score  coef*x + intercept , the same form logistic regression uses.
        # That makes predictions cheap and the weights comparable between models.
        sigma_inv = np.linalg.inv(self.sigma_)
        self.coef_ = sigma_inv @ (self.mu1_ - self.mu0_)
        self.intercept_ = float(
            -0.5 * (self.mu1_ @ sigma_inv @ self.mu1_ - self.mu0_ @ sigma_inv @ self.mu0_)
            + np.log(self.phi_ / (1.0 - self.phi_))      # accounts for one class being more common than the other
        )
        return self

    def predict_proba(self, X):
        """ROLE: probability of "not healthy" for new rows (same line-score-then-sigmoid recipe)."""
        return sigmoid(X @ self.coef_ + self.intercept_)


# ===========================================================================
# MODEL 3 - LINEAR SUPPORT VECTOR MACHINE (SVM)   [paper Eq. 5, linear kernel]
# ===========================================================================
class LinearSVM(BaseClassifier):
    """
    HOW IT THINKS
        Of all the straight lines that separate the two groups, pick the one with the
        WIDEST EMPTY STREET around it. A wide gap means the boundary is not hugging
        either group, so new points are less likely to land on the wrong side.

    The setting C controls how strict it is:
        large C = "do not tolerate any wrongly placed training point" (narrower street)
        small C = "allow a few mistakes in exchange for a wider street"

    We let scikit-learn do the heavy lifting (it is a well-tested implementation),
    and this class just wraps it so it behaves like our other two models.

    NOTE ON SCORES: an SVM does not naturally output probabilities. We squash its
    "distance from the line" through the sigmoid. That is fine for ranking links and
    for the 0.5 cut-off, but do not read it as a true calibrated probability.
    """

    name = "linear_svm"

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        """ROLE: find the widest-street line using scikit-learn, then keep its weights."""
        _check_binary(y)
        self._svc = SVC(kernel="linear", C=self.C).fit(X, y)     # "linear" = straight-line boundary
        self.coef_ = self._svc.coef_[0].copy()
        self.intercept_ = float(self._svc.intercept_[0])
        return self

    def predict_proba(self, X):
        """ROLE: score new rows by how far they are from the line (squashed into 0..1)."""
        return sigmoid(self._svc.decision_function(X))


# ===========================================================================
# THE MODEL "MENU"
# ===========================================================================
# A lookup table from the NAME you type in settings to the class that implements it.
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionNewton,
    "gda": GDA,
    "linear_svm": LinearSVM,
}


def build_model(name, params=None):
    """
    ROLE: create a model from its name, e.g. build_model("linear_svm", {"C": 10}).
    This is how the rest of the program gets a model without knowing which class it is.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](**(params or {}))
