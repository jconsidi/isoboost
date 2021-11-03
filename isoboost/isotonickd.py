# isotonickd.py

import logging

from numpy import asarray
from numpy import vectorize
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import check_array
from sklearn.metrics import r2_score

from .isotonic1d import regress_isotonic_1d
from .isotonic2d import regress_isotonic_2d
from .isotonicreduce import reduce_isotonic_l2
from .piecewise import PiecewiseLinear


class IsotonicKdRegression(RegressorMixin, TransformerMixin):
    """Approximate k-dimensional isotonic regression.

    Uses repeated 2D regression to approximate the full kD regression.

    Interface based on sklearn.isotonic.IsotonicRegression
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/isotonic.py
    """

    def __init__(self, n_estimators=None, n_values=None):
        self.fs = None
        self.rs = None
        self.k = None
        self.n_estimators = n_estimators
        self.n_values = n_values

    def fit(self, X, y, sample_weight=None):
        # TODO: shape checks
        X = check_array(X)
        y = check_array(y, ensure_2d=False)

        self.fs = []
        self.k = X.shape[1]

        if self.k == 0:
            raise ValueError("cannot fit 0D data")
        elif self.k == 1:
            self.fs.append(
                regress_isotonic_1d(
                    xs=X[:, 0], vs=y, ws=sample_weight, n_values=self.n_values
                )
            )
        else:
            if not self.n_estimators:
                self.n_estimators = self.k * 2 - 1 if self.k > 2 else 1

            if self.n_values:
                X_original = X
                X = X.copy()
                self.rs = [None for _ in range(self.k)]
                for i in range(self.k):
                    if len(set(X[:, i])) > self.n_values:
                        reduced = reduce_isotonic_l2(
                            X[:, i],
                            sample_weight
                            if sample_weight
                            else [1 for _ in range(X.shape[0])],
                            self.n_values,
                        )
                        self.rs[i] = vectorize(
                            PiecewiseLinear(reduced.items()).interpolate
                        )
                        X[:, i] = self.rs[i](X[:, i])

                if all(f is None for f in self.rs):
                    # forget about reducing if none of the inputs were reduced
                    self.rs = None
                    del X
                    X = X_original

            self.fs.append(
                regress_isotonic_2d(
                    xs=X[:, 0],
                    ys=X[:, 1],
                    vs=list(y),
                    ws=sample_weight,
                    n_values=self.n_values,
                )
            )
            if len(y) <= 1:
                # degenerate case - just one sample, so stop immediately
                # LATER: move this earlier
                return

            previous_prediction = [self.fs[0](*r) for r in zip(X[:, 0], X[:, 1])]

            training_scores = []
            training_scores.append(r2_score(y, previous_prediction))
            logging.warning(
                "IsotonicKdRegression.fit() score %.6f after initial model",
                training_scores[0],
            )

            for i in range(1, self.n_estimators):
                current_input = X[:, (i + 1) % self.k]
                self.fs.append(
                    regress_isotonic_2d(
                        xs=previous_prediction,
                        ys=current_input,
                        vs=y,
                        ws=sample_weight,
                        n_values=self.n_values,
                    )
                )

                previous_prediction = [
                    self.fs[-1](*r) for r in zip(previous_prediction, current_input)
                ]
                training_scores.append(r2_score(y, previous_prediction))
                logging.warning(
                    "IsotonicKdRegression.fit() score %.6f after %d models",
                    training_scores[-1],
                    len(self.fs),
                )

                if training_scores[-1] >= 1.0:
                    logging.warning(
                        "IsotonicKdRegression.fit() stopping after %d models",
                        len(self.fs),
                    )
                    break

                if len(training_scores) >= self.k + 1:
                    # check if the last round through input columns improved the score
                    if training_scores[-1] <= training_scores[-1 - self.k]:
                        # training score did not improve, so drop the last round of models
                        self.fs[-self.k :] = []
                        logging.warning(
                            "IsotonicKdRegression.fit() rolling back to first %d models",
                            len(self.fs),
                        )
                        break

    def predict(self, T):
        """Predict new data by bilinear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples, k)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self.transform(T)

    def transform(self, T):
        """Transform new data by bilinear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples, k)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """

        T = check_array(T)
        if T.shape[1] != self.k:
            raise ValueError("wrong shape")

        if self.k == 1:
            T = T.reshape((-1,))
            return asarray([self.fs[0](x) for x in T])

        if self.rs:
            T = T.copy()
            for (i, f) in enumerate(self.rs):
                if f:
                    T[:, i] = f(T[:, i])

        prediction = [self.fs[0](x, y) for (x, y) in zip(T[:, 0], T[:, 1])]

        for i in range(1, len(self.fs)):
            f = self.fs[i]
            current_input = T[:, (i + 1) % self.k]
            prediction = [f(x, y) for (x, y) in zip(prediction, current_input)]

        return asarray(prediction)
