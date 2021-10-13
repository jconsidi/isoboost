# isotonicboost.py

import logging

from numpy import asarray
from numpy import hstack
from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import r2_score

from .isotonic2d import Isotonic2dRegression

class IsotonicBoostRegressor(RegressorMixin, BaseEnsemble):
    """An IsotonicBoost regressor that uses isotonic regression to combine
    an initial regressor with additional regressor trained against
    previous errors.

    Using isotonic regression guarantees that the training loss will
    not decrease with each additional model. Only L2 is supported for
    the loss function.

    Code loosely based on AdaBoostRegressor at
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_weight_boosting.py

    """

    def __init__(
        self,
        base_estimator=None,
        *,
        n_estimators=50,
        estimator_params=tuple(),
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_estimator()
        
        self.estimators_ = []
        self.isotonic_regressions = []

        predictions = 0.0
        for iboost in range(self.n_estimators):
            previous_predictions = asarray(predictions)
            
            delta_estimator = self._make_estimator()
            delta_estimator.fit(X, y - previous_predictions, sample_weight=sample_weight)
            delta_predictions = delta_estimator.predict(X)
            if iboost == 0:
                predictions = delta_predictions
                logging.info("IsotonicBoostRegressor.fit() score %.6f after initial model",
                             r2_score(predictions, y))
                continue

            isotonic_X = hstack((previous_predictions, delta_predictions))
            isotonic_regression = Isotonic2dRegression()
            isotonic_regression.fit(isotonic_X, y, sample_weight)
            self.isotonic_regressions.append(isotonic_regression)

            predictions = isotonic_regression.predict(isotonic_X)

            score = r2_score(predictions, y)
            logging.info("IsotonicBoostRegressor.fit() score %.6f after %d models",
                            score,
                            iboost+1)
            if score >= 1.0:
                break

    def predict(self, X):
        predictions = 0
        for iboost in range(len(self.estimators_)):
            previous_predictions = predictions

            delta_predictions = self.estimators_[iboost].predict(X)
            if iboost == 0:
                predictions = delta_predictions
            else:
                predictions = self.isotonic_regressions[iboost-1].predict(hstack((previous_predictions, delta_predictions)))

        return predictions
