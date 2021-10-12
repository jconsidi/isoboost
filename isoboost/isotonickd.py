# isotonickd.py

from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import check_array

from .isotonic1d import regress_isotonic_1d
from .isotonic2d import regress_isotonic_2d

class IsotonicKdRegression(RegressorMixin, TransformerMixin):
    """Approximate k-dimensional isotonic regression.

    Uses repeated 2D regression to approximate the full kD regression.

    Interface based on sklearn.isotonic.IsotonicRegression
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/isotonic.py
    """

    def __init__(self, n_estimators=None):
        self.fs = None
        self.k = None
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None):
        # TODO: shape checks
        X = check_array(X)
        y = check_array(y, ensure_2d=False)

        self.fs = []
        self.k = X.shape[1]
        
        if self.k == 0:
            raise ValueError("cannot fit 0D data")
        elif self.k == 1:
            self.fs.append(regress_isotonic_1d(xs=X[:,0], vs=y, ws=sample_weight))
        else:
            if not self.n_estimators:
                self.n_estimators = self.k * 2 - 1 if self.k > 2 else 1

            self.fs.append(regress_isotonic_2d(xs=X[:,0], ys=X[:,1], vs=list(y), ws=sample_weight))
            previous_prediction = [self.fs[0](*r) for r in zip(X[:,0], X[:,1])]

            for i in range(1, self.n_estimators):
                current_input = X[:,(i+1) % self.k]
                self.fs.append(regress_isotonic_2d(xs=previous_prediction,
                                                   ys=current_input,
                                                   vs=y,
                                                   ws=sample_weight))

                previous_prediction = [self.fs[-1](*r) for r in zip(previous_prediction, current_input)]

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
            return [self.fs[0](x) for x in T]

        prediction = [self.fs[0](x, y) for (x, y) in zip(T[:,0], T[:,1])]

        for i in range(1, self.n_estimators):
            f = self.fs[i]
            current_input = T[:,(i+1) % self.k]
            prediction = [f(x, y) for (x, y) in zip(prediction, current_input)]

        return prediction
