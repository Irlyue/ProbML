import numpy as np
from .base import SupervisedBaseModel
from scipy import stats


class KNN(SupervisedBaseModel):

    def __init__(self, k, n_classes=None):
        """
        :param k: int,
        :param n_classes: int, default to None. If None, `n_classes` will be
        inferred from the training label `y`.
        """
        self.k = k
        self.n_classes = n_classes

    def __repr__(self):
        properties = f'k={self.k}, n_classes={self.n_classes}'
        return '{}({})'.format(type(self).__name__, properties)

    def fit(self, X, y):
        super().fit(X, y)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        return self

    def predict(self, X):
        """
        :param X: np.array, shape(n_samples, n_features)
        :return: np.array, with shape(n_samples,) and type(np.int32).
        """
        Xt, yt = self.X, self.y
        k = min(self.k, len(Xt))

        Xt, X = Xt.astype(np.float32), X.astype(np.float32)

        # d[i, j] = <X[i], X[i]> - 2*<X[i], Xt[j]> + <Xt[j], Xt[j]>
        # only the last two terms are needed
        distance = -2 * np.dot(X, Xt.T) + np.sum(Xt**2, axis=1)
        y_pred = yt[distance.argmin(axis=1)]
        if k >= 2:
            # get the smallest k indices (not sorted)
            smallest_k_indices = np.argpartition(distance, kth=k, axis=1)[:, :k]
            # get the corresponding classes
            corresponding_classes = np.take(yt, smallest_k_indices, axis=0)
            # get the class with the most frequent occurrences
            mode, count = stats.mode(corresponding_classes, axis=1)
            mode, count = mode.ravel(), count.ravel()
            flag = (count != 1)
            y_pred[flag] = mode[flag]
        y_pred = y_pred.astype(np.int32)
        return y_pred
