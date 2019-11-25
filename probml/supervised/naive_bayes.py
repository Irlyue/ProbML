import utils
import numpy as np
from .base import ClassificationBaseModel


class NaiveBayes(ClassificationBaseModel):

    def __init__(self, pseudo_counts=1):
        """
        :param pseudo_counts: int, 0 for MLE and 1 for MAP
        """
        super().__init__()
        self.pseudo_counts = pseudo_counts

    def fit(self, X, y):
        super().fit(X, y)
        n_examples, n_features = X.shape
        classes, class_counts = np.unique(y, return_counts=True)
        theta = np.zeros((len(classes), n_features))
        for i, cls in enumerate(classes):
            assert i == cls, 'Make sure the class index starts from 0!'
            cls_X = X[y == cls]
            theta[i] = (cls_X.sum(axis=0) + self.pseudo_counts) / (class_counts[i] + 2 * self.pseudo_counts)

        self.n_classes = len(classes)
        self.class_prior = class_counts / n_examples  # (classes,)
        self.theta = theta  # (n_classes, n_features)
        return self

    def predict(self, X):
        scores = self.predict_score(X)
        return scores.argmax(axis=1)

    def predict_score(self, X):
        """
            Output the log probability. You could use utils.softmax to compute the actual
        probability.
        :return np.array, shape(n_examples, n_classes)
        """
        n_examples, _ = X.shape
        scores = np.zeros((n_examples, self.n_classes))

        # make sure none of the probability reaches 0 or 1.
        # this will ease the subsequent computation of logarithm
        eps = 1e-6
        theta = np.clip(self.theta, eps, 1. - eps)

        log_class_prior = np.log(self.class_prior)
        for cls in range(self.n_classes):
            cls_theta = theta[cls]
            # log p(x|y=c) = sum_i log p(x_i|y=c)
            log_lik = np.log(cls_theta * X + (1. - cls_theta) * (1. - X)).sum(axis=1)
            # log p(x|y=c) + log p(y=c)
            scores[:, cls] = log_lik + log_class_prior[cls]
        return scores

    def __repr__(self):
        return utils.common_repr(self, ['n_classes'])
