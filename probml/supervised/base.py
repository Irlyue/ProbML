import numpy as np
from abc import ABC


class SupervisedBaseModel:

    def __repr__(self):
        return '{}()'.format(type(self).__name__)

    def fit(self, X, y):
        self.X, self.y = X, y

    def predict(self, X):
        raise NotImplementedError


class ClassificationBaseModel(SupervisedBaseModel, ABC):

    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_classes = len(np.unique(y))

    def predict_score(self, X):
        raise NotImplementedError
