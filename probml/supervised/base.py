class SupervisedBaseModel:

    def __repr__(self):
        return '{}()'.format(type(self).__name__)

    def fit(self, X, y):
        self.X, self.y = X, y

    def predict(self, X):
        raise NotImplementedError
