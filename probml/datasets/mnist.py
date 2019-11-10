import numpy as np


def load_mnist(flatten=False, nb_train=None, nb_test=None):
    """
    :param nb_train: int, default to None. If given, only that number
    of images are loaded for training.
    :param nb_test: int,
    :param flatten: bool, whether to flatten the image to 1d vector.
    :return: (X_train, y_train), (X_test, y_test).
        X_train: 60000*28*28, np.uint8
        y_train: 60000, np.int64
        X_test: 10000*28*28, np.uint8
        y_test: 10000, np.int64
    """
    train = dict(np.load('data/mnist/train.npz').items())
    test = dict(np.load('data/mnist/test.npz').items())
    if flatten:
        train['images'] = np.reshape(train['images'], (-1, 28*28))
        test['images'] = np.reshape(test['images'], (-1, 28*28))

    if nb_train:
        train['images'] = train['images'][:nb_train]
        train['labels'] = train['labels'][:nb_train]

    if nb_test:
        test['images'] = test['images'][:nb_test]
        test['labels'] = test['labels'][:nb_test]

    return (train['images'], train['labels']), (test['images'], test['labels'])
