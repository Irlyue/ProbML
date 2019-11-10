import utils

from probml import datasets
from probml.supervised import KNN


def main():
    (X_train, y_train), (X_test, y_test) = datasets.load_mnist(flatten=True,
                                                               nb_train=10000,
                                                               nb_test=10000)
    print('{} training samples and {} testing samples!'.format(len(X_train), len(X_test)))
    clr = KNN(k=5)
    clr.fit(X_train, y_train)

    print(clr)

    with utils.Timer() as tr:
        print((y_test == clr.predict(X_test)).mean())

    print('Done in {:.3f} seconds!'.format(tr.elapsed))


if __name__ == '__main__':
    main()