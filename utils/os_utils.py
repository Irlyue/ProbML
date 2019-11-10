import os
from time import time


class Timer:
    def __enter__(self):
        self._tic = time()
        return self

    def __exit__(self, *args):
        self.elapsed = time() - self._tic


def path_exists(path):
    return os.path.exists(path)


def create_if_not_exists(path):
    if not path_exists(path):
        os.makedirs(path)
