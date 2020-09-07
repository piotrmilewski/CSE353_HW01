import numpy as np


def question1(X, y):
    return {'mu0': np.mean(X[y == 0], axis=0),
            'var0': np.var(X[y == 0], axis=0),
            'mu1': np.mean(X[y == 1], axis=0),
            'var1': np.var(X[y == 1], axis=0)}
