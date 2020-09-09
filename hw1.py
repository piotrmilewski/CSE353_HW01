import numpy as np
from sklearn.linear_model import LinearRegression


def question1(X, y):
    return {'mu0': np.mean(X[y == 0], axis=0),
            'var0': np.var(X[y == 0], axis=0),
            'mu1': np.mean(X[y == 1], axis=0),
            'var1': np.var(X[y == 1], axis=0)}


def question2(x, y):
    if len(x) < 7:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0
    init1 = np.flipud(x[0: 7])
    init2 = np.flipud(y[0: 7])
    init = np.concatenate((init1, init2))
    linRegX = np.array([init])
    index = 8
    while index < len(x):
        data1 = np.flipud(x[index - 7: index])
        data2 = np.flipud(y[index - 7: index])
        data = np.concatenate((data1, data2))
        linRegX = np.vstack([linRegX, data])
        index = index + 1
    linRegY = y[7:len(y)]
    reg = LinearRegression().fit(linRegX, linRegY)
    return {'w': reg.coef_,
            'b': reg.intercept_}
