import math
import random


def parseData(data):
    Y = []
    X = [[1] for _ in range(len(data))]
    for i, point in enumerate(data):
        X[i] += data[i][:-1]
        Y.append(data[i][-1])
    return (X, Y)


def lm(x, beta):
    """multivariate linear model with beta coefficients"""
    n = len(x)
    yhat = 0
    for i in range(n):
        yhat += x[i] * beta[i]
    return yhat


def mse(X, Y, beta):
    """mean squared error loss function"""
    m = len(X)
    se = 0
    for i in range(m):
        se += (lm(X[i], beta) - Y[i]) ** 2
    return se / m


def rmse(X, Y, beta):
    return math.sqrt(mse(X, Y, beta))


def mseprime(X, Y, beta, j):
    """partial derivative of mse function with respect to beta[j]"""
    m = len(X)
    esp = 0

    for i in range(m):
        esp += (lm(X[i], beta) - Y[i]) * X[i][j]
    return (2 / m) * esp


def gd(X, Y, epoc, alpha):
    """gradient descent algorithm"""
    n = len(X[0])
    temp = [0 for _ in range(n)]
    b = [0 for _ in range(n)]

    for _ in range(epoc):
        for j in range(n):
            temp[j] = b[j] - alpha * mseprime(X, Y, b, j)
        for k in range(n):
            b[k] = temp[k]
    return b


def sgd(X, Y, epoc, alpha):
    """stochastic gradient descent algorithm"""
    n = len(X[0])
    temp = [0 for _ in range(n)]
    b = [0 for _ in range(n)]

    for _ in range(epoc):
        for j in range(n):
            r = random.randint(0, len(X) - 1)
            temp[j] = b[j] - alpha * (lm(X[r], b) - Y[r]) * X[r][j]
        for k in range(n):
            b[k] = temp[k]
    return b


data = [[1, 1, 1], [2, 2, 2]]

X = parseData(data)[0]
Y = parseData(data)[1]


print(sgd(X, Y, 100, 0.01))
print(rmse(X, Y, sgd(X, Y, 100, 0.01)))
print(gd(X, Y, 100, 0.01))
print(rmse(X, Y, gd(X, Y, 100, 0.01)))
