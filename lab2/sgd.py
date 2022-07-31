import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def gradient(x, y, params):
    w, b = params
    diff = y - x * w - b
    dw = - 2 * (x * diff).mean()
    db = - 2 * diff.mean()
    return np.array([dw, db])


def loss_(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def predict(x, params):
    return x * params[0] + params[1]


def sgd_regressor(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5):
    params = np.random.randn(2)  # Randomly initializing weights

    points = []
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params)
        params += -learning_rate * new_grad

        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def sgd_regressor_momentum(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3):
    params = np.random.randn(2)  # Randomly initializing weights
    change = 0.0
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params)
        new_change = momentum * change - learning_rate * new_grad
        params += new_change
        change = new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def sgd_regressor_nesterov(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3):
    params = np.random.randn(2)  # Randomly initializing weights
    change = 0.0
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params + momentum * change)
        new_change = momentum * change - learning_rate * new_grad
        params += new_change
        change = new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def sgd_regressor_adagrad(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    G = np.zeros(2)
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params)
        # G = np.sum(new_grad ** 2)
        G += new_grad ** 2
        new_change = - learning_rate * new_grad / np.sqrt(G + eps)
        params += new_change
        # change = new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def sgd_regressor_rmsprop(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3, forget=0.2):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    sq_grad_avg = np.zeros(2)
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params)

        sq_grad_avg = sq_grad_avg * forget + (new_grad ** 2) * (1 - forget)
        new_change = - learning_rate * new_grad / np.sqrt(sq_grad_avg + eps)
        params += new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def sgd_regressor_adam(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5,
                       momentum=0.3, forget_m=0.99, forget_g=0.99):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    sq_grad_avg = np.zeros(2)
    sq_m_avg = np.zeros(2)
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        new_grad = gradient(x_tr, y_tr, params)

        sq_grad_avg = sq_grad_avg * forget_g + (new_grad ** 2) * (1 - forget_g)
        sq_m_avg = sq_m_avg * forget_m + (new_grad) * (1 - forget_m)

        m_new = sq_m_avg / (1 - forget_m)
        grad_new = sq_grad_avg / (1 - forget_g)

        new_change = - learning_rate * m_new / np.sqrt(grad_new + eps)
        params += new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, *params))
            return params

    return params


def generate_points(n, a, b):
    x = np.arange(n)
    x = preprocessing.normalize([x])[0]
    y = a * x + b
    xy = np.array([x, y])
    xy = pd.DataFrame(xy.T, columns=['x', 'y'])

    return xy


xy = generate_points(n=100, a=-20, b=10)
#
# params = sgd_regressor(xy, learning_rate=0.4, n_epochs=10000, k=20, epsilon=1e-7)
# print(params)

# params = sgd_regressor_momentum(xy, learning_rate=0.4, n_epochs=10000, k=20, epsilon=1e-7, momentum=0.95)
# print(params)

params = sgd_regressor_nesterov(xy, learning_rate=0.4, n_epochs=10000, k=10, epsilon=1e-7, momentum=0.95)
print(params)

params = sgd_regressor_adagrad(xy, learning_rate=1, n_epochs=10000, k=10, epsilon=1e-7, momentum=0.95)
print(params)

params = sgd_regressor_rmsprop(xy, learning_rate=0.01, n_epochs=10000, k=10, epsilon=1e-7, forget=0.99)
print(params)

params = sgd_regressor_adam(xy, learning_rate=1, n_epochs=10000, k=10, epsilon=1e-7, forget_m=0.9, forget_g=0.99)
print(params)
