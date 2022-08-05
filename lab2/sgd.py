import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def gradient(x, y, grad):
    w, b = grad
    diff = y - x * w - b
    dw = - 2 * (x * diff).mean()
    db = - 2 * diff.mean()
    return np.array([dw, db])


def predict(x, grad):
    return x * grad[0] + grad[1]


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
        points.append((*params, loss))
        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def sgd_regressor_momentum(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3):
    params = np.random.randn(2)  # Randomly initializing weights
    change = 0.0
    points = []
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
        points.append((*params, loss))

        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def sgd_regressor_nesterov(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, momentum=0.3):
    params = np.random.randn(2)  # Randomly initializing weights
    change = 0.0
    points = []
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
        points.append((*params, loss))
        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def sgd_regressor_adagrad(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    sq_grad = np.zeros(2)
    points = []
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        grad = gradient(x_tr, y_tr, params)

        sq_grad += grad ** 2

        new_change = - learning_rate * grad / np.sqrt(sq_grad + eps)
        params += new_change

        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        points.append((*params, loss))
        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def sgd_regressor_rmsprop(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5, forget=0.95):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    sq_grad_avg = np.zeros(2)
    points = []
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        grad = gradient(x_tr, y_tr, params)

        sq_grad_avg = sq_grad_avg * forget + (grad ** 2) * (1 - forget)

        new_change = - learning_rate * grad / np.sqrt(sq_grad_avg + eps)
        params += new_change

        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)
        points.append((*params, loss))
        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def sgd_regressor_adam(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5,
                       forget_g=0.9, forget_sq_g=0.99):
    params = np.random.randn(2)  # Randomly initializing weights
    eps = 1e-8
    sq_grad_avg = np.zeros(2)
    grad_avg = np.zeros(2)
    points = []
    for epoch in range(n_epochs):
        temp = X.sample(k)

        x_tr = temp['x'].values
        y_tr = temp['y'].values

        grad = gradient(x_tr, y_tr, params)

        sq_grad_avg = sq_grad_avg * forget_sq_g + (grad ** 2) * (1 - forget_sq_g)
        grad_avg = grad_avg * forget_g + grad * (1 - forget_g)

        grad_ = grad_avg / (1 - forget_g)
        sq_grad = sq_grad_avg / (1 - forget_sq_g)

        new_change = - learning_rate * grad_ / np.sqrt(sq_grad + eps)
        params += new_change
        y_predicted = predict(x_tr, params)
        loss = mean_squared_error(y_predicted, y_tr)

        points.append((*params, loss))
        if loss < epsilon:
            print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, params[0], params[1]))
            return params, points

    return params, points


def generate_points(n, a, b):
    x = np.arange(n)
    x = preprocessing.normalize([x])[0]
    y = a * x + b
    xy = np.array([x, y])
    xy = pd.DataFrame(xy.T, columns=['x', 'y'])

    return xy
