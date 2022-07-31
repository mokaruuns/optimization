import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


def sgd_regressor(X, learning_rate=0.2, n_epochs=1000, k=20, epsilon=1e-5):
    w = random.random()  # Randomly initializing weights
    b = random.random()  # Random intercept value
    epoch = 1

    while epoch <= n_epochs:
        temp = X.sample(k)

        X_tr = temp['x'].values
        y_tr = temp['y'].values

        Lw = (-2 / k * X_tr) * (y_tr - X_tr * w - b)
        Lb = (-2 / k) * (y_tr - X_tr * w - b)

        w += np.sum(-learning_rate * Lw)
        b += np.sum(-learning_rate * Lb)

        y_predicted = X_tr * w + b
        y_pred = y_predicted

        loss = mean_squared_error(y_pred, y_tr)
        print("Epoch: %d, Loss: %.6f, w: %.3f b: %.3f" % (epoch, loss, w, b))
        epoch += 1
        if loss < epsilon:
            return w, b

    return w, b


def generate_points(n, a, b):
    x = np.arange(n)
    x = preprocessing.normalize([x])[0]
    y = a * x + b
    xy = np.array([x, y])
    xy = pd.DataFrame(xy.T, columns=['x', 'y'])
    return xy


xy = generate_points(n=1000, a=-3, b=12)
w, b = sgd_regressor(xy, learning_rate=0.9, n_epochs=10000, k=20, epsilon=1e-7)
print(w, b)
