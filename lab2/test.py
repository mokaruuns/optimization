from lab2.print_counters import draw
from lab2.sgd import generate_points, sgd_regressor, sgd_regressor_momentum, sgd_regressor_nesterov, \
    sgd_regressor_adagrad, sgd_regressor_rmsprop, sgd_regressor_adam, predict


def test1():
    xy = generate_points(n=200, a=-20, b=10)
    mini_batch = 5
    lr = 0.4
    #
    params = sgd_regressor(xy, learning_rate=lr, n_epochs=10000, k=mini_batch)
    print("sgd", params)

    params = sgd_regressor_momentum(xy, learning_rate=lr, n_epochs=10000, k=mini_batch, momentum=0.95)
    print("momentum", params)

    params = sgd_regressor_nesterov(xy, learning_rate=lr, n_epochs=10000, k=mini_batch, momentum=0.95)
    print("nesterov", params)

    params = sgd_regressor_adagrad(xy, learning_rate=1, n_epochs=10000, k=mini_batch)
    print("adagrad", params)

    params = sgd_regressor_rmsprop(xy, learning_rate=1, n_epochs=10000, k=mini_batch, forget=0.9)
    print('rmsprop', params)

    params = sgd_regressor_adam(xy, learning_rate=lr, n_epochs=10000, k=mini_batch)
    print('adam', params)


def get_data():
    xy = generate_points(n=200, a=-5, b=3)
    mini_batch = 5
    lr = 0.4
    x_tr = xy['x'].values
    y_tr = xy['y'].values
    params, points = sgd_regressor_momentum(xy, learning_rate=lr, n_epochs=10000, k=mini_batch, momentum=0.95)
    return x_tr, y_tr, points


# test1()

data = get_data()
draw(data)
