import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from sklearn.metrics import mean_squared_error


def makeData():
    # Строим сетку в интервале от -10 до 10, имеющую 100 отсчетов по обоим координатам
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # Создаем двумерную матрицу-сетку
    xgrid, ygrid = np.meshgrid(x, y)

    # В узлах рассчитываем значение функции
    # z = np.sin(xgrid) * np.sin(ygrid) / (xgrid * ygrid)
    z = 16 * xgrid ** 2 + 20 * ygrid ** 2 - 4 * xgrid - 8 * ygrid + 1
    # print(xgrid)
    return xgrid, ygrid, z


if __name__ == '__main__':
    x, y, z = makeData()
    # fig = plt.figure()
    # axes = fig.add_subplot(projection='3d')
    # axes.plot_surface(x, y, z)
    plt.contour(x, y, z, levels=10)
    plt.show()


def get_y(x, w, b, y_expected):
    res = np.zeros(w.shape)
    for i in range(len(w)):
        for j in range(len(w[i])):
            y_pred = w[i][j] * x + b[i][j]
            # print(y_pred, y_expected)
            res[i][j] = mean_squared_error(y_pred, y_expected)
    return res


def draw(points):
    x, y_expected, p = points
    w_p, b_p, l_p = np.array(p).T
    w = np.linspace(-6, 1, 500)
    b = np.linspace(2.25, 3.25, 500)
    wgrid, bgrid = np.meshgrid(w, b)  # точки для построения линий

    # zgrid = 16 * wgrid ** 2 + 20 * bgrid ** 2 - 4 * wgrid - 8 * bgrid + 5  # нужно придумать функцию

    z_loss = get_y(x, wgrid, bgrid, y_expected)
    print(z_loss)

    # print(y_pred)
    """
        Нужны x
        Функция - loss = mean_squared_error(y_predicted, y_expected)
        
        для каждого w и b нужно построить y_predicted, y_expected - известен при заданных w0 b0
        
    """
    # zgrid = 64 * xgrid ** 2 + 64 * ygrid ** 2 + 126 * xgrid * ygrid - 10 * xgrid + 30 * ygrid + 13
    # print(z)
    # print(sorted(z))
    plt.contour(wgrid, bgrid, z_loss, levels=[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1])
    pylab.plt.plot(w_p, b_p, color='r')
    # ax.clabel(colors='black')
    # pylab.plt.savefig(name+ '.png')
    plt.show()
