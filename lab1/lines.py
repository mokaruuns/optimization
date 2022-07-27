import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab


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


def draw(points, func, optimizer):
    x1, y1, z = points
    x = np.linspace(-0.5, 1, 1000)
    y = np.linspace(-0.5, 1, 1000)
    xgrid, ygrid = np.meshgrid(x, y)

    zgrid = 16 * xgrid ** 2 + 20 * ygrid ** 2 - 4 * xgrid - 8 * ygrid + 5
    # zgrid = 64 * xgrid ** 2 + 64 * ygrid ** 2 + 126 * xgrid * ygrid - 10 * xgrid + 30 * ygrid + 13
    # print(z)
    # print(sorted(z))
    plt.contour(xgrid, ygrid, zgrid, levels=sorted(z))
    pylab.plt.plot(x1, y1, color='r', marker='o')
    # ax.clabel(colors='black')
    pylab.plt.savefig(str(func).replace(' ', '_') + type(optimizer(1, 0, 1)).__name__ + '.png')
    plt.show()
