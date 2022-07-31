import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import statistics

from lab1.gd.functions import DiagFunction
from lab1.gd.base_gradient import BaseGradient
from lab1.gd.dichotomy_optimizer import DichotomyOptimizer
from lab1.gd.golden_ratio_optimizer import GoldenRatioOptimizer
from lab1.gd.fibonacci_optimizer import FibonacciOptimizer


def generate_diag(n, k):
    a = np.sort(np.random.randint(1, k + 1, n))
    a[0] = 1
    a[n - 1] = k
    return a


# Function(lambda x, y: 16 * x ** 2 + 20 * y ** 2 - 4 * x - 8 * y + 5)
# min fn = -1/20, x = 1/8, y = 1/5
def function1(n, k):
    return {'a': generate_diag(n, k), 'dim': n}


colors = {2: 'orange', 10: 'green', 100: 'red', 1000: 'black', 10000: 'blue'}
titles = ['Градиентный спуск',
          'Наискорейший спуск + дихотомия',
          'Наискорейший спуск + золотое сечение',
          'Наискорейший спуск + Фибоначчи']


def draw_plots(d):
    for i, row in d.iterrows():
        ax = plt.subplot()
        ax.grid()
        ax.set_xlabel('число обусловленности')
        ax.set_ylabel('итерации')
        for indx, j in enumerate(row[1:]):
            ax.plot(j.keys(), j.values(), label=d.columns[indx + 1])
        leg = plt.legend(loc='upper left', shadow=True)
        leg.get_frame().set_alpha(0.5)
        plt.title(titles[i])
        plt.savefig('problem6_median_' + d['index'].iloc[i] + '.png')
        plt.show()


def get_mean(amount, method, n, k, optimizer=None):
    x = []

    for i in range(amount):
        fn = DiagFunction(**function1(n, k))
        if optimizer is None:
            x.append(method(fn)[2])
        else:
            x.append(method(fn, optimizer)[2])
    return statistics.median(x)


def run_test(n, k):
    di_n = {10: dict(), 100: dict(), 1000: dict(), 10000: dict()}
    for i in n:
        di = {'gradient_descent': dict(),
              'DichotomyOptimizer': dict(),
              'GoldenRatioOptimizer': dict(),
              'FibonacciOptimizer': dict()
              }
        optimizer_params = {"epsilon": 1e-5, "start": np.ones(i), "lr": 1}
        optimizer = BaseGradient(**optimizer_params)
        for j in k:
            di['gradient_descent'][j] = get_mean(7, optimizer.gradient_descent, i, j)
            di['DichotomyOptimizer'][j] = get_mean(7, optimizer.steepest_descent, i, j, DichotomyOptimizer)
            di['GoldenRatioOptimizer'][j] = get_mean(7, optimizer.steepest_descent, i, j, GoldenRatioOptimizer)
            di['FibonacciOptimizer'][j] = get_mean(7, optimizer.steepest_descent, i, j, FibonacciOptimizer)
            di_n[i] = di
        print(i, "is done")
    return di_n


x = run_test(n=[10, 100, 1000, 10000], k=range(5, 2506, 50))
df_x = pd.DataFrame(x).reset_index()

draw_plots(df_x)
print(df_x)
