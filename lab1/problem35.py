import numpy as np

from functions import BiFunction
from base_gradient import BaseGradient
from dichotomy_optimizer import DichotomyOptimizer
from golden_ratio_optimizer import GoldenRatioOptimizer
from fibonacci_optimizer import FibonacciOptimizer


# Function(lambda x, y: 16 * x ** 2 + 20 * y ** 2 - 4 * x - 8 * y + 5)
# min fn = -1/20, x = 1/8, y = 1/5
def function1():
    optimizer_params = {"epsilon": 1e-5, "start": np.array([0, 0]), "lr": 1}
    fn_params = {'a': np.array([[32, 0], [0, 40]]),
                 'b': np.array([-4, -8]),
                 'c': 5,
                 'dim': 2}
    return optimizer_params, fn_params


# z = 64x^2 + 64y^2 + 128xy - 10x + 30y + 13
def function2():
    optimizer_params = {"epsilon": 1e-5, "start": np.array([0, 0]), "lr": 1}
    fn_params = {'a': np.array([[128, 126], [126, 128]]),
                 'b': np.array([-10, 30]),
                 'c': 13,
                 'dim': 2}
    return optimizer_params, fn_params


data = function2()
optimizer = BaseGradient(**data[0])
fn = BiFunction(**data[1])
print(optimizer.gradient_descent(fn))
print(optimizer.steepest_descent(fn, DichotomyOptimizer))
print(optimizer.steepest_descent(fn, GoldenRatioOptimizer))
print(optimizer.steepest_descent(fn, FibonacciOptimizer))
