from copy import copy
from typing import Tuple, Any

import numpy as np

from lines import draw
from functions import BiFunction, Function
from optimizer import Optimizer
from dichotomy_optimizer import DichotomyOptimizer
from golden_ratio_optimizer import GoldenRatioOptimizer
from fibonacci_optimizer import FibonacciOptimizer
from typing import Union


def get_new_alpha(func: BiFunction, start_point: np.ndarray, grad: np.ndarray, lr: float,
                  my_optimizer) -> float:
    x = Function(lambda l: func.apply(start_point + grad * (-l)))
    optimizer = my_optimizer(1e-5, 0, 1)
    return optimizer.optimize(x)[0]


def mod(vector: np.ndarray):
    return np.sqrt(vector.dot(vector))


def draw_lines(points, func, optimizer):
    arr = np.array(points).T
    draw(arr, func, optimizer)


class BaseGradient:
    def __init__(self, epsilon, start, lr):
        self.epsilon = epsilon
        self.start = start
        self.lr = lr

    def optimize(self, func: BiFunction, my_optimizer, stp) -> np.ndarray:
        func.reset_applying()
        stop = False
        start_point = copy(self.start)
        lr = self.lr
        xyz = []
        while not stop:
            grad = func.count_gradient(start_point)
            ln = mod(grad)
            # print(grad, ln)
            grad = grad / ln
            if stp:
                lr = get_new_alpha(func, start_point, grad, lr, my_optimizer)
            next_point = start_point + grad * (-lr)
            fx_current = func.apply(start_point)
            fx_next = func.apply(next_point)
            # print([*start_point, fx_current])
            xyz.append([*start_point, fx_current])
            if abs(fx_next - fx_current) < self.epsilon:
                stop = True
            if fx_current < fx_next:
                lr /= 2
            start_point = next_point
        # draw_lines(xyz, func, my_optimizer)
        return start_point

    def steepest_descent(self, func: BiFunction, my_optimizer):
        x = self.optimize(func, my_optimizer, stp=True)
        return x, func.amount_applying, func.amount_applying_grad

    def gradient_descent(self, func: BiFunction):
        x = self.optimize(func, my_optimizer=Optimizer, stp=False)
        return x, func.amount_applying, func.amount_applying_grad
