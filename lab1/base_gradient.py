from copy import copy
from typing import Tuple, Any

import numpy as np

from functions import BiFunction, Function
from dichotomy_optimizer import DichotomyOptimizer
from golden_ratio_optimizer import GoldenRatioOptimizer
from fibonacci_optimizer import FibonacciOptimizer
from typing import Union


def get_new_alpha(func: BiFunction, start_point: np.ndarray, grad: np.ndarray) -> float:
    x = Function(lambda l: func.apply(start_point + grad * (-l)))
    optimizer = FibonacciOptimizer(1e-8, 0, 1)
    return optimizer.optimize(x)[0]

def mod(vector: np.ndarray):
    return np.sqrt(vector.dot(vector))

class BaseGradient:
    def __init__(self, epsilon=1e-4, start=None, lr=0.5):
        self.epsilon = epsilon
        self.start = start
        self.lr = lr

    def optimize(self, func: BiFunction, stp=True) -> float:
        func.reset_applying()
        stop = False
        start_point = copy(self.start)
        next_point = copy(self.start)
        lr = self.lr
        while not stop:
            print(start_point)
            grad = func.count_gradient(start_point)
            ln = mod(grad)
            grad = grad / ln
            lr *= 0.5
            if stp:
                lr = get_new_alpha(func, start_point, grad)
            next_point = start_point + grad * (-lr)
            if ln < self.epsilon:
                stop = True
            start_point = next_point
        return start_point

    def steepest_descent(self, func: BiFunction):
        return self.optimize(func, stp=True), func.amount_applying, func.amount_applying_grad

    def gradient_descent(self, func: BiFunction):
        return self.optimize(func, stp=False), func.amount_applying, func.amount_applying_grad
