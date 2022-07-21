from typing import Tuple, Any

from optimizer import Optimizer
from functions import Function
from math import ceil, log10

SQRT5 = 5 ** (1 / 2)


def fib(n):
    a, b = 0, 1
    for __ in range(n):
        a, b = b, a + b
    return a


class FibonacciOptimizer(Optimizer):
    def __init__(self, epsilon: float, left_border: float, right_border: float) -> None:
        super().__init__(epsilon, left_border, right_border)
        self.__get_initial_fn()

    def optimize(self, func: Function) -> tuple[float | Any, int]:
        func.reset_applying()
        a = self.left_border
        b = self.right_border
        delta = b - a
        x1 = a + (self.fn / self.fn2) * delta
        x2 = a + (self.fn1 / self.fn2) * delta
        fx1 = func.apply(x1)
        fx2 = func.apply(x2)
        for i in range(self.n, 1, -1):
            if fx1 > fx2:
                a = x1
                x1 = x2
                x2 = a + b - x1
                fx1, fx2 = fx2, func.apply(x2)
            else:
                b = x2
                x2 = x1
                x1 = a + b - x2
                fx1, fx2 = func.apply(x1), fx1
            # print(x1, x2)
            if (x1 - x2) / 2 > self.epsilon:
                break
        return (x1 + x2) / 2, func.get_amount_applying()

    def __get_initial_fn(self) -> None:
        delta = self.right_border - self.left_border
        n = ceil(log10(SQRT5 * delta / self.epsilon) / log10((1 + SQRT5) / 2)) - 2
        fn = fib(n)
        fn1 = fib(n + 1)
        fn2 = fn + fn1
        tmp = abs(delta) / self.epsilon
        while tmp >= fn2:
            n += 1
            fn, fn1, fn2 = fn1, fn2, fn1 + fn2
        self.n = n
        self.fn, self.fn1, self.fn2 = fn, fn1, fn2
