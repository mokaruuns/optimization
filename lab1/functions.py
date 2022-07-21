from typing import Union

import numpy as np


class BaseFunction:
    def __init__(self) -> None:
        self.amount_applying = 0

    def get_amount_applying(self) -> int:
        return self.amount_applying

    def inc_amount_applying(self) -> None:
        self.amount_applying += 1

    def reset_applying(self) -> None:
        self.amount_applying = 0


class Function(BaseFunction):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def apply(self, args) -> Union[int, float]:
        self.inc_amount_applying()
        if isinstance(args, dict):
            return self.func(**args)
        else:
            return self.func(args)


class BiFunction(BaseFunction):
    def __init__(self, dim: int, a: np.ndarray, b: np.ndarray, c: float) -> None:
        super().__init__()
        self.amount_applying_grad = 0
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c

    def count_gradient(self, vector: np.ndarray) -> np.ndarray:
        self.amount_applying_grad += 1
        return np.dot(self.a, vector) + self.b

    def mod(self, vector: np.ndarray) -> float:
        return np.dot(vector, vector) ** 0.5

    def apply(self, vector: np.ndarray) -> float:
        self.inc_amount_applying()
        ans = np.sum(np.outer(vector, vector) * self.a) / 2
        ans += np.sum(vector * self.b)
        ans += self.c
        return ans
