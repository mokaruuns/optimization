from typing import Tuple, Any

from optimizer import Optimizer
from functions import Function


class DichotomyOptimizer(Optimizer):
    def __init__(self, epsilon: float, left_border: float, right_border: float) -> None:
        super().__init__(epsilon, left_border, right_border)

    def optimize(self, func: Function) -> tuple[float, int]:
        func.reset_applying()
        left = self.left_border
        right = self.right_border
        while right - left > self.epsilon:
            x1 = (left + right - self.epsilon / 2) / 2
            x2 = (left + right + self.epsilon / 2) / 2
            fx1 = func.apply(x1)
            fx2 = func.apply(x2)
            if fx1 > fx2:
                left = x1
            else:
                right = x2
        return (left + right) / 2, func.get_amount_applying()
