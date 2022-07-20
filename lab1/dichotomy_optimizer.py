from optimizer import Optimizer
from function import Function


class DichotomyOptimizer(Optimizer):
    def __init__(self, epsilon: float, left_border: float, right_border: float) -> None:
        super().__init__(epsilon, left_border, right_border)

    def optimize(self, func: Function) -> float:
        func.reset_applying()
        left = self.left_border
        right = self.right_border
        while right - left > self.epsilon:
            delta = (right - left) / 100
            x1 = (left + right - delta) / 2
            x2 = (left + right + delta) / 2
            fx1 = func.apply(x1)
            fx2 = func.apply(x2)
            if fx1 > fx2:
                left = x1
            else:
                right = x2
        return (left + right) / 2
