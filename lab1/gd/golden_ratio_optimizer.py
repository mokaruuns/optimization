from lab1.gd.optimizer import Optimizer
from lab1.gd.functions import Function


class GoldenRatioOptimizer(Optimizer):
    def __init__(self, epsilon: float, left_border: float, right_border: float) -> None:
        super().__init__(epsilon, left_border, right_border)

    def optimize(self, func: Function) -> tuple[float, int]:
        func.reset_applying()
        tau = (5 ** 0.5 - 1) / 2
        left = self.left_border
        right = self.right_border
        delta = right - left
        x1 = left + (1.0 - tau) * delta
        x2 = left + tau * delta
        fx1 = func.apply(x1)
        fx2 = func.apply(x2)
        current_epsilon = (right - left) / 2
        while current_epsilon > self.epsilon:
            if fx1 < fx2:
                right = x2
                x2 = x1
                x1 = left + (1.0 - tau) * (right - left)
                fx2 = fx1
                fx1 = func.apply(x1)
            else:
                left = x1
                x1 = x2
                fx1 = fx2
                x2 = left + tau * (right - left)
                fx2 = func.apply(x2)
            current_epsilon *= tau
        return (left + right) / 2, func.get_amount_applying()
