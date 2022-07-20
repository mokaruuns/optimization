from bi_function import BiFunction
from function import Function
from lab1.dichotomy_optimizer import DichotomyOptimizer


def get_new_alpha(func, start_point, grad) -> float:
    x = Function(lambda l: func.apply(start_point + grad * (-l)))
    optimizer = DichotomyOptimizer(1e-5, 0, 1)
    return optimizer.optimize(x)


class SteepestDescent:
    def __init__(self, epsilon=1e-4, start=None, lr=1):
        self.epsilon = epsilon
        self.start = start
        self.lr = lr

    def optimize(self, func: BiFunction) -> float:
        stop = False
        start_point = self.start
        lr = self.lr
        while not stop:
            grad = func.count_gradient(start_point)
            ln = func.mod(grad)
            grad = grad / ln
            lr = get_new_alpha(func, start_point, grad)
            next_point = start_point + grad * (-lr)
            if ln < self.epsilon:
                stop = True
            print(start_point)
            start_point = next_point
        return start_point
