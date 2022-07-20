from bi_function import BiFunction


class GradientDescent:
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
            next_point = start_point + grad * (-lr)
            if ln < self.epsilon:
                stop = True
            if func.apply(start_point) < func.apply(next_point):
                lr /= 2
            print(start_point)
            start_point = next_point
        return start_point
