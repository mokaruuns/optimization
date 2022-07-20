import numpy as np


class BiFunction:
    def __init__(self, dim, a, b, c):
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c

    def count_gradient(self, vector):
        return np.dot(self.a, vector) + self.b

    def mod(self, vector):
        return np.dot(vector, vector) ** 0.5

    def apply(self, vector):
        ans = np.sum(np.outer(vector, vector) * self.a) / 2
        ans += np.sum(vector * self.b)
        ans += self.c
        return ans
