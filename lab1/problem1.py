import numpy as np

from functions import BiFunction
from base_gradient import BaseGradient

# Function(lambda x, y: 16 * x ** 2 + 20 * y ** 2 - 4 * x - 8 * y + 1)
# min fn = -1/20, x = 1/8, y = 1/5
optimizer_params = {"epsilon": 1e-10, "start": np.array([0, 0]), "lr": 1}
fn_params = {'a': np.array([[32, 0], [0, 40]]),
             'b': np.array([-4, -8]),
             'c': 1,
             'dim': 2}

optimizer1 = BaseGradient(**optimizer_params)
fn = BiFunction(**fn_params)
print(optimizer1.gradient_descent(fn))
# print(optimizer1.steepest_descent(fn))
