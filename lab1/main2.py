import numpy as np

from lab1.bi_function import BiFunction
from lab1.steepest_descent import SteepestDescent

# Function(lambda x, y: 16 * x ** 2 + 20 * y ** 2 - 4 * x - 8 * y + 5)
optimizer_params = {"epsilon": 1e-4, "start": np.array([0, 0]), "lr": 1}
fn_params = {'a': np.array([[32, 0], [0, 40]]),
             'b': np.array([-4, -8]),
             'c': 5,
             'dim': 2}

optimizer1 = SteepestDescent(**optimizer_params)
fn = BiFunction(**fn_params)
print(optimizer1.optimize(fn))
