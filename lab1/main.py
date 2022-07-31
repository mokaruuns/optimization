from lab1.gd.functions import Function
from lab1.gd.dichotomy_optimizer import DichotomyOptimizer
from lab1.gd.golden_ratio_optimizer import GoldenRatioOptimizer
from lab1.gd.fibonacci_optimizer import FibonacciOptimizer

function1 = Function(lambda x: 24 * x - 25 * x ** 2 + (35 * x ** 3) / 3 - (5 * x ** 4) / 2 + x ** 5 / 5)
function2 = Function(lambda x: x ** 2 - 3 * x + 3)
function3 = Function(lambda x: 3 * x ** 4 + x ** 3 - 10 * x ** 2 + 3 * x)

params = {"left_border": 0, "right_border": 5, "epsilon": 1e-10}
optimizer1 = DichotomyOptimizer(**params)
optimizer2 = GoldenRatioOptimizer(**params)
optimizer3 = FibonacciOptimizer(**params)
print(optimizer1.optimize(function1))
print(optimizer2.optimize(function1))
print(optimizer3.optimize(function1))
