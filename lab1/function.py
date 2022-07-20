from typing import Union
from collections import namedtuple


class Function:
    """
    you can use x,y,z in function
    """

    def __init__(self, func) -> None:
        self.func = func
        self.amount_applying = 0

    def apply(self, args) -> Union[int, float]:
        self.inc_amount_applying()
        if isinstance(args, dict):
            return self.func(**args)
        else:
            return self.func(args)

    def get_amount_applying(self):
        return self.amount_applying

    def inc_amount_applying(self):
        self.amount_applying += 1

    def reset_applying(self):
        self.amount_applying = 0

# Func = namedtuple("Func", "function grad")
# un = Func(lambda x: x ** 2, lambda x: 2 * x)
# print(un.function(1))
