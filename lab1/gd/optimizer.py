class Optimizer:
    def __init__(self, epsilon: float, left_border: float, right_border: float) -> None:
        self.epsilon = epsilon
        self.left_border = left_border
        self.right_border = right_border
