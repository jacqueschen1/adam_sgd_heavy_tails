"""Easing functions for transitions"""


class Easing:
    def __init__(self, e):
        self.e = e

    def __call__(self, p):
        return self.e(p)

    def compose(self, other):
        return Easing(lambda p: self.e(other.e(p)))


reverse = Easing(lambda p: 1 - p)
inQuad = Easing(lambda p: p ** 2)
inOutQuad = Easing(
    lambda p: (2 * p ** 2) if (p < 0.5) else (1 - 0.5 * (2 - 2 * p) ** 2)
)
