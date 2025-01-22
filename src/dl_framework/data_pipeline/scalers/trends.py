import numpy as np


class ITrend:
    """
    Contract for trends
    """
    name: str = "base"

    @staticmethod
    def predict(x, *args):
        pass

    @staticmethod
    def transform(x, y, *args):
        pass

    @staticmethod
    def inverse_transform(x, y, *args):
        pass

    @staticmethod
    def initial_guess(y):
        pass


class LinearTrend(ITrend):
    name: str = "linear"

    @staticmethod
    def predict(x, a, b):
        return a + b*x

    @staticmethod
    def transform(x, y, a, b):
        return y - (a + b*x)

    @staticmethod
    def inverse_transform(x, y, a, b):
        return y + a + b*x

    @staticmethod
    def initial_guess(y):
        a = y[0]
        b = (y[-1] - y[0]) / len(y)
        return a, b


class ExponentialTrend(ITrend):
    name: str = "exponential"

    @staticmethod
    def predict(x, a, b, c):
        return a + b*np.exp(c*x)

    @staticmethod
    def transform(x, y, a, b, c):
        return (y - a) / (b*np.exp(c*x))

    @staticmethod
    def inverse_transform(x, y, a, b, c):
        return y*b*np.exp(c*x) + a

    @staticmethod
    def initial_guess(y):
        a = y.min()
        b = 0.1
        c = 0.05
        return a, b, c