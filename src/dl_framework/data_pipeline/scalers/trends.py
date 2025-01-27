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
    def residuals(x, y, *args):
        pass

    @staticmethod
    def transform(x, y, *args):
        pass

    @staticmethod
    def inverse_transform(x, y, *args):
        pass

    @staticmethod
    def initial_guess(x, y):
        pass

    @staticmethod
    def bounds(x, y):
        pass


class LinearTrend(ITrend):
    name: str = "linear"

    @staticmethod
    def predict(x, a, b):
        return a + b * x

    @staticmethod
    def residuals(x, y, a, b):
        return LinearTrend.transform(x, y, a, b)

    @staticmethod
    def transform(x, y, a, b):
        return y - (a + b * x)

    @staticmethod
    def inverse_transform(x, y, a, b):
        return y + a + b * x

    @staticmethod
    def initial_guess(x, y):
        a = y[0]
        b = (y[-1] - y[0]) / len(y)
        return a, b

    @staticmethod
    def bounds(x, y):
        return ([-np.inf, -np.inf],
                [np.inf, np.inf])


class ExponentialTrend(ITrend):
    name: str = "exponential"
    eps: float = 1e-5

    @staticmethod
    def predict(x, a, b, c):
        return a + b * np.exp(c * x)

    @staticmethod
    def residuals(x, y, a, b, c):
        exp_value = np.exp(c * x)
        model = a + b * exp_value
        return (y - model) / (np.minimum(exp_value, np.abs(y) * 1.3) + ExponentialTrend.eps)

    @staticmethod
    def transform(x, y, a, b, c):
        exp_value = np.exp(c * x)
        model = a + b * exp_value
        return (y - model) / (exp_value + ExponentialTrend.eps)

    @staticmethod
    def inverse_transform(x, y, a, b, c):
        exp_value = np.exp(c * x)
        model = a + b * exp_value
        return y * (exp_value + ExponentialTrend.eps) + model

    @staticmethod
    def initial_guess(x, y):
        c = 1e-4 # Small value
        x_start = np.mean(x[:5])
        x_end = np.mean(x[-5:])
        y_start = np.mean(y[:5])
        y_end = np.mean(y[-5:])

        # Solve for a, b, c in the exponential model y = a + b * exp(c * x)
        # y_start = a + b * exp(c * x_start)
        # y_end = a + b * exp(c * x_end)
        b = (y_end - y_start) / (np.exp(c * x_end) - np.exp(c * x_start))
        a = y_start - b * np.exp(c * x_start)
        return a, b, c

    @staticmethod
    def bounds(x, y):
        c_min = 1e-6
        c_max = 8e-3

        return ([-10, 0, c_min],
                [10, np.inf, c_max])


class InverseExponentialTrend(ITrend):
    name: str = "inverse_exponential"
    eps: float = 1e-5

    @staticmethod
    def predict(x, a, b, c):
        # a + b exp(-c * x)
        return a + b * np.exp(-c * x)

    @staticmethod
    def residuals(x, y, a, b, c):
        model = InverseExponentialTrend.predict(x, a, b, c)
        return (y - model) / (model + InverseExponentialTrend.eps)

    @staticmethod
    def transform(x, y, a, b, c):
        """
        Similar to ExponentialTrend, we do (y - model)/(model + eps).
        """
        model = InverseExponentialTrend.predict(x, a, b, c)
        return (y - model) / (model + InverseExponentialTrend.eps)

    @staticmethod
    def inverse_transform(x, y, a, b, c):
        """
        The inverse of transform: y*(model+eps) + model.
        """
        model = InverseExponentialTrend.predict(x, a, b, c)
        return y * (model + InverseExponentialTrend.eps) + model

    @staticmethod
    def initial_guess(x, y):
        """
        For an inverse exp, a is the 'plateau', b is the amplitude, c is small positive.
        """
        c = 1e-4
        x_start = np.mean(x[:5])
        x_end = np.mean(x[-5:])
        y_start = np.mean(y[:5])
        y_end = np.mean(y[-5:])

        # Solve y = a + b * exp(-c * x_end)
        # y_start = a + b * exp(-c * x_start)
        # y_end = a + b * exp(-c * x_end)
        b = (y_end - y_start) / (np.exp(-c * x_end) - np.exp(-c * x_start))
        a = y_start - b * np.exp(-c * x_start)
        return (a, b, c)

    @staticmethod
    def bounds(x, y):
        """
        a in [min(y)-0.2, max(y)+0.2],
        b in some negative to positive range,
        c in [1e-6, 1e-1].
        Adjust to taste/experience.
        """
        y_min, y_max = np.min(y), np.max(y)

        lower_c = 1e-6
        upper_c = 8e-3

        return ([-np.inf, -np.inf, lower_c],
                [np.inf, np.inf, upper_c])
