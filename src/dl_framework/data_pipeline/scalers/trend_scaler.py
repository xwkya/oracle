from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, ClassVar

import numpy as np
from scipy.optimize import least_squares
from src.dl_framework.data_pipeline.scalers.trends import LinearTrend, ExponentialTrend, ITrend, InverseExponentialTrend


@dataclass
class TrendRemovalConfig:
    include_exponential: bool = True
    include_linear: bool = True
    include_inverse_exponential: bool = True

    # The higher the scale the more penalized the model is for having this trend
    linear_mse_scale: float = 1.0
    exponential_mse_scale: float = 1.0
    inverse_exponential_mse_scale: float = 1.0

    # Number of iterations
    max_iter: Optional[int] = 2000

class TrendRemovalConfigError(Exception):
    pass

class TrendRemovalScaler:
    """
    Removes per-column drift (linear or exponential), then applies a single StandardScaler
    on the drift-removed training portion.

    :param train_cutoff_idx: Index separating training rows from test rows.
    :type train_cutoff_idx: int
    """

    def __init__(self, train_cutoff_idx: int, config: Optional[TrendRemovalConfig] = None):
        self.train_cutoff_idx = train_cutoff_idx
        if config is None:
            self.config = TrendRemovalConfig()
        else:
            self.config = config

        self.col_info = {}  # {col_i: {'model': ITrend, 'params': Tuple}}

    def fit(self, data: np.ndarray) -> TrendRemovalScaler:
        """
        Fits drift models for each column on valid training samples and then
        fits one StandardScaler over the entire drift-removed training subset.
        Nans are ignored during fitting.

        :param data: Array of shape (num_samples, num_cols) with possible NaNs.
        :type data: np.ndarray
        :return: Fitted TrendRemovalScaler instance.
        :rtype: TrendRemovalScaler
        """
        n_samples, n_cols = data.shape
        x = np.arange(n_samples)

        for col_i in range(n_cols):
            col_data = data[:, col_i]
            train_mask = (x < self.train_cutoff_idx) & ~np.isnan(col_data)

            if not np.any(train_mask):
                self.col_info[col_i] = {'model': None, 'params': None}
                continue

            x_train = x[train_mask]
            y_train = col_data[train_mask]

            best_model, best_params, mse = self._fit_best_trend_model(x_train, y_train, self.config)
            if best_model is None:
                best_model = LinearTrend
                best_params = [np.mean(y_train), 0.0]
                mse = np.mean((y_train - best_params[0]) ** 2)

            self.col_info[col_i] = {'model': best_model, 'params': best_params, "mse": mse}

        return self

    def transform(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Removes drift per column for all rows.
        Nans are kept but ignored.

        :param data: Array of shape (num_samples, num_cols).
        :type data: np.ndarray
        :param x: Optional array of shape (num_samples,) for x-axis values.
        :type x: Optional[np.ndarray]
        :return: Drift-removed and scaled array of the same shape, with NaNs preserved.
        :rtype: np.ndarray
        """
        if x is None:
            x = np.arange(data.shape[0])

        n_samples, n_cols = data.shape
        removed_matrix = np.full_like(data, np.nan, dtype=np.float32)

        for col_i in range(n_cols):
            info = self.col_info.get(col_i)
            if info is None or info['model'] is None:
                continue

            model, params = info['model'], info['params']
            col_data = data[:, col_i]
            valid_mask = ~np.isnan(col_data)

            x_valid = x[valid_mask]
            y_valid = col_data[valid_mask]
            y_removed = model.transform(x_valid, y_valid, *params)
            removed_matrix[valid_mask, col_i] = y_removed

        nan_mask = np.isnan(data)
        removed_matrix[nan_mask] = np.nan

        return removed_matrix

    def inverse_transform(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Re-applies drift per column.
        Nans are kept but ignored.

        :param data: Scaled array of shape (num_samples, num_cols).
        :type data: np.ndarray
        :param x: Optional array of shape (num_samples,) for x-axis values.
        :type x: Optional[np.ndarray]
        :return: Original-scale array with NaNs re-inserted.
        :rtype: np.ndarray
        """
        if x is None:
            x = np.arange(data.shape[0])

        n_samples, n_cols = data.shape
        nan_mask = np.isnan(data)

        for col_i in range(n_cols):
            info = self.col_info.get(col_i)
            if info is None or info['model'] is None:
                continue

            model, params = info['model'], info['params']
            col_data = data[:, col_i]
            valid_mask = ~nan_mask[:, col_i]

            x_valid = x[valid_mask]
            y_unscaled = col_data[valid_mask]
            y_original = model.inverse_transform(x_valid, y_unscaled, *params)
            data[valid_mask, col_i] = y_original

        return data

    def output_trends(self, x: np.ndarray) -> np.ndarray:
        """
        Outputs the trend for each column.
        :param x: 1D array of x-axis values.
        :type x: np.ndarray
        :return: 2D array of shape (num_samples, num_cols) with the trend for each column.
        :rtype: np.ndarray
        """
        n_samples = x.shape[0]
        trends_matrix = np.full((n_samples, len(self.col_info)), np.nan, dtype=np.float32)

        for col_i, info in self.col_info.items():
            if info['model'] is None:
                continue

            model, params = info['model'], info['params']
            trends_matrix[:, col_i] = model.predict(x, *params)

        return trends_matrix


    @staticmethod
    def _fit_best_trend_model(
        x: np.ndarray, y: np.ndarray, config: TrendRemovalConfig
    ) -> Tuple[Optional[ITrend], Optional[Tuple[float, ...]], float]:
        """
        Attempts to fit linear and exponential models to find the best MSE.

        :param x: 1D array of x-axis values with no NaNs.
        :type x: np.ndarray
        :param y: 1D array of y-axis values with no NaNs.
        :type y: np.ndarray
        :return: (best_model_class, best_params, best_mse).
        :rtype: Tuple[Optional[ITrend], Optional[Tuple[float, ...]], float]
        """
        logger = logging.getLogger(TrendRemovalScaler.__name__)
        # Get candidates according to config
        candidates: List[Tuple[str, ClassVar]] = []

        if config.include_exponential:
            candidates.append((ExponentialTrend.name, ExponentialTrend))

        if config.include_linear:
            candidates.append((LinearTrend.name, LinearTrend))

        if config.include_inverse_exponential:
            candidates.append((InverseExponentialTrend.name, InverseExponentialTrend))

        # Initialize metrics
        best_mse = np.inf
        best_model = None
        best_params = None

        # Track the linear slope to discard other models if it is too high
        linear_parameters = None

        for (name, trend_class) in candidates:
            def residuals(params, x_r, y_r):
                r = trend_class.residuals(x_r, y_r, *params)
                return r

            # Discard exponential models if the slope is too low
            if ((linear_parameters is not None)
                    and (name == ExponentialTrend.name or name == InverseExponentialTrend.name)
                    and (linear_parameters[1] < 0.2/(x[-1]-x[0]))):
                continue

            p0 = trend_class.initial_guess(x, y)
            is_valid = True
            for i, p in enumerate(p0):
                bounds = trend_class.bounds(x, y)
                if p < bounds[0][i] or p > bounds[1][i]:
                    is_valid = False
                    break

            if not is_valid:
                continue

            try:
                res = least_squares(
                    fun=residuals,
                    x0=p0,
                    args=(x, y),
                    max_nfev=config.max_iter,
                    bounds=trend_class.bounds(x, y)
                )

                params = res.x
                final_residuals = residuals(params, x, y)
                mse = np.mean(final_residuals ** 2)

                if name == ExponentialTrend.name:
                    a, b, c = params
                    # we solve for the exponential model y = a + b * exp(c * x) where dy/dx = 0
                    # we want this to be at least 24 units before the maximum train cutoff or a very long time after
                    # this is only relevant when c is not too small
                    x_star = -np.log(b) / c - np.max(x)
                    if c > 8e-3 and (x_star > -24) and (x_star < 12*10):
                        mse += 1000

                    mse *= config.exponential_mse_scale

                if name == LinearTrend.name:
                    linear_parameters = params
                    mse *= config.linear_mse_scale

                if name == InverseExponentialTrend.name:
                    mse *= config.inverse_exponential_mse_scale


                if mse < best_mse:
                    best_mse = mse
                    best_model = trend_class
                    best_params = params

            except RuntimeError as e:
                raise e
            except Exception as e:
                print(f"Error fitting {name}")
                print(f"Initial guess: {p0}")
                raise e

        return best_model, best_params, best_mse

    @staticmethod
    def _validate_config(config: TrendRemovalConfig):
        if not config.include_exponential and not config.include_linear:
            raise TrendRemovalConfigError("At least one trend type must be included in the config")