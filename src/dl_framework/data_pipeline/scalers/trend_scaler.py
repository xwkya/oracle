from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from src.dl_framework.data_pipeline.scalers.trends import LinearTrend, ExponentialTrend, ITrend

@dataclass
class TrendRemovalConfig:
    include_exponential: bool = True
    include_linear: bool = True

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
        self.config = config or TrendRemovalConfig()

        self.table_scaler = None
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

        train_removed = np.full((self.train_cutoff_idx, n_cols), np.nan, dtype=np.float32)

        for col_i in range(n_cols):
            col_data = data[:, col_i]
            train_mask = (x < self.train_cutoff_idx) & ~np.isnan(col_data)
            if not np.any(train_mask):
                self.col_info[col_i] = {'model': None, 'params': None}
                continue

            x_train = x[train_mask]
            y_train = col_data[train_mask]

            best_model, best_params, _ = self._fit_best_trend_model(x_train, y_train, self.config)
            if best_model is None:
                best_model = LinearTrend
                best_params = [np.mean(y_train), 0.0]

            self.col_info[col_i] = {'model': best_model, 'params': best_params}

            y_train_removed = best_model.transform(x_train, y_train, *best_params)
            train_removed[train_mask, col_i] = y_train_removed

        for col_i in range(n_cols):
            col_vals = train_removed[:, col_i]
            not_nan_mask = ~np.isnan(col_vals)
            if np.any(not_nan_mask):
                mean_val = np.mean(col_vals[not_nan_mask])
                col_vals[~not_nan_mask] = mean_val

        self.table_scaler = StandardScaler()
        self.table_scaler.fit(train_removed)

        return self

    def transform(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Removes drift per column for all rows, then applies the single StandardScaler.
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

        filled_removed_matrix = removed_matrix.copy()
        col_means = self.table_scaler.mean_
        for col_i in range(n_cols):
            valid_mask = ~np.isnan(filled_removed_matrix[:, col_i])
            if not np.all(valid_mask):
                filled_removed_matrix[~valid_mask, col_i] = col_means[col_i]

        scaled_matrix = self.table_scaler.transform(filled_removed_matrix)
        nan_mask = np.isnan(data)
        scaled_matrix[nan_mask] = np.nan

        return scaled_matrix

    def inverse_transform(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reverses the scaling, then re-applies drift per column.
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
        filled_scaled_matrix = data.copy()
        nan_mask = np.isnan(data)
        col_means = self.table_scaler.mean_

        for col_i in range(n_cols):
            mask = np.isnan(filled_scaled_matrix[:, col_i])
            if np.any(mask):
                filled_scaled_matrix[mask, col_i] = col_means[col_i]

        unscaled_matrix = self.table_scaler.inverse_transform(filled_scaled_matrix)
        original_matrix = np.full_like(unscaled_matrix, np.nan, dtype=np.float32)

        for col_i in range(n_cols):
            info = self.col_info.get(col_i)
            if info is None or info['model'] is None:
                continue

            model, params = info['model'], info['params']
            col_data = unscaled_matrix[:, col_i]
            valid_mask = ~nan_mask[:, col_i]

            x_valid = x[valid_mask]
            y_unscaled = col_data[valid_mask]
            y_original = model.inverse_transform(x_valid, y_unscaled, *params)
            original_matrix[valid_mask, col_i] = y_original

        original_matrix[nan_mask] = np.nan
        return original_matrix

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

        # Get candidates according to config
        candidates = []

        if config.include_exponential:
            candidates.append((ExponentialTrend.name, ExponentialTrend))

        if config.include_linear:
            candidates.append((LinearTrend.name, LinearTrend))

        # Initialize metrics
        best_mse = np.inf
        best_model = None
        best_params = None
        y_scale = max(np.nanstd(y), (np.nanmax(y) - np.nanmin(y)) / 2)

        for (name, trend_class) in candidates:
            try:
                p0 = trend_class.initial_guess(y)
                params, _ = curve_fit(trend_class.predict, x, y, p0=p0, maxfev=5000)
                y_pred = trend_class.predict(x, *params)
                mse = np.mean((y_pred - y) ** 2)

                if name == 'exponential':
                    a, b, c = params
                    if b < 0 and c > 0:
                        continue
                    if (b > 0) and abs(-np.log(b) / c - np.max(x)) < 36:
                        mse += 1000
                    if abs(c) > 1e-1:
                        mse += 1.0 * y_scale

                if mse < best_mse:
                    best_mse = mse
                    best_model = trend_class
                    best_params = params

            except RuntimeError:
                continue

        return best_model, best_params, best_mse

    @staticmethod
    def _validate_config(config: TrendRemovalConfig):
        if not config.include_exponential and not config.include_linear:
            raise TrendRemovalConfigError("At least one trend type must be included in the config")