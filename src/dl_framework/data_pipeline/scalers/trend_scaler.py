import pandas as pd
import numpy as np

from typing import Tuple, Optional

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from src.dl_framework.data_pipeline.scalers.trends import LinearTrend, ExponentialTrend, ITrend

class TrendRemovalScaler:
    """
    For each column, we do:
      - Fit a drift model (linear or exponential) on train portion.
      - Remove that drift from the entire column (train+test).
      - Fit a StandardScaler on the drift-removed train portion.
      - In transform(), remove drift then apply the standard scaler.
      - In inverse_transform(), reverse the standard scaling and reapply drift.

    We'll store a dict {col_index: (model_name, model_params, standard_scaler)}.
    We also store the train_cutoff_idx to know which portion is "train".
    """
    def __init__(self, train_cutoff_idx: int):
        self.train_cutoff_idx = train_cutoff_idx
        self.stds = None
        self.table_scaler = None
        self.col_info = {}  # col -> dict with { 'model', 'params' }

    def fit(self, data: np.ndarray):
        """
        data: shape (num_samples, num_cols), with possible NaNs in test portion.
              We assume train portion can be forward/back-filled for the fit.
        """
        n_samples, n_cols = data.shape
        full_x = np.arange(n_samples)
        x_train = full_x[:self.train_cutoff_idx]
        train_data = data[:self.train_cutoff_idx, :]

        # Compute std on the original data and clamp
        stds = np.nanstd(train_data, axis=0)
        self.stds = np.maximum(stds, 1e-2)

        filled_train_data = pd.DataFrame(data[:self.train_cutoff_idx, :].copy()).ffill().bfill().values

        filled_train_data = filled_train_data / self.stds
        train_removed = np.zeros_like(filled_train_data)

        for col_i in range(n_cols):
            y_train = filled_train_data[:, col_i]

            # Fit best trend model
            best_model, best_params, best_mse = TrendRemovalScaler._fit_best_trend_model(x_train, y_train)
            if best_model is None:
                # Fallback: assume linear with slope=0
                best_model = LinearTrend
                best_params = [np.mean(y_train), 0.0]

            # Create drift-removed version of training data
            y_train_removed = best_model.transform(x_train, y_train, *best_params)
            train_removed[:, col_i] = y_train_removed

            self.col_info[col_i] = {
                'model': best_model,
                'params': best_params,
            }

        # Replace filled values with mean for standard scaler
        mask = np.isnan(train_data)
        train_removed[mask] = np.nan

        # Train standard scaler
        self.table_scaler = StandardScaler()
        self.table_scaler.fit(train_removed)
        return self

    def transform(self, data: np.ndarray, x: Optional[np.ndarray]=None) -> np.ndarray:
        """
        data: shape (num_samples, num_cols)
        x: optional array of shape (num_samples,) for the x-axis.
        Return the drift-removed + standard-scaled version of data.
        """
        n_samples, n_cols = data.shape
        original_data = data
        data = data.copy()
        if x is None:
            x = np.arange(n_samples)

        removed_matrix = np.zeros_like(data, dtype=np.float32)
        stds = self.stds
        data = data / stds
        data_filled = pd.DataFrame(data).ffill().bfill().values

        for col_i in range(n_cols):
            col_data_filled = data_filled[:, col_i]

            info = self.col_info[col_i]
            model, params = info['model'], info['params']

            # Remove the drift
            col_removed = model.transform(x, col_data_filled, *params)
            removed_matrix[:, col_i] = col_removed

        # Now standard-scale
        scaled_matrix = self.table_scaler.transform(removed_matrix)

        # Put NaNs back in their positions
        nan_mask = np.isnan(original_data)
        scaled_matrix[nan_mask] = np.nan

        return scaled_matrix

    def inverse_transform(self, data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        data: shape (num_samples, num_cols), presumably scaled by transform().
        x: optional array of shape (num_samples,).

        Return: the original-scale data (with drift re-applied).
        """
        n_samples, n_cols = data.shape
        input_data = data
        data = data.copy()
        if x is None:
            x = np.arange(n_samples)

        removed_matrix = self.table_scaler.inverse_transform(data)  # shape (num_samples, n_cols)
        removed_matrix = np.nan_to_num(removed_matrix, nan=0.)

        for col_i in range(n_cols):
            info = self.col_info[col_i]
            model = info['model']
            params = info['params']

            col_removed = removed_matrix[:, col_i]
            # Add drift
            col_original = model.inverse_transform(x, col_removed, *params)

            removed_matrix[:, col_i] = col_original

        # Scale back by stds
        original_matrix = removed_matrix * self.stds

        # Re-insert NaNs from the input data
        mask = np.isnan(input_data)
        original_matrix[mask] = np.nan

        return original_matrix

    @staticmethod
    def _fit_best_trend_model(x, y) -> Tuple[ITrend, Tuple, float]:
        """
        Given 1D arrays x and y (no NaNs), try fitting the trend models.
        Currently, supports LinearTrend and ExponentialTrend.

        :param x: 1D array of x values
        :param y: 1D array of y values
        :return: (best_model, best_params, best_mse, best_mse)
        """
        candidates = [
            (LinearTrend.name, LinearTrend),
            (ExponentialTrend.name, ExponentialTrend),
        ]

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

                # Some penalty heuristics to avoid degenerate exponentials

                if name == 'exponential':
                    a, b, c = params[0], params[1], params[2]
                    # b < 0 and c > 0 means exponential decline, extremely unlikely
                    if b < 0 and c > 0:
                        continue

                    # y = a + b*e^(cx) <=> y = a + e^(cx + log(b))
                    # we will impose that growth cannot become > 1 between the 3 years around X.
                    # e.g cx + log(b) gives the x at which the slope is 1, compute absolute difference to x_max
                    if (b > 0) and abs(-np.log(b) / c - np.max(x)) < 36:
                        mse += 1000

                    # Data is normalized, penalize large c
                    if abs(c) > 1e-1:
                        mse += 1. * y_scale

                if mse < best_mse:
                    best_mse = mse
                    best_model = trend_class
                    best_params = params


            except RuntimeError as e:
                raise e

        return best_model, best_params, best_mse