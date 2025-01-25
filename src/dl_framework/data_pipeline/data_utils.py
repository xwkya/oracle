from typing import List

import numpy as np

from src.date_utils import DateUtils
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class DataUtils:
    @staticmethod
    def is_expected_mask_for_col(freq_str: str, months_of_date: np.ndarray) -> np.ndarray:
        """
        Given a freq_str in {'A','Q','M'} and an array of month integers,
        return a boolean mask of where we expect a value for that freq.
        E.g. for 'Q', we might only expect months 3,6,9,12; for 'A', only month 1, etc.
        """
        if freq_str == 'M':
            return np.ones_like(months_of_date, dtype=bool)
        elif freq_str == 'T':
            return np.isin(months_of_date, [1, 4, 7, 10])
        elif freq_str == 'A':
            return months_of_date == 1
        else:
            raise ValueError(f"Unknown frequency string: {freq_str}")

    @staticmethod
    def generate_expected_and_truly_missing_masks_vectorized(data_state: InseeDataState, date_list: List[str]) \
            -> (np.ndarray, np.ndarray):
        """
        Generates two boolean masks of shape (L, k_i):
          - expected_missing_mask: True where the value is *missing* but that is *expected*
                                   (due to freq not reporting that month)
          - truly_missing_mask:    True where the value is missing but it *should* have been reported
        :param data_state: The InseeDataState (L, k_i)
        :param date_list: A list of date strings, e.g. ['2020-01-01', '2020-02-01', ...]
        :return: A tuple of two boolean masks (expected_missing_mask, truly_missing_mask)
        """
        L, k_i = data_state.data.shape

        months_of_date = np.array([DateUtils.parse_date(date).month for date in date_list], dtype=int)  # shape (L,)
        freq_array = np.array([data_state.col_to_freq[c] for c in range(k_i)], dtype=object)  # shape (k_i,)

        expected_array = np.zeros((L, k_i), dtype=bool)
        for c in range(k_i):
            # noinspection PyTypeChecker
            freq_str: str = freq_array[c]
            expected_array[:, c] = DataUtils.is_expected_mask_for_col(freq_str, months_of_date)

        # Check where table_array is NaN => "missing"
        missing_mask = np.isnan(data_state.data)  # shape (L, k_i)

        truly_missing_mask = missing_mask & expected_array
        expected_missing_mask = missing_mask & ~expected_array

        return expected_missing_mask, truly_missing_mask

    @staticmethod
    def interpolate_missing_values(data: np.ndarray) -> np.ndarray:
        """
        Interpolates missing values in a 2D array.
        :param data: the 2D array to interpolate
        :return: the interpolated 2D array
        """
        interpolated_data = data.copy()
        valid_masks = ~np.isnan(interpolated_data)

        # Loop over each column in val_columns
        for c in range(interpolated_data.shape[1]):
            col = interpolated_data[:, c]
            valid_mask = valid_masks[:, c]

            # Proceed only if we have at least two valid points to interpolate between
            if valid_mask.sum() > 1:
                col_x = np.flatnonzero(valid_mask)
                col_y = col[valid_mask]

                # Index of non zero values in valid mask (missing points)
                missing_x = np.flatnonzero(~valid_mask)

                # Interpolate and fill those missing points
                col[~valid_mask] = np.interp(missing_x, col_x, col_y)

        return interpolated_data
