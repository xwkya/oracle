import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List

from src.dl_framework.data_pipeline.data_processor import DataProcessor
from src.dl_framework.data_pipeline.data_utils import DataUtils

class InseeDataset(Dataset):
    def __init__(self,
                 data_processor: DataProcessor,
                 min_window_length_year: int = 1,
                 max_window_length_year: int = 4,
                 number_of_samples: int = 100_000,
                 interpolate: bool = False,
                 # -- Masking probabilities
                 p_1_none: float = 0.1,
                 p_2_uniform: float = 0.2,
                 p_3_last1yr: float = 0.3,
                 p_4_table: float = 0.3,
                 p_uniform: float = 0.3,  # Probability to mask each cell in uniform masking
                 seed: Optional[int] = 42,
                 inference_mode: bool = False):
        """
        Dataset for generating synthetic data for the INSEE time series.
        :param data_processor: DataProcessor instance
        :param min_window_length_year: Minimum window length in years
        :param max_window_length_year: Maximum window length in years
        :param number_of_samples: Number of samples to generate
        :param interpolate: Whether to interpolate missing values
        :param p_1_none: Probability of no masking
        :param p_2_uniform: Probability of uniform masking (chunks of 1 year)
        :param p_3_last1yr: Probability of masking the last year
        :param p_4_table: Probability of masking a table
        :param p_uniform: Probability of masking each cell in uniform masking
        :param seed: Random seed
        :param inference_mode: Whether to generate data for inference (use train data and test targets)
        """
        super().__init__()

        logger = logging.getLogger(InseeDataset.__name__)
        self.logger = logger

        if inference_mode and interpolate:
            logger.warning("Interpolation in inference mode leads to data leakage. Setting interpolate=False.")
            interpolate = False

        # Normalize the probabilities
        p_sum = p_1_none + p_2_uniform + p_3_last1yr + p_4_table
        p_1_none = p_1_none / p_sum
        p_2_uniform = p_2_uniform / p_sum
        p_3_last1yr = p_3_last1yr / p_sum
        p_4_table = p_4_table / p_sum

        if not data_processor.is_fitted:
            logger.info("Fitting data processor from provider")
            data_processor.fit_from_provider()

        self.month_range = data_processor.date_range
        self.num_months = len(data_processor.date_range)
        self.test_start_idx = data_processor.cutoff_idx
        self.inference_mode = inference_mode

        self.min_window_length_months = 12 * min_window_length_year
        self.max_window_length_months = 12 * max_window_length_year

        # -- Masking probabilities
        self.p_1_none = p_1_none
        self.p_2_uniform = p_2_uniform
        self.p_3_last1yr = p_3_last1yr
        self.p_4_table = p_4_table
        self.p_uniform = p_uniform

        # Number of training examples to be generated
        self.number_of_samples = number_of_samples

        # Optionally set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        table_data_dict: Dict[str, np.ndarray] = {}
        table_column_names: Dict[str, Dict[int, str]] = {}

        # Transform each table from (L, k_i) => (L, 3*k_i)
        for data_state in data_processor.transform_from_provider():
            table_name = data_state.table_name
            table_column_names[table_name] = data_state.col_to_name
            raw_table = data_state.data  # shape (L, k_i)
            L, k_i = raw_table.shape

            # Interpolate missing values if needed
            if interpolate:
                interpolated_value = DataUtils.interpolate_missing_values(raw_table)
            else:
                interpolated_value = None

            exp_missing, true_missing = DataUtils.generate_expected_and_truly_missing_masks_vectorized(data_state, self.month_range)

            is_nan = np.isnan(raw_table)
            transformed_table = np.zeros((L, 3 * k_i), dtype=raw_table.dtype)

            # 1) First feature: if not NaN => x, if NaN => 0.0 or interpolated value
            if interpolate:
                transformed_table[:, 0::3] = np.where(is_nan, interpolated_value, raw_table)
            else:
                transformed_table[:, 0::3] = np.where(is_nan, 0.0, raw_table)

            # 2) Second feature: 1 if NaN & expected == True, else 0
            transformed_table[:, 1::3] = np.where(is_nan & exp_missing, 1.0, 0.0)
            
            # 3) Third feature: 1 if NaN & expected == False, else 0
            transformed_table[:, 2::3] = np.where(is_nan & true_missing, 1.0, 0.0)

            # Store the transformed table
            table_data_dict[table_name] = transformed_table.astype(np.float32)

        self.table_column_names = table_column_names
        self.table_data_dict = table_data_dict
        self.num_tables = len(table_data_dict)

        # Get the table names and shapes in a consistent order
        self.table_names = list(table_data_dict.keys())
        self.table_names.sort()
        self.table_shapes = [table_data_dict[tn].shape[1] // 3 for tn in self.table_names]

    def __len__(self):
        return self.number_of_samples

    def normalize_probabilities(self):
        p_sum = self.p_1_none + self.p_2_uniform + self.p_3_last1yr + self.p_4_table
        self.p_1_none = self.p_1_none / p_sum
        self.p_2_uniform = self.p_2_uniform / p_sum
        self.p_3_last1yr = self.p_3_last1yr / p_sum
        self.p_4_table = self.p_4_table / p_sum

    def __getitem__(self, idx):
        """
        Returns a dict:
            {
              "full_data": { table_name -> np.ndarray of shape (window_length, 3*k_i) },
              "mask": np.ndarray of shape (window_length, num_tables),
            }
        """
        # Sample a random window length in years (inclusive)
        window_length = np.random.randint(self.min_window_length_months // 12,
                                          self.max_window_length_months // 12 + 1)
        window_length *= 12

        # Sample a random start index if not in inference mode
        if self.inference_mode:
            start_idx = self.test_start_idx - window_length + 12
        else:
            max_start = self.test_start_idx - window_length
            if max_start < 0:
                raise ValueError("Not enough months for the requested window in training set.")

            start_idx = (np.random.randint(0, max_start + 1) // 12) * 12
        end_idx = start_idx + window_length

        # Prepare output data structures
        full_data = {}
        mask = np.zeros((window_length, self.num_tables), dtype=np.float32)

        # Decide which masking mode to apply
        r = np.random.rand()
        if self.inference_mode:
            mask_mode = "last1yr"
        else:
            if r < self.p_1_none:
                mask_mode = "none"
            elif r < self.p_1_none + self.p_2_uniform:
                mask_mode = "uniform"
            elif r < self.p_1_none + self.p_2_uniform + self.p_3_last1yr:
                mask_mode = "last1yr"
            else:
                mask_mode = "table"

        # Slice out the transformed data (now shape = (L, 3*k_i))
        for i, tn in enumerate(self.table_data_dict.keys()):
            table_array = self.table_data_dict[tn]
            full_data[tn] = table_array[start_idx:end_idx, :]

        # Fill the mask
        if mask_mode == "none":
            # do nothing
            pass

        elif mask_mode == "uniform":
            # For each (t, i), mask with probability p_uniform
            random_matrix = np.random.rand(window_length // 12, self.num_tables) # (window_length // 12, num_tables)
            random_matrix = np.repeat(random_matrix, 12, axis=0) # (window_length, num_tables)

            mask[random_matrix < self.p_uniform] = 1.0

        elif mask_mode == "last1yr":
            omit_start = max(0, window_length - 12)
            mask[omit_start:, :] = 1.0

        elif mask_mode == "table":
            n_mask_tables = np.random.randint(1, self.num_tables + 1)
            table_indices_to_mask = np.random.choice(self.num_tables,
                                                     size=n_mask_tables,
                                                     replace=False)
            mask[:, table_indices_to_mask] = 1.0

        # 7) Return the sample
        return {
            "full_data": full_data,  # dict[str, np.ndarray], each (window_length, 3*k_i)
            "mask": mask,  # np.ndarray of shape (window_length, num_tables)
        }
