import numpy as np

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor, IProcessorFactory


class LowVarianceDrop(IProcessor):
    def __init__(self, cutoff_idx: int, variance_threshold: float = 1e-5):
        super().__init__("LowVarianceDrop", invertible=False, description="Drops columns with low variance.")
        self.cutoff_idx = cutoff_idx
        self.variance_threshold = variance_threshold

    def fit(self, data: InseeDataState):
        pass

    def transform(self, data: InseeDataState) -> InseeDataState:
        data.data, data.col_to_freq, data.col_names = self._drop_low_variance_columns(
            data.data,
            data.col_names,
            data.col_to_freq,
            self.cutoff_idx,
            self.variance_threshold
        )

        return data

    @staticmethod
    def _drop_low_variance_columns(
            array_data: np.ndarray,
            col_names: dict,
            col_to_freq: dict,
            train_cutoff_idx: int,
            epsilon: float = 1e-5
    ):
        """
        Identify columns whose variance is below `epsilon` and drop them.
        :param array_data: The full data array (num_months, k_i)
        :param col_names: dict of {col_index -> col_name} for the current columns
        :param col_to_freq: dict of {col_index -> freq_str} for the current columns
        :param train_cutoff_idx: index representing the cutoff for training data
        :param epsilon: threshold for dropping columns by variance
        :return: (scaled_data, new_col_to_freq, new_col_names)
        """
        # Temporarily fill NaNs in train portion to fit an initial StandardScaler
        train_data = array_data[:train_cutoff_idx, :].copy()
        stds = np.nanstd(train_data, axis=0)

        keep_mask = stds >= epsilon

        if not np.any(keep_mask):
            print("[WARNING] All columns have variance below epsilon. Keeping them all.")
            keep_mask = np.ones_like(keep_mask, dtype=bool)

        array_data = array_data[:, keep_mask]

        old_cols = np.where(keep_mask)[0]
        new_col_to_freq = {}
        new_col_names = {}
        for new_i, old_i in enumerate(old_cols):
            new_col_to_freq[new_i] = col_to_freq[old_i]
            new_col_names[new_i] = col_names[old_i]

        return array_data, new_col_to_freq, new_col_names

class LowVarianceDropFactory(IProcessorFactory):
    def __init__(self, cutoff_idx: int, variance_threshold: float = 1e-5):
        super().__init__()
        self.cutoff_idx = cutoff_idx
        self.variance_threshold = variance_threshold

    def create(self) -> IProcessor:
        return LowVarianceDrop(cutoff_idx=self.cutoff_idx, variance_threshold=self.variance_threshold)