from typing import Optional

import numpy as np

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor, IProcessorFactory


class RangeScalerProcessor(IProcessor):
    def __init__(self, cutoff_idx: int):
        super().__init__(
            name='RangeScaler',
            description='Standardize features by dividing the value by the absolute maximum value in the training set',
            invertible=True
        )

        self.cutoff_idx = cutoff_idx
        self.ranges: Optional[np.ndarray] = None

    def fit(self, data: InseeDataState):
        max_array = np.nanmax(data.data[:self.cutoff_idx, :], axis=0)
        min_array = np.nanmin(data.data[:self.cutoff_idx, :], axis=0)

        # Max of 1e-3 to avoid division by zero
        self.ranges = np.maximum(np.maximum(np.abs(max_array), np.abs(min_array)), 1e-3)

    def transform(self, data: InseeDataState) -> InseeDataState:
        data.data = data.data / self.ranges
        return data

    def inverse_transform(self, data: InseeDataState) -> InseeDataState:
        data.data = data.data * self.ranges
        return data

class RangeScalerProcessorFactory(IProcessorFactory):
    def __init__(self, cutoff_idx: int):
        super().__init__()
        self.cutoff_idx = cutoff_idx

    def create(self, **kwargs) -> IProcessor:
        return RangeScalerProcessor(self.cutoff_idx)