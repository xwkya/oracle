from typing import Optional

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor
from src.dl_framework.data_pipeline.scalers.trend_scaler import TrendRemovalScaler


class TrendRemovalProcessor(IProcessor):
    def __init__(self, cutoff_idx: int):
        super().__init__(
            name='TrendRemoval',
            description='Remove the trend from the data',
            invertible=True
        )

        self.scaler = TrendRemovalScaler(cutoff_idx)
        self.cutoff_idx = cutoff_idx

    def fit(self, data: InseeDataState):
        self.scaler.fit(data.data[:self.cutoff_idx, :])

    def transform(self, data):
        data.data = self.scaler.transform(data.data)
        return data

    def inverse_transform(self, data):
        pass