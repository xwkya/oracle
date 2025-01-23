from typing import Optional

from sklearn.preprocessing import StandardScaler

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor, IProcessorFactory


class StandardScalerProcessor(IProcessor):
    def __init__(self, cutoff_idx: int):
        super().__init__(
            name='StandardScaler',
            description='Standardize features by removing the mean and scaling to unit variance',
            invertible=True
        )

        self.cutoff_idx = cutoff_idx
        self.scaler = StandardScaler()

    def fit(self, data: InseeDataState):
        self.scaler.fit(data.data[:self.cutoff_idx, :])

    def transform(self, data: InseeDataState) -> InseeDataState:
        data.data = self.scaler.transform(data.data)
        return data

    def inverse_transform(self, data: InseeDataState) -> InseeDataState:
        data.data = self.scaler.inverse_transform(data.data)
        return data

class StandardScalerProcessorFactory(IProcessorFactory):
    def __init__(self, cutoff_idx: int):
        super().__init__()
        self.cutoff_idx = cutoff_idx

    def create(self, **kwargs) -> IProcessor:
        return StandardScalerProcessor(self.cutoff_idx)