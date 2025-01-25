import logging

import numpy as np

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor, IProcessorFactory
from src.dl_framework.data_pipeline.scalers.trend_scaler import TrendRemovalScaler, TrendRemovalConfigError, \
    TrendRemovalConfig


class TrendRemovalProcessor(IProcessor):
    def __init__(self, cutoff_idx: int, config: TrendRemovalConfig=None):
        super().__init__(
            name='TrendRemoval',
            description='Remove the trend from the data',
            invertible=True
        )
        self.logger = logging.getLogger(TrendRemovalProcessor.__name__)

        try:
            self.scaler = TrendRemovalScaler(cutoff_idx, config)
        except TrendRemovalConfigError as e:
            self.logger.error(f"Error while creating TrendRemovalScaler: {e}")
            self.logger.info("Error is not critical, proceeding with the rest of the pipeline")
        except Exception as e:
            self.logger.error(f"Unexpected error while creating TrendRemovalScaler: {e}")
            raise e

        self.cutoff_idx = cutoff_idx

    def fit(self, data: InseeDataState):
        self.scaler.fit(data.data[:self.cutoff_idx, :])

    def transform(self, data_state: InseeDataState) -> InseeDataState:
        x = np.arange(data_state.start_index, data_state.start_index + data_state.data.shape[0])
        data_state.data = self.scaler.transform(data_state.data, x)
        return data_state

    def inverse_transform(self, data_state: InseeDataState) -> InseeDataState:
        x = np.arange(data_state.start_index, data_state.start_index + data_state.data.shape[0])
        data_state.data = self.scaler.inverse_transform(data_state.data, x)
        return data_state

    def get_num_features(self):
        return len(self.scaler.col_info)

    def create_visualisation(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.output_trends(x)


class TrendRemovalProcessorFactory(IProcessorFactory):
    def __init__(self, cutoff_idx: int, config: TrendRemovalConfig=None):
        super().__init__()
        self.cutoff_idx = cutoff_idx
        self.config = config

    def create(self, **kwargs) -> IProcessor:
        return TrendRemovalProcessor(self.cutoff_idx, config=self.config)