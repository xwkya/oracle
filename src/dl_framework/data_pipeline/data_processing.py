from __future__ import annotations
import logging
from typing import Tuple, List, Dict, Generator

from datetime import datetime
from src.data_sources.data_provider import DataProvider
from src.data_sources.data_source import DataSource
from src.date_utils import DateUtils
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor, IProcessorFactory
from src.dl_framework.data_pipeline.processors.low_variance_drop import LowVarianceDropFactory
from src.dl_framework.data_pipeline.processors.range_scaler_processor import RangeScalerProcessorFactory
from src.dl_framework.data_pipeline.processors.standard_scaler import StandardScalerProcessorFactory
from src.dl_framework.data_pipeline.processors.trend_removal_processor import TrendRemovalProcessorFactory
from src.dl_framework.data_pipeline.scalers.trend_scaler import TrendRemovalConfig


class DataProcessing:
    def __init__(self, data_source: DataSource, min_date: datetime, max_date: datetime, train_cutoff: datetime):
        self.logger = logging.getLogger(DataProcessing.__name__)
        self.processor_factories: List[IProcessorFactory] = []

        self.data_source = data_source
        self.min_date = min_date
        self.max_date = max_date
        self.train_cutoff = train_cutoff
        self.date_range = DateUtils.month_range(min_date, max_date)

        date_to_idx = {d: i for i, d in enumerate(self.date_range)}
        self.cutoff_idx = date_to_idx[train_cutoff.strftime('%Y-%m-01')]
        self.data_provider = DataProvider()
        self.processor_cache: Dict[str, List[IProcessor]] = {}


    def add_scaler(self, with_mean: bool=True, with_std: bool=True) -> DataProcessing:
        self.processor_factories.append(
            StandardScalerProcessorFactory(cutoff_idx=self.cutoff_idx, with_mean=with_mean, with_std=with_std))
        return self

    def add_trend_removal(self, config: TrendRemovalConfig=None) -> DataProcessing:
        self.processor_factories.append(TrendRemovalProcessorFactory(cutoff_idx=self.cutoff_idx, config=config))
        return self

    def add_variance_drop(self, variance_threshold: float=1e-5) -> DataProcessing:
        self.processor_factories.append(LowVarianceDropFactory(cutoff_idx=self.cutoff_idx, variance_threshold=variance_threshold))
        return self

    def add_range_scaler(self) -> DataProcessing:
        self.processor_factories.append(RangeScalerProcessorFactory(cutoff_idx=self.cutoff_idx))
        return self

    # ---------------------------------------------
    # Fit methods
    # ---------------------------------------------
    def fit(self, generator: Generator[InseeDataState, None, None]):
        for data_state in generator:
            table_name = data_state.table_name

            self.processor_cache[table_name] = []
            for processor_factory in self.processor_factories:
                processor = processor_factory.create()
                processor.fit(data_state)
                data_state = processor.transform(data_state)

                self.processor_cache[table_name].append(processor)

            self.logger.info(f"Finished fitting processors for table {table_name}")

    def fit_from_provider(self):
        return self.fit(self.data_provider.iter_data(self.data_source, self.min_date, self.max_date))

    # ---------------------------------------------
    # Transform methods
    # ---------------------------------------------
    def transform_from_provider(self):
        yield from self.transform(self.data_provider.iter_data(self.data_source, self.min_date, self.max_date))

    def transform(self, generator: Generator[InseeDataState, None, None]) \
            -> Generator[Dict[str, InseeDataState], None, None]:
        for data_state in generator:
            table_name = data_state.table_name
            for processor in self.processor_cache[table_name]:
                data_state = processor.transform(data_state)

            yield data_state

    # ---------------------------------------------
    # Inverse transform methods
    # ---------------------------------------------
    def inverse_transform(self, generator: Generator[InseeDataState, None, None])\
            -> Generator[Dict[str, InseeDataState], None, None]:
        """
        Transforms the data back to the last non-invertible processor (non-invertible processors lose the data forever)
        """
        for data_state in generator:
            table_name = data_state.table_name
            for processor in reversed(self.processor_cache[table_name]):
                if not processor.invertible:
                    break

                data_state = processor.inverse_transform(data_state)

            yield data_state