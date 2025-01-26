from __future__ import annotations

import pickle
import uuid
from functools import wraps
from typing import Tuple, List, Dict, Generator, Optional, Callable, Any
import logging

from datetime import datetime

import numpy as np
from dateutil.relativedelta import relativedelta

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


def with_processor_id(add_step_method: Callable[[Any, ...], DataProcessor]) -> Callable[[Any, ...], DataProcessor]:
    """
    Decorator to add a processor ID to a processor in the data pipeline.
    Processor IDs will be stored in the pipeline's attribute _processor_ids.
    _processor_ids is a dictionary mapping each processor id to its index in the pipeline.
    :param add_step_method: the method to decorate
    """

    @wraps(add_step_method)
    def wrapper(self: DataProcessor, *args, processor_id: Optional[str] = None, **kwargs) -> DataProcessor:
        result = add_step_method(self, *args, **kwargs)
        if processor_id is None:
            # Create a random GUID for the processor ID
            processor_id = str(uuid.uuid4())

        processor_index = len(self.processor_factories) - 1
        self._processor_ids[processor_id] = processor_index

        return result

    return wrapper


class DataProcessor:
    def __init__(
            self,
            data_source: DataSource,
            min_date: datetime,
            max_date: datetime,
            train_cutoff: datetime,
            max_elements: Optional[int] = None):
        """
        Initializes the `DataProcessor` object with the given parameters.

        Example:
            >>> data_processor = DataProcessor(DataSource.INSEE, datetime(1970, 1, 1), datetime(2014, 1, 1), datetime(2010, 1, 1))
            >>> data_processor.add_scaler().add_trend_removal().add_variance_drop().add_range_scaler()
            >>> data_processor.fit_from_provider()
            >>> for data_state in data_processor.transform_from_provider():
            ...     print(data_state.data[:10, :])

        :param data_source: The data source to use (e.g. `DataSource.INSEE`)
        :param min_date: The minimum date to consider
        :param max_date: The maximum date to consider
        :param train_cutoff: The date to use as the training cutoff
        :param max_elements: The maximum number of tables to process
        """

        self.logger = logging.getLogger(DataProcessor.__name__)
        self.processor_factories: List[IProcessorFactory] = []

        # Keep track of the state
        self.is_fitted = True
        self.table_names: List[str] = []

        self.data_source = data_source
        self.min_date = min_date
        self.max_date = max_date
        self.train_cutoff = train_cutoff
        self.date_range = DateUtils.month_range(min_date, max_date)

        self.date_to_idx = {d: i for i, d in enumerate(self.date_range)}
        self.cutoff_idx = self.date_to_idx[train_cutoff.strftime('%Y-%m-01')]
        self.data_provider = DataProvider()
        self.processor_cache: Dict[str, List[IProcessor]] = {}
        self.max_elements = max_elements

        self._processor_ids: Dict[
            str, int] = {}  # Dict of processor names to their index in the processor_factories list

    @with_processor_id
    def add_scaler(self, with_mean: bool = True, with_std: bool = True) -> DataProcessor:
        self.processor_factories.append(
            StandardScalerProcessorFactory(cutoff_idx=self.cutoff_idx, with_mean=with_mean, with_std=with_std))
        self.is_fitted = False
        return self

    @with_processor_id
    def add_trend_removal(self, config: TrendRemovalConfig = None) -> DataProcessor:
        self.processor_factories.append(
            TrendRemovalProcessorFactory(cutoff_idx=self.cutoff_idx, config=config))
        self.is_fitted = False
        return self

    @with_processor_id
    def add_variance_drop(self, variance_threshold: float = 1e-5) -> DataProcessor:
        self.processor_factories.append(
            LowVarianceDropFactory(cutoff_idx=self.cutoff_idx, variance_threshold=variance_threshold))
        # Does not require fitting
        return self

    @with_processor_id
    def add_range_scaler(self) -> DataProcessor:
        self.processor_factories.append(
            RangeScalerProcessorFactory(cutoff_idx=self.cutoff_idx))
        self.is_fitted = False
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

            self.table_names.append(table_name)
            self.logger.info(f"Finished fitting processors for table {table_name}")

        self.is_fitted = True

    def fit_from_provider(self):
        self.fit(self._yield_from_data_provider())

    # ---------------------------------------------
    # Transform methods
    # ---------------------------------------------
    def transform(self, generator: Generator[InseeDataState, None, None]) \
            -> Generator[InseeDataState, None, None] | None:
        if not self.is_fitted:
            self.logger.error("DataProcessing object has not been fitted. Please call fit() first.")
            return

        for data_state in generator:
            table_name = data_state.table_name
            for processor in self.processor_cache[table_name]:
                data_state = processor.transform(data_state)

            yield data_state

    def transform_from_provider(self) \
            -> Generator[InseeDataState, None, None]:
        yield from self.transform(
            self._yield_from_data_provider()
        )

    # ---------------------------------------------
    # Inverse transform methods
    # ---------------------------------------------
    def inverse_transform(self, generator: Generator[InseeDataState, None, None]) \
            -> Generator[InseeDataState, None, None] | None | None:
        """
        Transforms the data back to the last non-invertible processor (non-invertible processors lose the data forever)
        :param generator: The generator to transform
        :return: A generator of the transformed data
        """
        if not self.is_fitted:
            self.logger.error("DataProcessing object has not been fitted. Please call fit() first.")
            return

        for data_state in generator:
            table_name = data_state.table_name
            for processor in reversed(self.processor_cache[table_name]):
                if not processor.invertible:
                    break

                data_state = processor.inverse_transform(data_state)

            yield data_state

    # ---------------------------------------------
    # Visualisation methods
    # ---------------------------------------------
    def obtain_trends(self, processor_id: str, min_date: datetime = None, max_date: datetime = None) \
            -> Generator[InseeDataState, None, None] | None:
        if processor_id not in self._processor_ids:
            raise ValueError(f"Processor with id {processor_id} has not been registered."
                             f" Call .add_trend_removal(processor_id={processor_id}) to register it.")

        if not self.is_fitted:
            self.logger.error("DataProcessing object has not been fitted. Please call fit() first.")
            return

        processor_index = self._processor_ids[processor_id]

        if not isinstance(self.processor_factories[processor_index], TrendRemovalProcessorFactory):
            raise ValueError(f"Processor with id {processor_id} is not a TrendRemovalProcessor. No trend to obtain.")

        # Resolve min/max date
        if min_date is None:
            min_date = self.min_date
        if max_date is None:
            max_date = self.max_date

        # Remove 1 month (standard is we exclude the upper bound)
        max_date = max_date - relativedelta(months=1)

        for table_name in self.processor_cache:
            # Create a dummy insee Data state with the trend
            min_date_idx = self.date_to_idx[min_date.strftime('%Y-%m-01')]
            max_date_idx = self.date_to_idx[max_date.strftime('%Y-%m-01')]

            x = np.arange(min_date_idx, max_date_idx + 1)
            trends = self.processor_cache[table_name][processor_index].create_visualisation(x)

            data_state = InseeDataState.from_data(
                data=trends,
                col_to_freq={i: 'M' for i in range(trends.shape[1])},
                col_to_name={i: f'Trend_{i}' for i in range(trends.shape[1])},
                dates=[DateUtils.parse_date(d) for d in self.date_range[min_date_idx:max_date_idx + 1]],
                table_name=table_name,
                start_index=min_date_idx,
            )

            for backward_processors in reversed(self.processor_cache[table_name][:processor_index]):
                if not backward_processors.invertible:
                    break

                data_state = backward_processors.inverse_transform(data_state)

            yield data_state

    # ---------------------------------------------
    # Save/Load methods
    # ---------------------------------------------
    def save(self, file_path: str) -> None:
        """
        Saves the entire DataProcessor state to a file, including factories,
        fitted processor cache, hyperparameters, etc.
        """
        # Build a dictionary of everything we need to recreate the DataProcessor.
        # (We typically skip self.logger and self.data_provider
        #  because they can be re-instantiated.)
        state_dict = {
            "data_source": self.data_source,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "train_cutoff": self.train_cutoff,
            "max_elements": self.max_elements,
            "processor_factories": self.processor_factories,
            "processor_cache": self.processor_cache,
            "table_names": self.table_names,
            "is_fitted": self.is_fitted,
            "_processor_ids": self._processor_ids,
            "date_range": self.date_range,
            "date_to_idx": self.date_to_idx,
            "cutoff_idx": self.cutoff_idx
        }

        with open(file_path, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, file_path: str) -> DataProcessor:
        """
        Loads a DataProcessor instance from a file path.
        Restores all the internal state, including factories,
        fitted processors, hyperparameters, etc.
        """
        with open(file_path, "rb") as f:
            state_dict = pickle.load(f)

        # Create a new instance using the constructor parameters
        new_instance = cls(
            data_source=state_dict["data_source"],
            min_date=state_dict["min_date"],
            max_date=state_dict["max_date"],
            train_cutoff=state_dict["train_cutoff"],
            max_elements=state_dict["max_elements"]
        )

        # Now set all the other attributes that we saved
        new_instance.processor_factories = state_dict["processor_factories"]
        new_instance.processor_cache = state_dict["processor_cache"]
        new_instance.table_names = state_dict["table_names"]
        new_instance.is_fitted = state_dict["is_fitted"]
        new_instance._processor_ids = state_dict["_processor_ids"]
        new_instance.date_range = state_dict["date_range"]
        new_instance.date_to_idx = state_dict["date_to_idx"]
        new_instance.cutoff_idx = state_dict["cutoff_idx"]

        # Re-instantiate a logger and data_provider so they're available
        new_instance.logger = logging.getLogger(DataProcessor.__name__)
        new_instance.data_provider = DataProvider()

        return new_instance

    def _yield_from_data_provider(self) -> Generator[InseeDataState, None, None]:
        return self.data_provider.iter_data(
            self.data_source,
            self.min_date,
            self.max_date,
            max_elements=self.max_elements)
