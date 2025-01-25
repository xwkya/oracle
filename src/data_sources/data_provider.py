import logging
from typing import Dict, Generator, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_sources.data_provider_contract import IDataProvider
from src.data_sources.data_source import DataSource
from src.data_sources.insee.insee_data_provider import InseeDataProvider
from src.date_utils import DateUtils
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class DataProvider:
    data_provider: Dict[DataSource, IDataProvider] = {
        DataSource.INSEE: InseeDataProvider(),
    }
    logger = logging.getLogger("DataProvider")

    @classmethod
    def iter_data(
            cls,
            source: DataSource,
            min_date: datetime,
            max_date: datetime,
            monthly=True,
            max_elements: Optional[int]=None) -> Generator[InseeDataState, None, None]:
        """
        Iterate over the data from the specified source between the specified dates.
        :param source: The data source (e.g. DataSource.INSEE)
        :param min_date: The minimum date to fetch.
        :param max_date: The maximum date to fetch.
        :param monthly: Whether to fetch the data monthly or relative to its original frequency.
        :param max_elements: The maximum number of elements to fetch.
        :return: A generator of InseeDataState objects.
        """

        streamed_elements = 0

        for data_state in cls.data_provider[source].iter_data(min_date, max_date):
            if max_elements is not None and streamed_elements >= max_elements:
                break

            if monthly:
                monthly_dates = DateUtils.month_range(min_date, max_date)
                date_to_idx = {d: i for i, d in enumerate(monthly_dates)}

                # Create array_data (num_months, number_of_features)
                num_months = len(monthly_dates)
                num_features = data_state.data.shape[1]

                array_data = np.full((num_months, num_features), np.nan, dtype=np.float32)
                for i, date in enumerate(data_state.dates):
                    date_str = date.strftime("%Y-%m-01")
                    if date_str in date_to_idx:
                        idx = date_to_idx[date_str]
                        array_data[idx] = data_state.data[i, :]

                data_state.data = array_data
                data_state.dates = [DateUtils.parse_date(d) for d in monthly_dates]

            try:
                data_state.check_sanity()
            except AssertionError as e:
                cls.logger.error(f"Data state sanity check failed")
                raise e

            streamed_elements += 1
            yield data_state

        cls.logger.info(f"Streamed {streamed_elements} elements from {source.name}")