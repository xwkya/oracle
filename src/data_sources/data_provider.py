from typing import Dict, Generator, Tuple
from datetime import datetime
import pandas as pd

from src.data_sources.data_provider_contract import IDataProvider
from src.data_sources.data_source import DataSource
from src.data_sources.insee.insee_data_provider import InseeDataProvider
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class DataProvider:
    data_provider: Dict[DataSource, IDataProvider] = {
        DataSource.INSEE: InseeDataProvider(),
    }

    @staticmethod
    def iter_data(source: DataSource, min_date: datetime, max_date: datetime) -> Generator[InseeDataState, None, None]:
        yield from DataProvider.data_provider[source].iter_data(min_date, max_date)