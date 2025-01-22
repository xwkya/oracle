from typing import Dict, Generator, Tuple

import pandas as pd

from src.data_sources.data_source import DataSource
from src.data_sources.insee.insee_data_provider import InseeDataProvider


class IDataProvider:
    def iter_data(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame]]:
        pass

class DataProvider:
    data_provider: Dict[DataSource, IDataProvider] = {
        DataSource.INSEE: InseeDataProvider(),
    }

    @staticmethod
    def iter_data(source: DataSource) -> Generator[Tuple[pd.DataFrame, pd.DataFrame]]:
        yield from DataProvider.data_provider[source].iter_data()