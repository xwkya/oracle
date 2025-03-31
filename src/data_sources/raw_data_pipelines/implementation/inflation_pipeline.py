import logging

import pandas as pd

from src.core_utils import CoreUtils
from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import IDataFetcher, DataPipeline


class InflationFetcher(IDataFetcher):
    """
    Get inflation index from FRED: https://fred.stlouisfed.org/series/CPIAUCNS
    This script loads the csv from the filepath specified in appsettings.ini
    """

    def fetch_data(self) -> pd.DataFrame:
        """
        Load the inflation data.
        :return: The inflation data as a DataFrame.
        """
        config = CoreUtils.load_ini_config()
        inflation_file_path = config["datasets"]["InflationIndexFilePath"]
        df = pd.read_csv(inflation_file_path)
        return df


class InflationDataPipeline(DataPipeline):
    def __init__(self):
        super().__init__([InflationFetcher()], DataSource.INFLATION)
        self.logger = logging.getLogger(InflationDataPipeline.__name__)

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Changing inflation index to datetime...")
        data["Period"] = pd.PeriodIndex(data["observation_date"], freq='Y')
        data.drop(columns=["observation_date"], inplace=True)

        return data