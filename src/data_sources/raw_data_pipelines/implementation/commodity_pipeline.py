import pandas as pd

from src.core_utils import CoreUtils
from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import IDataFetcher, DataPipeline


class CommodityDataFetcher(IDataFetcher):
    """
    Fetches commodity data from the World Bank commodity-market csv.
    CSV file can be downloaded from https://www.worldbank.org/en/research/commodity-markets
    """
    def fetch_data(self) -> pd.DataFrame:
        """
        Load the commodity data.
        :return: The commodity data as a DataFrame.
        """
        config = CoreUtils.load_ini_config()
        commodity_file_path = config["datasets"]["CommodityFilePath"]
        df = pd.read_csv(commodity_file_path, sep="\t")
        return df

class CommodityDataPipeline(DataPipeline):
    def __init__(self):
        super().__init__([CommodityDataFetcher()], DataSource.COMMODITY)

    """
    Preprocesses the commodity data.
    """
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the commodity data.
        :param df: The commodity data.
        :return: The preprocessed commodity data.
        """
        value_columns = list(df.columns[1:])

        # Convert the value columns to numeric
        df[value_columns] = df[value_columns].apply(pd.to_numeric, errors='coerce')
        return CommodityDataPipeline.period_to_datetime(df)

    @staticmethod
    def period_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
        # Period format is 1960M02, 1960M03, etc. Convert to datetime then period object
        df['Period'] = pd.to_datetime(df['Period'], format='%YM%m')
        df['Period'] = pd.PeriodIndex(df['Period'], freq='M')
        return df