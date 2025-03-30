import pandas as pd

from src.core_utils import CoreUtils


class InflationFetcher:
    """
    Get inflation index from FRED: https://fred.stlouisfed.org/series/CPIAUCNS
    This script loads the csv from the filepath specified in appsettings.ini
    """
    @staticmethod
    def load_inflation_data() -> pd.DataFrame:
        """
        Load the inflation data.
        :return: The inflation data as a DataFrame.
        """
        config = CoreUtils.load_ini_config()
        inflation_file_path = config["datasets"]["InflationIndexFilePath"]
        df = pd.read_csv(inflation_file_path)
        return df
