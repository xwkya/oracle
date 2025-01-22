import pandas as pd

from typing import Generator, Tuple, List
from src.core_utils import CoreUtils
from src.data_sources.data_provider import IDataProvider


class InseeDataProvider(IDataProvider):
    data_path: str = "Data/insee/"

    def iter_data(self) -> Generator[Tuple[pd.DataFrame, pd.DataFrame]]:
        for csv_file, meta_csv_file in InseeDataProvider._get_csv_list():
            df = pd.read_csv(csv_file)
            meta_df = pd.read_csv(meta_csv_file)

            yield df, meta_df


    @staticmethod
    def _get_csv_list() -> Tuple[List[str], List[str]]:
        """
        Gets the list of pivot/meta CSV files in the data_dir.
        :return: a tuple with the list of pivot CSV files and the list of metadata CSV files
        """

        root = CoreUtils.get_root()
        data_dir = root / InseeDataProvider.data_path

        csv_files = list(data_dir.glob("*.csv"))
        metadata_files = [f for f in csv_files if "_meta.csv" in f.name]
        pivot_files = [f for f in csv_files if "_meta.csv" not in f.name]

        return pivot_files, metadata_files
