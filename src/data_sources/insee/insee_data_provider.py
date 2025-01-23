import datetime
from pathlib import Path

import pandas as pd

from typing import Generator, Tuple, List
from src.core_utils import CoreUtils
from src.data_sources.data_provider_contract import IDataProvider
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class InseeDataProvider(IDataProvider):
    data_path: str = "Data/insee/"

    def __init__(self):
        super().__init__()
        pass

    def iter_data(self, min_date: datetime, max_date: datetime) -> Generator[InseeDataState, None, None]:
        for table_name, csv_file, meta_csv_file in zip(*InseeDataProvider._get_csv_list()):
            df = pd.read_csv(csv_file, index_col='DATE_PARSED')
            # Convert the index to datetime
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

            # Filter the DataFrame
            df = df.loc[min_date:max_date]
            meta_df = pd.read_csv(meta_csv_file)

            yield InseeDataState(df, meta_df, table_name, 0)

    @staticmethod
    def _get_csv_list() -> Tuple[List[str], List[Path], List[Path]]:
        """
        Gets the list of pivot/meta CSV files in the data_dir.

        Returns:
            Tuple containing:
            - List[str]: Table names (stems of the pivot files)
            - List[Path]: Paths to pivot CSV files
            - List[Path]: Paths to corresponding metadata CSV files
        """
        root = CoreUtils.get_root()
        data_dir = root / InseeDataProvider.data_path

        # glob() returns an iterator of Path objects
        csv_files = list(data_dir.glob("*.csv"))

        # Filter pivot files (Path objects)
        pivot_files = [f for f in csv_files if "_meta.csv" not in f.name]
        pivot_files.sort()

        # Create metadata paths (keeping them as Path objects)
        metadata_files = [data_dir / f"{x.stem}_meta.csv" for x in pivot_files]

        # Extract table names as strings
        table_names = [f.stem for f in pivot_files]

        return table_names, pivot_files, metadata_files
