from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Type, List

import pandas as pd

from azureorm.BaseTable import BaseTable
from src.core_utils import CoreUtils


class IDataFetcher:
    """
    Contract implemented by every source to fetch its data.
    The data is just a DataFrame with the csv data.
    """
    def fetch_data(self) -> pd.DataFrame:
        pass

class IDataPipeline:
    def output_to_file(self, file_path: str) -> DataPipeline:
        pass

    def output_to_db(self, db_model) -> DataPipeline:
        pass

    def run_pipeline(self):
        pass


class DataPipeline(ABC, IDataPipeline):
    """
    Contract implemented by every pipeline to process the data.
    The output format should always follow this pattern:
    - The first column should be the period, and the rest should be the values.
    - The period column should be named 'Period' and be of type pd.Period
    """
    def __init__(self, data_fetchers: List[IDataFetcher]):
        self.data_fetchers = data_fetchers
        self.countries = CoreUtils.get_countries_of_interest()

        # Saving parameters
        self.save_to_file = False
        self.file_path = None
        self.save_to_db = False
        self.db_model: Optional[Type[BaseTable]] = None

    def output_to_file(self, file_path: str) -> DataPipeline:
        self.save_to_file = True
        self.file_path = file_path
        return self

    def output_to_db(self, db_model) -> DataPipeline:
        self.save_to_db = True
        self.db_model = db_model
        return self

    @abstractmethod
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def run_pipeline(self):
        merged_df = None
        for data_fetcher in self.data_fetchers:
            data = data_fetcher.fetch_data()
            processed_data = self._process_data(data)
            if merged_df is None:
                merged_df = processed_data
            else:
                merged_df = pd.concat([merged_df, processed_data])

        if self.save_to_file:
            # Create directory if not exists
            file_dir = os.path.dirname(self.file_path)
            os.makedirs(file_dir, exist_ok=True)

            merged_df.to_csv(self.file_path, index=False)

        if self.save_to_db:
            orm = CoreUtils.get_orm()
            orm.bulk_insert_records_with_progress(
                self.db_model,
                merged_df.to_dict(orient='records'),
                chunk_size=1_000,
                log_progress=True,
                count=len(merged_df)
            )
