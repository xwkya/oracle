import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class InseeDataState:
    table_name: str
    data: np.ndarray
    col_to_freq: Dict[int, str]
    col_to_name: Dict[int, str]
    start_index: int = 0

    def __init__(self, df_pivot: pd.DataFrame, df_meta: pd.DataFrame, table_name: str, start_index: int = 0):
        """
        Create a new InseeDataState object. The data is expected to be a pivoted DataFrame and metadata DataFrame.
        :param df_pivot: the pivoted DataFrame
        :param df_meta: the metadata DataFrame
        :param table_name: the name of the table
        :param start_index: the starting index for the data (used for transforming/reverse transforming slices)
        """
        self.table_name = table_name
        self.data = df_pivot.values
        self.meta_data = df_meta.values
        self.start_index = start_index
        self._generate_col_infos(df_pivot, df_meta)

    def _generate_col_infos(self, df_pivot: pd.DataFrame, df_meta: pd.DataFrame):
        logger = logging.getLogger(InseeDataState.__name__)
        col_names = {}
        col_to_freq = {}
        feature_cols = [col for col in df_pivot.columns if col != 'DATE_PARSED']

        index_to_keep = []

        for col_index, col_name in enumerate(feature_cols):

            row_meta = df_meta[df_meta['TITLE_FR'] == col_name]
            if len(row_meta) == 0:
                print(f"Could not find metadata for column {col_name}")
                print(f"Table name: {self.table_name}")
                print(f"Available columns: {df_meta['TITLE_FR'].values}")
                raise Exception("Metadata not found for column")

            index_to_keep.append(col_index)
            freq_str = row_meta['FREQ'].values[0]
            col_to_freq[col_index] = freq_str
            col_names[col_index] = col_name

        # Map the broken indexes
        new_index = list(range(len(index_to_keep)))
        for old_index, new_index in zip(index_to_keep, new_index):
            col_to_freq[new_index] = col_to_freq.pop(old_index)
            col_names[new_index] = col_names.pop(old_index)

        self.data = self.data[:, index_to_keep]
        self.col_to_freq = col_to_freq
        self.col_names = col_names