from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class InseeDataState:
    table_name: str
    data: np.ndarray
    col_to_freq: Dict[int, str]
    col_to_name: Dict[int, str]
    dates: List[datetime]
    start_index: int = 0

    def __init__(
            self,
            table_name: str,
            df_pivot: pd.DataFrame=None,
            df_meta: pd.DataFrame=None,
            data: np.ndarray=None,
            dates: List[datetime]=None,
            col_to_freq: Dict[int, str]=None,
            col_to_name: Dict[int, str]=None,
            start_index: int = 0):
        """
        Create a new InseeDataState object. The data is expected to be a pivoted DataFrame and metadata DataFrame.
        :param df_pivot: the pivoted DataFrame
        :param df_meta: the metadata DataFrame
        :param table_name: the name of the table
        :param start_index: the starting index for the data (used for transforming/reverse transforming slices)
        """
        self.table_name = table_name
        self.start_index = start_index

        if data is not None:
            self.data = data
        else:
            self.data = df_pivot.values

        if dates is not None:
            self.dates = dates
        else:
            self.dates = [pd.to_datetime(x) for x in df_pivot.index]

        if col_to_freq is not None:
            self.col_to_freq = col_to_freq
            self.col_to_name = col_to_name
        else:
            self._generate_col_infos(df_pivot, df_meta)

    @classmethod
    def from_dataframe(cls, df_pivot: pd.DataFrame, df_meta: pd.DataFrame, table_name: str, start_index: int = 0):
        """
        Create a new InseeDataState object from the dataframes and metadata.
        :param df_pivot: the pivoted DataFrame
        :param df_meta: the metadata DataFrame
        :param table_name: the name of the table
        :param start_index: the starting index for the data (used for transforming/reverse transforming slices)
        """
        return InseeDataState(
            table_name=table_name,
            df_pivot=df_pivot,
            df_meta=df_meta,
            start_index=start_index)

    @classmethod
    def from_data(
            cls,
            data: np.ndarray,
            col_to_freq: Dict[int, str],
            col_to_name: Dict[int, str],
            dates: List[datetime],
            table_name: str,
            start_index: int = 0):
        """
        Create a new InseeDataState object from the data and metadata.
        :param data: the data array
        :param col_to_freq: dict of {col_index -> freq_str} for the current columns
        :param col_to_name: dict of {col_index -> col_name} for the current columns
        :param dates: the list of dates for the data
        :param table_name: the name of the table
        :param start_index: the starting index for the data (used for transforming/reverse transforming slices)
        """
        return InseeDataState(
            table_name=table_name,
            data=data,
            col_to_freq=col_to_freq,
            col_to_name=col_to_name,
            dates=dates,
            start_index=start_index)

    def check_sanity(self):
        """
        Assert that the data respects the expected shape.
        """
        assert self.data.ndim == 2, "Data must be 2D"
        assert self.data.shape[1] == len(self.col_to_name), "Data must have the same number of columns as the metadata"
        assert len(self.dates) == self.data.shape[0], "Data must have the same number of rows as the dates"

    def copy(self) -> InseeDataState:
        """
        Create a deep copy of the current object.
        """
        return InseeDataState.from_data(
            table_name=self.table_name,
            data=self.data.copy(),
            col_to_freq=self.col_to_freq.copy(),
            col_to_name=self.col_to_name.copy(),
            dates=self.dates.copy(),
            start_index=self.start_index)

    def _generate_col_infos(self, df_pivot: pd.DataFrame, df_meta: pd.DataFrame):
        col_names = {}
        col_to_freq = {}
        feature_cols = [col for col in df_pivot.columns if col != 'DATE_PARSED']

        index_to_keep = []

        for col_index, col_name in enumerate(feature_cols):

            row_meta = df_meta[df_meta['TITLE_FR'] == col_name]
            if len(row_meta) == 0:
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
        self.col_to_name = col_names
