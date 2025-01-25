import logging
from dataclasses import dataclass
from typing import Tuple, Union
import pandas as pd

from src.data_sources.insee.data_fetcher import InseeDataFetcher
from src.date_utils import DateUtils

@dataclass
class DataFilterConfig:
    stopped_before: Union[str, pd.Timestamp]
    start_after: Union[str, pd.Timestamp]
    zeros_before: Union[str, pd.Timestamp]
    zeros_threshold: float

class InseeDataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(InseeDataPipeline.__name__)

    def preprocess_data(self, source_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download the data from the specified source table and return the pivoted DataFrame and the metadata DataFrame.
        :param source_df: the df returned by the data_fetcher
        :return: a tuple with the pivoted DataFrame and the metadata DataFrame
        """
        # Process the column types
        df = self._process_column_types(source_df)

        # Pivot the DataFrame
        df_pivot, df_metadata = self._pivot_df(df)

        return df_pivot, df_metadata

    def filter_data(
            self,
            df_pivot: pd.DataFrame,
            df_metadata: pd.DataFrame,
            config: DataFilterConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the data from the pivoted DataFrame and the metadata DataFrame and return the filtered DataFrames.
        :param df_pivot: the pivoted DataFrame to filter
        :param df_metadata: the metadata DataFrame to filter
        :param config: the configuration to filter the data
        :return: a tuple with the filtered pivoted DataFrame and the filtered metadata DataFrame
        """
        # Drop series that stopped before the specified date
        df_pivot, df_metadata = self._drop_stopped_before(df_pivot, df_metadata, config.stopped_before)

        # Drop series that started after the specified date (full of zeros or Nan)
        df_pivot, df_metadata = self._drop_start_after(df_pivot, df_metadata, config.start_after, config.zeros_before, config.zeros_threshold)

        return df_pivot, df_metadata

    def _drop_stopped_before(self, df_pivot: pd.DataFrame, df_metadata: pd.DataFrame, min_date: Union[str, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drops series (columns) from df_pivot and the corresponding rows in df_metadata
        whose last valid date is strictly earlier than the provided min_date.

        Parameters
        ----------
        df_pivot : pd.DataFrame
            Pivoted time series DataFrame with DateTime index and one column per time series.
            The column names should match df_metadata's index (usually TITLE_FR).
        df_metadata : pd.DataFrame
            Metadata table indexed by the same series identifiers (e.g. TITLE_FR).
        min_date : str or pd.Timestamp
            The cutoff date as a string or pd.Timestamp. Series with a last valid
            observation strictly before this date will be dropped.

        Returns
        -------
        df_pivot_filtered : pd.DataFrame
            The filtered pivot table with columns that pass the cutoff date check.
        df_metadata_filtered : pd.DataFrame
            The filtered metadata with matching rows retained.
        """

        # Ensure min_date is a Timestamp
        if not isinstance(min_date, pd.Timestamp):
            min_date = pd.to_datetime(min_date, errors='coerce')

        # Build a table of valid observation info for the pivot columns
        valid_obs_info = []
        for col in df_pivot.columns:
            last_valid = df_pivot[col].last_valid_index()
            valid_obs_info.append((col, last_valid))

        df_valid_obs = pd.DataFrame(valid_obs_info, columns=['TITLE_FR', 'last_valid'])

        # Filter columns whose last_valid is >= min_date
        keep_mask = df_valid_obs['last_valid'] >= min_date
        keep_columns = df_valid_obs.loc[keep_mask, 'TITLE_FR']

        # Now filter the pivot DataFrame
        df_pivot_filtered = df_pivot[keep_columns]

        # Filter the metadata by these same columns (TITLE_FR)
        df_metadata_filtered = df_metadata.loc[df_metadata.index.intersection(keep_columns)]

        self.logger.info(f"Kept {df_pivot_filtered.shape[1]} series out of {df_pivot.shape[1]} with last valid date >= {min_date}")
        return df_pivot_filtered, df_metadata_filtered

    def _drop_start_after(
            self,
            df_pivot: pd.DataFrame,
            df_metadata: pd.DataFrame,
            min_date: Union[str, pd.Timestamp],
            min_date_zeros: Union[str, pd.Timestamp],
            zeros_max_ratio: float=0.5):
        """
        Drops series (columns) from df_pivot and the corresponding rows in df_metadata
        whose first valid date is strictly later than the provided min_date.

        :param df_pivot: pd.DataFrame
            Pivoted time series DataFrame with DateTime index and one column per time series.
            The column names should match df_metadata's index (usually TITLE_FR).
        :param df_metadata: pd.DataFrame
            Metadata table indexed by the same series identifiers (e.g. TITLE_FR).
        :param min_date: str or pd.Timestamp
            The cutoff date as a string or pd.Timestamp. Series with a first valid
            observation strictly after this date will be dropped.
        :param min_date_zeros: str or pd.Timestamp
            The cutoff date as a string or pd.Timestamp. Series with more than 70% of zeros
            before this date will be dropped
        :param zeros_max_ratio: float
            The maximum ratio of zeros allowed before min_date_zeros
        :param zeros_max_ratio: float
            The maximum ratio of zeros allowed before min_date_zeros
        :return: (df_pivot_filtered, df_metadata_filtered)
        """

        # Ensure min_date is a Timestamp
        if not isinstance(min_date, pd.Timestamp):
            min_date = pd.to_datetime(min_date, errors='coerce')
        if not isinstance(min_date_zeros, pd.Timestamp):
            min_date_zeros = pd.to_datetime(min_date_zeros, errors='coerce')

        # Build a table of valid observation info for the pivot columns
        valid_obs_info = []
        for col in df_pivot.columns:
            first_valid = df_pivot[col].first_valid_index()
            valid_obs_info.append((col, first_valid))

        df_valid_obs = pd.DataFrame(valid_obs_info, columns=['TITLE_FR', 'first_valid'])

        # Filter columns whose first_valid is <= min_date
        keep_mask = df_valid_obs['first_valid'] <= min_date
        columns_to_drop = list(df_valid_obs.loc[~keep_mask, 'TITLE_FR'])
        keep_columns = df_valid_obs.loc[keep_mask, 'TITLE_FR']

        # Drop columns with at least 70% zeros before min_date_zeros
        for col in keep_columns:
            col_series = df_pivot.loc[:min_date_zeros, col]

            col_non_nan = col_series.dropna()

            if len(col_non_nan) == 0:
                columns_to_drop.append(col)
                continue

            frac_zeros = (col_non_nan == 0).mean()
            if frac_zeros >= zeros_max_ratio:
                columns_to_drop.append(col)

        self.logger.info(f"Dropping {len(columns_to_drop)} column(s) with >= {int(zeros_max_ratio * 100)}% zeros before {min_date_zeros}")

        df_pivoted_filtered = df_pivot.drop(columns=columns_to_drop)
        df_metadata_filtered = df_metadata.loc[df_metadata.index.difference(columns_to_drop)]

        return df_pivoted_filtered, df_metadata_filtered


    @staticmethod
    def _pivot_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse time period and returns a pivoted DataFrame and a metadata DataFrame.
        :param df: the DataFrame to process
        :return: a tuple with the pivoted DataFrame and the metadata DataFrame
        """
        # ----------------------------------------------------------------------------
        # Create the pivoted table:
        #    index = DATE_PARSED
        #    columns = TITLE_FR
        #    values = OBS_VALUE
        # ----------------------------------------------------------------------------

        df_pivot = df.pivot_table(
            index='DATE_PARSED',
            columns='TITLE_FR',
            values='OBS_VALUE',
            aggfunc='first'  # We assume there is only one value per cell
        )

        # Metadata columns
        metadata_cols = ['TITLE_FR', 'IDBANK', 'CORRECTION', 'NATURE', 'UNIT', 'STOPPED', 'LAST_UPDATE', 'FREQ']
        df_metadata = df[metadata_cols].drop_duplicates(subset='TITLE_FR').set_index('TITLE_FR')

        return df_pivot, df_metadata

    @staticmethod
    def _process_column_types(df: pd.DataFrame):
        """
        Process the column types of the DataFrame in place.
        :param df: the DataFrame to process
        """

        # If MULT is missing or empty, let's assume 0 (no multiplication)
        df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce').fillna(0)
        df['DATE_PARSED'] = df['TIME_PERIOD'].astype(str).apply(DateUtils.parse_time_period)

        return df