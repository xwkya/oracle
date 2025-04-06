from __future__ import annotations
import logging
from typing import Optional, List

import pandas as pd

from src.data_sources.raw_data_pipelines.implementation.inflation_pipeline import InflationDataPipeline


class InflationAdjuster:
    """
    Provides methods to adjust monetary values for inflation using CPI data.
    """

    # Cache for loaded CPI data to avoid reloading repeatedly
    _cpi_data: Optional[pd.DataFrame] = None
    _logger: Optional[logging.Logger] = logging.getLogger("InflationAdjuster")

    @classmethod
    def _get_cpi_data(cls) -> pd.DataFrame:
        """Loads CPI data using InflationFetcher or returns cached data."""
        if cls._cpi_data is None:
            print("INFO: Loading CPI data for the first time...")
            pipeline = InflationDataPipeline()
            cls._cpi_data = pipeline._process_data(pipeline.data_fetchers[0].fetch_data())
            # Ensure index is PeriodIndex for reliable year filtering
            cls._cpi_data.index = cls._cpi_data.index.to_period('M')
            print("INFO: CPI data loaded and cached.")

        return cls._cpi_data

    @staticmethod
    def adjust_for_inflation(
            df: pd.DataFrame,
            date_col: str,
            value_cols: List[str],
            base_year: int = 2015
    ) -> pd.DataFrame:
        """
        Adjusts specified monetary columns in a DataFrame for inflation to a base year.

        Uses the monthly CPI-U NSA series (CPIAUCNS). Finds the average CPI
        for the base year and adjusts values based on the CPI of the month
        corresponding to the date in 'date_col'.

        Args:
            df (pd.DataFrame): The input DataFrame.
            date_col (str): The name of the column containing datetime objects or
                              parsable date strings.
            value_cols (List[str]): A list of column names containing the monetary
                                    values to adjust.
            base_year (int, optional): The year to which all values will be adjusted.
                                       Defaults to 2015.

        Returns:
            pd.DataFrame: A new DataFrame with added columns for the adjusted values.
                          Adjusted columns will be named '{original_col}_adjusted'.

        Raises:
            ValueError: If input columns are not found, date column cannot be
                        parsed, value columns are not numeric, base year CPI
                        cannot be determined, or CPI data is missing required columns.
            TypeError: If df is not a DataFrame or value_cols is not a list.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if not isinstance(value_cols, list):
            raise TypeError("'value_cols' must be a list of column names.")
        if not value_cols:
            raise ValueError("'value_cols' cannot be empty.")

        # --- Input Validation and Preparation ---
        df_adj = df.copy()  # Work on a copy to avoid modifying the original

        if date_col not in df_adj.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

        for col in value_cols:
            if col not in df_adj.columns:
                raise ValueError(f"Value column '{col}' not found in DataFrame.")

        # Convert date column to datetime, coercing errors to NaT (Not a Time)
        df_adj[date_col] = pd.to_datetime(df_adj[date_col], errors='coerce')
        if df_adj[date_col].isnull().any():
            print(
                f"WARNING: Column '{date_col}' contained values that could not be parsed into dates (converted to NaT). Corresponding adjusted values will be NaN.")

        # --- Load and Prepare CPI Data ---
        cpi_df = InflationAdjuster._get_cpi_data()
        if 'CPIAUCNS' not in cpi_df.columns:
            raise ValueError("Loaded CPI data is missing the 'CPIAUCNS' column.")


        # --- Calculate Base CPI ---
        # Filter CPI data for the base year and calculate the average
        try:
            # Ensure base_year is treated correctly against PeriodIndex
            base_period_start = pd.Period(f'{base_year}-01', freq='M')
            base_period_end = pd.Period(f'{base_year}-12', freq='M')
            cpi_base_year = cpi_df.loc[base_period_start:base_period_end, 'CPIAUCNS']

            if cpi_base_year.empty:
                raise ValueError(
                    f"No CPI data found for the base year {base_year}. Available years: {cpi_df.index.year.min()}-{cpi_df.index.year.max()}")

            cpi_base_value = cpi_base_year.mean()
            print(f"INFO: Average CPI for base year {base_year} = {cpi_base_value:.3f}")

        except Exception as e:
            raise ValueError(f"Could not determine average CPI for base year {base_year}: {e}")

        # --- Map Original CPI to Input DataFrame ---
        # Create a key column in df_adj based on year-month to match CPI index
        # Use PeriodIndex ('M' for month) for robust matching
        df_adj['_YearMonth'] = df_adj[date_col].dt.to_period('M')

        # Map the CPI value from the corresponding month in the CPI data
        cpi_series = cpi_df['CPIAUCNS']
        df_adj['_CPI_Original'] = df_adj['_YearMonth'].map(cpi_series)

        # Check if any dates were outside the CPI data range
        if df_adj['_CPI_Original'].isnull().any():
            raise ValueError("Some dates in the input data were outside the CPI data range. "
                             "Adjustment cannot be performed for these dates.")

        # --- Calculate Adjusted Values ---
        for col in value_cols:
            adj_col_name = f"{col}_adjusted_{base_year}"
            InflationAdjuster._logger.info(f"Calculating: {adj_col_name} = {col} * ({cpi_base_value:.3f} / _CPI_Original)")

            # Perform calculation - result will be NaN if _CPI_Original is NaN or 0
            df_adj[adj_col_name] = df_adj[col] * (cpi_base_value / df_adj['_CPI_Original'])

        # --- Clean Up and Return ---
        df_adj = df_adj.drop(columns=['_YearMonth', '_CPI_Original'])

        return df_adj