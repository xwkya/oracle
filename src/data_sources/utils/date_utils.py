import re
from argparse import ArgumentTypeError
from datetime import datetime

import pandas as pd


class DateUtils:
    @staticmethod
    def parse_date(date_str) -> datetime:
        """
        Parse a date string into a datetime object, used for argparse.
        """
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ArgumentTypeError(f"Invalid date format. Use YYYY-MM-DD")

    @staticmethod
    def parse_time_period(x: str) -> pd.Timestamp:
        """
        Parse a time period into a pandas Timestamp object.
        :param x: the time period string to parse (must be in the format YYYY, YYYY-Q1, or YYYY-MM)
        :return: a pandas pd.Timestamp object
        """
        x = str(x).strip()

        match_year = re.match(r'^(\d{4})$', x)
        match_quarter = re.match(r'^(\d{4})-Q([1-4])$', x)
        match_month = re.match(r'^(\d{4})-(\d{2})$', x)

        if match_year:
            year = int(match_year.group(1))
            return pd.to_datetime(f'{year}-01-01')

        elif match_quarter:
            year = int(match_quarter.group(1))
            quarter = int(match_quarter.group(2))

            month_start = 3 * (quarter - 1) + 1  # Q1=1 -> Month=1, Q2=2 -> Month=4, etc.
            return pd.to_datetime(f'{year}-{month_start:02d}-01')

        elif match_month:
            year = int(match_month.group(1))
            month = int(match_month.group(2))
            return pd.to_datetime(f'{year}-{month:02d}-01')

        else:
            raise ValueError(f"Invalid time period format: {x}")