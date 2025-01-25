import logging
import re
from argparse import ArgumentTypeError
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class DateUtils:
    logger = logging.getLogger("DateUtils")
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
        The format is for INSEE data, which can be either a year, a quarter, or a month.
        :param x: the time period string to parse (must be in the format YYYY, YYYY-Q1, or YYYY-MM)
        :return: a pandas pd.Timestamp object
        """
        x = str(x).strip()

        match_year = re.match(r'^(\d{4})$', x)
        match_quarter = re.match(r'^(\d{4})-(S|Q)([1-4])$', x)
        match_month = re.match(r'^(\d{4})-(\d{2})$', x)

        if match_year:
            year = int(match_year.group(1))
            return pd.to_datetime(f'{year}-01-01')

        elif match_quarter:
            year = int(match_quarter.group(1))
            quarter = int(match_quarter.group(3))

            month_start = 3 * (quarter - 1) + 1  # Q1=1 -> Month=1, Q2=2 -> Month=4, etc.
            return pd.to_datetime(f'{year}-{month_start:02d}-01')

        elif match_month:
            year = int(match_month.group(1))
            month = int(match_month.group(2))
            return pd.to_datetime(f'{year}-{month:02d}-01')

        else:
            return pd.NaT

    @staticmethod
    def month_range(start: datetime, end: datetime) -> List[str]:
        """
        Return a list of monthly dates (as strings 'YYYY-MM-01')
        from start to end inclusive.
        :param start: the start date in datetime format
        :param end: the end date in datetime format
        :return: a list of monthly dates
        """
        dates = []
        cur = start
        stop = end
        while cur < stop:
            dates.append(cur.strftime("%Y-%m-01"))
            # move forward one month
            cur += relativedelta(months=1)

        return dates

    @staticmethod
    def is_expected(freq_str: str, date: datetime) -> bool:
        """
        :param freq_str: 'A', 'Q', or 'M'
        :param date: e.g. datetime(2021, 3, 1)
        :return: True if the date is expected for the frequency
        """
        m = date.month
        if freq_str == 'M':
            return True
        elif freq_str == 'Q':
            return m in [3, 6, 9, 12]
        elif freq_str == 'A':
            return m == 1
        else:
            return True