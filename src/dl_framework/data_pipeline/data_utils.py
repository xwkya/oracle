from datetime import datetime
from typing import List
from dateutil.relativedelta import relativedelta


class DataUtils:
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
        while cur <= stop:
            dates.append(cur.strftime("%Y-%m-01"))
            # move forward one month
            cur += relativedelta(months=1)

        return dates