import logging

import pandas as pd


class InflationDataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(InflationDataPipeline.__name__)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Changing inflation index to datetime...")
        data["observation_date"] = pd.to_datetime(data["observation_date"])
        data.set_index("observation_date", inplace=True)
        data.sort_index(inplace=True)
        return data