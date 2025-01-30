from typing import Generator, Tuple

import pandas as pd

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class IDataProvider:
    """
    Contract implemented by every source to iterate over its data.
    """
    def iter_data(self, min_date, max_date) -> Generator[InseeDataState, None, None]:
        pass