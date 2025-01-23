from typing import Generator, Tuple

import pandas as pd

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class IDataProvider:
    def iter_data(self, min_date, max_date) -> Generator[InseeDataState, None, None]:
        pass