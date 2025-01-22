from dataclasses import dataclass
from typing import Dict

import numpy as np

@dataclass
class InseeDataState:
    data: np.ndarray
    meta_data: np.ndarray
    col_to_freq: Dict[int, str]
    col_to_name: Dict[int, str]