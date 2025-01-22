from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from datetime import datetime

import pandas as pd

from src.data_sources.data_provider import DataProvider
from src.data_sources.data_source import DataSource
from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState
from src.dl_framework.data_pipeline.data_utils import DataUtils
from src.dl_framework.data_pipeline.processors.base_processor import IProcessor
from src.dl_framework.data_pipeline.processors.low_variance_drop import LowVarianceDrop
from src.dl_framework.data_pipeline.processors.standard_scaler import StandardScalerProcessor
from src.dl_framework.data_pipeline.processors.trend_removal_processor import TrendRemovalProcessor


class DataProcessing:
    # TODO: Use processor factories instead of direct instantiation (we can stream the tables one by one instead..)
    def __init__(self, data_source: DataSource, min_date: datetime, max_date: datetime, train_cutoff: datetime):
        self.processors: List[IProcessor] = []
        self.data: InseeDataState
        self.data_source = data_source
        self.min_date = min_date
        self.max_date = max_date
        self.train_cutoff = train_cutoff
        self.date_range = DataUtils.month_range(min_date, max_date)

        date_to_idx = {d: i for i, d in enumerate(self.date_range)}
        self.cutoff_idx = date_to_idx[train_cutoff.strftime('%Y-%m-01')]


    def add_scaler(self) -> DataProcessing:
        self.processors.append(StandardScalerProcessor(cutoff_idx=self.cutoff_idx))
        return self

    def add_trend_removal(self) -> DataProcessing:
        self.processors.append(TrendRemovalProcessor(cutoff_idx=self.cutoff_idx))
        return self

    def add_variance_drop(self) -> DataProcessing:
        self.processors.append(LowVarianceDrop(cutoff_idx=self.cutoff_idx))
        return self
