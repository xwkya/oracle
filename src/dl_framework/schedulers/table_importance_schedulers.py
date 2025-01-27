from typing import Dict, Optional

import numpy as np


class TableImportanceScheduler:
    def __init__(self, num_steps: int):
        self.step_count = 0
        self.num_steps = num_steps
        self.table_schedule: Dict[str, np.ndarray] = {}

    def add_exponential_schedule(self, table_name: str, start_value: float, end_value: float):
        schedule = np.geomspace(start_value, end_value, self.num_steps)
        self._add_table_schedule(table_name, schedule)

    def add_linear_schedule(self, table_name: str, start_value: float, end_value: float):
        schedule = np.linspace(start_value, end_value, self.num_steps)
        self._add_table_schedule(table_name, schedule)

    def step(self):
        self.step_count += 1

    def get_tables_importance(self) -> Dict[str, float]:
        tables_importance = {}
        for table_name, schedule in self.table_schedule.items():
            tables_importance[table_name] = schedule[min(self.step_count, self.num_steps - 1)]

        return tables_importance

    def _add_table_schedule(self, table_name: str, schedule: np.ndarray):
        self.table_schedule[table_name] = schedule