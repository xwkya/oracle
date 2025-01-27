from typing import Optional, Dict, Any

import numpy as np


class TimeImportanceScheduler:
    decay_frequency_to_num_steps = {
        "yearly": 12,
        "quarterly": 4,
        "monthly": 1,
    }

    def __init__(self,
                 num_steps:int,
                 decay_frequency: str = "yearly",
                 initial_decay: float = 1.0,
                 final_decay: float = 0.8):
        self.step_count = 0
        self.num_steps = num_steps
        self.decay_frequency = decay_frequency
        self.decays = np.linspace(initial_decay, final_decay, num_steps)
        self.time_schedule: Dict[str, np.ndarray[float]] = {}

    def step(self):
        self.step_count += 1

    def get_time_importance(self, window_length: int) -> np.ndarray[np.float32]:
        num_decays = window_length // self.decay_frequency_to_num_steps[self.decay_frequency]
        importance = [self.decays[min(self.step_count, self.num_steps - 1)]**i for i in range(num_decays)]

        # Repeat and remove elements to match window length
        importance_array = np.array(importance[::-1], dtype=np.float32)
        importance_array = np.repeat(importance_array, self.decay_frequency_to_num_steps[self.decay_frequency])
        importance_array = importance_array[len(importance_array) - window_length:]

        return importance_array
