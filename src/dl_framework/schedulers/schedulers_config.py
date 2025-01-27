from dataclasses import dataclass


@dataclass
class SchedulersConfig:
    table_min_importance: float
    table_max_importance: float
    time_min_decay: float
    time_max_decay: float
    p3_min_value: float
    p3_max_value: float

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            table_min_importance=config_dict["table_importance"]["start"],
            table_max_importance=config_dict["table_importance"]["end"],
            time_min_decay=config_dict["time_decay"]["start"],
            time_max_decay=config_dict["time_decay"]["end"],
            p3_min_value=config_dict["p3"]["start"],
            p3_max_value=config_dict["p3"]["end"]
        )