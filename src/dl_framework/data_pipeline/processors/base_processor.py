import numpy as np

from src.dl_framework.data_pipeline.data_states.insee_data_state import InseeDataState


class IProcessor:
    def __init__(self, name: str, invertible: bool, description: str = None):
        self.name = name
        self.description = description
        self.invertible = invertible

    def fit(self, data: InseeDataState):
        raise NotImplementedError

    def transform(self, data: InseeDataState) -> InseeDataState:
        raise NotImplementedError

    def inverse_transform(self, data: InseeDataState) -> InseeDataState:
        raise NotImplementedError

    def create_visualisation(self, x: np.ndarray):
        raise NotImplementedError

    def get_num_features(self):
        raise NotImplementedError

class IProcessorFactory:
    def __init__(self):
        pass

    def create(self, **kwargs) -> IProcessor:
        raise NotImplementedError