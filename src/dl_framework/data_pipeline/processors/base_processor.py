class IProcessor:
    def __init__(self, name: str, invertible: bool, description: str = None):
        self.name = name
        self.description = description
        self.invertible = invertible

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError