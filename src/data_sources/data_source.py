from enum import Enum

class DataSource(Enum):
    INSEE = 1 # Deprecated
    BACI = 2
    GRAVITY = 3
    INFLATION = 4
    WDI = 5
    COMMODITY = 6
