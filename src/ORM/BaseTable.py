from src.ORM.ORMWrapper import Base
from sqlalchemy.orm.attributes import InstrumentedAttribute


class BaseTable(Base):
    """
    Base class for all tables in the ORM. Provides a preprocessing step for string columns.
    """
    __abstract__ = True

    def __setattr__(self, key, value):
        if isinstance(value, str):
            column = self.__class__.__dict__.get(key)
            if isinstance(column, InstrumentedAttribute):
                # Check if column has length constraint
                if hasattr(column.type, 'length') and column.type.length:
                    value = value[:column.type.length]
        super().__setattr__(key, value)