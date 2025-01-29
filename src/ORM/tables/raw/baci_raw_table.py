from sqlalchemy import Column, String, Float, Boolean, DateTime, SmallInteger

from src.ORM.BaseTable import BaseTable


class BaciRaw(BaseTable):
    __tablename__ = "BaciRaw"

    Country = Column(String(3), primary_key=True, nullable=False)
    Partner = Column(String(3), primary_key=True, nullable=False)
    Year = Column(SmallInteger, primary_key=True, nullable=False)
    ProductCcode = Column(String(6), primary_key=True, nullable=False)
    Value = Column(Float, nullable=True)
    Volume = Column(Float, nullable=True)

    def __init__(
            self,
            country: str,
            partner: str,
            year: int,
            product_code: str,
            value: float,
            volume: float
    ):
        self.Country = country
        self.Partner = partner
        self.Year = year
        self.ProductCcode = product_code
        self.Value = value
        self.Volume = volume
