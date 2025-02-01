from sqlalchemy import Column, String, Float, SmallInteger

from ..BaseTable import BaseTable


class BaciTradeByProduct(BaseTable):
    __tablename__ = "BaciTradeByProduct"

    Importer = Column(String(3), primary_key=True, nullable=False)
    Exporter = Column(String(3), primary_key=True, nullable=False)
    Year = Column(SmallInteger, primary_key=True, nullable=False)
    ProductCode = Column(String(2), primary_key=True, nullable=False)
    ValueBillionUSD = Column(Float, nullable=True)
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
        self.Importer = country
        self.Exporter = partner
        self.Year = year
        self.ProductCode = product_code
        self.ValueBillionUSD = value
        self.Volume = volume
