from sqlalchemy import Column, String, Float, SmallInteger

from ..BaseTable import BaseTable


class BaciSparseTradeVolume(BaseTable):
    __tablename__ = "BaciSparseTradeVolume"

    Importer = Column(String(3), primary_key=True, nullable=False)
    Exporter = Column(String(3), primary_key=True, nullable=False)
    Year = Column(SmallInteger, primary_key=True, nullable=False)
    ValueBillionUSD = Column(Float, nullable=True)
