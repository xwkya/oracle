from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Float, Boolean, DateTime, INTEGER

from src.ORM.BaseTable import BaseTable


class CountryProductTrade(BaseTable):
    __tablename__ = "news_summaries"

    country = Column(String(3), primary_key=True, nullable=False)
    partner = Column(String(3), primary_key=True, nullable=False)
    year = Column(DateTime, primary_key=True, nullable=False)
    product_code = Column(INTEGER, primary_key=True, nullable=False)
    intensity_index = Column(Float, nullable=True)
    complementarity_index = Column(Float, nullable=True)

    @staticmethod
    def hash_url(url: str) -> str:
        return hash_url(url)

    def __init__(
            self,
            news_url: str,
            website_base_url: Optional[str],
            topic: Optional[str],
            title: str,
            summary: Optional[str],
            relevance: Optional[float],
            valid: bool,
            window_end_date: datetime,
            publish_date: Optional[datetime]
    ):
        self.id = hash_url(news_url)
        self.news_url = news_url
        self.website_base_url = website_base_url
        self.topic = topic
        self.title = title
        self.summary = summary
        self.relevance = relevance
        self.valid = valid
        self.window_end_date = window_end_date
        self.publish_date = publish_date

