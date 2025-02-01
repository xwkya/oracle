import hashlib
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Float, Boolean, DateTime

from ..BaseTable import BaseTable


def hash_url(url: str) -> str:
    """
    Returns a SHA-256 hash of the provided URL as a hex string.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

class NewsSummary(BaseTable):
    __tablename__ = "news_summaries"

    id = Column(String(64), primary_key=True, unique=True, nullable=False)
    news_url = Column(String(1024), nullable=False)
    website_base_url = Column(String(256), nullable=True)
    topic = Column(String(64), nullable=True)
    title = Column(String(350), nullable=False)
    summary = Column(String, nullable=True)
    relevance = Column(Float, nullable=True)
    valid = Column(Boolean, nullable=False, default=False)
    window_end_date = Column(DateTime, nullable=False)
    publish_date = Column(DateTime, nullable=True)

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

