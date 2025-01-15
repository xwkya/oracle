"""
news_pipeline.py

Orchestrates the producer-consumer pattern with an async queue:
 - Creates the SharedORM instance.
 - Ensures the NewsSummary table is created.
 - Launches the producer and consumer tasks.
 - Waits for both to finish.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from src.ORM.ORMWrapper import SharedORM
from src.ORM.news_summary import NewsSummary
from src.exogen_sources.news.producer import Producer
from src.exogen_sources.news.consumer import Consumer

class NewsPipeline:
    def __init__(
        self,
        topics: List[str],
        start_date: datetime,
        end_date: datetime,
        window_size_days: int,
        max_results_per_topic: int,
        db_url: str = None,
        language: str = "fr"
    ):
        """
        Initializes the pipeline with producer, consumer, and the required DB setup.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting the ORM")
        self.orm = SharedORM(db_url=db_url)
        self.orm.create_table(NewsSummary)

        self.producer = Producer(
            topics=topics,
            start_date=start_date,
            end_date=end_date,
            window_size_days=window_size_days,
            max_results_per_topic=max_results_per_topic,
            language=language,
        )
        self.consumer = Consumer(self.orm)

    async def run(self):
        """
        Runs the producer-consumer pipeline:
         - Creates an async queue
         - Schedules producer and consumer
         - Waits for both tasks to complete
        """
        queue = asyncio.Queue()

        producer_task = asyncio.create_task(self.producer.produce(queue))
        consumer_task = asyncio.create_task(self.consumer.consume(queue))

        await producer_task
        await queue.join()  # Ensures all items are processed by consumer
        # Send None sentinel is already done by producer
        await consumer_task
