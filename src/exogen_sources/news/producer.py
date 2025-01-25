"""
producer.py

Producer that:
 1) Splits a date range into rolling windows.
 2) Retrieves Google News URLs synchronously for each topic and window (to avoid spamming).
 3) Decodes each URL synchronously using new_decoderv1.
 4) Uses crawl4ai to fetch HTML asynchronously (arun_many) for the decoded URLs in batches.
 5) Places results into an async queue for consumption.

This module should be invoked by an async task in the pipeline.
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

from googlenewsdecoder import new_decoderv1
from src.exogen_sources.news.news_url_getter import NewsUrlGetter, NYTNewsGetter
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig

class Producer:
    def __init__(
        self,
        topics: List[str],
        start_date: datetime,
        end_date: datetime,
        window_size_days: int,
        max_results_per_topic: int,
        language: str = "fr"
    ):
        """
        Initializes the producer with topic and date window configurations.

        :param topics: List of topics to search for.
        :param start_date: Global start date for retrieving news.
        :param end_date: Global end date for retrieving news.
        :param window_size_days: Number of days in each rolling window.
        :param max_results_per_topic: Max number of news URLs to retrieve per topic in each window.
        :param language: Language for NewsUrlGetter.
        """
        self.topics = topics
        self.start_date = start_date
        self.end_date = end_date
        self.window_size_days = window_size_days
        self.max_results_per_topic = max_results_per_topic
        self.language = language
        self.logger = logging.getLogger(Producer.__name__)
        self.headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
                'Referer': 'https://www.google.com/'
            }

    def _generate_windows(self) -> List[Tuple[datetime, datetime]]:
        """
        Splits the date range into consecutive windows of window_size_days.
        Returns a list of (start, end) date tuples.
        """
        self.logger.info("Generating windows...")
        windows = []
        current_start = self.start_date
        while current_start < self.end_date:
            window_end = current_start + timedelta(days=self.window_size_days)
            if window_end > self.end_date:
                window_end = self.end_date
            windows.append((current_start, window_end))
            current_start = window_end
        self.logger.info(f"Generated {len(windows)} windows for {self.start_date} to {self.end_date}")

        # Temporary override of windows, generate a window per month.
        # year = self.start_date.year
        # month = self.start_date.month
        # while year <= self.end_date.year or (month <= self.end_date.month - 1):
        #     window_start = datetime(year, month, 1)
        #     month += 1
        #     if month > 12:
        #         month = 1
        #         year += 1
        #     window_end = datetime(year, month, 1)
        #     windows.append((window_start, window_end))

        return windows

    async def produce(self, queue: asyncio.Queue):
        """
        Main entry point for the producer. Synchronously fetches & decodes Google News URLs,
        then asynchronously crawls HTML, and places the results into an async queue.
        """
        windows = self._generate_windows()

        config = CrawlerRunConfig(
            cache_mode=CacheMode.DISABLED,
            excluded_tags=['nav', 'footer', 'aside'],
            remove_overlay_elements=True,
            simulate_user=True,
            delay_before_return_html=1,
            semaphore_count=1,
        )

        async with AsyncWebCrawler(verbose=False, config=BrowserConfig(headers=self.headers)) as crawler:
            for (win_start, win_end) in windows:
                self.logger.info(f"Fetching {win_start} to {win_end}...")
                decoded_urls = []
                url_info_map = []  # Will store (decoded_url, topic, window_end)

                # Retrieve and decode URLs synchronously for each topic
                for topic in self.topics:
                    getter = NewsUrlGetter(
                        language=self.language,
                        max_results=self.max_results_per_topic,
                        start_date=(win_start.year, win_start.month, win_start.day),
                        end_date=(win_end.year, win_end.month, win_end.day),
                    )
                    raw_urls = getter.get_news_url(topic)  # synchronous
                    for uinfo in raw_urls:
                        original_url = uinfo.url
                        decoded_urls.append(original_url)
                        url_info_map.append((original_url, topic, win_end))
                self.logger.info(f"Fetched {len(decoded_urls)} URLs for topics {self.topics}.")

                if not decoded_urls:
                    continue

                self.logger.info(f"Crawling..")
                print(decoded_urls[0])
                time.sleep(5)

                # Asynchronously fetch HTML for all decoded URLs in this window
                results = await crawler.arun_many(
                    urls=decoded_urls,
                    config=config,
                )

                self.logger.info(f"Crawling done.")

                # Place each crawled result into the queue
                for i, r in enumerate(results):
                    # The i-th item in url_info_map corresponds to the same index in results
                    # because arun_many returns results in the same order as input
                    (this_url, this_topic, this_window_end) = url_info_map[i]
                    await queue.put({
                        "url": this_url,
                        "html": r.html,
                        "topic": this_topic,
                        "window_end": this_window_end,
                    })

        # Signal to the consumer that production is done
        print("Producer done.")
        await queue.put(None)
