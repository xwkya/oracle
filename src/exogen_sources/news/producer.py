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

import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple

from googlenewsdecoder import new_decoderv1
from src.exogen_sources.news.news_url_getter import NewsUrlGetter
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


def decode_url(url: str):
    """
    Decodes a URL using new_decoderv1 synchronously.
    Returns the decoded URL if successful, None otherwise.
    """
    try:
        decoded = new_decoderv1(url)
        if decoded.get("status"):
            return decoded["decoded_url"]
        else:
            print("Error decoding URL:", decoded.get("message"))
            return None
    except Exception as e:
        print(f"Error occurred while decoding {url}: {e}")
        return None

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

    def _generate_windows(self) -> List[Tuple[datetime, datetime]]:
        """
        Splits the date range into consecutive windows of window_size_days.
        Returns a list of (start, end) date tuples.
        """
        windows = []
        current_start = self.start_date
        while current_start < self.end_date:
            window_end = current_start + timedelta(days=self.window_size_days)
            if window_end > self.end_date:
                window_end = self.end_date
            windows.append((current_start, window_end))
            current_start = window_end
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
            remove_forms=True,
            simulate_user=True,
            override_navigator=True,
            magic=True,
        )

        async with AsyncWebCrawler(verbose=False) as crawler:
            for (win_start, win_end) in windows:
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
                        decoded = decode_url(original_url)
                        if decoded:
                            decoded_urls.append(decoded)
                            url_info_map.append((decoded, topic, win_end))

                if not decoded_urls:
                    continue

                # Asynchronously fetch HTML for all decoded URLs in this window
                results = await crawler.arun_many(
                    urls=decoded_urls,
                    config=config,
                )

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
