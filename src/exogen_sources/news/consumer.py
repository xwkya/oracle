import re
import json
import asyncio
import torch
from transformers import pipeline
from newspaper import Article
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, Dict
from src.ORM.news_summary import NewsSummary
from src.ORM.ORMWrapper import SharedORM

class Consumer:
    """
    Consumer that:
     - Reads items from an async queue in parallel to the producer.
     - For each item, uses the LLM pipeline to summarize.
     - Parses the output JSON from the LLM.
     - Inserts or upserts the result in the database.

    The consumer runs until it encounters a None sentinel.
    """
    def __init__(self, orm: SharedORM):
        """
        Initializes the consumer. Sets up the LLM pipeline and a reference to the ORM instance.
        """
        self.orm = orm
        self.pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token="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",
        )

    async def consume(self, queue: asyncio.Queue):
        """
        Continuously reads from the queue, summarizes each article, and saves to DB.
        Exits when a None sentinel is encountered.
        """
        while True:
            item = await queue.get()
            if item is None:
                # No more items to process
                queue.task_done()
                break

            url = item["url"]
            html = item["html"]
            topic = item["topic"]
            window_end = item["window_end"]

            data = await self._process_article(url, html, window_end)
            if data:
                await self._save_to_db_async(data)
            queue.task_done()

    async def _process_article(self, url: str, html: str, window_end: datetime) -> Optional[dict]:
        """
        Parses the article using newspaper, sends it to the LLM for summarization,
        and extracts JSON data (Title, Author, Summary, Valid, Relevance).

        Returns a dictionary with the final data or None on failure.
        """
        article_obj = Article(url)
        try:
            article_obj.download(input_html=html)
            article_obj.parse()
            article_obj.nlp()
        except Exception as e:
            print(f"Article parsing error for {url}: {e}")
            return None

        news_text = article_obj.text
        news_title = article_obj.title

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert news summarizer. You should summarize the news in a json "
                    "with the following fields: Title, Author, Summary, Valid, Relevance. "
                    "Valid is a boolean that is true if it's actually a news, false if there was "
                    "an error and it's not a news. Relevance is a float between 0 and 1 that says "
                    "how impactful the news is on the macro-economic context. Only output the json "
                    "and nothing else. Your summary should be several sentences, that contain "
                    "technical detail, facts if present, potential outcomes if present and context."
                ),
            },
            {
                "role": "user",
                "content": news_text
            },
        ]

        try:
            outputs = self.pipe(messages, max_new_tokens=1024)
        except Exception as e:
            print(f"LLM pipeline error for {url}: {e}")
            return None

        if not outputs or not isinstance(outputs, list) or "generated_text" not in outputs[0]:
            print(f"Invalid LLM output structure for {url}.")
            return None

        llm_output = outputs[0]["generated_text"]
        json_match = re.search(r"\{[\s\S]*\}", llm_output.strip())
        if not json_match:
            print(f"No JSON found in LLM output for {url}.")
            return None

        raw_json_str = json_match.group(0)
        try:
            parsed_data = json.loads(raw_json_str)
        except Exception as e:
            print(f"JSON parsing error for {url}: {e}")
            return None

        website_base_url = self._extract_website_base_url(url)

        result = {
            "title": parsed_data.get("Title", "Unknown Title"),
            "summary": parsed_data.get("Summary", ""),
            "relevance": parsed_data.get("Relevance", 0.0),
            "valid": parsed_data.get("Valid", False),
            "window_end_date": window_end,
            "publish_date": article_obj.publish_date,  # might be None
            "news_url": url,
            "website_base_url": website_base_url,
        }
        return result

    async def _save_to_db_async(self, data: dict):
        """
        Inserts or upserts the news summary record asynchronously.
        In a real production setting, consider run_in_executor or an async DB driver.
        """
        await asyncio.to_thread(self._save_to_db_sync, data)

    def _save_to_db_sync(self, data: dict):
        """
        Synchronous insertion/upsertion with the shared ORM.
        Called via to_thread for async usage.
        """
        pk_value = NewsSummary.hash_url(data["news_url"])
        self.orm.upsert_record(
            model_class=NewsSummary,
            pk_field="id",
            pk_value=pk_value,
            news_url=data["news_url"],
            website_base_url=data["website_base_url"],
            title=data["title"],
            summary=data["summary"],
            relevance=data["relevance"],
            valid=data["valid"],
            window_end_date=data["window_end_date"],
            publish_date=data["publish_date"]
        )

    def _extract_website_base_url(self, url: str) -> str:
        """
        Extracts the base domain from a URL. For example:
        'https://example.com/path/abc' -> 'https://example.com'
        """
        import re
        pattern = r'^(https?:\/\/[^\/]+)'
        match = re.match(pattern, url.strip())
        if match:
            return match.group(1)
        return url
