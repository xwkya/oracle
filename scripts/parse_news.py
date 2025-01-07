import asyncio
from datetime import datetime
from src.exogen_sources.news.news_pipeline import NewsPipeline

if __name__ == "__main__":
    topics = ["industrie", "finance"]
    start_date = datetime(2021, 2, 2)
    end_date = datetime(2021, 3, 2)
    window_size_days = 5
    max_results_per_topic = 3

    pipeline = NewsPipeline(
        topics=topics,
        start_date=start_date,
        end_date=end_date,
        window_size_days=window_size_days,
        max_results_per_topic=max_results_per_topic,
        db_url=None,  # default SQLite
        language="fr"
    )

    asyncio.run(pipeline.run())
