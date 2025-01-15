import asyncio
import logging
import argparse
from datetime import datetime
from src.exogen_sources.news.news_pipeline import NewsPipeline
from src.logging_config import setup_logging
import nltk


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format. Use YYYY-MM-DD")


def parse_args():
    parser = argparse.ArgumentParser(description='News Pipeline Runner')

    parser.add_argument(
        '--topics',
        nargs='+',
        default=["industrie", "finance"],
        help='List of topics to search for (space-separated)'
    )

    parser.add_argument(
        '--start-date',
        type=parse_date,
        default='2000-01-01',
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        type=parse_date,
        default='2000-02-01',
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Window size in days'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='Maximum results per topic'
    )

    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='Database URL (default: SQLite)'
    )

    parser.add_argument(
        '--language',
        type=str,
        default='fr',
        help='Language for the search (default: fr)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    nltk.download('punkt_tab')
    logger = logging.getLogger(__name__)

    args = parse_args()

    pipeline = NewsPipeline(
        topics=args.topics,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size_days=args.window_size,
        max_results_per_topic=args.max_results,
        db_url=args.db_url,
        language=args.language
    )

    asyncio.run(pipeline.run())