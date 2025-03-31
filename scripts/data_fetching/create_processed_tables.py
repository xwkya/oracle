"""
This script is used to process raw data files from different sources into a processed format.
The result can be saved to DB, CSV or blob storage.
"""
import argparse
import logging
import os
import sys

from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.source_to_pipeline import DataSourceToPipelineMatcher

from src.logging_config import setup_logging


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Raw data pipeline processing script')
    # --- Command Line Arguments ---
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='ProcessedData/',  # Specific output dir for WDI
        help='Directory to store output CSV files and reports'
    )

    # --- Data Sources ---
    parser.add_argument(
        '--source',
        type=str,
        default='wdi',  # Default data source
        choices=['WDI', 'INFLATION', 'COMMODITY', 'BACI', 'GRAVITY'],
        help='Data source to process'
    )

    # Control report generation (defaults to False) if applicable
    parser.add_argument(
        '--export_reports',
        action='store_true',
        default=False,
        help='Export detailed report files if applicable.'
    )

    # Whether to place it inside the database (NOT IMPLEMENTED)
    parser.add_argument(
        '--db',
        action='store_true',
        help='Store results in the database (NOT IMPLEMENTED)'
    )

    parser.add_argument(
        '--csv',
        action='store_true',
        help='Store results in CSV files'
    )

    parser.add_argument(
        '--blob',
        action='store_true',
        help='Store results in blob storage'
    )

    return parser.parse_args()

def get_arguments(data_source: DataSource):
    match data_source:
        case DataSource.WDI:
            return {
                "start_year_filter": 1990,
            }
        case DataSource.INFLATION:
            return {}
        case DataSource.COMMODITY:
            return {}
        case DataSource.BACI:
            return {
                'min_year': 1995,
                'max_year': 2023,
            }
        case DataSource.GRAVITY:
            return {}
        case _:
            raise ValueError(f"Data source {data_source} has no argument implemented for this script, add it to the get_arguments.")


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger()
    args = parse_args()

    data_source: DataSource = DataSource[args.source.upper()]
    pipeline_constructor = DataSourceToPipelineMatcher.get_pipeline(data_source.value)

    logger.info(f"Processing data source: {data_source.name}")

    # --- Instantiate and Run Pipeline ---
    try:
        pipeline = pipeline_constructor(**get_arguments(data_source))

        if args.db:
            raise NotImplementedError("Database output is not implemented yet.")

        if args.csv:
            pipeline.output_to_file(os.path.join(args.output_base_dir, f"{data_source.name.lower()}.csv"))

        if args.blob:
            pipeline.output_to_blob()

        pipeline.run_pipeline()

        if args.export_reports and hasattr(pipeline, 'export_filtering_report'):
            pipeline.export_filtering_report(args.output_base_dir + f"/reports/{data_source.name.lower()}")

    except NotImplementedError as e:
        logger.error(f"Functionality not implemented: {e}")
        sys.exit(1)

    logger.info("WDI data processing script finished.")