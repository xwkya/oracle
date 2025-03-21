import argparse
import json
import logging
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from src.core_utils import CoreUtils
from src.data_sources.insee.data_fetcher import InseeDataFetcher
from src.data_sources.insee.data_pipeline import InseeDataPipeline, DataFilterConfig
from src.date_utils import DateUtils
from src.logging_config import setup_logging
from src.data_sources.insee.inputs import SOURCE_TABLES


def parse_args():
    parser = argparse.ArgumentParser(description='Insee tables creation')
    config = CoreUtils.load_ini_config()

    parser.add_argument(
        '--output',
        default=config["datasets"]["INSEEFolderPath"],
        help='Location to save the output files. Defaults to the pipeline standard path.'
    )

    parser.add_argument(
        '--stopped-before',
        type=DateUtils.parse_date,
        default='2019-01-01',
        help='Drop every series that stops before this date (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--start-after',
        type=DateUtils.parse_date,
        default='1990-01-01',
        help='Drop every series that starts after this date (YYYY-MM-DD format'
    )

    parser.add_argument(
        '--zeros-before',
        type=DateUtils.parse_date,
        default='2005-01-01',
        help='Drop every series that has too many zeros before this date (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--zeros-threshold',
        type=float,
        default=0.5,
        help='Drop every series that has more than this ratio of zeros before the zeros-before date'
    )

    parser.add_argument(
        '--remove-stopped',
        default=False,
        action='store_true',
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Create the INSEE tables from the raw data and save them to the output folder (by default this will be the pipeline's standard path).
    /!\ This script takes a long time to run (~1 hour)
    """
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)

    # Create output folder if it doesn't exist
    path = CoreUtils.get_root()
    path = path / args.output
    path.mkdir(parents=True, exist_ok=True)

    # Initialize data fetcher and pipeline
    params = None
    data_fetcher = InseeDataFetcher()
    data_pipeline = InseeDataPipeline()

    filter_config = DataFilterConfig(
        stopped_before=pd.Timestamp(args.stopped_before),
        start_after=pd.Timestamp(args.start_after),
        zeros_before=pd.Timestamp(args.zeros_before),
        zeros_threshold=args.zeros_threshold
    )

    all_tables = []

    # Fetch and process data for each table
    with logging_redirect_tqdm():
        for table in tqdm(SOURCE_TABLES, desc="Processing tables"):
            raw_df = data_fetcher.fetch_dataflow_data(table, params, args.remove_stopped)
            df_pivot, df_metadata = data_pipeline.preprocess_data(raw_df)
            df_pivot_filtered, df_metadata_filtered = data_pipeline.filter_data(df_pivot, df_metadata, filter_config)

            if df_pivot.shape[1] == 0:
                logger.warning(f"Table {table} has no data left after filtering, skipping it.")
                continue

            all_tables.append(table)
            df_pivot_filtered.to_csv(f"{args.output}/{table}.csv")
            df_metadata_filtered.to_csv(f"{args.output}/{table}_meta.csv")

    logger.info(f"Finished processing {len(all_tables)} tables")

    with open(f"{args.output}/all_data.json", "w") as f:
        f.write(json.dumps(all_tables))