import argparse
import logging
import os
import sys

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from azureorm.tables import BaciTradeByProduct
from src.core_utils import CoreUtils
from src.data_sources.raw_data_pipelines.baci.data_fetcher import BaciDataFetcher, BACIDataPipeline
from src.logging_config import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Baci tables creation')
    # --- Command Line Arguments ---
    parser.add_argument(
        '--min_year', default=1995, type=int, help='Minimum year to process (inclusive)'
    )
    parser.add_argument(
        '--max_year', default=2023, type=int, help='Maximum year to process (inclusive)'
    )
    parser.add_argument(
        '--local_csv', action='store_true',
        help='Store results in a local CSV instead of the database'
    )
    parser.add_argument(
        '--output_path', type=str, default='ProcessedData/baci_trade_by_product.csv',
        help='Output CSV file path (used if --local_csv is set)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    '''
    Processes BACI trade data yearly, either inserting into a database
    or appending to a single CSV file without loading all years into memory.
    Handles overwriting existing CSV files with user confirmation.
    '''
    setup_logging()
    logger = logging.getLogger()
    args = parse_args()
    countries = CoreUtils.get_countries_of_interest()

    # --- CSV Output Setup (if requested) ---
    output_file_exists_at_start = False  # Track initial state for header writing logic
    if args.local_csv:
        logger.info(f"Output configured to local CSV: {args.output_path}")

        # Handle potentially existing output file
        if os.path.exists(args.output_path):
            logger.warning(f"Output file '{args.output_path}' already exists.")
            while True:
                response = input("Delete existing file and proceed? (y/n): ").strip().lower()
                if response == 'y':
                    try:
                        os.remove(args.output_path)
                        logger.info(f"Deleted existing file: {args.output_path}")
                        output_file_exists_at_start = False  # File is gone
                    except OSError as e:
                        logger.error(f"Error deleting file {args.output_path}: {e}", exc_info=True)
                        sys.exit(f"Exiting due to file deletion error: {e}")
                    break  # Confirmed deletion
                elif response == 'n':
                    logger.info("Exiting script as requested without modifying existing file.")
                    sys.exit(0)  # User chose not to overwrite
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        else:
            output_file_exists_at_start = False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    # --- Database Setup (if requested) ---
    elif not args.local_csv:
        logger.info("Output configured to database.")
        # Assuming CoreUtils handles DB connection details
        orm = CoreUtils.get_orm()

    # --- Main Processing Loop ---
    first_write_done = False  # Ensure header is only written once per script run for CSV

    with logging_redirect_tqdm():
        for year in tqdm(range(args.min_year, args.max_year + 1), desc="Processing years"):
            logger.info(f"Processing BACI data for year {year}")
            try:
                # Process data
                baci_pipeline = BACIDataPipeline(year)
                group_aggregate = baci_pipeline.preprocess_data(baci, countries)

                if group_aggregate is None or group_aggregate.empty:
                    logger.warning(f"No data after processing for year {year}, skipping.")
                    continue

                # Add year and transform units
                group_aggregate['Year'] = year
                group_aggregate['ValueBillionUSD'] = group_aggregate['ValueThousandUSD'] / 1e6
                group_aggregate.drop(columns=['ValueThousandUSD'], inplace=True)

                # --- Output Data ---
                if args.local_csv:
                    # Append to CSV, write header only on first successful write to a new/cleared file
                    write_header = not first_write_done and not output_file_exists_at_start
                    group_aggregate.to_csv(
                        args.output_path,
                        mode='a',  # Append mode
                        index=False,
                        header=write_header  # Control header writing
                    )
                    first_write_done = True  # Mark that a write operation has occurred

                else:
                    # Insert into database
                    orm.bulk_insert_records_with_progress(
                        BaciTradeByProduct,
                        group_aggregate.to_dict(orient='records'),
                        chunk_size=10000,  # Configurable chunk size for DB insert
                        log_progress=True,
                        count=len(group_aggregate)
                    )

            except FileNotFoundError:
                logger.warning(f"BACI source file for year {year} not found. Skipping.")
            except Exception as e:
                # Log unexpected errors during yearly processing and continue
                logger.error(f"An unexpected error occurred processing year {year}: {e}", exc_info=True)
                logger.warning(f"Skipping year {year} due to error.")
                continue  # Continue to next year on error

    logger.info("Processing finished.")