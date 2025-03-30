# scripts/process_wdi.py
import argparse
import logging
import os
import sys
import pandas as pd

# Ensure the src directory is in the Python path
# Adjust the number of '..' based on your project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from src.core_utils import CoreUtils
from src.data_sources.world_bank.wdi.data_fetcher import WDIDataFetcher, WDIDataPipeline
from src.logging_config import setup_logging


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='WDI Time Series Filtering Pipeline')
    # --- Command Line Arguments ---
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ProcessedData/wdi',  # Specific output dir for WDI
        help='Directory to store output CSV files and reports'
    )
    # Control report generation (defaults to False)
    parser.add_argument(
        '--export_reports',
        action='store_true',
        default=False,  # Default is NOT to export the reports
        help='Export detailed removed series list and redundancy report text file.'
    )
    # Whether to place it inside the database (NOT IMPLEMENTED) or the csv (default)
    parser.add_argument(
        '--db',
        action='store_true',
        help='Store results in the database (NOT IMPLEMENTED)'
    )

    # --- Filtering Parameters (Optional Overrides) ---
    # Step 1
    parser.add_argument('--overall_fill_threshold', type=float, default=0.25, help='Min overall fill ratio (Step 1)')
    # Step 2
    parser.add_argument('--poor_country_fill_threshold', type=float, default=0.50,
                        help='Threshold for poor country fill (Step 2)')
    parser.add_argument('--country_proportion_threshold', type=float, default=0.25,
                        help='Min proportion of poorly covered countries (Step 2)')
    # Step 3
    parser.add_argument('--stopped_year_threshold', type=int, default=2015,
                        help='Year threshold for stopped series (Step 3)')
    parser.add_argument('--stopped_fill_threshold', type=float, default=0.20,
                        help='Min fill ratio after stopped year (Step 3)')
    # Step 4
    parser.add_argument('--preferred_level_suffix', type=str, default=".KD", help='Preferred level suffix (Step 4)')
    parser.add_argument('--secondary_level_suffix', type=str, default=".PP.KD", help='Secondary level suffix (Step 4)')
    parser.add_argument('--preferred_growth_suffix', type=str, default=".KD.ZG",
                        help='Preferred growth suffix (Step 4)')

    return parser.parse_args()


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger()
    args = parse_args()

    # Determine output format for results summary
    if args.db:
        output_format = 'db'
        logger.info("Output configured to database (currently not implemented).")
    else:
        output_format = 'csv'  # Default to CSV if --db is not specified
        logger.info(f"Output summary configured to local CSV in directory: {args.output_dir}")

    # --- Get Countries of Interest ---
    try:
        countries = list(CoreUtils.get_countries_of_interest())
        if not countries:
            logger.error("Failed to retrieve countries of interest. Exiting.")
            sys.exit(1)
        logger.info(f"Retrieved {len(countries)} countries of interest.")
    except Exception as e:
        logger.error(f"Error getting countries of interest: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Data ---
    wdi_df = WDIDataFetcher.load_wdi_data()
    if wdi_df is None:
        logger.error("Failed to load WDI data. Exiting.")
        sys.exit(1)

    # --- Instantiate and Run Pipeline ---
    try:
        pipeline = WDIDataPipeline(
            overall_fill_threshold=args.overall_fill_threshold,
            poor_country_fill_threshold=args.poor_country_fill_threshold,
            country_proportion_threshold=args.country_proportion_threshold,
            stopped_year_threshold=args.stopped_year_threshold,
            stopped_fill_threshold=args.stopped_fill_threshold,
            preferred_level_suffix=args.preferred_level_suffix,
            secondary_level_suffix=args.secondary_level_suffix,
            preferred_growth_suffix=args.preferred_growth_suffix
        )

        final_indicator_status, redundancy_map = pipeline.process(wdi_df, countries)

        # --- Export Results ---
        if final_indicator_status.empty:
            logger.warning("Pipeline processing resulted in an empty status DataFrame. No results to export.")
        else:
            # Export the lists of kept/removed series and optionally the reports
            pipeline.export_results(
                final_indicator_status,
                redundancy_map,
                args.output_dir,
                output_format,
                export_reports=args.export_reports  # Pass the flag here
            )

            # Always export the filtered data CSV
            filtered_data_output_path = os.path.join(args.output_dir, "filtered_wdi_data.csv")
            pipeline.export_filtered_data(
                raw_df=wdi_df,  # Pass the original raw dataframe
                final_indicator_status_df=final_indicator_status,
                countries_of_interest=countries,
                output_path=filtered_data_output_path
            )

    except NotImplementedError as e:
        logger.error(f"Functionality not implemented: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline processing: {e}", exc_info=True)
        sys.exit(1)

    logger.info("WDI data processing script finished.")