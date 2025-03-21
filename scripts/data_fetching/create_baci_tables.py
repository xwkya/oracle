import argparse
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from azureorm.tables import BaciTradeByProduct
from src.core_utils import CoreUtils
from src.data_sources.baci.data_fetcher import BaciDataFetcher
from src.data_sources.baci.data_pipeline import BACIDataPipeline
import pandas as pd

from src.logging_config import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Baci tables creation')

    parser.add_argument(
        '--min_year',
        default=1995,
        type=int,
        help='Minimum year to consider (included)'
    )

    parser.add_argument(
        '--max_year',
        default='2023',
        type=int,
        help='Maximum year to consider (included)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    '''
    Loads the baci csv files and inserts them in the database
    Requires the baci csv files to be in the Data folder.
    '''
    setup_logging()
    logger = logging.getLogger()
    args = parse_args()
    countries = pd.read_csv('Data/top_gdp.csv')['id'].to_list()
    orm = CoreUtils.get_orm()

    with logging_redirect_tqdm():
        for year in tqdm(range(args.min_year, args.max_year + 1)):
            logger.info(f"Processing BACI data for year {year}")
            baci = BaciDataFetcher.load_baci_file(year)
            baci_pipeline = BACIDataPipeline()

            group_aggregate = baci_pipeline.preprocess_data(baci, countries)
            group_aggregate['Year'] = year

            # Convert to billion of USD
            group_aggregate['ValueBillionUSD'] = group_aggregate['ValueThousandUSD'] / 1e6
            group_aggregate.drop(columns=['ValueThousandUSD'], inplace=True)

            orm.bulk_insert_records_with_progress(
                BaciTradeByProduct,
                group_aggregate.to_dict(orient='records'),
                chunk_size=10000,
                log_progress=True,
                count=len(group_aggregate)
            )