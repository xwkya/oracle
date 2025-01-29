import logging

from src.data_sources.world_bank.tii.download_tii import download_and_unzip_trade_index
from src.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("DownloadTradeIndex")
    download_and_unzip_trade_index("Data/world_bank/trade_intensity_index/index.json", logger)
    download_and_unzip_trade_index("Data/world_bank/trade_complementarity_index/index.json", logger)
    logger.info("\nAll files downloaded, unzipped, and renamed successfully!")