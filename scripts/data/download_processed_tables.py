from src.core_utils import CoreUtils
from src.data_sources.data_blob_wrapper import DataBlobWrapper
from src.logging_config import setup_logging

if __name__ == '__main__':
    setup_logging()

    blob_container = DataBlobWrapper(DataBlobWrapper.PROCESSED_DATA_CONTAINER)
    config = CoreUtils.load_ini_config()
    save_path = config["datasets"]["ProcessedDataPath"]

    blob_container.download_folder(save_path)