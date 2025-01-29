import json
import logging
import os
import requests
import zipfile
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def download_and_unzip_trade_index(json_path: str, logger: logging.Logger):
    base_dir = os.path.dirname(json_path)

    # Load the index.json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with logging_redirect_tqdm():
        for detail in tqdm(data["Details"]):
            year = detail["Year"]
            zip_url = detail["Path"]

            zip_filename = zip_url.split("/")[-1]
            zip_filepath = os.path.join(base_dir, zip_filename)

            logger.info(f"----- Processing year {year} -----")
            logger.info(f"Downloading from: {zip_url}")

            # Download the zip file
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()  # throw error if download failed
            with open(zip_filepath, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)

            logger.info(f"Unzipping {zip_filename}...")
            # Unzip it
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(base_dir)
                extracted_files = zip_ref.namelist()

            # Rename the csv
            csv_found = False
            for extracted_file in extracted_files:
                if extracted_file.lower().endswith(".csv"):
                    old_path = os.path.join(base_dir, extracted_file)
                    new_path = os.path.join(base_dir, f"{year}.csv")
                    os.rename(old_path, new_path)
                    csv_found = True

            if not csv_found:
                logger.warning(f"No CSV file found for {year}! Check ZIP contents.")

            os.remove(zip_filepath)
            logger.info(f"Deleted {zip_filename}")
