import os

import pandas as pd

from src.core_utils import CoreUtils


class GravityDataFetcher:
    """
    The Gravity dataset can be found at https://www.cepii.fr/CEPII/fr/bdd_modele/bdd_modele_item.asp?id=8.
    The dataset does not expose an API, so it must be downloaded manually.
    To get the dataset directly from the Azure Blob Storage, use the DatasetUploader class.
    """

    @staticmethod
    def load_gravity_data() -> pd.DataFrame:
        """
        Load the Gravity dataset.
        :return: The Gravity dataset as a DataFrame.
        """

        config = CoreUtils.load_ini_config()
        gravity_folder_path = config["datasets"]["GravityFolderPath"]
        gravity_file_path = os.path.join(gravity_folder_path, "Gravity_V202211.csv")

        df = pd.read_csv(gravity_file_path)

        return df