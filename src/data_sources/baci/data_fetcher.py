import os.path

import numpy as np
import pandas as pd

from src.core_utils import CoreUtils


class BaciDataFetcher:
    """
    The BACI dataset can be found at https://www.cepii.fr/CEPII/fr/bdd_modele/bdd_modele_item.asp?id=37.
    The dataset does not expose an API, so it must be downloaded manually.
    To get the dataset directly from the Azure Blob Storage, use the DatasetUploader class.
    """

    @staticmethod
    def load_baci_file(year: int) -> pd.DataFrame:
        """
        Load the BACI dataset for a specific year.
        :param year: The year of the dataset.
        :return: The BACI dataset as a DataFrame.
        """

        config = CoreUtils.load_ini_config()
        baci_path = config["datasets"]["BACIFolderPath"]
        try:
            # noinspection PyTypeChecker
            baci_year = pd.read_csv(
                os.path.join(baci_path, f"{year}.csv"),
                dtype={'t': np.int16, 'i': np.int16, 'j': np.int16, 'k': np.int32, 'v': np.float32, 'q': np.float32},
                na_values=['NA', 'N/A', 'NULL', ' ', ''],
                skipinitialspace=True
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find BACI file for year {year} at path {baci_path}")

        except Exception as e:
            raise Exception(f"An error occurred while loading the BACI file for year {year}") from e

        return baci_year