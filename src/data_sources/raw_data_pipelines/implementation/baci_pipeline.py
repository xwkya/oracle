import logging
import os
import os.path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.core_utils import CoreUtils
from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import IDataFetcher, DataPipeline


class BaciDataFetcher(IDataFetcher):
    """
    The BACI dataset can be found at https://www.cepii.fr/CEPII/fr/bdd_modele/bdd_modele_item.asp?id=37.
    The dataset does not expose an API, so it must be downloaded manually.
    To get the dataset directly from the Azure Blob Storage, use the DatasetUploader class.
    """

    def __init__(self, year: int):
        self.year = year

    def fetch_data(self) -> pd.DataFrame:
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
                os.path.join(baci_path, f"{self.year}.csv"),
                dtype={'t': np.int16, 'i': np.int16, 'j': np.int16, 'k': np.int32, 'v': np.float64, 'q': np.float64},
                na_values=['NA', 'N/A', 'NULL', ' ', ''],
                skipinitialspace=True
            )

            baci_year['Year'] = self.year

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find BACI file for year {self.year} at path {baci_path}")

        except Exception as e:
            raise Exception(f"An error occurred while loading the BACI file for year {self.year}") from e

        return baci_year


class BACIDataPipeline(DataPipeline):
    def __init__(self, min_year: int, max_year: int):
        super().__init__([BaciDataFetcher(year) for year in range(min_year, max_year + 1)], DataSource.BACI)
        self.logger = logging.getLogger(BACIDataPipeline.__name__)
        self.config = CoreUtils.load_ini_config()

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Get the year
        year = data['Year'].iloc[0]

        # 1) Load the BACI country mapping and build a code -> iso3 dict
        baci_country_mapping = pd.read_csv(
            os.path.join(self.config['datasets']['BACIFolderPath'], 'country_codes.csv')
        )

        code_to_iso = baci_country_mapping[
            baci_country_mapping['country_iso3'].isin(self.countries)
        ][['country_code', 'country_iso3']]

        code_to_iso_dict = code_to_iso.set_index('country_code')['country_iso3'].to_dict()

        # 2) Filter relevant rows from the raw data
        filtered_baci: pd.DataFrame = data[data['i'].isin(code_to_iso_dict) & data['j'].isin(code_to_iso_dict)].copy()
        self.logger.info(
            f"Filtered BACI data for {len(self.countries)} countries. "
            f"Original data had {len(data)} rows, "
            f"filtered data has {len(filtered_baci)} rows."
        )

        # 3) Create product category from 'k'
        filtered_baci['ProductCode'] = filtered_baci['k'] // 10000
        filtered_baci['Exporter'] = filtered_baci['i'].map(code_to_iso_dict).astype('category')
        filtered_baci['Importer'] = filtered_baci['j'].map(code_to_iso_dict).astype('category')

        # 4) Group by and aggregate
        group_aggregate: pd.DataFrame = (
            filtered_baci
            .groupby(['Importer', 'Exporter', 'ProductCode'], observed=True)
            .agg({'v': 'sum', 'q': 'sum'})
            .reset_index()
        )
        group_aggregate.rename(columns={'v': 'ValueThousandUSD', 'q': 'Volume'}, inplace=True)

        group_aggregate["ValueBillionUSD"] = group_aggregate['ValueThousandUSD'].astype('float64') / 1e6
        group_aggregate.drop(columns=["ValueThousandUSD"], inplace=True)

        # 5) Insert the 'Period' column as the FIRST column
        #    We'll represent the entire year as a single yearly period
        #    e.g. 2024 => Period('2024', freq='Y') -> covers 2024-01-01 to 2024-12-31
        period_value = pd.Period(year, freq='Y')
        group_aggregate.insert(0, 'Period', period_value)

        return group_aggregate


    def compute_tii(self, group_aggregate: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Trade Intensity Index (TII) and Trade Complementarity Index (TCI) from trade value data.
        The TII is computed on a per-product basis.

        Equation:

        TII for product p:
          $$T_{ij}^p = \frac{ \bigl(x_{ij}^p / X_i^p\bigr)}{\bigl(x_{wj}^p / X_w^p\bigr)}$$

          where:
          - $x_{ij}^p$ is country i's exports of product p to country j.
          - $X_i^p$ is i's total exports of product p.
          - $x_{wj}^p$ is the world's exports of product p to country j.
          - $X_w^p$ is total world exports of product p.


        :param group_aggregate: The DataFrame containing the aggregated trade data.
        :return: The DataFrame containing the following colunns: exporter, importer, product_category, v, exporter_product_total, world_imports, product_world_exports, tii
        """
        # X_i^p: exporter i's total exports for product p
        exporter_product = (
            group_aggregate
            .groupby(["exporter", "product_category"])['v'].sum()
            .rename("exporter_product_total")
            .reset_index()
        )

        # x_{wj}^p: total world exports of product p to importer j
        importer_product = (
            group_aggregate
            .groupby(["importer", "product_category"])['v'].sum()
            .rename("world_imports")
            .reset_index()
        )

        # X_w^p: total world exports of product p
        product_world = (
            group_aggregate
            .groupby("product_category")['v'].sum()
            .rename("product_world_exports")
            .reset_index()
        )

        # Merge all into a df
        # x_{ij}^p = group_aggregate['v'] (the row-level trade)

        TII = (group_aggregate
                  .merge(exporter_product, on=["exporter", "product_category"], how="left")
                  .merge(importer_product, on=["importer", "product_category"], how="left")
                  .merge(product_world, on=["product_category"], how="left")
                  )

        # TII^p = ( x_{ij}^p / X_i^p ) / ( x_{wj}^p / X_w^p )
        TII["tii"] = (
                (TII["v"] / TII["exporter_product_total"])  # x_{ij}^p / X_i^p
                /
                (TII["world_imports"] / TII["product_world_exports"])  # x_{wj}^p / X_w^p
        )

        return TII

    def compute_tci(self, group_aggregate: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Trade Complementarity Index (TCI) from trade value data. The TCI is computed across all products.
        TCI:
          $$TCI_{ij} = 100\Bigl(1 - \frac{1}{2}\sum_p |m_i^p - x_j^p|\Bigr)$$

          where:
          - $m_i^p$ is the share of product p in i's total imports.
          - $x_j^p$ is the share of product p in j's total exports.
        :param group_aggregate:
        :return: The dataframe containing the following colunns: importer, exporter, abs_diff, tci
        """

        import_shares = (
            group_aggregate
            .groupby(["importer", "product_category"], as_index=False)['v'].sum()
        )
        importer_totals = (
            import_shares
            .groupby("importer")['v'].sum()
            .rename("importer_total")
            .reset_index()
        )
        import_shares = import_shares.merge(importer_totals, on="importer", how="left")
        import_shares["import_share"] = import_shares["v"] / import_shares["importer_total"]  # m_i^p
        import_shares = import_shares.rename(columns={"product_category": "p", "import_share": "m_i^p"})

        # For x_j^p
        export_shares = (
            group_aggregate
            .groupby(["exporter", "product_category"], as_index=False)['v'].sum()
        )
        exporter_totals = (
            export_shares
            .groupby("exporter")['v'].sum()
            .rename("exporter_total")
            .reset_index()
        )
        export_shares = export_shares.merge(exporter_totals, on="exporter", how="left")
        export_shares["export_share"] = export_shares["v"] / export_shares["exporter_total"]  # x_j^p
        export_shares = export_shares.rename(columns={"product_category": "p", "export_share": "x_j^p"})

        # Merge m_i^p with x_j^p across products
        merged_for_tci = import_shares.merge(
            export_shares,
            on="p",
            how="outer"
        ).fillna(0)

        # Summation of |m_i^p - x_j^p|
        merged_for_tci["abs_diff"] = (merged_for_tci["m_i^p"] - merged_for_tci["x_j^p"]).abs()

        TCI = (
            merged_for_tci.groupby(["importer", "exporter"], as_index=False)['abs_diff'].sum()
        )

        # TCI_{ij} = 100 (1 - sum(...)/2 )
        TCI["tci"] = 100 * (1 - TCI["abs_diff"] / 2)

        return TCI

    def compute_mean_imports_exports(self, group_aggregate: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the mean imports and exports for each country.
        :param group_aggregate: The DataFrame containing the aggregated trade data.
        :return: Two DataFrames: mean_imports and mean_exports
        """
        mean_imports = (
            group_aggregate
            .groupby("importer")
            .agg(mean_v=("v", "mean"), mean_q=("q", "mean"))
            .reset_index()
        )

        mean_exports = (
            group_aggregate
            .groupby("exporter")
            .agg(mean_v=("v", "mean"), mean_q=("q", "mean"))
            .reset_index()
        )

        return mean_imports, mean_exports