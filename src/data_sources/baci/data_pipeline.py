import logging
import os
from typing import Tuple, List

import pandas as pd

from src.core_utils import CoreUtils


class BACIDataPipeline:
    def __init__(self):
        self.logger = logging.getLogger(BACIDataPipeline.__name__)
        self.config = CoreUtils.load_ini_config()

    def preprocess_data(self, baci_year_df: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
        """
        Baci data has a consistent format across the years, so we will only clean the data here.
        :param baci_year_df: The DataFrame containing the Baci data for a specific year.
        :param countries: The list of countries to keep.
        :return: The cleaned DataFrame with the following columns: exporter, importer, product_category, v, q
        """

        # Load the BACI country mapping and filter the countries/rename them.
        baci_country_mapping = pd.read_csv(
            os.path.join(self.config['datasets']['BACIFolderPath'], 'country_codes.csv')
        )

        code_to_iso = baci_country_mapping[baci_country_mapping['country_iso3'].isin(countries)][
            ['country_code', 'country_iso3']]
        code_to_iso_dict = code_to_iso.set_index('country_code')['country_iso3'].to_dict()

        baci_2010_filtered = baci_year_df[
            (baci_year_df['i'].isin(code_to_iso_dict)) & (baci_year_df['j'].isin(code_to_iso_dict))].copy()

        self.logger.info(f"Filtered BACI data for {len(countries)} countries. "
                         f"Original data had {len(baci_year_df)} rows, "
                         f"filtered data has {len(baci_2010_filtered)} rows.")
        baci_2010_filtered['ProductCode'] = baci_2010_filtered['k'] // 10000
        baci_2010_filtered['Exporter'] = baci_2010_filtered['i'].map(code_to_iso_dict)
        baci_2010_filtered['Importer'] = baci_2010_filtered['j'].map(code_to_iso_dict)

        # Group by exporter, importer, product category and sum the values
        group_aggregate = baci_2010_filtered.groupby(['Importer', 'Exporter', 'ProductCode']).agg(
            {'v': 'sum', 'q': 'sum'}).reset_index()

        group_aggregate.rename(columns={'v': 'ValueBillionUSD', 'q': 'Volume'}, inplace=True)

        if len(set(countries) - set(group_aggregate['Importer'].unique())) > 0:
            self.logger.warning(f"Missing countries in the BACI data: {set(countries) - set(group_aggregate['Importer'].unique())}")

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