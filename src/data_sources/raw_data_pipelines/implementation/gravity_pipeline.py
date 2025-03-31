import logging
import os
from typing import List

import numpy as np
import pandas as pd

from src.core_utils import CoreUtils
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import IDataFetcher, DataPipeline


class GravityDataFetcher(IDataFetcher):
    """
    The Gravity dataset can be found at https://www.cepii.fr/CEPII/fr/bdd_modele/bdd_modele_item.asp?id=8.
    The dataset does not expose an API, so it must be downloaded manually.
    """

    def fetch_data(self) -> pd.DataFrame:
        config = CoreUtils.load_ini_config()
        gravity_folder_path = config["datasets"]["GravityFolderPath"]
        gravity_file_path = os.path.join(gravity_folder_path, "Gravity_V202211.csv")
        df = pd.read_csv(gravity_file_path)
        return df


class GravityDataPipeline(DataPipeline):
    """
    The pipeline for processing Gravity data.

    Ensures that the output DataFrame:
    - Has its first column as 'Period' of type pd.Period (freq='Y')
    - Contains only the relevant bilateral variables
    - Filters on countries, removes self-loops, and keeps years > 1970
    """

    def __init__(self):
        super().__init__([GravityDataFetcher()])
        self.logger = logging.getLogger(GravityDataPipeline.__name__)
        self.config = CoreUtils.load_ini_config()

    def _process_data(self, gravity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the gravity data by filtering on countries, filtering years,
        and casting columns to appropriate data types. Inserts a 'Period' column
        as the first column, with yearly frequency derived from the 'Year' column.
        """

        self.logger.info(f"Loaded Gravity data, has {len(gravity_df)} rows.")

        # Clean up the origin/destination country codes
        gravity_df["country_id_o_clean"] = gravity_df["country_id_o"].str.split(".").str[0]
        gravity_df["country_id_d_clean"] = gravity_df["country_id_d"].str.split(".").str[0]

        # Filter to keep only the countries of interest
        # self.countries is loaded from CoreUtils.get_countries_of_interest() at pipeline init
        df = gravity_df[
            gravity_df["iso3_o"].isin(self.countries) & gravity_df["iso3_d"].isin(self.countries)
            ].copy()

        # Filter for years > 1970
        df = df[df["year"] > 1970].copy()

        # Exclude rows where origin == destination
        df = df[df["iso3_o"] != df["iso3_d"]]

        self.logger.info(f"Filtered Gravity data on countries and dates, now has {len(df)} rows.")

        # Keep only certain bilateral variables
        bilateral_vars = [
            "contig",
            "distw_harmonic",
            "dist",
            "diplo_disagreement",
            "scaled_sci_2021",
            "comlang_off",
            "comlang_ethno",
            "comrelig",
            "heg_o",
            "heg_d",
            "col_dep_ever",
            "sibling_ever",
            "sibling",
            "fta_wto",
            "rta_coverage"
        ]

        keep_cols = ["year", "iso3_o", "iso3_d"] + bilateral_vars
        df = df[keep_cols].copy()

        # Deduplicate by (iso3_o, iso3_d, year), taking the first row
        df = df.groupby(["iso3_o", "iso3_d", "year"], as_index=False).first()

        # Rename columns
        rename_dict = {
            "year": "Year",
            "iso3_o": "Origin",
            "iso3_d": "Destination",
            "contig": "HasContiguousBorder",
            "distw_harmonic": "DistanceWeightedHarmonic",
            "dist": "DistanceCity",
            "diplo_disagreement": "DiplomaticDisagreement",
            "scaled_sci_2021": "ScaledSci2021",
            "comlang_off": "HasCommonOfficialLanguage",
            "comlang_ethno": "HasCommonEthnoLanguage",
            "comrelig": "ReligiousProximity",
            "heg_o": "HasHegemonyOrigin",
            "heg_d": "HasHegemonyDestination",
            "col_dep_ever": "HasColonialDependencyEver",
            "sibling_ever": "IsSiblingEver",
            "sibling": "IsSiblingNow",
            "fta_wto": "HasRegionalTradeAgreement",
            "rta_coverage": "RtaCoverage"
        }
        df: pd.DataFrame = df.rename(columns=rename_dict)

        # Decode RTA coverage
        df["HasRtaGoods"] = df["RtaCoverage"].apply(lambda x: x == 1 if pd.notna(x) else False)
        df["HasRtaServices"] = df["RtaCoverage"].apply(lambda x: x == 2 if pd.notna(x) else False)
        df["HasRtaGoodsAndServices"] = df["RtaCoverage"].apply(lambda x: x == 3 if pd.notna(x) else False)
        df.drop(columns=["RtaCoverage"], inplace=True)

        # Boolean columns
        bool_cols = [
            "HasContiguousBorder",
            "HasCommonOfficialLanguage",
            "HasCommonEthnoLanguage",
            "HasHegemonyOrigin",
            "HasHegemonyDestination",
            "HasColonialDependencyEver",
            "HasRegionalTradeAgreement",
            "IsSiblingEver",
            "IsSiblingNow",
            "HasRtaGoods",
            "HasRtaServices",
            "HasRtaGoodsAndServices"
        ]
        df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

        # Float columns
        float_cols = [
            "DistanceWeightedHarmonic",
            "DistanceCity",
            "DiplomaticDisagreement",
            "ScaledSci2021",
            "ReligiousProximity",
        ]
        for c in float_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

        # Final adjustments to data types
        df["Year"] = df["Year"].astype(np.int16)
        df["Origin"] = df["Origin"].astype("string")
        df["Destination"] = df["Destination"].astype("string")

        # -----------------------------------------------------------------------------------
        # Insert 'Period' as the first column (yearly frequency) according to the pipeline contract
        # -----------------------------------------------------------------------------------
        # Each row has a numeric 'Year'; we convert that to a period (e.g. 2021 -> Period('2021', freq='Y'))
        period_index = pd.PeriodIndex(df["Year"].astype(int), freq="Y")
        df.insert(0, "Period", period_index)
        # Ensure the column is recognized as period type
        df["Period"] = df["Period"].astype("period[Y]")
        df.drop(columns=["Year"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df