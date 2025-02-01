import logging
from typing import List

import numpy as np
import pandas as pd

from src.core_utils import CoreUtils


class GravityDataPipeline:
    def __init__(self):
        self.config = CoreUtils.load_ini_config()
        self.logger = logging.getLogger(GravityDataPipeline.__name__)

    def preprocess_data(self, gravity_df: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
        """
        Preprocess the gravity data by filtering on countries and years, and keeping only the relevant columns.
        This process can take quite a bit of RAM, so it is recommended to only run it for development purposes.
        :param gravity_df: The raw gravity data DataFrame.
        :param countries: The list of countries to keep. If None, all countries are kept (not recommended).
        :return: The preprocessed gravity data DataFrame.
        """
        # Extract the country code by removing .1, .2, etc.
        self.logger.info(f"Loaded Gravity data, has {len(gravity_df)} rows.")
        gravity_df["country_id_o_clean"] = gravity_df["country_id_o"].str.split(".").str[0]
        gravity_df["country_id_d_clean"] = gravity_df["country_id_d"].str.split(".").str[0]

        # Filter to keep only the countries of interest
        if countries is not None:
            gravity_df = gravity_df[gravity_df["iso3_o"].isin(countries) & gravity_df["iso3_d"].isin(countries)]

        # Filter to only select years post-1970
        df = gravity_df[gravity_df["year"] > 1970].copy()

        # Filter to remove countries that appear as the origin and destination
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

        # Group and take the first row per (iso3_o, iso3_d, year) group (the first non-nan value is chosen by default)
        df = df.groupby(["iso3_o", "iso3_d", "year"], as_index=False).first()

        # Rename columns to be more explicit
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
        df = df.rename(columns=rename_dict)

        # Boolean columns: fill missing with False, then cast to standard bool (np.bool_)
        bool_cols = [
            "HasContiguousBorder",
            "HasCommonOfficialLanguage",
            "HasCommonEthnoLanguage",
            "HasHegemonyOrigin",
            "HasHegemonyDestination",
            "HasColonialDependencyEver",
            "HasRegionalTradeAgreement",
            "HasSiblingEver",
            "HasSiblingNow",
        ]

        # Regional Trade Agreement decoding as per Gravity documentation.
        df["HasRtaGoods"] = df["RtaCoverage"].apply(lambda x: x == 1 if pd.notna(x) else False)
        df["HasRtaServices"] = df["RtaCoverage"].apply(lambda x: x == 2 if pd.notna(x) else False)
        df["HasRtaGoodsAndServices"] = df["RtaCoverage"].apply(lambda x: x == 3 if pd.notna(x) else False)
        bool_cols.extend(["HasRtaGoods", "HasRtaServices", "HasRtaGoodsAndServices"])

        df.drop(columns=["RtaCoverage"], inplace=True)
        df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

        # Float columns (store missing as np.nan):
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

        return df