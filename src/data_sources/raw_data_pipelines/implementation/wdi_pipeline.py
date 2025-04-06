# src/data_sources/wdi/data_fetcher.py
import os
from typing import Tuple, List, Dict, Optional

import pandas as pd
import logging

from src.core_utils import CoreUtils
from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import IDataFetcher, DataPipeline


class WDIDataFetcher(IDataFetcher):
    """
    Handles loading the raw WDI data.
    """
    _logger = logging.getLogger(__name__)

    def fetch_data(self) -> pd.DataFrame | None:
        """
        Loads the WDI data from the specified CSV file.

        Returns:
            pd.DataFrame | None: A pandas DataFrame containing the WDI data,
                                 or None if the file cannot be loaded.
        """
        config = CoreUtils.load_ini_config()
        file_path = config["datasets"]["WDIFilePath"]

        try:
            WDIDataFetcher._logger.info(f"Loading WDI data from: {file_path}")
            df = pd.read_csv(file_path)
            WDIDataFetcher._logger.info(f"Successfully loaded WDI data with shape: {df.shape}")
            # Basic validation
            expected_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
            if not all(col in df.columns for col in expected_cols):
                WDIDataFetcher._logger.error(f"WDI CSV missing expected columns. Found: {df.columns.tolist()}")
                return None
            # Check for year columns (basic check) - assumes years start from 1960 onwards
            year_cols = [col for col in df.columns if col.isdigit() and int(col) >= 1960]
            if not year_cols:
                WDIDataFetcher._logger.error("WDI CSV does not appear to contain year columns (e.g., '1960', '1961', ...).")
                return None
            WDIDataFetcher._logger.info(f"Found {len(year_cols)} potential year columns.")

            return df
        except FileNotFoundError:
            WDIDataFetcher._logger.error(f"WDI data file not found at: {file_path}")
            return None
        except Exception as e:
            WDIDataFetcher._logger.error(f"Error loading WDI data from {file_path}: {e}", exc_info=True)
            return None


UNIT_TRANSFORM_SUFFIXES = [
    '.PP.KD', '.PP.CD', '.KD.ZG', '.KN.ZG', '.CN.ZG', '.CD.ZG',  # Combined first
    '.KD', '.KN', '.CD', '.CN', '.ZG'  # Basic units/transforms
]
OTHER_KEEP_SUFFIXES = ['.ZS', '.XD', '.IN']  # .IN can be ambiguous, but often represents rates/indices
PCAP_SUFFIXES = ['.PCAP', '.PC']  # Per capita suffixes


class WDIDataPipeline(DataPipeline):
    """
    WDI Data Pipeline: Fetches, filters (by fill rate, country coverage, recency,
    redundancy), and processes WDI data into the standard pipeline format.

    Provides an optional method `export_filtering_report` for debugging the
    filtering steps.
    """

    def __init__(self,
                 start_year_filter: int = 1990,
                 overall_fill_threshold: float = 0.25,
                 poor_country_fill_threshold: float = 0.50,
                 country_proportion_threshold: float = 0.25,
                 stopped_year_threshold: int = 2015,
                 stopped_fill_threshold: float = 0.20,
                 preferred_level_suffix: str = ".KD",
                 secondary_level_suffix: str = ".PP.KD",
                 preferred_growth_suffix: str = ".KD.ZG"):
        """
        Initializes the WDI pipeline with filtering parameters.

        Args:
            overall_fill_threshold: Minimum overall data points ratio to keep an indicator.
            poor_country_fill_threshold: Fill ratio threshold below which a country is considered 'poorly covered' for an indicator.
            country_proportion_threshold: If the proportion of countries with 'poor coverage' exceeds this, remove the indicator.
            stopped_year_threshold: Year from which to check for recent data presence.
            stopped_fill_threshold: Minimum fill ratio required for data points >= stopped_year_threshold.
            preferred_level_suffix: Suffix indicating the most preferred type for level indicators (e.g., constant USD).
            secondary_level_suffix: Suffix indicating the second preference for level indicators (e.g., constant PPP).
            preferred_growth_suffix: Suffix indicating the preferred type for growth indicators.
        """
        # Pass the specific fetcher(s) to the base class in a list
        super().__init__([WDIDataFetcher()], DataSource.WDI)
        self.logger = logging.getLogger(__name__)

        # Filtering parameters
        self.start_year_filter = start_year_filter
        self.overall_fill_threshold = overall_fill_threshold
        self.poor_country_fill_threshold = poor_country_fill_threshold
        self.country_proportion_threshold = country_proportion_threshold
        self.stopped_year_threshold = stopped_year_threshold
        self.stopped_fill_threshold = stopped_fill_threshold
        self.preferred_level_suffix = preferred_level_suffix
        self.secondary_level_suffix = secondary_level_suffix
        self.preferred_growth_suffix = preferred_growth_suffix

        # --- Internal state for reporting ---
        # These will be populated during _process_data and used by export_filtering_report
        self.indicator_status_df: Optional[pd.DataFrame] = None
        self.redundancy_map: Optional[Dict[str, List[str]]] = None
        self.raw_indicator_names: Optional[Dict[str, str]] = None  # Store original names for report

        self.logger.info("WDIDataPipeline initialized with filtering parameters.")

    def _melt_data(self, df_wide: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Melts the wide-format WDI DataFrame to a long format."""
        self.logger.debug("Melting wide DataFrame to long format...")
        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        # Identify year columns reliably
        year_cols = sorted([col for col in df_wide.columns if col.isdigit() and len(col) == 4])

        if not year_cols:
            self.logger.error("No year columns found in the input DataFrame for melting.")
            # Return empty DataFrame and empty list to avoid crashing later stages
            return pd.DataFrame(columns=id_vars + ['Year', 'Value']), []

        try:
            df_melted = pd.melt(df_wide,
                                id_vars=id_vars,
                                value_vars=year_cols,
                                var_name='Year',
                                value_name='Value')
            # Convert Year to numeric, coercing errors (though shouldn't happen with check)
            df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
            # Drop rows where year conversion failed or value is inherently missing
            df_melted.dropna(subset=['Year', 'Value'], inplace=True)
            df_melted['Year'] = df_melted['Year'].astype(int)

            self.logger.debug(f"Melting complete. Shape of melted data (before filtering): {df_melted.shape}")
            return df_melted, year_cols
        except Exception as e:
            self.logger.error(f"Error during data melting: {e}", exc_info=True)
            # Return empty DataFrame and empty list
            return pd.DataFrame(columns=id_vars + ['Year', 'Value']), []

    def _apply_filters(self, df_melted: pd.DataFrame, countries_of_interest: List[str],
                       year_cols: List[str]) -> pd.DataFrame:
        """Applies sequential filtering logic (Steps 1-3) based on data availability."""
        if df_melted.empty or not year_cols:
            self.logger.warning("Cannot apply filters: Melted data or year columns are empty.")
            return pd.DataFrame(columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason'])

        # --- Initialize Status Tracking ---
        # Store original names map for reporting
        self.raw_indicator_names = \
        df_melted[['Indicator Code', 'Indicator Name']].drop_duplicates().set_index('Indicator Code')[
            'Indicator Name'].to_dict()
        all_indicator_codes = df_melted['Indicator Code'].unique()
        indicator_status = pd.DataFrame({'Indicator Code': all_indicator_codes})
        indicator_status['Indicator Name'] = indicator_status['Indicator Code'].map(self.raw_indicator_names)
        indicator_status['Status'] = 'kept'
        indicator_status['Removal Step'] = pd.NA
        indicator_status['Reason'] = pd.NA

        num_countries = len(countries_of_interest)
        num_years = len(year_cols)
        self.logger.info(
            f"Starting filtering with {len(indicator_status)} unique indicators across {num_countries} countries and {num_years} years.")

        # --- Step 1: Filter by Overall Fill Ratio ---
        self.logger.info(f"Applying Step 1: Overall Fill Ratio Filter (Threshold < {self.overall_fill_threshold:.2f})")
        total_possible_points = num_years * num_countries
        if total_possible_points == 0:
            self.logger.warning("Cannot calculate overall fill ratio: zero possible data points. Skipping Step 1.")
            removed_count_step1 = 0
        else:
            # Ensure we only consider relevant countries for fill ratio calculation
            fill_counts = df_melted[df_melted['Country Code'].isin(countries_of_interest)] \
                .groupby('Indicator Code')['Value'].count()
            fill_counts = fill_counts.reindex(all_indicator_codes, fill_value=0)  # Ensure all indicators are present
            overall_fill_ratios = fill_counts / total_possible_points
            indicators_to_remove_step1 = overall_fill_ratios[overall_fill_ratios < self.overall_fill_threshold].index
            mask_step1 = indicator_status['Indicator Code'].isin(indicators_to_remove_step1)
            indicator_status.loc[mask_step1, 'Status'] = 'removed'
            indicator_status.loc[mask_step1, 'Removal Step'] = '1. Overall Fill Ratio'
            ratios_for_removed = overall_fill_ratios.loc[indicators_to_remove_step1].apply(
                lambda x: f"{x:.2%}")  # Format as percentage
            reasons_step1 = "Overall Fill Ratio " + ratios_for_removed + f" < {self.overall_fill_threshold:.2%}"
            indicator_status.loc[mask_step1, 'Reason'] = reasons_step1.reindex(
                indicator_status.loc[mask_step1, 'Indicator Code']).values
            removed_count_step1 = mask_step1.sum()

        kept_count_step1 = len(indicator_status[indicator_status['Status'] == 'kept'])
        self.logger.info(f"Removed {removed_count_step1} indicators in Step 1. Kept: {kept_count_step1}")

        # --- Step 2: Filter by Poor Country Coverage Distribution ---
        self.logger.info(
            f"Applying Step 2: Poor Country Coverage Filter (>= {self.country_proportion_threshold * 100:.0f}% of countries have fill < {self.poor_country_fill_threshold:.2f})")
        indicators_to_check_step2 = indicator_status[indicator_status['Status'] == 'kept']['Indicator Code']
        removed_count_step2 = 0
        if not indicators_to_check_step2.empty and num_years > 0 and num_countries > 0:
            country_indicator_fill = df_melted[df_melted['Country Code'].isin(countries_of_interest)] \
                                         .groupby(['Indicator Code', 'Country Code'])['Value'].count() / num_years
            # Unstack and reindex carefully to handle indicators with zero counts in some countries
            country_indicator_fill = country_indicator_fill.unstack(level='Country Code')
            # Reindex rows (indicators) to match those being checked, and columns (countries) to match the full list
            country_indicator_fill = country_indicator_fill.reindex(index=indicators_to_check_step2,
                                                                    columns=countries_of_interest).fillna(0)

            if not country_indicator_fill.empty:
                poor_coverage_mask = country_indicator_fill < self.poor_country_fill_threshold
                proportion_poorly_covered = poor_coverage_mask.sum(axis=1) / num_countries
                indicators_to_remove_step2 = proportion_poorly_covered[
                    proportion_poorly_covered >= self.country_proportion_threshold].index
                mask_step2 = indicator_status['Indicator Code'].isin(indicators_to_remove_step2) & (
                        indicator_status['Status'] == 'kept')
                indicator_status.loc[mask_step2, 'Status'] = 'removed'
                indicator_status.loc[mask_step2, 'Removal Step'] = '2. Poor Country Coverage'
                props_for_removed = proportion_poorly_covered.loc[indicators_to_remove_step2].apply(
                    lambda x: f"{x:.2%}")  # Format as percentage
                reasons_step2 = f"Poor Coverage Proportion " + props_for_removed + f" >= {self.country_proportion_threshold:.2%} (for fill < {self.poor_country_fill_threshold:.2%})"
                indicator_status.loc[mask_step2, 'Reason'] = reasons_step2.reindex(
                    indicator_status.loc[mask_step2, 'Indicator Code']).values
                removed_count_step2 = mask_step2.sum()
            else:
                self.logger.warning(
                    "Could not calculate country fill ratios for remaining indicators (empty after unstack/reindex). Skipping Step 2.")
        elif num_years == 0 or num_countries == 0:
            self.logger.warning("Skipping Step 2 because number of years or countries is zero.")
        else:
            self.logger.info("No indicators left to check for Step 2.")
        kept_count_step2 = len(indicator_status[indicator_status['Status'] == 'kept'])
        self.logger.info(f"Removed {removed_count_step2} indicators in Step 2. Kept: {kept_count_step2}")

        # --- Step 3: Filter by Stopped Series (Recent Fill Ratio) ---
        self.logger.info(
            f"Applying Step 3: Stopped Series Filter (Fill Ratio >= {self.stopped_year_threshold} < {self.stopped_fill_threshold:.2f})")
        indicators_to_check_step3 = indicator_status[indicator_status['Status'] == 'kept']['Indicator Code']
        removed_count_step3 = 0
        numeric_years = pd.to_numeric(year_cols, errors='coerce')
        years_after_threshold = [year for year, num_year in zip(year_cols, numeric_years) if
                                 pd.notna(num_year) and num_year >= self.stopped_year_threshold]
        num_recent_years = len(years_after_threshold)

        if not years_after_threshold:
            self.logger.warning(f"No year columns found >= {self.stopped_year_threshold}. Skipping Step 3.")
        elif not indicators_to_check_step3.empty and num_countries > 0:
            df_recent = df_melted[
                (df_melted['Year'] >= self.stopped_year_threshold) &
                (df_melted['Country Code'].isin(countries_of_interest))
                ]
            total_possible_points_recent = num_countries * num_recent_years

            if total_possible_points_recent > 0:
                recent_fill_counts = df_recent.groupby('Indicator Code')['Value'].count()
                recent_fill_counts = recent_fill_counts.reindex(indicators_to_check_step3,
                                                                fill_value=0)  # Ensure all are present
                recent_fill_ratios = recent_fill_counts / total_possible_points_recent
                indicators_to_remove_step3 = recent_fill_ratios[recent_fill_ratios < self.stopped_fill_threshold].index
                mask_step3 = indicator_status['Indicator Code'].isin(indicators_to_remove_step3) & (
                        indicator_status['Status'] == 'kept')
                indicator_status.loc[mask_step3, 'Status'] = 'removed'
                indicator_status.loc[mask_step3, 'Removal Step'] = '3. Stopped Series (Recent Fill)'
                ratios_for_removed_step3 = recent_fill_ratios.loc[indicators_to_remove_step3].apply(
                    lambda x: f"{x:.2%}")  # Format as percentage
                reasons_step3 = f"Recent Fill Ratio (>= {self.stopped_year_threshold}) " + ratios_for_removed_step3 + f" < {self.stopped_fill_threshold:.2%}"
                indicator_status.loc[mask_step3, 'Reason'] = reasons_step3.reindex(
                    indicator_status.loc[mask_step3, 'Indicator Code']).values
                removed_count_step3 = mask_step3.sum()
            else:
                self.logger.warning("Cannot calculate recent fill ratio: zero possible data points. Skipping Step 3.")
        elif num_countries == 0:
            self.logger.warning("Skipping Step 3 because number of countries is zero.")
        else:
            self.logger.info("No indicators left to check for Step 3.")
        kept_count_step3 = len(indicator_status[indicator_status['Status'] == 'kept'])
        self.logger.info(f"Removed {removed_count_step3} indicators in Step 3. Kept: {kept_count_step3}")

        return indicator_status

    def _apply_redundancy_filter(self, indicator_status_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Applies the redundancy filter (Step 4), choosing preferred indicator types."""
        self.logger.info(f"Applying Step 4: Redundancy Filter")
        # Operate on a copy to avoid modifying the input df directly if it's reused
        current_status_df = indicator_status_df.copy()
        kept_indicators_df = current_status_df[current_status_df['Status'] == 'kept'].copy()

        if kept_indicators_df.empty:
            self.logger.warning("No indicators left after Steps 1-3. Skipping Step 4 (Redundancy).")
            return current_status_df, {}

        kept_indicators_df['Base_Code'] = kept_indicators_df['Indicator Code'].apply(self.get_base_code)
        grouped = kept_indicators_df.groupby('Base_Code')

        final_keep_codes_set = set()
        redundancy_map = {}
        codes_removed_in_step4_set = set()

        for base_code, group in grouped:
            group_codes = group['Indicator Code'].tolist()
            kept_in_group_this_base = []
            removed_in_group_map = {}  # Maps removed code -> code it was removed in favour of

            # --- Categorize codes within the group ---
            levels = [c for c in group_codes if
                      not any(c.endswith(s) for s in ['.ZG'] + OTHER_KEEP_SUFFIXES) and not any(
                          c.endswith(pc) for pc in PCAP_SUFFIXES)]
            growths = [c for c in group_codes if c.endswith('.ZG') and not any(c.endswith(pc) for pc in PCAP_SUFFIXES)]
            pc_levels = [c for c in group_codes if
                         any(c.endswith(pc) for pc in PCAP_SUFFIXES) and not c.endswith('.ZG') and not any(
                             c.endswith(s) for s in OTHER_KEEP_SUFFIXES)]
            pc_growths = [c for c in group_codes if any(c.endswith(pc) for pc in PCAP_SUFFIXES) and c.endswith('.ZG')]
            other_keep = [c for c in group_codes if any(c.endswith(s) for s in OTHER_KEEP_SUFFIXES)]

            # --- Process Levels (Apply Preference) ---
            chosen_level = None
            levels_pref = [c for c in levels if c.endswith(self.preferred_level_suffix)]
            levels_sec = [c for c in levels if c.endswith(self.secondary_level_suffix)]
            if levels_pref:
                chosen_level = sorted(levels_pref)[0]  # Sort for consistency if multiple somehow exist
            elif levels_sec:
                chosen_level = sorted(levels_sec)[0]
            elif levels:  # Fallback: Keep the alphabetically first non-preferred level if no preferred exist
                chosen_level = sorted(levels)[0]
            if chosen_level:
                kept_in_group_this_base.append(chosen_level)
                for code in levels:
                    if code != chosen_level: removed_in_group_map[code] = chosen_level

            # --- Process Growth (Apply Preference) ---
            chosen_growth = None
            growths_pref = [g for g in growths if g.endswith(self.preferred_growth_suffix)]
            if growths_pref:
                chosen_growth = sorted(growths_pref)[0]
            elif growths:  # Fallback
                chosen_growth = sorted(growths)[0]
            if chosen_growth:
                kept_in_group_this_base.append(chosen_growth)
                for code in growths:
                    if code != chosen_growth: removed_in_group_map[code] = chosen_growth

            # --- Process Per Capita Levels (Apply Preference) ---
            chosen_pc_level = None
            pc_levels_pref = [c for c in pc_levels if c.endswith(self.preferred_level_suffix)]
            pc_levels_sec = [c for c in pc_levels if c.endswith(self.secondary_level_suffix)]
            if pc_levels_pref:
                chosen_pc_level = sorted(pc_levels_pref)[0]
            elif pc_levels_sec:
                chosen_pc_level = sorted(pc_levels_sec)[0]
            elif pc_levels:  # Fallback
                chosen_pc_level = sorted(pc_levels)[0]
            if chosen_pc_level:
                kept_in_group_this_base.append(chosen_pc_level)
                for code in pc_levels:
                    if code != chosen_pc_level: removed_in_group_map[code] = chosen_pc_level

            # --- Process Per Capita Growth (Apply Preference) ---
            chosen_pc_growth = None
            pc_growths_pref = [g for g in pc_growths if g.endswith(self.preferred_growth_suffix)]
            if pc_growths_pref:
                chosen_pc_growth = sorted(pc_growths_pref)[0]
            elif pc_growths:  # Fallback
                chosen_pc_growth = sorted(pc_growths)[0]
            if chosen_pc_growth:
                kept_in_group_this_base.append(chosen_pc_growth)
                for code in pc_growths:
                    if code != chosen_pc_growth: removed_in_group_map[code] = chosen_pc_growth

            # --- Keep Other Types ---
            # These are generally not considered redundant with levels/growths
            kept_in_group_this_base.extend(other_keep)

            # --- Update final sets and maps ---
            final_keep_codes_set.update(kept_in_group_this_base)
            for removed_code, kept_code in removed_in_group_map.items():
                # Ensure the 'kept_code' actually made it to the final list for this base group
                if kept_code in kept_in_group_this_base:
                    if kept_code not in redundancy_map:
                        redundancy_map[kept_code] = []
                    redundancy_map[kept_code].append(removed_code)
                    codes_removed_in_step4_set.add(removed_code)

        removed_count_step4 = len(codes_removed_in_step4_set)
        final_kept_count = len(final_keep_codes_set)
        self.logger.info(
            f"Removed {removed_count_step4} indicators in Step 4 (Redundancy). Final Kept: {final_kept_count}")

        # --- Update the main status DataFrame ---
        # Identify all codes that were initially 'kept' but are not in the final set
        initially_kept_codes = set(kept_indicators_df['Indicator Code'])
        codes_to_mark_removed = initially_kept_codes - final_keep_codes_set
        assert codes_to_mark_removed == codes_removed_in_step4_set, "Mismatch in Step 4 removal logic!"

        for code in codes_removed_in_step4_set:
            reason_suffix = "Removed due to redundancy"
            # Find the code it was removed in favour of (more informative reason)
            removed_for_code = None
            for kept_c, removed_l in redundancy_map.items():
                if code in removed_l:
                    removed_for_code = kept_c
                    reason_suffix = f"Removed in favor of {removed_for_code}"
                    break

            # Find the row in the *original* status df to update
            mask_step4 = (current_status_df['Indicator Code'] == code) & (current_status_df['Status'] == 'kept')
            if mask_step4.any():
                current_status_df.loc[mask_step4, 'Status'] = 'removed'
                current_status_df.loc[mask_step4, 'Removal Step'] = '4. Redundancy'
                current_status_df.loc[mask_step4, 'Reason'] = f"Redundant: {reason_suffix}"
            else:
                # This case should ideally not happen if logic is correct
                self.logger.warning(
                    f"Tried to mark code '{code}' as removed in Step 4, but it was not found or already removed.")

        # Final check: Ensure all codes in final_keep_codes_set have status 'kept'
        final_check_mask = current_status_df['Indicator Code'].isin(final_keep_codes_set)
        if not current_status_df.loc[final_check_mask, 'Status'].eq('kept').all():
            self.logger.error("Inconsistency found: Some final kept codes are marked as removed in status DF!")

        return current_status_df, redundancy_map

    def _generate_redundancy_report_text(self, final_status_df: pd.DataFrame, redundancy_map: Dict[str, list]) -> str:
        """Generates a text report detailing the redundancy removal decisions."""
        report_lines = []
        report_lines.append("--- Redundancy Removal Report (Step 4) ---\n")
        report_lines.append(
            "This report shows which indicators were kept after the redundancy filter and which were removed in their favor.")
        report_lines.append(
            f"Level Preference: '{self.preferred_level_suffix}' > '{self.secondary_level_suffix}' > Other > Fallback (alphabetical first).")
        report_lines.append(
            f"Growth Preference: '{self.preferred_growth_suffix}' > Other > Fallback (alphabetical first).")
        report_lines.append(f"PCap Level Preference: Like Level.")
        report_lines.append(f"PCap Growth Preference: Like Growth.")
        report_lines.append(f"Other types ({', '.join(OTHER_KEEP_SUFFIXES)}) generally kept.")
        report_lines.append(
            "[Fallback Selection] indicates a non-preferred type was chosen as no preferred type was available.\n")

        # Use the stored raw names if available, otherwise use names from final status df
        indicator_names = self.raw_indicator_names or \
                          final_status_df.set_index('Indicator Code')['Indicator Name'].to_dict()

        kept_indicators_final_df = final_status_df[final_status_df['Status'] == 'kept'].sort_values('Indicator Code')

        for _, row in kept_indicators_final_df.iterrows():
            kept_code = row['Indicator Code']
            kept_name = row['Indicator Name']  # Name from the final status df
            report_lines.append(f"Kept: {kept_code} | {kept_name}")

            # Determine if it was a fallback selection based on its suffix and type
            is_level = (not any(kept_code.endswith(s) for s in ['.ZG'] + OTHER_KEEP_SUFFIXES) and not any(
                kept_code.endswith(pc) for pc in PCAP_SUFFIXES))
            is_growth = (kept_code.endswith('.ZG') and not any(kept_code.endswith(pc) for pc in PCAP_SUFFIXES))
            is_pc_level = (any(kept_code.endswith(pc) for pc in PCAP_SUFFIXES) and not any(
                kept_code.endswith(s) for s in ['.ZG'] + OTHER_KEEP_SUFFIXES))
            is_pc_growth = (any(kept_code.endswith(pc) for pc in PCAP_SUFFIXES) and kept_code.endswith('.ZG'))

            is_fallback = False
            if is_level and not any(
                    kept_code.endswith(s) for s in [self.preferred_level_suffix, self.secondary_level_suffix]):
                is_fallback = True
            elif is_growth and not kept_code.endswith(self.preferred_growth_suffix):
                is_fallback = True
            elif is_pc_level and not any(
                    kept_code.endswith(s) for s in [self.preferred_level_suffix, self.secondary_level_suffix]):
                is_fallback = True
            elif is_pc_growth and not kept_code.endswith(self.preferred_growth_suffix):
                is_fallback = True

            if is_fallback:
                report_lines[-1] += "  **[Fallback Selection]**"

            removed_list = redundancy_map.get(kept_code, [])
            if removed_list:
                report_lines.append(f"  Removed:")
                for rem_code in sorted(removed_list):
                    rem_name = indicator_names.get(rem_code, "Name not found")
                    report_lines.append(f"    - {rem_code} | {rem_name}")
            else:
                report_lines.append(f"  Removed: None")
            report_lines.append("-" * 20)

        return "\n".join(report_lines)

    @staticmethod
    def get_base_code(indicator_code: str) -> str:
        """Identifies the 'base' part of an indicator code by removing known suffixes."""
        base = indicator_code
        # Sort suffixes by length descending to match longest first (e.g., .PP.KD before .KD)
        temp_strip_suffixes = sorted(UNIT_TRANSFORM_SUFFIXES + OTHER_KEEP_SUFFIXES + PCAP_SUFFIXES, key=len,
                                     reverse=True)
        stripped = False
        for suffix in temp_strip_suffixes:
            if base.endswith(suffix):
                # Ensure removing the suffix leaves a meaningful part (avoid stripping everything)
                potential_base = base[:-len(suffix)]
                # Heuristic: Base code usually contains '.' and doesn't end with '.'
                if len(potential_base) > 1 and '.' in potential_base and potential_base[-1] != '.':
                    base = potential_base
                    stripped = True
                    # Once a valid suffix is stripped, assume it's the main one and break
                    # This handles cases like 'ABC.XYZ.PP.KD' -> 'ABC.XYZ' correctly
                    break
        return base

    # --- Core Pipeline Method ---
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single chunk of raw WDI data according to the pipeline steps.

        Args:
            data: Raw WDI data (wide format) from the fetcher.

        Returns:
            pd.DataFrame: Processed data in the standard format:
                          'Period' (pd.Period) column first, 'Country Code' second,
                          followed by columns for each kept 'Indicator Code'.
        """
        self.logger.info(f"Processing WDI data chunk with shape: {data.shape}")

        # 1. Filter by Countries of Interest
        df_subset = data[data['Country Code'].isin(self.countries)].copy()
        if df_subset.empty:
            self.logger.warning(
                "Data chunk is empty after filtering by countries of interest. Returning empty DataFrame.")
            self.indicator_status_df = pd.DataFrame(
                columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason'])
            self.redundancy_map = {}
            self.raw_indicator_names = {}
            # Return DF with standard columns but no data
            return pd.DataFrame({'Period': pd.Series(dtype='period[A-DEC]'), 'Country Code': pd.Series(dtype='object')})
        self.logger.info(f"Subsetted data for {len(self.countries)} countries. Shape: {df_subset.shape}")

        # 2. Melt Data
        df_melted, original_year_cols = self._melt_data(df_subset)
        if df_melted.empty:
            self.logger.warning("Melted data is empty. Returning empty DataFrame.")
            self.indicator_status_df = pd.DataFrame(
                columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason'])
            self.redundancy_map = {}
            self.raw_indicator_names = {}
            return pd.DataFrame({'Period': pd.Series(dtype='period[A-DEC]'), 'Country Code': pd.Series(dtype='object')})

        # 2.5 Apply Start Year Filter
        self.logger.info(f"Applying start year filter: Keeping data from year {self.start_year_filter} onwards.")
        initial_melted_rows = len(df_melted)
        df_melted = df_melted[df_melted['Year'] >= self.start_year_filter].copy()  # Use .copy()
        rows_after_year_filter = len(df_melted)
        self.logger.info(
            f"Filtered {initial_melted_rows - rows_after_year_filter} rows based on start year. Shape after year filter: {df_melted.shape}")

        if df_melted.empty:
            self.logger.warning(
                f"Melted data is empty after applying start year filter ({self.start_year_filter}). Returning empty DataFrame.")
            self.indicator_status_df = pd.DataFrame(
                columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason'])
            self.redundancy_map = {}
            self.raw_indicator_names = {}
            return pd.DataFrame({'Period': pd.Series(dtype='period[A-DEC]'), 'Country Code': pd.Series(dtype='object')})

        # Update the list of year columns relevant for filtering statistics
        filtered_year_cols = sorted([col for col in original_year_cols if int(col) >= self.start_year_filter])
        if not filtered_year_cols:
            raise ValueError("No year columns remain after applying start year filter.")

        # 3. Apply Filters (Steps 1-3) and Redundancy Filter (Step 4)
        status_df_step3 = self._apply_filters(df_melted, self.countries, filtered_year_cols)
        self.indicator_status_df, self.redundancy_map = self._apply_redundancy_filter(status_df_step3)

        # 4. Identify Kept Indicators
        kept_indicator_codes = self.indicator_status_df[self.indicator_status_df['Status'] == 'kept'][
            'Indicator Code'].tolist()
        if not kept_indicator_codes:
            self.logger.warning("No indicators kept after all filtering steps. Returning empty DataFrame.")
            return pd.DataFrame({'Period': pd.Series(dtype='period[A-DEC]'), 'Country Code': pd.Series(dtype='object')})
        self.logger.info(f"Identified {len(kept_indicator_codes)} indicators to keep after all filters.")

        # 5. Filter Melted Data to Final Set
        final_melted_df = df_melted[
            df_melted['Indicator Code'].isin(kept_indicator_codes)
        ].copy()

        # 6. Pivot to Required Output Format (Corrected)
        self.logger.debug("Pivoting filtered data to final format (Period | Country Code | IndicatorCode columns)...")
        try:
            # Pivot with Year and Country Code as index, Indicator Code as columns
            output_df = final_melted_df.pivot_table(
                index=['Year', 'Country Code'],  # Use multi-index for pivot
                columns='Indicator Code',  # Indicator codes become columns
                values='Value'
            )

            # Reset index to turn 'Year' and 'Country Code' into columns
            output_df = output_df.reset_index()

            # Convert 'Year' column to 'Period'
            output_df['Period'] = pd.PeriodIndex(output_df['Year'], freq='Y')

            # Drop the original integer 'Year' column and Indicator Code column
            output_df = output_df.drop(columns=['Year'])
            output_df.reset_index(drop=True, inplace=True)

            # Reorder columns: Period, Country Code, then sorted indicator codes
            indicator_cols = sorted([col for col in output_df.columns if col not in ['Period']])
            final_cols = ['Period', 'Country Code'] + indicator_cols
            output_df = output_df[final_cols]

            self.logger.info(f"Successfully processed data. Final shape: {output_df.shape}")
            return output_df

        except Exception as e:
            raise e

    # --- Optional Report Export Method ---
    def export_filtering_report(self, output_dir: str, base_filename: str = "wdi_filtering_report"):
        """
        Exports detailed reports about which indicators were kept or removed
        during the pipeline run. This is intended for debugging/analysis and
        is NOT called automatically by `run_pipeline`.

        Requires `run_pipeline()` to have been executed first to populate the
        necessary internal state.

        Args:
            output_dir: The directory where report files will be saved.
            base_filename: Base name for the output files (e.g., 'wdi_filtering').
                           '.csv' and '.txt' extensions will be added.
        """
        self.logger.info(f"Attempting to export filtering reports to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if self.indicator_status_df is None or self.redundancy_map is None:
            self.logger.error("Cannot export report: Pipeline has not been run yet or failed during processing. "
                              "Internal status data is missing.")
            return

        # --- Prepare DataFrames for Export ---
        try:
            kept_indicators_final_df = self.indicator_status_df[
                self.indicator_status_df['Status'] == 'kept'
                ][['Indicator Code', 'Indicator Name']].sort_values('Indicator Code').reset_index(drop=True)

            removed_indicators_final_df = self.indicator_status_df[
                self.indicator_status_df['Status'] == 'removed'
                ][['Indicator Code', 'Indicator Name', 'Removal Step', 'Reason']].sort_values(
                'Removal Step').reset_index(drop=True)

            # --- Export Removed Indicators List (CSV) ---
            removed_csv_path = os.path.join(output_dir, f"{base_filename}_removed_indicators.csv")
            removed_indicators_final_df.to_csv(removed_csv_path, index=False)
            self.logger.info(
                f"Saved list of {len(removed_indicators_final_df)} removed indicators to: {removed_csv_path}")

            # --- Generate and Export Redundancy Report (TXT) ---
            redundancy_report_path = os.path.join(output_dir, f"{base_filename}_redundancy_report.txt")
            report_content = self._generate_redundancy_report_text(self.indicator_status_df, self.redundancy_map)
            with open(redundancy_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Saved redundancy analysis report to: {redundancy_report_path}")

            # --- Log Summary ---
            self.logger.info("--- Filtering Summary (from report export) ---")
            removal_counts = self.indicator_status_df['Removal Step'].value_counts(dropna=False).rename(
                index={pd.NA: 'Kept'})
            for step, count in removal_counts.items():
                self.logger.info(f"  {step}: {count}")

        except Exception as e:
            self.logger.error(f"Error occurred during report export: {e}", exc_info=True)