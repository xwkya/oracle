# src/data_sources/wdi/data_fetcher.py
import os
from typing import Tuple, List, Dict

import pandas as pd
import logging

from src.core_utils import CoreUtils


class WDIDataFetcher:
    """
    Handles loading the raw WDI data.
    """
    _logger = logging.getLogger(__name__)

    @staticmethod
    def load_wdi_data() -> pd.DataFrame | None:
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


def get_base_code(indicator_code: str) -> str:
    """
    Attempts to identify the 'base' conceptual part of an indicator code
    by removing known unit/transformation/per-capita/other suffixes for grouping purposes.

    Args:
        indicator_code (str): The full WDI indicator code.

    Returns:
        str: The inferred base code.
    """
    base = indicator_code
    temp_strip_suffixes = sorted(UNIT_TRANSFORM_SUFFIXES + OTHER_KEEP_SUFFIXES + PCAP_SUFFIXES, key=len, reverse=True)
    stripped = False
    for suffix in temp_strip_suffixes:
        if base.endswith(suffix):
            potential_base = base[:-len(suffix)]
            if len(potential_base) > 1 and '.' in potential_base and potential_base[-1] != '.':
                base = potential_base
                stripped = True
                break
    return base


class WDIDataPipeline:
    """
    Encapsulates the WDI data filtering and processing pipeline.
    Includes methods for exporting results and the filtered dataset.
    """

    def __init__(self,
                 overall_fill_threshold: float = 0.25,
                 poor_country_fill_threshold: float = 0.50,
                 country_proportion_threshold: float = 0.25,
                 stopped_year_threshold: int = 2015,
                 stopped_fill_threshold: float = 0.20,
                 preferred_level_suffix: str = ".KD",
                 secondary_level_suffix: str = ".PP.KD",
                 preferred_growth_suffix: str = ".KD.ZG"):
        """
        Initializes the pipeline with filtering parameters.
        (Args description omitted for brevity - see previous version)
        """
        self.logger = logging.getLogger(__name__)
        self.overall_fill_threshold = overall_fill_threshold
        self.poor_country_fill_threshold = poor_country_fill_threshold
        self.country_proportion_threshold = country_proportion_threshold
        self.stopped_year_threshold = stopped_year_threshold
        self.stopped_fill_threshold = stopped_fill_threshold
        self.preferred_level_suffix = preferred_level_suffix
        self.secondary_level_suffix = secondary_level_suffix
        self.preferred_growth_suffix = preferred_growth_suffix
        self.logger.info("WDIDataPipeline initialized with parameters:")
        # Log parameters (omitted for brevity)

    def _melt_data(self, df_subset: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Melts the DataFrame from wide to long format."""
        # (Implementation remains the same as before)
        self.logger.debug("Melting DataFrame...")
        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        value_vars = [col for col in df_subset.columns if col.isdigit() and int(col) >= 1960]
        if not value_vars:
            raise ValueError("No year columns found in the subsetted DataFrame.")

        df_melted = pd.melt(df_subset, id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
        df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
        df_melted.dropna(subset=['Year'], inplace=True)
        df_melted['Year'] = df_melted['Year'].astype(int)

        self.logger.debug(f"Melting complete. Shape: {df_melted.shape}")
        return df_melted, value_vars

    def _apply_filters(self, df_melted: pd.DataFrame, countries: List[str], value_vars: List[str]) -> pd.DataFrame:
        """Applies the sequential filtering logic (Steps 1-3)."""
        # (Implementation remains the same as before)
        # --- Initialize Status Tracking ---
        indicator_names = df_melted[['Indicator Code', 'Indicator Name']].drop_duplicates().set_index('Indicator Code')
        all_indicator_codes = df_melted['Indicator Code'].unique()
        indicator_status = pd.DataFrame({'Indicator Code': all_indicator_codes})
        indicator_status = indicator_status.merge(indicator_names, on='Indicator Code', how='left')
        indicator_status['Status'] = 'kept'
        indicator_status['Removal Step'] = pd.NA  # Use pandas NA
        indicator_status['Reason'] = pd.NA

        self.logger.info(f"Starting filtering with {len(indicator_status)} unique indicators.")

        # --- Step 1: Filter by Overall Fill Ratio ---
        self.logger.info(f"Applying Step 1: Overall Fill Ratio Filter (Threshold < {self.overall_fill_threshold:.2f})")
        total_possible_points = len(value_vars) * len(countries)
        if total_possible_points == 0:
            self.logger.warning("Cannot calculate overall fill ratio: zero possible data points. Skipping Step 1.")
            removed_count_step1 = 0
        else:
            overall_fill_ratios = df_melted.groupby('Indicator Code')['Value'].count() / total_possible_points
            indicators_to_remove_step1 = overall_fill_ratios[overall_fill_ratios < self.overall_fill_threshold].index
            mask_step1 = indicator_status['Indicator Code'].isin(indicators_to_remove_step1)
            indicator_status.loc[mask_step1, 'Status'] = 'removed'
            indicator_status.loc[mask_step1, 'Removal Step'] = '1. Overall Fill Ratio'
            ratios_for_removed = overall_fill_ratios.loc[indicators_to_remove_step1].apply(lambda x: f"{x:.2f}")
            reasons_step1 = "Overall Fill Ratio " + ratios_for_removed + f" < {self.overall_fill_threshold:.2f}"
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
        if not indicators_to_check_step2.empty:
            country_indicator_fill = df_melted.groupby(['Indicator Code', 'Country Code'])['Value'].count() / len(
                value_vars)
            country_indicator_fill = country_indicator_fill.unstack()
            country_indicator_fill = country_indicator_fill.reindex(index=indicators_to_check_step2,
                                                                    columns=countries).fillna(0)
            if not country_indicator_fill.empty:
                poor_coverage_mask = country_indicator_fill < self.poor_country_fill_threshold
                proportion_poorly_covered = poor_coverage_mask.sum(axis=1) / len(countries)
                indicators_to_remove_step2 = proportion_poorly_covered[
                    proportion_poorly_covered >= self.country_proportion_threshold].index
                mask_step2 = indicator_status['Indicator Code'].isin(indicators_to_remove_step2) & (
                            indicator_status['Status'] == 'kept')
                indicator_status.loc[mask_step2, 'Status'] = 'removed'
                indicator_status.loc[mask_step2, 'Removal Step'] = '2. Poor Country Coverage'
                props_for_removed = proportion_poorly_covered.loc[indicators_to_remove_step2].apply(
                    lambda x: f"{x:.2f}")
                reasons_step2 = f"Poor Coverage Proportion " + props_for_removed + f" >= {self.country_proportion_threshold:.2f} (for fill < {self.poor_country_fill_threshold:.2f})"
                indicator_status.loc[mask_step2, 'Reason'] = reasons_step2.reindex(
                    indicator_status.loc[mask_step2, 'Indicator Code']).values
                removed_count_step2 = mask_step2.sum()
            else:
                self.logger.warning("Could not calculate country fill ratios for remaining indicators. Skipping Step 2.")
        else:
            self.logger.info("No indicators left to check for Step 2.")
        kept_count_step2 = len(indicator_status[indicator_status['Status'] == 'kept'])
        self.logger.info(f"Removed {removed_count_step2} indicators in Step 2. Kept: {kept_count_step2}")

        # --- Step 3: Filter by Stopped Series (Recent Fill Ratio) ---
        self.logger.info(
            f"Applying Step 3: Stopped Series Filter (Fill Ratio after {self.stopped_year_threshold} < {self.stopped_fill_threshold:.2f})")
        indicators_to_check_step3 = indicator_status[indicator_status['Status'] == 'kept']['Indicator Code']
        removed_count_step3 = 0
        numeric_years = pd.to_numeric(value_vars, errors='coerce')
        years_after_threshold = [year for year, num_year in zip(value_vars, numeric_years) if
                                 pd.notna(num_year) and num_year >= self.stopped_year_threshold]
        if not years_after_threshold:
            self.logger.warning(f"No year columns found >= {self.stopped_year_threshold}. Skipping Step 3.")
        elif not indicators_to_check_step3.empty:
            df_recent = df_melted[df_melted['Year'] >= self.stopped_year_threshold]
            total_possible_points_recent = len(countries) * len(years_after_threshold)
            if total_possible_points_recent > 0:
                recent_fill_counts = df_recent.groupby('Indicator Code')['Value'].count()
                recent_fill_counts = recent_fill_counts.reindex(indicators_to_check_step3, fill_value=0)
                recent_fill_ratios = recent_fill_counts / total_possible_points_recent
                indicators_to_remove_step3 = recent_fill_ratios[recent_fill_ratios < self.stopped_fill_threshold].index
                mask_step3 = indicator_status['Indicator Code'].isin(indicators_to_remove_step3) & (
                            indicator_status['Status'] == 'kept')
                indicator_status.loc[mask_step3, 'Status'] = 'removed'
                indicator_status.loc[mask_step3, 'Removal Step'] = '3. Stopped Series (Recent Fill)'
                ratios_for_removed_step3 = recent_fill_ratios.loc[indicators_to_remove_step3].apply(
                    lambda x: f"{x:.2f}")
                reasons_step3 = f"Recent Fill Ratio (>= {self.stopped_year_threshold}) " + ratios_for_removed_step3 + f" < {self.stopped_fill_threshold:.2f}"
                indicator_status.loc[mask_step3, 'Reason'] = reasons_step3.reindex(
                    indicator_status.loc[mask_step3, 'Indicator Code']).values
                removed_count_step3 = mask_step3.sum()
            else:
                self.logger.warning("Cannot calculate recent fill ratio: zero possible data points. Skipping Step 3.")
        else:
            self.logger.info("No indicators left to check for Step 3.")
        kept_count_step3 = len(indicator_status[indicator_status['Status'] == 'kept'])
        self.logger.info(f"Removed {removed_count_step3} indicators in Step 3. Kept: {kept_count_step3}")

        return indicator_status

    def _apply_redundancy_filter(self, indicator_status_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Applies the redundancy filter (Step 4)."""
        # (Implementation remains the same as before)
        self.logger.info(f"Applying Step 4: Redundancy Filter (with Fallback)")
        kept_indicators_df = indicator_status_df[indicator_status_df['Status'] == 'kept'].copy()
        if kept_indicators_df.empty:
            logger.warning("No indicators left to apply redundancy filter. Skipping Step 4.")
            return indicator_status_df, {}

        kept_indicators_df['Base_Code'] = kept_indicators_df['Indicator Code'].apply(get_base_code)
        grouped = kept_indicators_df.groupby('Base_Code')

        final_keep_codes_set = set()
        redundancy_map = {}
        codes_removed_in_step4_set = set()

        for base_code, group in grouped:
            group_codes = group['Indicator Code'].tolist()
            kept_in_group_this_base = []
            removed_in_group_map = {}

            levels = [c for c in group_codes if
                      not any(c.endswith(s) for s in ['.ZG'] + OTHER_KEEP_SUFFIXES) and not any(
                          c.endswith(pc) for pc in PCAP_SUFFIXES)]
            growths = [c for c in group_codes if c.endswith('.ZG') and not any(c.endswith(pc) for pc in PCAP_SUFFIXES)]
            pc_levels = [c for c in group_codes if
                         any(c.endswith(pc) for pc in PCAP_SUFFIXES) and not c.endswith('.ZG') and not any(
                             c.endswith(s) for s in OTHER_KEEP_SUFFIXES)]
            pc_growths = [c for c in group_codes if any(c.endswith(pc) for pc in PCAP_SUFFIXES) and c.endswith('.ZG')]
            other_keep = [c for c in group_codes if any(c.endswith(s) for s in OTHER_KEEP_SUFFIXES)]

            # Process Levels
            chosen_level = None
            levels_kd = [c for c in levels if c.endswith(self.preferred_level_suffix)]
            levels_ppkd = [c for c in levels if c.endswith(self.secondary_level_suffix)]
            if levels_kd:
                chosen_level = levels_kd[0]
            elif levels_ppkd:
                chosen_level = levels_ppkd[0]
            elif levels:
                chosen_level = sorted(levels)[0]
            if chosen_level:
                kept_in_group_this_base.append(chosen_level)
                for code in levels:
                    if code != chosen_level: removed_in_group_map[code] = chosen_level

            # Process Growth
            chosen_growth = None
            growths_kd_zg = [g for g in growths if g.endswith(self.preferred_growth_suffix)]
            if growths_kd_zg:
                chosen_growth = growths_kd_zg[0]
            elif growths:
                chosen_growth = sorted(growths)[0]
            if chosen_growth:
                kept_in_group_this_base.append(chosen_growth)
                for code in growths:
                    if code != chosen_growth: removed_in_group_map[code] = chosen_growth

            # Process Per Capita Levels
            chosen_pc_level = None
            pc_levels_kd = [c for c in pc_levels if c.endswith(self.preferred_level_suffix)]
            pc_levels_ppkd = [c for c in pc_levels if c.endswith(self.secondary_level_suffix)]
            if pc_levels_kd:
                chosen_pc_level = pc_levels_kd[0]
            elif pc_levels_ppkd:
                chosen_pc_level = pc_levels_ppkd[0]
            elif pc_levels:
                chosen_pc_level = sorted(pc_levels)[0]
            if chosen_pc_level:
                kept_in_group_this_base.append(chosen_pc_level)
                for code in pc_levels:
                    if code != chosen_pc_level: removed_in_group_map[code] = chosen_pc_level

            # Process Per Capita Growth
            chosen_pc_growth = None
            pc_growths_kd_zg = [g for g in pc_growths if g.endswith(self.preferred_growth_suffix)]
            if pc_growths_kd_zg:
                chosen_pc_growth = pc_growths_kd_zg[0]
            elif pc_growths:
                chosen_pc_growth = sorted(pc_growths)[0]
            if chosen_pc_growth:
                kept_in_group_this_base.append(chosen_pc_growth)
                for code in pc_growths:
                    if code != chosen_pc_growth: removed_in_group_map[code] = chosen_pc_growth

            # Keep Other Types
            kept_in_group_this_base.extend(other_keep)

            # Update final sets and map
            final_keep_codes_set.update(kept_in_group_this_base)
            for removed_code, kept_code in removed_in_group_map.items():
                if kept_code in kept_in_group_this_base:
                    if kept_code not in redundancy_map:
                        redundancy_map[kept_code] = []
                    redundancy_map[kept_code].append(removed_code)
                    codes_removed_in_step4_set.add(removed_code)

        removed_count_step4 = len(codes_removed_in_step4_set)
        final_kept_count = len(final_keep_codes_set)
        self.logger.info(f"Removed {removed_count_step4} indicators in Step 4 (Redundancy). Final Kept: {final_kept_count}")

        # Update the status DataFrame
        for code in codes_removed_in_step4_set:
            removed_for_code = None
            reason_suffix = "Removed due to redundancy"
            for kept_c, removed_l in redundancy_map.items():
                if code in removed_l:
                    removed_for_code = kept_c
                    reason_suffix = f"Removed in favor of {removed_for_code}"
                    break
            mask_step4 = (indicator_status_df['Indicator Code'] == code) & (indicator_status_df['Status'] == 'kept')
            if mask_step4.any():
                indicator_status_df.loc[mask_step4, 'Status'] = 'removed'
                indicator_status_df.loc[mask_step4, 'Removal Step'] = '4. Redundancy'
                indicator_status_df.loc[mask_step4, 'Reason'] = f"Redundant: {reason_suffix}"

        return indicator_status_df, redundancy_map

    def process(self, raw_df: pd.DataFrame, countries_of_interest: List[str]) -> Tuple[
        pd.DataFrame, Dict[str, List[str]]]:
        """
        Executes the full WDI data filtering pipeline.

        """
        self.logger.info("Starting WDI data processing pipeline.")
        df_subset = raw_df[raw_df['Country Code'].isin(countries_of_interest)].copy()
        if df_subset.empty:
            self.logger.error("DataFrame is empty after filtering by countries of interest. No data to process.")
            return pd.DataFrame(columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason']), {}
        df_subset.reset_index(drop=True, inplace=True)
        self.logger.info(f"Subsetted data for {len(countries_of_interest)} countries. Shape: {df_subset.shape}")

        try:
            df_melted, value_vars = self._melt_data(df_subset)
        except ValueError as e:
            self.logger.error(f"Error during data melting: {e}")
            return pd.DataFrame(columns=['Indicator Code', 'Indicator Name', 'Status', 'Removal Step', 'Reason']), {}

        indicator_status_df = self._apply_filters(df_melted, countries_of_interest, value_vars)
        final_indicator_status_df, redundancy_map = self._apply_redundancy_filter(indicator_status_df)
        self.logger.info("WDI data processing pipeline finished.")
        return final_indicator_status_df, redundancy_map

    def _generate_redundancy_report(self, final_kept_df: pd.DataFrame, redundancy_map: Dict[str, list],
                                    original_status_df: pd.DataFrame) -> str:
        """Generates a text report detailing the redundancy removal decisions."""
        report_lines = []
        report_lines.append("--- Redundancy Removal Report (v3 - with Fallback) ---\n")
        report_lines.append(
            "This report shows which indicators were kept and which potentially redundant indicators were removed in their favor.")
        report_lines.append(
            f"Preference order: Constant USD ({self.preferred_level_suffix}) > Constant PPP ({self.secondary_level_suffix}) > Other Levels (e.g., .CN, .CD) for levels.")
        report_lines.append(
            f"Preference order: Growth of Constant USD ({self.preferred_growth_suffix}) > Other Growths for growth rates.")
        report_lines.append(
            "If preferred versions were not available, the first available non-preferred version is kept as a fallback.")
        report_lines.append(
            f"Ratio ({', '.join(OTHER_KEEP_SUFFIXES)}), etc. indicators are generally kept alongside the selected level/growth.\n")

        original_names = original_status_df.set_index('Indicator Code')['Indicator Name'].to_dict()

        for _, row in final_kept_df.iterrows():
            kept_code = row['Indicator Code']
            kept_name = row['Indicator Name']
            report_lines.append(f"Kept time series: {kept_code} | {kept_name}")

            is_level_fallback = (not any(
                kept_code.endswith(s) for s in [self.preferred_level_suffix, self.secondary_level_suffix]) and not any(
                kept_code.endswith(s) for s in ['.ZG'] + OTHER_KEEP_SUFFIXES) and not any(
                kept_code.endswith(pc) for pc in PCAP_SUFFIXES))
            is_growth_fallback = (
                        kept_code.endswith('.ZG') and not kept_code.endswith(self.preferred_growth_suffix) and not any(
                    kept_code.endswith(pc) for pc in PCAP_SUFFIXES))
            is_pc_level_fallback = (any(kept_code.endswith(pc) for pc in PCAP_SUFFIXES) and not any(
                kept_code.endswith(s) for s in
                [self.preferred_level_suffix, self.secondary_level_suffix, '.ZG'] + OTHER_KEEP_SUFFIXES))
            is_pc_growth_fallback = (any(kept_code.endswith(pc) for pc in PCAP_SUFFIXES) and kept_code.endswith(
                '.ZG') and not kept_code.endswith(self.preferred_growth_suffix))

            if is_level_fallback or is_growth_fallback or is_pc_level_fallback or is_pc_growth_fallback:
                report_lines[-1] += "  **[Fallback Selection]**"
            report_lines.append("")

            removed_list = redundancy_map.get(kept_code, [])
            if removed_list:
                report_lines.append(f"  Removed time series: // duplicates removed in favor of the above")
                for rem_code in sorted(removed_list):
                    rem_name = original_names.get(rem_code, "Name not found")
                    report_lines.append(f"    {rem_code} | {rem_name}")
            else:
                report_lines.append(f"  Removed time series: // empty")
            report_lines.append("-" * 20 + "\n")

        return "\n".join(report_lines)

    def export_results(self,
                       indicator_status_df: pd.DataFrame,
                       redundancy_map: Dict[str, list],
                       output_dir: str,
                       output_format: str = 'csv',
                       export_reports: bool = False):  # Added export_reports flag
        """
        Exports the filtering results (kept/removed lists, optional reports).

        Args:
            indicator_status_df (pd.DataFrame): DataFrame with the final status of all indicators.
            redundancy_map (Dict[str, list]): Dictionary mapping kept codes to removed redundant codes.
            output_dir (str): The directory to save output files.
            output_format (str): 'csv' or 'db'.
            export_reports (bool): If True, generate and save the removed list and redundancy report txt file.
                                   Defaults to False.
        """
        self.logger.info(f"Exporting filtering results in '{output_format}' format to directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Separate Kept and Removed Indicators based on final status
        kept_indicators_final_df = indicator_status_df[indicator_status_df['Status'] == 'kept'][
            ['Indicator Code', 'Indicator Name']].sort_values('Indicator Code')
        removed_indicators_final_df = indicator_status_df[indicator_status_df['Status'] == 'removed'][
            ['Indicator Code', 'Indicator Name', 'Removal Step', 'Reason']].sort_values('Indicator Code')

        if output_format == 'csv':
            # Always export the list of kept indicator codes/names
            kept_series_list_file = os.path.join(output_dir, "final_kept_series_list_wdi.csv")
            try:
                kept_indicators_final_df.to_csv(kept_series_list_file, index=False)
                self.logger.info(
                    f"Successfully saved {len(kept_indicators_final_df)} final kept indicator codes/names to '{kept_series_list_file}'")
            except Exception as e:
                self.logger.error(f"Error saving final kept indicators list file: {e}", exc_info=True)

            # Export reports only if requested
            if export_reports:
                self.logger.info("Exporting detailed removal reports as requested.")
                removed_series_file = os.path.join(output_dir, "final_removed_series_wdi.csv")
                redundancy_report_file = os.path.join(output_dir, "redundancy_removal_report_wdi.txt")

                try:
                    removed_indicators_final_df.to_csv(removed_series_file, index=False)
                    self.logger.info(
                        f"Successfully saved {len(removed_indicators_final_df)} removed indicators to '{removed_series_file}'")
                except Exception as e:
                    self.logger.error(f"Error saving removed indicators file: {e}", exc_info=True)

                try:
                    report_content = self._generate_redundancy_report(kept_indicators_final_df, redundancy_map,
                                                                      indicator_status_df)
                    with open(redundancy_report_file, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    self.logger.info(f"Successfully saved redundancy removal report to '{redundancy_report_file}'")
                except Exception as e:
                    self.logger.error(f"Error saving redundancy report file: {e}", exc_info=True)
            else:
                self.logger.info(
                    "Skipping export of detailed removal reports (final_removed_series_wdi.csv, redundancy_removal_report_wdi.txt).")

            # Display Summary (always displayed)
            self.logger.info("--- Final Filtering Summary ---")
            removal_counts = indicator_status_df['Removal Step'].value_counts(dropna=False).rename(
                index={pd.NA: 'Kept'})
            for step, count in removal_counts.items():
                self.logger.info(f"  {step}: {count}")


        elif output_format == 'db':
            self.logger.error("Database output is not yet implemented for WDI results.")
            raise NotImplementedError("Database export for WDI results is not implemented.")
        else:
            self.logger.error(f"Unsupported output format: {output_format}. Choose 'csv'.")

    def export_filtered_data(self,
                             raw_df: pd.DataFrame,
                             final_indicator_status_df: pd.DataFrame,
                             countries_of_interest: List[str],
                             output_path: str):
        """
        Exports the filtered WDI data (selected countries, kept indicators) to a CSV file
        in the original wide format.

        Args:
            raw_df (pd.DataFrame): The original raw WDI DataFrame (wide format).
            final_indicator_status_df (pd.DataFrame): DataFrame with final indicator statuses.
            countries_of_interest (List[str]): List of country codes to keep.
            output_path (str): The full path for the output CSV file.
        """
        self.logger.info(f"Exporting filtered WDI data to: {output_path}")

        kept_indicator_codes = final_indicator_status_df[final_indicator_status_df['Status'] == 'kept'][
            'Indicator Code'].tolist()

        if not kept_indicator_codes:
            self.logger.warning("No indicators were kept after filtering. Exporting an empty file.")
            header_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + \
                          sorted([col for col in raw_df.columns if col.isdigit() and int(col) >= 1960])
            empty_df = pd.DataFrame(columns=header_cols)
            try:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                empty_df.to_csv(output_path, index=False)
                self.logger.info("Saved empty file with headers as no indicators were kept.")
            except Exception as e:
                self.logger.error(f"Error saving empty filtered data file: {e}", exc_info=True)
            return

        try:
            # Filter rows (countries and indicators)
            filtered_df = raw_df[
                raw_df['Country Code'].isin(countries_of_interest) &
                raw_df['Indicator Code'].isin(kept_indicator_codes)
                ].copy()

            # Ensure correct columns are present (ID columns + year columns)
            id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
            year_cols = sorted([col for col in raw_df.columns if col.isdigit() and int(col) >= 1960])
            final_cols = id_cols + year_cols

            # Reorder and select final columns - handle potential missing year columns gracefully if needed, though unlikely here
            filtered_df = filtered_df[final_cols]

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            # Save to CSV
            filtered_df.to_csv(output_path, index=False)
            self.logger.info(
                f"Successfully saved filtered WDI data ({filtered_df.shape[0]} rows, {len(kept_indicator_codes)} indicators) to '{output_path}'")

        except KeyError as e:
            self.logger.error(
                f"Missing expected columns during final export: {e}. Available columns: {raw_df.columns.tolist()}")
        except Exception as e:
            self.logger.error(f"Error exporting filtered WDI data: {e}", exc_info=True)
