import logging
from typing import List, Optional
import requests
import xml.etree.ElementTree as ET
import pandas as pd

from src.data_sources.insee.model.dataflow import Dataflow


class InseeDataFetcher:
    """
    INSEE data can be fetched from the INSEE API.
    This code serves for future compatibility and update with the INSEE API.
    The data can be fetched from the Azure Blob Storage as well.
    """
    INSEE_BASE_URL: str = "https://bdm.insee.fr"

    def __init__(self):
        self.logger = logging.getLogger(InseeDataFetcher.__name__)
        self.dataflow_url = InseeDataFetcher.INSEE_BASE_URL + "/series/sdmx/dataflow"
        self.sdmx_url = InseeDataFetcher.INSEE_BASE_URL + "/series/sdmx"

    def list_dataflows(self) -> List[Dataflow]:
        """
        Fetch the list of dataflows from the insee API
        :return: List of Dataflow objects
        """
        url = self.dataflow_url
        response = requests.get(url)
        response.raise_for_status()

        # Namespaces in the xml
        ns = {
            'mes': "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
            'str': "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
            'com': "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common"
        }

        # Parse the XML format
        root = ET.fromstring(response.content)

        dataflows = []

        for df in root.findall(".//str:Dataflow", ns):
            # Id (e.g. 'CNA-2020-PIB')
            dataflow_id = df.get("id")

            # Name (in French)
            name_fr = None
            for name_node in df.findall("com:Name", ns):
                if name_node.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == 'fr':
                    name_fr = name_node.text
                    break

            # URL
            annotation_url_nodes = df.findall(".//com:AnnotationURL", ns)
            annotation_url = annotation_url_nodes[0].text if annotation_url_nodes else None

            # Create a Dataflow object
            dataflow = Dataflow(dataflow_id, name_fr, annotation_url)
            dataflows.append(dataflow)

        self.logger.info(f"Found {len(dataflows)} dataflows")
        return dataflows

    def fetch_dataflow_data(self, data_id: str, params=None, remove_stopped: bool=False):
        """
        Fetch the dataflow data from the given Dataflow object, with fallback to split queries if needed.
        :param data_id: the id of the table (Dataflow.id)
        :param params: the parameters to pass to the request
        :param remove_stopped: whether to remove stopped series
        """
        url = self.sdmx_url + f"/data/{data_id}/all"
        return self._fetch_dataflow_data(url, params=params, remove_stopped=remove_stopped)

    def _fetch_dataflow_data(self, base_url, params=None, remove_stopped: bool=False) -> pd.DataFrame:
        """
        Fetch the dataflow from the given URL and parse it, with fallback to split queries if needed.
        :param base_url: the base URL of the dataflow to fetch
        :param params: the parameters to pass to the request
        :param remove_stopped: whether to remove stopped series
        :return: a pandas DataFrame with time series observations
        """
        content = self.fetch_dataflow_xml(base_url, params=params)

        split_queries = self.parse_split_queries(content)

        # If we get 200, parse the content directly
        if not split_queries:
            return self.parse_xml_content(content)

        if split_queries:
            dfs = []
            for path in split_queries:
                # Construct the full URL (sub-query)
                sub_url = self.sdmx_url + path
                sub_content = self.fetch_dataflow_xml(sub_url, params=params)

                # Parse the sub-query content
                df_sub = self.parse_xml_content(sub_content, remove_stopped)
                dfs.append(df_sub)

            return pd.concat(dfs, ignore_index=True)

        else:
            return self.parse_xml_content(content, remove_stopped)

    def fetch_dataflow_xml(self, url, params=None) -> bytes:
        """
        Fetch the XML file from the given URL
        If the response code is 200 or 510, return the content
        :param url: the URL of the dataflow to fetch
        :param params: the parameters to pass to the request
        """
        if params is None:
            params = {}
        r = requests.get(url, params=params)

        # 200: OK, 510: Request is too large
        if r.status_code == 200:
            return r.content
        else:
            content = r.content
            root = ET.fromstring(content)

            error_msg = root.find(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message}ErrorMessage")
            if error_msg is not None and error_msg.get("code") == "510":
                self.logger.warning("Request is too large, splitting the query")
                return content
            else:
                r.raise_for_status()

    import xml.etree.ElementTree as ET
    import pandas as pd

    def parse_xml_content(self, xml_content: bytes, remove_stopped=False) -> pd.DataFrame:
        """
        Parse the StructureSpecific SDMX XML response from Insee
        and return a pandas DataFrame with time series observations.
        :param xml_content: the XML content to parse
        :param remove_stopped: whether to remove stopped series
        :return: a pandas DataFrame with time series observations
        """

        root = ET.fromstring(xml_content)
        all_rows = []

        # Find all <Series> elements (no namespace)
        # e.g.  <Series IDBANK="xxx"> <Obs TIME_PERIOD="yyy" /> ...
        series_list = root.findall(".//Series")  # no namespace prefix

        for series in series_list:
            s_attrib = series.attrib  # dictionary of attributes on <Series>
            if s_attrib.get("SERIE_ARRETEE") == "TRUE" and remove_stopped:
                self.logger.info(f"Skipping stopped series {s_attrib.get('IDBANK')}")
                continue

            # Find all <Obs> children (no namespace)
            obs_list = series.findall("./Obs")

            for obs in obs_list:
                time_period = obs.get("TIME_PERIOD")
                obs_value_str = obs.get("OBS_VALUE")
                obs_status = obs.get("OBS_STATUS")

                # Convert obs_value to float (unless it's 'NaN')
                if obs_value_str == "NaN":
                    obs_value = None
                else:
                    try:
                        obs_value = float(obs_value_str)
                    except (ValueError, TypeError):
                        self.logger.warning(f"OBS_VALUE could not be parsed to float: {obs_value_str}")
                        obs_value = None

                row = {
                    "TITLE_FR": s_attrib.get("TITLE_FR"),
                    "TITLE_EN": s_attrib.get("TITLE_EN"),
                    "STOPPED": s_attrib.get("SERIE_ARRETEE"),
                    "LAST_UPDATE": s_attrib.get("LAST_UPDATE"),
                    "FREQ": s_attrib.get("FREQ"),
                    "LAST_UPDATED": s_attrib.get("LAST_UPDATED"),
                    "IDBANK": s_attrib.get("IDBANK"),
                    "INDICATEUR": s_attrib.get("INDICATEUR"),
                    "CORRECTION": s_attrib.get("CORRECTION"),
                    "NATURE": s_attrib.get("NATURE"),
                    "UNIT": s_attrib.get("UNIT_MEASURE"),
                    "MULT": s_attrib.get("UNIT_MULT"),
                    "TIME_PERIOD": time_period,
                    "OBS_VALUE": obs_value,
                    "OBS_STATUS": obs_status,

                }
                all_rows.append(row)

        return pd.DataFrame(all_rows)

    @staticmethod
    def parse_split_queries(xml_content: bytes) -> Optional[List[str]]:
        """
        Returns a list of split queries if the XML content contains a 510 error, None otherwise.
        :param xml_content: the XML content to parse
        :return: a list of split queries or None
        """
        root = ET.fromstring(xml_content)

        error_msg = root.find(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message}ErrorMessage")
        if error_msg is None:
            return None

        split_queries = []
        if error_msg.get("code") == "510":
            ns_com = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common}"
            text_elements = error_msg.findall(f".//{ns_com}Text")

            for el in text_elements:
                text_val = el.text or ""
                text_val = text_val.strip()
                if text_val.startswith("/data/"):
                    split_queries.append(text_val)

        return split_queries