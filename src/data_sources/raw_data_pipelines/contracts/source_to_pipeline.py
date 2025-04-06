from typing import Dict, Type

from src.data_sources.data_source import DataSource
from src.data_sources.raw_data_pipelines.contracts.pipelines_contracts import DataPipeline
from src.data_sources.raw_data_pipelines.implementation.baci_pipeline import BACIDataPipeline
from src.data_sources.raw_data_pipelines.implementation.commodity_pipeline import CommodityDataPipeline
from src.data_sources.raw_data_pipelines.implementation.gravity_pipeline import GravityDataPipeline
from src.data_sources.raw_data_pipelines.implementation.inflation_pipeline import InflationDataPipeline
from src.data_sources.raw_data_pipelines.implementation.wdi_pipeline import WDIDataPipeline


class DataSourceToPipelineMatcher:
    mapping: Dict[str, Type[DataPipeline]] = {
        DataSource.WDI.value: WDIDataPipeline,
        DataSource.INFLATION.value: InflationDataPipeline,
        DataSource.BACI.value: BACIDataPipeline,
        DataSource.GRAVITY.value: GravityDataPipeline,
        DataSource.COMMODITY.value: CommodityDataPipeline
    }

    @staticmethod
    def get_pipeline(source: DataSource) -> Type[DataPipeline]:
        return DataSourceToPipelineMatcher.mapping[source]