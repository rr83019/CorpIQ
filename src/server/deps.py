from pathlib import Path
from functools import lru_cache


from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.utils.helper import create_spark_session
from src.storage.iceberg_manager import IcebergTableManager
from src.storage.mlflow_manager import MLflowManager
from src.utils.helper import load_config
from src.pipelines.ml_pipeline import MLPipeline


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@lru_cache
def get_config():
    return load_config(str(CONFIG_PATH))


@lru_cache
def get_mlflow_manager():
    return MLflowManager(get_config())


@lru_cache
def get_iceberg_manager():
    spark = create_spark_session(get_config())
    mgr = IcebergTableManager(get_config(), spark)
    mgr.initialize_catalog_and_database()
    return mgr


@lru_cache
def get_ml_pipeline():
    return MLPipeline(config_path=str(CONFIG_PATH))


@lru_cache
def get_ingestion_pipeline():
    return IngestionPipeline(config_path=str(CONFIG_PATH))
