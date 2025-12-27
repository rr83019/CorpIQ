from datetime import datetime, timedelta
from typing import Dict, Any
import os
import yaml
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.operators.python import PythonOperator

BASE_PATH = os.environ.get("AIRFLOW__CORE__DAGS_FOLDER", "/src/airflow")
CONFIG_PATH = os.path.join(BASE_PATH, "../config/config.yaml")
INGESTION_APP = os.path.join(BASE_PATH, "../src/pipelines/ingestion_pipeline.py")
ML_APP = os.path.join(BASE_PATH, "../src/pipelines/ml_pipeline.py")


def load_config(**context) -> Dict[str, Any]:
    config_path = context["params"].get("config_path", CONFIG_PATH)
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config not found: {config_path}")
        return {}


def build_spark_args(config: Dict[str, Any]) -> Dict[str, Any]:
    spark_cfg = config.get("spark", {})
    spark_conf = spark_cfg.get("configs", {})

    packages = spark_conf.get("spark.jars.packages", "")
    conf = {k: str(v) for k, v in spark_conf.items() if k != "spark.jars.packages"}

    resources = config.get("resources", {}).get("spark", {})

    return {
        "packages": packages,
        "conf": conf,
        "driver_memory": resources.get("driver_memory", "2g"),
        "executor_memory": resources.get("executor_memory", "2g"),
        "executor_cores": resources.get("executor_cores", 1),
        "num_executors": resources.get("num_executors", 1),
    }


default_args = {
    "owner": "data-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="corporate_data_ml_pipeline",
    description="Corporate Data → ML Pipeline (Spark/Iceberg/MLflow)",
    schedule="0 2 * * *",  # 2 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags={"spark", "iceberg", "mlflow"},
    params={"config_path": CONFIG_PATH},
) as dag:
    load_config_task = PythonOperator(
        task_id="load_and_validate_config",
        python_callable=load_config,
    )

    run_ingestion = SparkSubmitOperator(
        task_id="data_harmonization_ingestion",
        conn_id="spark_default",
        application=INGESTION_APP,
        application_args=["--config", CONFIG_PATH],
        packages="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1",
        conf={
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.test_catalog": "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.test_catalog.type": "hadoop",
        },
        driver_memory="2g",
        executor_memory="2g",
        executor_cores=1,
        num_executors=1,
        env_vars={
            "PYTHONPATH": "/src",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        dag=dag,
    )

    run_ml = SparkSubmitOperator(
        task_id="ml_model_training",
        conn_id="spark_default",
        application=ML_APP,
        application_args=["--config", CONFIG_PATH],
        packages="org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1",
        conf={
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        },
        driver_memory="4g",
        executor_memory="4g",
        executor_cores=2,
        num_executors=2,
        env_vars={
            "PYTHONPATH": "/src",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        dag=dag,
    )

    load_config_task >> run_ingestion >> run_ml
