import json
from typing import Dict, Any

import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, coalesce, lit

from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.pipelines.ml_pipeline import MLPipeline
from src.storage.iceberg_manager import IcebergTableManager
from src.storage.mlflow_manager import MLflowManager


class IcebergService:
    def __init__(self, iceberg_manager: IcebergTableManager):
        self.iceberg_manager = iceberg_manager

    def sanitize_dataframe(self, df: DataFrame) -> DataFrame:
        cleaned_cols = []
        for field in df.schema.fields:
            c = col(field.name)
            field_type = field.dataType.simpleString().lower()

            if field_type in ["double", "float"]:
                # Numeric: handle NaN + null
                cleaned = (
                    when(isnull(c) | isnan(c), None).otherwise(c).alias(field.name)
                )
            elif field_type.startswith("array"):
                # Arrays: null-safe coalesce only
                cleaned = coalesce(c, lit(None)).alias(field.name)
            else:
                # Strings, timestamps, booleans: null-safe only
                cleaned = coalesce(c, lit(None).cast(field.dataType)).alias(field.name)

            cleaned_cols.append(cleaned)

        return df.select(*cleaned_cols)

    def query(self, filter_condition=None, limit=100):
        df = self.iceberg_manager.query_table(filter_condition, limit)
        df_clean = self.sanitize_dataframe(df)

        return [json.loads(row) for row in df_clean.toJSON().collect()]

    def stats(self):
        return self.iceberg_manager.get_table_statistics()

    def time_travel(self, snapshot_id=None, timestamp=None):
        df = self.iceberg_manager.time_travel_query(snapshot_id, timestamp)
        return df.toPandas().to_dict(orient="records")


class IngestionService:

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def run_ingestion(
            self,
            run_adverse_media: bool = True,
            optimize_table: bool = False
    ) -> Dict[str, Any]:
        result = self.pipeline.run(
            run_adverse_media=run_adverse_media,
            optimize_table=optimize_table
        )
        return result


class ModelService:
    def __init__(
            self,
            spark: SparkSession,
            mlflow_manager: MLflowManager,
            pipeline: MLPipeline
    ):
        self.spark = spark
        self.mlflow_manager = mlflow_manager
        self.pipeline = pipeline

    def predict(self, model_name: str, records: list, stage: str):
        model = self.mlflow_manager.load_model(model_name=model_name, stage=stage)

        pdf = pd.DataFrame(records)
        sdf = self.spark.createDataFrame(pdf)

        predictions = model.transform(sdf).select("prediction").toPandas()

        return predictions["prediction"].tolist()

    def run_pipeline(
            self,
            train_only: bool = False,
            skip_validation: bool = False,
            resume_from_checkpoint: bool = False
    ) -> Dict:

        return self.pipeline.run(
            train_only=train_only,
            skip_validation=skip_validation,
            resume_from_checkpoint=resume_from_checkpoint,
        )
