from typing import Dict, Optional
from pyspark.sql import DataFrame, SparkSession, Column
from pyspark.sql import functions as F
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    DoubleType,
    IntegerType,
)


class DataLoader:
    """
    Handles loading data
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.source_config = config["data_sources"]
        self.data_config = config["data"]

    def get_source1_spark_schema(self) -> StructType:
        """
        Schema for Source 1 (supply chain).
        """
        return StructType(
            [
                StructField("corporate_name_S1", StringType(), nullable=True),
                StructField("address", StringType(), nullable=True),
                StructField(
                    "activity_places",
                    ArrayType(StringType(), containsNull=False),
                    nullable=True,
                ),
                StructField(
                    "top_suppliers",
                    ArrayType(StringType(), containsNull=False),
                    nullable=True,
                ),
                StructField("country", StringType(), nullable=True),
                StructField("postal_code", StringType(), nullable=True),
                StructField("industry_sector", StringType(), nullable=True),
            ]
        )

    def get_source2_spark_schema(self) -> StructType:
        """
        Schema for Source 2 (financials).
        """
        return StructType(
            [
                StructField("corporate_name_S2", StringType(), nullable=True),
                StructField(
                    "main_customers",
                    ArrayType(StringType(), containsNull=False),
                    nullable=True,
                ),
                StructField("revenue", DoubleType(), nullable=True),
                StructField("profit", DoubleType(), nullable=True),
                StructField("country", StringType(), nullable=True),
                StructField("postal_code", StringType(), nullable=True),
                StructField("industry_sector", StringType(), nullable=True),
                StructField("fiscal_year", IntegerType(), nullable=True),
            ]
        )

    def load_source1(self) -> DataFrame:
        return self._load_source(
            source_key="source1",
            schema_fn=self.get_source1_spark_schema,
            source_name="Source1",
        )

    def load_source2(self) -> DataFrame:
        return self._load_source(
            source_key="source2",
            schema_fn=self.get_source2_spark_schema,
            source_name="Source2",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 2, 10), reraise=True)
    def _load_source(self, source_key: str, schema_fn, source_name: str) -> DataFrame:
        config = self.source_config[source_key]
        logger.info(f"Loading {source_name}")

        df = self._read_raw(config, schema_fn())

        if config.get("schema_validation", False):
            df = self._apply_schema(df, schema_fn(), source_name)

        return self._add_metadata(df, source_name)

    def _read_raw(self, cfg: Dict, schema: Optional[StructType]) -> DataFrame:
        return (
            self.spark.read.format(cfg["format"])
            # .schema(schema)
            .option("header", "true")
            .option("inferSchema", "true")
            .load(cfg["path"])
        )

    def _apply_schema(
        self, df: DataFrame, expected_schema: StructType, source_name: str
    ) -> DataFrame:
        diff = self._analyze_schema_diff(df.columns, expected_schema)
        self._log_schema_diff(diff, source_name)

        df = self._add_missing_columns(df, diff["missing"], expected_schema)
        return self._select_and_cast_columns(df, expected_schema)

    def _analyze_schema_diff(
        self, actual_columns: list, expected_schema: StructType
    ) -> Dict[str, set]:
        expected = {f.name for f in expected_schema.fields}
        actual = set(actual_columns)

        return {"missing": expected - actual, "extra": actual - expected}

    def _log_schema_diff(self, diff: Dict[str, set], source_name: str):
        if diff["missing"]:
            logger.warning(f"{source_name} missing columns: {diff['missing']}")
        if diff["extra"]:
            logger.warning(f"{source_name} extra columns ignored: {diff['extra']}")

    def _add_missing_columns(
        self, df: DataFrame, missing: set, schema: StructType
    ) -> DataFrame:
        for field in schema.fields:
            if field.name in missing:
                df = df.withColumn(field.name, F.lit(None).cast(field.dataType))
        return df

    def _select_and_cast_columns(self, df: DataFrame, schema: StructType) -> DataFrame:
        expressions = [
            self._build_cast_expression(field, df.columns) for field in schema.fields
        ]
        return df.select(*expressions)

    def _build_cast_expression(self, field: StructField, available_columns: list):
        if field.name not in available_columns:
            return F.lit(None).cast(field.dataType).alias(field.name)

        col = F.col(field.name)

        if (
            isinstance(field.dataType, ArrayType)
            and field.dataType.elementType == StringType()
        ):
            return self._parse_string_array(col).alias(field.name)

        return col.cast(field.dataType).alias(field.name)

    def _parse_string_array(self, col: Column) -> Column:
        parsed_json = F.from_json(col, ArrayType(StringType()))

        cleaned = F.regexp_replace(F.regexp_replace(col, r"^\[|\]$", ""), r'"', "")

        split_fallback = F.when(F.trim(cleaned) == "", F.expr("array()")).otherwise(
            F.split(cleaned, r"\s*[,;]\s*")
        )

        return (
            F.when(parsed_json.isNotNull(), parsed_json)
            .otherwise(split_fallback)
            .cast(ArrayType(StringType()))
        )

    def _add_metadata(self, df: DataFrame, source_name: str) -> DataFrame:
        return df.withColumn("_source_system", F.lit(source_name)).withColumn(
            "_load_timestamp", F.current_timestamp()
        )

    def load_from_iceberg(self) -> DataFrame:
        table_name = self.data_config["source_table"]
        min_records = int(self.data_config["quality_checks"]["min_records"])

        logger.info(f"Loading data from Iceberg table: {table_name}")

        try:
            df = self.spark.table(table_name)

            self._validate_min_records(df, min_records)

            logger.info("Loaded records from Iceberg table")
            return df

        except Exception as e:
            logger.error(f"Error loading data from Iceberg: {e}")
            raise

    def _validate_min_records(self, df: DataFrame, min_records: int) -> None:
        sampled_count = df.limit(min_records + 1).count()

        if sampled_count < min_records:
            raise ValueError(
                f"Insufficient data: {sampled_count} records (minimum: {min_records})"
            )
