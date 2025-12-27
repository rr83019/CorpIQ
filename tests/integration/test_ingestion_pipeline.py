from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, current_timestamp

from src.ingestion.entity_resolution import EntityMatcher
from src.ingestion.data_quality import DataQualityValidator
from src.storage.iceberg_manager import IcebergTableManager


def mock_adverse_media_simple(spark, input_df):
    mock_data = spark.createDataFrame(
        [
            (
                "Acme Corporation",
                False,
                "No issues found",
                0.1,
                ["risk"],
                "2025-12-27T00:00:00Z",
            ),
            (
                "Global Tech Inc",
                True,
                "Regulatory warning",
                0.75,
                ["sanctions"],
                "2025-12-27T00:00:00Z",
            ),
        ],
        [
            "corporate_name",
            "has_adverse_media",
            "summary",
            "risk_score",
            "keywords_found",
            "checked_at",
        ],
    )

    result = input_df.join(
        mock_data, input_df.corporate_name_S1 == mock_data.corporate_name, "left"
    ).drop("corporate_name")

    return result.withColumnRenamed(
        "checked_at", "adverse_media_last_checked"
    ).withColumnRenamed("summary", "adverse_media_summary")


def add_metadata(df: DataFrame, source_label: str) -> DataFrame:
    return (
        df.withColumn("source", lit(source_label))
        .withColumn("_source_system", lit(f"System{source_label.replace('S', '')}"))
        .withColumn("_load_timestamp", current_timestamp())
        .withColumn("created_at", current_timestamp())
        .withColumn("updated_at", current_timestamp())
    )


def test_end_to_end_pipeline(
    spark, test_config, sample_source1_data, sample_source2_data
):
    matcher = EntityMatcher(test_config, spark)
    validator = DataQualityValidator(test_config, spark)

    sample_source1_data = add_metadata(sample_source1_data, "source1")
    sample_source2_data = add_metadata(sample_source2_data, "source2")

    assert validator.validate(sample_source1_data, "source1")
    assert validator.validate(sample_source2_data, "source2")

    harmonized = matcher.resolve_entities(sample_source1_data, sample_source2_data)

    assert harmonized.count() > 0


def test_resolve_entities_output_matches_iceberg_schema(
    spark, test_config, sample_source1_data, sample_source2_data
):
    matcher = EntityMatcher(test_config, spark)

    sample_source1_data = add_metadata(sample_source1_data, "source1")
    sample_source2_data = add_metadata(sample_source2_data, "source2")

    harmonized_df = matcher.resolve_entities(sample_source1_data, sample_source2_data)

    harmonized_df = mock_adverse_media_simple(spark, harmonized_df)

    harmonized_df = harmonized_df.withColumn(
        "created_at", current_timestamp()
    ).withColumn("updated_at", current_timestamp())

    expected_columns = set(IcebergTableManager.HARMONIZED_COLUMNS)
    actual_columns = set(harmonized_df.columns)

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    assert not missing, f"Missing columns after resolve_entities: {missing}"
    assert not extra, f"Unexpected extra columns after resolve_entities: {extra}"
