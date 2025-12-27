from pyspark.sql import functions as F
from pyspark.sql.functions import lit, current_timestamp

from src.ingestion.entity_resolution import EntityMatcher


class TestEntityResolution:
    """Test entity resolution functionality"""

    def test_standardize_dataframe(self, spark, test_config, sample_source1_data):
        """Test data standardization"""
        matcher = EntityMatcher(test_config, spark)

        standardized = matcher._prepare_source(sample_source1_data, source="S1")

        assert "corporate_name_clean" in standardized.columns
        assert "address_clean" in standardized.columns
        assert "country_clean" in standardized.columns

        clean_names = standardized.select("corporate_name_clean").collect()
        assert all(
            name["corporate_name_clean"].islower()
            for name in clean_names
            if name["corporate_name_clean"]
        )

    def test_blocking_keys(self, spark, test_config, sample_source1_data):
        matcher = EntityMatcher(test_config, spark)

        standardized = matcher._prepare_source(sample_source1_data, source="S1")
        with_blocking = matcher._add_blocking_keys(standardized)
        row = with_blocking.select("blocking_key").first()[0]

        assert "blocking_key" in with_blocking.columns
        assert "||" in row

        null_count = with_blocking.filter(F.col("blocking_key").isNull()).count()
        assert null_count == 0

    def test_clean_text_udf(self, spark, test_config):
        matcher = EntityMatcher(test_config, spark)

        df = spark.createDataFrame([("Acme Corp. Ltd",), ("  Foo-Bar LLC ",)], ["raw"])

        cleaned = df.withColumn("clean", matcher.clean_text_udf(F.col("raw"))).collect()

        assert cleaned[0]["clean"] == "acme"
        assert cleaned[1]["clean"] == "foobar"

    def test_address_cleaning(self, spark, test_config):
        matcher = EntityMatcher(test_config, spark)

        df = spark.createDataFrame([("12/B, MG Road!",)], ["address"])

        out = df.withColumn("clean", matcher.clean_text_udf("address")).collect()[0][
            "clean"
        ]

        assert out == "12b mg road"

    def test_entity_resolution(
        self, spark, test_config, sample_source1_data, sample_source2_data
    ):
        matcher = EntityMatcher(test_config, spark)

        sample_source1_data = sample_source1_data.withColumn("source", lit("S1"))
        sample_source2_data = sample_source2_data.withColumn("source", lit("S2"))

        sample_source1_data = sample_source1_data.withColumn(
            "_source_system", lit("System1")
        )
        sample_source2_data = sample_source2_data.withColumn(
            "_source_system", lit("System2")
        )

        sample_source1_data = sample_source1_data.withColumn(
            "_load_timestamp", current_timestamp()
        )
        sample_source2_data = sample_source2_data.withColumn(
            "_load_timestamp", current_timestamp()
        )

        harmonized = matcher.resolve_entities(sample_source1_data, sample_source2_data)

        assert harmonized.count() > 0

        assert "corporate_id" in harmonized.columns

        unique_ids = harmonized.select("corporate_id").distinct().count()
        total_source_records = sample_source1_data.count() + sample_source2_data.count()

        assert unique_ids <= total_source_records
