from src.ingestion.data_quality import DataQualityValidator


class TestDataQualityValidator:
    """Unit tests for DataQualityValidator"""

    def test_not_null_rule(self, spark, test_config):
        validator = DataQualityValidator(test_config, spark)

        df = spark.createDataFrame(
            [
                {"corporate_name": "Company A"},
                {"corporate_name": None},
                {"corporate_name": "Company B"},
            ]
        )

        agg_expr, rule_plan = validator._build_rule_expressions(
            {"corporate_name": [{"type": "not_null"}]}
        )

        row = df.agg(*agg_expr).first()
        results = validator._evaluate_results(row, rule_plan)

        assert results[0]["error_count"] == 1

    def test_min_length_rule(self, spark, test_config):
        validator = DataQualityValidator(test_config, spark)

        df = spark.createDataFrame(
            [
                {"corporate_name": "AB"},
                {"corporate_name": "A"},  # invalid
                {"corporate_name": "Company"},
            ]
        )

        agg_expr, rule_plan = validator._build_rule_expressions(
            {"corporate_name": [{"type": "min_length", "value": 2}]}
        )

        row = df.agg(*agg_expr).first()
        results = validator._evaluate_results(row, rule_plan)

        assert results[0]["error_count"] == 1

    def test_positive_rule(self, spark, test_config):
        validator = DataQualityValidator(test_config, spark)

        df = spark.createDataFrame(
            [
                {"revenue": 100.0},
                {"revenue": -50.0},  # invalid
                {"revenue": 0.0},
            ]
        )

        agg_expr, rule_plan = validator._build_rule_expressions(
            {"revenue": [{"type": "positive"}]}
        )

        row = df.agg(*agg_expr).first()
        results = validator._evaluate_results(row, rule_plan)

        assert results[0]["error_count"] == 1

    def test_unique_rule(self, spark, test_config):
        validator = DataQualityValidator(test_config, spark)

        df = spark.createDataFrame(
            [
                {"corporate_name": "A"},
                {"corporate_name": "B"},
                {"corporate_name": "A"},  # duplicate
            ]
        )

        agg_expr, rule_plan = validator._build_rule_expressions(
            {"corporate_name": [{"type": "unique", "approx": False}]}
        )

        row = df.agg(*agg_expr).first()
        results = validator._evaluate_results(row, rule_plan)

        assert results[0]["error_count"] == 1

    def test_range_rule(self, spark, test_config):
        validator = DataQualityValidator(test_config, spark)

        df = spark.createDataFrame(
            [
                {"revenue": 10},
                {"revenue": 2000},  # invalid
                {"revenue": 500},
            ]
        )

        agg_expr, rule_plan = validator._build_rule_expressions(
            {"revenue": [{"type": "range", "min": 0, "max": 1000}]}
        )

        row = df.agg(*agg_expr).first()
        results = validator._evaluate_results(row, rule_plan)

        assert results[0]["error_count"] == 1
