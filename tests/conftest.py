import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("entity-matcher-tests")
        .config("spark.ui.enabled", "false")
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1,org.apache.iceberg:iceberg-aws-bundle:1.7.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.523,graphframes:graphframes:0.8.3-spark3.5-s_2.12",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(
            "spark.sql.catalog.test_catalog", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config("spark.sql.catalog.test_catalog.type", "hadoop")
        .config("spark.sql.catalog.test_catalog.warehouse", "/tmp/test_warehouse")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-test-checkpoint")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("/tmp/spark-test-checkpoint")

    yield spark
    spark.stop()


@pytest.fixture
def test_config(tmp_path):
    config = {
        "entity_resolution": {
            "matching_threshold": 0.85,
            "blocking_fields": ["country"],
            "fuzzy_match_fields": [
                {"name": "corporate_name", "weight": 0.6, "method": "jaro_winkler"},
                {"name": "address", "weight": 0.3, "method": "token_sort"},
                {"name": "postal_code", "weight": 0.1, "method": "exact"},
            ],
            "deduplication_strategy": "weighted_score",
        },
        "data_sources": {
            "source1": {
                "path": str(tmp_path / "test_data/source1"),
                "format": "csv",
                "schema_validation": False,
            },
            "source2": {
                "path": str(tmp_path / "test_data/source2"),
                "format": "csv",
                "schema_validation": False,
            },
        },
        "iceberg": {
            "catalog_name": "test_catalog",
            "database": "test_db",
            "table_name": "test_corporate_registry",
            "partition_spec": [],
            "sort_order": [],
        },
        "data_quality": {
            "validation_rules": {
                "country": [{"type": "not_null"}, {"type": "min_length", "value": 2}],
                "revenue": [{"type": "not_null"}, {"type": "positive"}],
            },
            "error_threshold": 0.05,
        },
        "performance": {"cache_intermediate_results": True},
        "data": {
            "source_table": "test_corporate_data",
            "target_column": "profit_above_threshold",
            "profit_threshold": 1000000,
            "features": {
                "numerical": ["revenue", "profit", "num_top_suppliers"],
                "categorical": ["country", "industry_sector"],
                "text": [],
            },
            "train_test_split": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_seed": 42,
                "stratify": True,
            },
            "quality_checks": {
                "min_records": 10,
                "max_null_percentage": 0.5,
                "min_positive_class_ratio": 0.1,
                "max_positive_class_ratio": 0.9,
            },
        },
    }
    return config


@pytest.fixture
def sample_source1_data(spark):
    data = [
        {
            "corporate_name_S1": "Acme Corporation",
            "address": "123 Main St, New York, NY",
            "activity_places": ["New York", "Los Angeles"],
            "top_suppliers": ["Supplier A", "Supplier B"],
            "country": "USA",
            "postal_code": "10001",
            "industry_sector": "Manufacturing",
        },
        {
            "corporate_name_S1": "ACME Corp",
            "address": "123 Main Street, New York, NY",
            "activity_places": ["New York"],
            "top_suppliers": ["Supplier A"],
            "country": "USA",
            "postal_code": "10001",
            "industry_sector": "Manufacturing",
        },
        {
            "corporate_name_S1": "Global Tech Inc",
            "address": "456 Tech Blvd, San Francisco, CA",
            "activity_places": ["San Francisco"],
            "top_suppliers": ["Tech Supplier"],
            "country": "USA",
            "postal_code": "94105",
            "industry_sector": "Technology",
        },
    ]

    return spark.createDataFrame(data)


@pytest.fixture
def sample_source2_data(spark):
    data = [
        {
            "corporate_name_S2": "Acme Corporation",
            "main_customers": ["Customer X", "Customer Y"],
            "revenue": 150000000.0,
            "profit": 25000000.0,
            "country": "USA",
            "postal_code": "10001",
            "industry_sector": "Manufacturing",
            "fiscal_year": 2024,
        },
        {
            "corporate_name_S2": "Global Tech Incorporated",
            "main_customers": ["Client A"],
            "revenue": 500000000.0,
            "profit": 100000000.0,
            "country": "USA",
            "postal_code": "94105",
            "industry_sector": "Technology",
            "fiscal_year": 2024,
        },
    ]

    return spark.createDataFrame(data)


@pytest.fixture
def ml_config():
    config = {
        "ml_pipeline": {"name": "test-pipeline", "version": "1.0.0"},
        "data": {
            "source_table": "test_table",
            "target_column": "profit_above_threshold",
            "profit_threshold": 1000000,
            "features": {
                "numerical": ["revenue", "profit", "num_top_suppliers"],
                "categorical": ["country", "industry_sector"],
                "text": [],
            },
            "train_test_split": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_seed": 42,
                "stratify": True,
            },
            "quality_checks": {
                "min_records": 10,
                "max_null_percentage": 0.5,
                "min_positive_class_ratio": 0.1,
                "max_positive_class_ratio": 0.9,
            },
        },
        "feature_engineering": {
            "derived_features": [
                {"name": "profit_margin", "expression": "profit / NULLIF(revenue, 0)"}
            ],
            "scaling": {"method": "standard", "features": ["revenue", "profit"]},
            "categorical_encoding": {"method": "one_hot", "max_categories": 50},
            "feature_selection": {"enabled": False},
        },
        "model": {
            "algorithm": "logistic_regression",
            "hyperparameters": {
                "logistic_regression": {
                    "maxIter": 10,
                    "regParam": 0.01,
                    "elasticNetParam": 0.0,
                    "family": "binomial",
                    "threshold": 0.5
                }
            },
        },
        "hyperparameter_tuning": {
            "enabled": False,
            "method": "grid_search",
            "param_grid": {
                "logistic_regression": {
                    "regParam": [0.01, 0.1, 0.5],
                    "elasticNetParam": [0.0, 0.5, 1.0],
                    "maxIter": [10, 50, 100],
                    "family": "binomial",
                    "threshold": 0.5
                }
            },
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "thresholds": {"min_accuracy": 0.6, "min_precision": 0.6},
        },
        "reproducibility": {"seed": 42},
    }
    return config


@pytest.fixture
def sample_data(spark):
    schema = StructType(
        [
            StructField("corporate_id", StringType(), False),
            StructField("corporate_name", StringType(), False),
            StructField("revenue", DoubleType(), True),
            StructField("profit", DoubleType(), True),
            StructField("country", StringType(), True),
            StructField("industry_sector", StringType(), True),
            StructField("num_top_suppliers", IntegerType(), True),
            StructField("num_main_customers", IntegerType(), True),
        ]
    )

    data = []

    for i in range(100):
        if i < 50:
            data.append(
                (
                    f"CORP-{i:04d}",
                    f"Company {i}",
                    float(10000000 + i * 100000),
                    float(2000000 + i * 50000),
                    "USA" if i % 2 == 0 else "UK",
                    "Technology" if i % 3 == 0 else "Manufacturing",
                    10 + i % 5,
                    5 + i % 3,
                )
            )

        else:
            data.append(
                (
                    f"CORP-{i:04d}",
                    f"Company {i}",
                    float(1000000 + i * 10000),
                    float(100000 + i * 5000),
                    "USA" if i % 2 == 0 else "UK",
                    "Technology" if i % 3 == 0 else "Manufacturing",
                    3 + i % 3,
                    2 + i % 2,
                )
            )

    df = spark.createDataFrame(data, schema)

    return df
