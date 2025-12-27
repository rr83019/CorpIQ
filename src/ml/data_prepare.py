from typing import Dict, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from loguru import logger


class DataPreparer:
    """
    Load and prepare data from Iceberg table for ML pipeline
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.data_config = config["data"]

    def prepare_training_data(
        self, df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        logger.info("Preparing training data")

        df, stats = self._filter_valid_records(df)

        train_df, val_df, test_df = self._split_data(df)

        logger.info("Data split complete")

        return train_df, val_df, test_df

    def _filter_valid_records(self, df: DataFrame) -> Tuple[DataFrame, Dict]:
        logger.info("Filtering valid records")

        numerical_features = self.data_config["features"]["numerical"]
        max_null_pct = float(self.data_config["quality_checks"]["max_null_percentage"])

        initial_count = df.count()

        df = df.filter(F.col("revenue").isNotNull() & F.col("profit").isNotNull())

        agg_expression = [F.count(F.lit(1)).alias("filtered_count")]

        for feature in numerical_features:
            if feature in df.columns:
                agg_expression.append(
                    F.sum(F.when(F.col(feature).isNull(), 1).otherwise(0)).alias(
                        f"{feature}_nulls"
                    )
                )

        stats_row = df.agg(*agg_expression).first()
        filtered_count = stats_row["filtered_count"]

        for feature in numerical_features:
            key = f"{feature}_nulls"
            if key in stats_row and initial_count > 0:
                null_pct = stats_row[key] / initial_count
                if null_pct > max_null_pct:
                    logger.warning(
                        f"Feature {feature} has {null_pct:.2%} nulls "
                        f"(threshold: {max_null_pct:.2%})"
                    )

        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed:,} records with insufficient data")

        return df, {"initial_count": initial_count, "filtered_count": filtered_count}

    def _split_data(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        config = self.data_config["train_test_split"]

        test_size = float(config["test_size"])
        val_size = float(config["validation_size"])
        seed = int(config["random_seed"])
        stratify = bool(config["stratify"])

        train_size = 1.0 - test_size - val_size

        logger.info(
            f"Splitting data: train={train_size:.2f}, "
            f"val={val_size:.2f}, test={test_size:.2f}"
        )

        if stratify:
            df = self._ensure_target_column(df)
            train_df, val_df, test_df = self._stratified_split(
                df, train_size, val_size, test_size, seed
            )
        else:
            train_df, val_df, test_df = self._random_split(
                df, train_size, val_size, test_size, seed
            )

        return (
            train_df,
            val_df,
            test_df,
        )

    def _ensure_target_column(self, df: DataFrame) -> DataFrame:
        target_col = self.data_config["target_column"]
        threshold = int(self.data_config["profit_threshold"])

        if target_col not in df.columns:
            df = df.withColumn(
                target_col, F.when(F.col("profit") > threshold, 1.0).otherwise(0.0)
            )

        return df

    def _stratified_split(
        self,
        df: DataFrame,
        train_size: float,
        val_size: float,
        test_size: float,
        seed: int,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        target_col = self.data_config["target_column"]

        df0 = df.filter(F.col(target_col) == 0.0)
        df1 = df.filter(F.col(target_col) == 1.0)

        train_0, temp_0 = df0.randomSplit([train_size, test_size + val_size], seed)
        val_0, test_0 = temp_0.randomSplit(
            [val_size / (test_size + val_size), test_size / (test_size + val_size)],
            seed,
        )

        train_1, temp_1 = df1.randomSplit([train_size, test_size + val_size], seed)
        val_1, test_1 = temp_1.randomSplit(
            [val_size / (test_size + val_size), test_size / (test_size + val_size)],
            seed,
        )

        return (
            train_0.union(train_1),
            val_0.union(val_1),
            test_0.union(test_1),
        )

    def _random_split(
        self,
        df: DataFrame,
        train_size: float,
        val_size: float,
        test_size: float,
        seed: int,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        train_df, temp_df = df.randomSplit([train_size, test_size + val_size], seed)
        val_df, test_df = temp_df.randomSplit(
            [val_size / (test_size + val_size), test_size / (test_size + val_size)],
            seed,
        )

        return train_df, val_df, test_df

    def check_class_balance(self, df: DataFrame) -> Dict:
        target_col = self.data_config["target_column"]

        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return {}

        class_counts = df.groupBy(target_col).count().collect()
        total = sum(row["count"] for row in class_counts)

        balance_info = {}

        for row in class_counts:
            label = row[target_col]
            count = row["count"]
            ratio = count / total if total > 0 else 0

            balance_info[f"class_{label}"] = {
                "count": count,
                "ratio": ratio,
            }

            logger.info(f"Class {label}: {count:,} samples ({ratio:.2%})")

        quality = self.data_config["quality_checks"]
        ratios = [v["ratio"] for v in balance_info.values()]

        if ratios:
            min_ratio, max_ratio = min(ratios), max(ratios)

            if min_ratio < float(quality["min_positive_class_ratio"]):
                logger.warning(
                    f"Severe class imbalance: minority class {min_ratio:.2%} "
                    f"(min: {float(quality['min_positive_class_ratio']):.2%})"
                )

            if max_ratio > float(quality["max_positive_class_ratio"]):
                logger.warning(
                    f"Severe class imbalance: majority class {max_ratio:.2%} "
                    f"(max: {float(quality['max_positive_class_ratio']):.2%})"
                )

        return balance_info
