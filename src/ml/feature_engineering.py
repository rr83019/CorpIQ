from typing import Dict, List, Tuple, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    StringIndexer,
    OneHotEncoder,
    ChiSqSelector,
)
from pyspark.ml import PipelineModel
from loguru import logger
import mlflow


class FeatureEngineer:
    """
    Feature engineering for ML pipeline
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.feature_config = config["feature_engineering"]
        self.data_config = config["data"]
        self.pipeline: Optional[PipelineModel] = None
        self.feature_names: List[str] = []

    def prepare_features(
        self, df: DataFrame, is_training: bool = True
    ) -> Tuple[DataFrame, List[str], List]:
        logger.info("Starting feature engineering")

        df = self._create_target_variable(df)
        df = self._create_derived_features(df)
        df = self._handle_missing_values(df)

        cat_stages = self._build_categorical_stages(df)
        scale_stages = self._build_scaling_stages(df)

        df, feature_names, assembler = self._assemble_features(df)

        if bool(self.feature_config["feature_selection"]["enabled"]) and is_training:
            df = self._select_features(df)

        self.feature_names = feature_names
        stages = cat_stages + scale_stages + [assembler]

        logger.info(
            f"Feature engineering complete. Total features: {len(feature_names)}"
        )
        return df, feature_names, stages

    def _create_target_variable(self, df: DataFrame) -> DataFrame:
        target_col = self.data_config["target_column"]
        threshold = int(self.data_config["profit_threshold"])

        if target_col in df.columns:
            return df

        df = df.withColumn(
            target_col, F.when(F.col("profit") > threshold, 1.0).otherwise(0.0)
        )

        return df

    def _create_derived_features(self, df: DataFrame) -> DataFrame:
        for feature in self.feature_config.get("derived_features", []):
            try:
                df = df.withColumn(feature["name"], F.expr(feature["expression"]))
            except Exception as e:
                logger.warning(f"Failed to create feature {feature['name']}: {e}")
        return df

    def _handle_missing_values(self, df: DataFrame) -> DataFrame:
        logger.info("Handling missing values")

        numerical = self.data_config["features"]["numerical"]
        categorical = self.data_config["features"]["categorical"]

        num_cols = [c for c in numerical if c in df.columns]

        medians = {c: df.approxQuantile(c, [0.5], 0.01)[0] for c in num_cols}

        df = df.fillna(medians)

        cat_cols = [c for c in categorical if c in df.columns]
        df = df.fillna({c: "unknown" for c in cat_cols})

        return df

    def _build_categorical_stages(self, df: DataFrame) -> List:
        stages = []
        encoding = self.feature_config["categorical_encoding"]["method"]
        categorical = self.data_config["features"]["categorical"]

        for col in categorical:
            if col not in df.columns:
                continue

            index_col = f"{col}_indexed"
            stages.append(
                StringIndexer(inputCol=col, outputCol=index_col, handleInvalid="keep")
            )

            if encoding == "one_hot":
                stages.append(
                    OneHotEncoder(inputCols=[index_col], outputCols=[f"{col}_encoded"])
                )

        return stages

    def _build_scaling_stages(self, df: DataFrame) -> List:
        config = self.feature_config["scaling"]
        method = config["method"]
        features = [f for f in config.get("features", []) if f in df.columns]

        if not features:
            return []

        assembler = VectorAssembler(
            inputCols=features, outputCol="features_to_scale", handleInvalid="keep"
        )

        if method == "standard":
            scaler = StandardScaler(
                inputCol="features_to_scale",
                outputCol="scaled_features",
                withMean=True,
                withStd=True,
            )
        elif method == "minmax":
            scaler = MinMaxScaler(
                inputCol="features_to_scale", outputCol="scaled_features"
            )
        elif method == "robust":
            scaler = RobustScaler(
                inputCol="features_to_scale", outputCol="scaled_features"
            )
        else:
            logger.warning(f"Unknown scaling method: {method}")
            return []

        return [assembler, scaler]

    def _assemble_features(
        self, df: DataFrame
    ) -> Tuple[DataFrame, List[str], VectorAssembler]:
        feature_cols = []

        numerical = self.data_config["features"]["numerical"]
        scaled = set(self.feature_config["scaling"].get("features", []))

        for f in numerical:
            if f in df.columns and f not in scaled:
                feature_cols.append(f)

        if "scaled_features" in df.columns:
            feature_cols.append("scaled_features")

        for col in self.data_config["features"]["categorical"]:
            if f"{col}_encoded" in df.columns:
                feature_cols.append(f"{col}_encoded")
            elif f"{col}_indexed" in df.columns:
                feature_cols.append(f"{col}_indexed")

        for f in self.feature_config.get("derived_features", []):
            name = f["name"]
            if name in df.columns and name not in scaled:
                feature_cols.append(name)

        for col in ["num_top_suppliers", "num_main_customers", "num_activity_places"]:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)

        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features", handleInvalid="keep"
        )

        return df, feature_cols, assembler

    def _select_features(self, df: DataFrame) -> DataFrame:
        method = self.feature_config["feature_selection"]["method"]
        target_col = self.data_config["target_column"]

        if method == "chi2":
            selector = ChiSqSelector(
                numTopFeatures=20,
                featuresCol="features",
                outputCol="selected_features",
                labelCol=target_col,
            )
            df = selector.fit(df).transform(df)
            df = df.withColumnRenamed("selected_features", "features")

        return df

    def validate_features(self, df: DataFrame) -> bool:
        logger.info("Validating features...")

        target = self.data_config["target_column"]
        if target not in df.columns:
            logger.error(f"Target column {target} missing")
            return False

        null_expr = sum(
            F.sum(F.col(c).isNull().cast("int"))
            for c in self.feature_names
            if c in df.columns
        )

        nulls = df.select(null_expr.alias("nulls")).first()["nulls"]
        if nulls > 0:
            logger.warning(f"Found {nulls} null feature values")

        counts = df.groupBy(target).count().collect()
        total = sum(r["count"] for r in counts)

        for r in counts:
            ratio = r["count"] / total
            logger.info(f"Class {r[target]}: {r['count']} ({ratio:.2%})")

            if ratio < 0.01 or ratio > 0.99:
                logger.warning(f"Severe class imbalance detected: {ratio:.2%}")

        return True

    def log_feature_engineering_artifacts(self, df: DataFrame):
        try:
            stats = {}
            for col in self.feature_names:
                if col in df.columns:
                    row = df.select(
                        F.mean(col), F.stddev(col), F.min(col), F.max(col)
                    ).first()

                    stats[col] = {
                        "mean": row[0],
                        "stddev": row[1],
                        "min": row[2],
                        "max": row[3],
                    }

            mlflow.log_dict(stats, "feature_statistics.json")
            mlflow.log_dict({"features": self.feature_names}, "feature_names.json")

        except Exception as e:
            logger.error(f"Error logging feature artifacts: {e}")
