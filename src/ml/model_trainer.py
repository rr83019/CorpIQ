from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from loguru import logger
import mlflow
import json


class ModelTrainer:
    """
    ML Model Training and Evaluation using Spark ML
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.model_config = config["model"]
        self.eval_config = config["evaluation"]
        self.data_config = config["data"]

        self.model: Optional[PipelineModel] = None
        self.metrics: Dict = {}

    def train(
        self,
        feature_stages: List,
        train_df: DataFrame,
        val_df: Optional[DataFrame] = None,
    ) -> PipelineModel:
        logger.info("Starting model training...")
        algorithm = self.model_config["algorithm"]

        estimator = self._build_estimator(algorithm)

        if self.config["hyperparameter_tuning"]["enabled"] == 'true':
            estimator = self._tune_hyperparameters(estimator, train_df, val_df)

        pipeline = Pipeline(stages=feature_stages + [estimator])

        start = datetime.now()
        self.model = pipeline.fit(train_df)
        duration = (datetime.now() - start).total_seconds()

        mlflow.log_metric("training_duration_seconds", duration)
        mlflow.log_param("algorithm", algorithm)

        # Count only once
        train_count = train_df.count()
        mlflow.log_param("num_training_samples", train_count)

        logger.info(f"Training complete in {duration:.2f}s ({train_count:,} samples)")

        return self.model

    def _build_estimator(self, algorithm: str):
        target = self.data_config["target_column"]
        params = self.model_config["hyperparameters"].get(algorithm, {})
        seed = int(self.config["reproducibility"]["seed"])

        if algorithm == "logistic_regression":
            return LogisticRegression(
                labelCol=target,
                featuresCol="features",
                maxIter=int(params["maxIter"]),
                regParam=float(params["regParam"]),
                elasticNetParam=float(params["elasticNetParam"]),
                family=params["family"],
                threshold=float(params["threshold"]),
            )

        if algorithm == "random_forest":
            return RandomForestClassifier(
                labelCol=target,
                featuresCol="features",
                seed=seed,
                numTrees=int(params["numTrees"]),
                maxDepth=int(params["maxDepth"]),
                minInstancesPerNode=int(params["minInstancesPerNode"]),
                subsamplingRate=float(params["subsamplingRate"]),
            )

        if algorithm == "gradient_boosted_trees":
            return GBTClassifier(
                labelCol=target,
                featuresCol="features",
                seed=seed,
                maxIter=int(params["maxIter"]),
                maxDepth=int(params["maxDepth"]),
                stepSize=float(params["stepSize"]),
            )

        raise ValueError(f"Unknown algorithm: {algorithm}")

    def _tune_hyperparameters(
        self, estimator, train_df: DataFrame, val_df: Optional[DataFrame]
    ):
        tuning_cfg = self.config["hyperparameter_tuning"]
        algorithm = self.model_config["algorithm"]
        target = self.data_config["target_column"]

        grid_cfg = tuning_cfg["param_grid"].get(algorithm, {})
        grid = ParamGridBuilder()

        for name, values in grid_cfg.items():
            grid = grid.addGrid(getattr(estimator, name), values)

        param_grid = grid.build()
        evaluator = BinaryClassificationEvaluator(
            labelCol=target, metricName=tuning_cfg["optimization"]["metric"]
        )

        if val_df is not None:
            tuner = TrainValidationSplit(
                estimator=estimator,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                trainRatio=0.8,
                parallelism=int(tuning_cfg["cross_validation"].get("parallelism", 1)),
                seed=int(self.config["reproducibility"]["seed"]),
            )
        else:
            tuner = CrossValidator(
                estimator=estimator,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=int(tuning_cfg["cross_validation"].get("num_folds", 5)),
                parallelism=int(tuning_cfg["cross_validation"].get("parallelism", 1)),
                seed=int(self.config["reproducibility"]["seed"]),
            )

        model = tuner.fit(train_df)
        best_model = model.bestModel

        for p, v in best_model.extractParamMap().items():
            mlflow.log_param(f"best_{p.name}", v)

        return best_model

    def evaluate(self, test_df: DataFrame, stage: str = "test") -> Dict:
        if self.model is None:
            raise ValueError("Model not trained")

        target = self.data_config["target_column"]

        preds = self.model.transform(test_df).cache()

        metrics = self._compute_metrics(preds, target)

        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(f"{stage}_{k}", v)

        self.metrics[stage] = metrics
        preds.unpersist()

        return metrics

    def _compute_metrics(self, preds: DataFrame, target: str) -> Dict:
        binary_eval = BinaryClassificationEvaluator(labelCol=target)
        multi_eval = MulticlassClassificationEvaluator(labelCol=target)

        metrics = {
            "areaUnderROC": binary_eval.evaluate(
                preds, {binary_eval.metricName: "areaUnderROC"}
            ),
            "areaUnderPR": binary_eval.evaluate(
                preds, {binary_eval.metricName: "areaUnderPR"}
            ),
            "accuracy": multi_eval.evaluate(preds, {multi_eval.metricName: "accuracy"}),
            "precision": multi_eval.evaluate(
                preds, {multi_eval.metricName: "weightedPrecision"}
            ),
            "recall": multi_eval.evaluate(
                preds, {multi_eval.metricName: "weightedRecall"}
            ),
            "f1": multi_eval.evaluate(preds, {multi_eval.metricName: "f1"}),
            "confusion_matrix": self._confusion_matrix_single_pass(preds, target),
        }

        return metrics

    def _confusion_matrix_single_pass(self, preds: DataFrame, target: str) -> Dict:
        agg = preds.select(
            F.sum(
                ((F.col(target) == 1) & (F.col("prediction") == 1)).cast("int")
            ).alias("TP"),
            F.sum(
                ((F.col(target) == 0) & (F.col("prediction") == 0)).cast("int")
            ).alias("TN"),
            F.sum(
                ((F.col(target) == 0) & (F.col("prediction") == 1)).cast("int")
            ).alias("FP"),
            F.sum(
                ((F.col(target) == 1) & (F.col("prediction") == 0)).cast("int")
            ).alias("FN"),
        ).first()

        return agg.asDict()

    def validate_model(self, metrics: Dict) -> Tuple[bool, List[str]]:
        thresholds = self.eval_config["thresholds"]
        failures = []

        for k, t in thresholds.items():
            key = k.replace("min_", "")
            if key in metrics and metrics[key] < float(t):
                failures.append(f"{key}: {metrics[key]:.4f} < {t:.4f}")

        return len(failures) == 0, failures

    def predict(self, df: DataFrame) -> DataFrame:
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.transform(df)

    def get_model_summary(self) -> Dict:
        if self.model is None:
            return {}

        trained = self.model.stages[-1]

        return {
            "algorithm": self.model_config["algorithm"],
            "model_type": type(trained).__name__,
            "parameters": {p.name: v for p, v in trained.extractParamMap().items()},
            "metrics": self.metrics,
        }

    def save_model_artifacts(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.write().overwrite().save(path)

        summary = json.dumps(self.get_model_summary(), indent=2, default=str)
        with open(f"{path}/model_summary.json", "w") as f:
            f.write(summary)
