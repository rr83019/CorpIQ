from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json
import tempfile

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class MLflowManager:
    """
    Manages MLflow experiment tracking and model registry
    """

    def __init__(self, config: Dict):
        self.config = config
        self.mlflow_config = config["mlflow"]
        self.client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.run_id: Optional[str] = None

        self._initialize_mlflow()

    def _initialize_mlflow(self):
        tracking_uri = self.mlflow_config["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"MLflow tracking URI: {tracking_uri}")

        experiment_name = self.mlflow_config["experiment_name"]

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name, tags=self.mlflow_config.get("tags", {})
                )
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")

            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error initializing MLflow: {str(e)}")
            raise

        self.client = MlflowClient()

    def start_run(self, run_name: Optional[str] = None) -> str:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id

        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")

        tags = self.mlflow_config.get("tags", {})
        for key, value in tags.items():
            mlflow.set_tag(key, value)

        mlflow.set_tag("run_timestamp", datetime.now().isoformat())
        mlflow.set_tag("run_name", run_name)

        return self.run_id

    def log_config(self, config: Dict):
        logger.info("Logging configuration to MLflow")

        flat_params = self._flatten_dict(config, max_depth=2)

        for key, value in flat_params.items():
            try:
                value_str = str(value)[:250]
                mlflow.log_param(key, value_str)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {str(e)}")

        config_json = json.dumps(config, indent=2, default=str)
        mlflow.log_text(config_json, "config.json")

    def log_dataset_info(self, df: DataFrame, dataset_name: str = "training"):
        logger.info(f"Logging {dataset_name} dataset info")

        try:
            num_records = df.count()
            num_features = len(df.columns)

            mlflow.log_metric(f"{dataset_name}_num_records", num_records)
            mlflow.log_metric(f"{dataset_name}_num_features", num_features)

            schema_str = df._jdf.schema().treeString()
            mlflow.log_text(schema_str, f"{dataset_name}_schema.txt")

            target_col = self.config["data"]["target_column"]
            if target_col in df.columns:
                class_dist = df.groupBy(target_col).count().toPandas()

                for _, row in class_dist.iterrows():
                    label = row[target_col]
                    count = row["count"]
                    mlflow.log_metric(f"{dataset_name}_class_{label}_count", count)

                self._plot_class_distribution(class_dist, dataset_name, target_col)

        except Exception as e:
            logger.error(f"Error logging dataset info: {str(e)}")

    def log_feature_importance(
        self, feature_importance: Dict[str, float], top_n: int = 20
    ):
        logger.info("Logging feature importance")

        try:
            for feature, importance in list(feature_importance.items())[:top_n]:
                mlflow.log_metric(f"feature_importance_{feature}", importance)

            importance_json = json.dumps(feature_importance, indent=2)
            mlflow.log_text(importance_json, "feature_importance.json")

            self._plot_feature_importance(feature_importance, top_n)

        except Exception as e:
            logger.error(f"Error logging feature importance: {str(e)}")

    def log_confusion_matrix(self, confusion_matrix: Dict, stage: str = "test"):
        logger.info(f"Logging confusion matrix for {stage}")

        try:
            for key, value in confusion_matrix.items():
                mlflow.log_metric(f"{stage}_confusion_{key}", value)

            self._plot_confusion_matrix(confusion_matrix, stage)

        except Exception as e:
            logger.error(f"Error logging confusion matrix: {str(e)}")

    def log_model(
        self, model: PipelineModel, train_df: DataFrame, artifact_path: str = "model"
    ):
        logger.info("Logging model to MLflow")

        try:
            predictions = model.transform(train_df.limit(100))
            signature = infer_signature(
                train_df.limit(100).toPandas(),
                predictions.select(["prediction"]).limit(100).toPandas(),
            )

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name=None,
            )

            logger.info(f"Model logged to MLflow at {artifact_path}")

        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def register_model(self, metrics: Dict, stage: str = "staging") -> str:
        logger.info("Registering model to MLflow Model Registry")

        try:
            registry_config = self.mlflow_config["registry"]
            model_name = registry_config["model_name"]

            model_uri = f"runs:/{self.run_id}/model"

            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

            version_number = model_version.version

            logger.info(f"Model registered: {model_name} version {version_number}")

            description = self._generate_model_description(metrics)
            self.client.update_model_version(
                name=model_name, version=version_number, description=description
            )

            if registry_config.get("staging_enabled", True):
                self.client.transition_model_version_stage(
                    name=model_name, version=version_number, stage="Staging"
                )
                logger.info(f"Model version {version_number} moved to Staging")

            if self._should_promote_to_production(metrics):
                logger.info("Model meets production criteria")

                if not registry_config["production_promotion"]["approval_required"]:
                    self.client.transition_model_version_stage(
                        name=model_name, version=version_number, stage="Production"
                    )
                    logger.info(
                        f"Model version {version_number} promoted to Production"
                    )
                else:
                    logger.info("Manual approval required for production promotion")

            return str(version_number)

        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise

    def _should_promote_to_production(self, metrics: Dict) -> bool:
        registry_config = self.mlflow_config["registry"]
        promotion_config = registry_config["production_promotion"]

        if not promotion_config["auto_promote"]:
            return False

        min_accuracy = promotion_config.get("min_accuracy", 0.8)

        return metrics.get("accuracy", 0) >= min_accuracy

    def _generate_model_description(self, metrics: Dict) -> str:
        desc_parts = [
            f"Model trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Algorithm: {self.config['model']['algorithm']}",
            "\nMetrics:",
            f"  Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"  Precision: {metrics.get('precision', 0):.4f}",
            f"  Recall: {metrics.get('recall', 0):.4f}",
            f"  F1 Score: {metrics.get('f1', 0):.4f}",
            f"  AUC-ROC: {metrics.get('areaUnderROC', 0):.4f}",
        ]

        return "\n".join(desc_parts)

    def end_run(self):
        if self.run_id:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = "",
        sep: str = ".",
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> Dict:
        items = []

        if current_depth >= max_depth:
            return {parent_key: str(d)}

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(
                        v, new_key, sep, max_depth, current_depth + 1
                    ).items()
                )
            else:
                items.append((new_key, v))

        return dict(items)

    def _plot_class_distribution(
        self, class_dist: pd.DataFrame, dataset_name: str, target_col: str
    ):
        try:
            plt.figure(figsize=(8, 6))
            plt.bar(class_dist[target_col].astype(str), class_dist["count"])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(f"{dataset_name.capitalize()} Class Distribution")

            for i, v in enumerate(class_dist["count"]):
                plt.text(i, v, str(v), ha="center", va="bottom")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                mlflow.log_artifact(tmp.name, f"{dataset_name}_class_distribution.png")
                Path(tmp.name).unlink()

            plt.close()

        except Exception as e:
            logger.warning(f"Could not create class distribution plot: {str(e)}")

    def _plot_feature_importance(
        self, feature_importance: Dict[str, float], top_n: int
    ):
        try:
            top_features = dict(list(feature_importance.items())[:top_n])

            plt.figure(figsize=(10, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())

            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances")
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                mlflow.log_artifact(tmp.name, "feature_importance.png")
                Path(tmp.name).unlink()

            plt.close()

        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {str(e)}")

    def _plot_confusion_matrix(self, confusion_matrix: Dict, stage: str):
        try:
            matrix = [
                [confusion_matrix["TN"], confusion_matrix["FP"]],
                [confusion_matrix["FN"], confusion_matrix["TP"]],
            ]

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
            )
            plt.title(f"{stage.capitalize()} Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches="tight")
                mlflow.log_artifact(tmp.name, f"{stage}_confusion_matrix.png")
                Path(tmp.name).unlink()

            plt.close()

        except Exception as e:
            logger.warning(f"Could not create confusion matrix plot: {str(e)}")

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/Production"

            model = mlflow.spark.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        try:
            runs_data = []

            for run_id in run_ids:
                run = self.client.get_run(run_id)

                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    **run.data.metrics,
                    **run.data.params,
                }

                runs_data.append(run_data)

            comparison_df = pd.DataFrame(runs_data)

            logger.info(f"Compared {len(run_ids)} model runs")

            return comparison_df

        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
