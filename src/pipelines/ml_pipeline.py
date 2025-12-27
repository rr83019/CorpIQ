import sys
import os
from pathlib import Path
from typing import Dict
from datetime import datetime

from pyspark.sql import DataFrame

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.storage.data_loader import DataLoader
from src.ml.data_prepare import DataPreparer
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_trainer import ModelTrainer
from src.storage.mlflow_manager import MLflowManager
from src.monitoring.ml_monitoring import MLMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.helper import load_config, setup_logging, create_spark_session


class MLPipeline:
    """
    Main Orchestrator for the ML Pipeline
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        setup_logging(self.config)

        self.spark = create_spark_session(self.config)

        self.metrics = MLMetrics(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)

        self.data_loader = DataLoader(self.config, self.spark)
        self.data_preparer = DataPreparer(self.config, self.spark)
        self.feature_engineer = FeatureEngineer(self.config, self.spark)
        self.model_trainer = ModelTrainer(self.config, self.spark)
        self.mlflow_manager = MLflowManager(self.config)

        logger.info("ML Pipeline initialized")

    def run(
        self,
        train_only: bool = False,
        skip_validation: bool = False,
        resume_from_checkpoint: bool = False,
    ) -> Dict:
        start = datetime.now()
        self.metrics.start_timer("total_pipeline")

        try:
            logger.info("Starting ML Pipeline")
            self.mlflow_manager.start_run(f"run_{start:%Y%m%d_%H%M%S}")
            self.mlflow_manager.log_config(self.config)

            df = self._load_data()
            splits = self._prepare_data(df)
            features = self._feature_engineering(splits)
            model = self._train_model(features)

            metrics = {}
            if not train_only:
                metrics = self._evaluate_model(features, skip_validation)

            self._finalize(model, metrics, features)

            return self._build_success_result(metrics, start)

        except Exception as e:
            logger.error("ML Pipeline failed", exc_info=True)
            self.metrics.record_pipeline_completion(success=False)
            self._persist_failure_metrics()
            return {
                "success": False,
                "error": str(e),
                "pipeline_metrics": self.metrics.export_metrics_to_dict(),
            }

        finally:
            self.mlflow_manager.end_run()

    def _load_data(self) -> DataFrame:
        self.metrics.start_timer("data_loading")
        df = self.data_loader.load_from_iceberg()
        self.metrics.stop_timer("data_loading")
        return df

    def _prepare_data(self, df: DataFrame) -> Dict[str, DataFrame]:
        self.metrics.start_timer("data_preparation")

        train_df, val_df, test_df = self.data_preparer.prepare_training_data(df)

        counts = {
            "train": train_df.count(),
            "val": val_df.count(),
            "test": test_df.count(),
        }

        for k, v in counts.items():
            self.metrics.record_metric(f"{k}_records", v)

        balance = self.data_preparer.check_class_balance(train_df)
        self.metrics.record_metric("class_balance", balance)
        self.metrics.record_data_quality_check("class_balance", True)

        self.metrics.stop_timer("data_preparation")

        self.checkpoint_manager.save_checkpoint(
            "data_preparation", {**counts, "class_balance": balance}
        )

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "counts": counts,
        }

    def _feature_engineering(self, splits: Dict) -> Dict:
        self.metrics.start_timer("feature_engineering")

        train_df, feature_names, stages = self.feature_engineer.prepare_features(
            splits["train"], is_training=True
        )

        self.metrics.record_metric("num_features", len(feature_names))

        if not self.feature_engineer.validate_features(train_df):
            raise ValueError("Feature validation failed")

        val_df, _, _ = self.feature_engineer.prepare_features(
            splits["val"], is_training=False
        )
        test_df, _, _ = self.feature_engineer.prepare_features(
            splits["test"], is_training=False
        )

        self.feature_engineer.log_feature_engineering_artifacts(train_df)

        self.metrics.stop_timer("feature_engineering")

        self.checkpoint_manager.save_checkpoint(
            "feature_engineering",
            {"num_features": len(feature_names), "sample_features": feature_names[:20]},
        )

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "features": feature_names,
            "stages": stages,
        }

    def _train_model(self, data: Dict):
        self.metrics.start_timer("model_training")

        model = self.model_trainer.train(data["stages"], data["train"], data["val"])

        duration = self.metrics.stop_timer("model_training")

        self.checkpoint_manager.save_model_checkpoint(
            model,
            os.path.join(
                self.checkpoint_manager.checkpoint_location,
                f"model_{datetime.now():%Y%m%d_%H%M%S}",
            ),
            {
                "training_time": duration,
                "num_features": len(data["features"]),
                "train_records": data["train"].count(),
            },
        )

        return model

    def _evaluate_model(self, data: Dict, skip_validation: bool) -> Dict:
        self.metrics.start_timer("model_evaluation")

        val_metrics = self.model_trainer.evaluate(data["val"], stage="validation")
        test_metrics = self.model_trainer.evaluate(data["test"], stage="test")

        self.metrics.record_model_metrics(val_metrics, "validation")
        self.metrics.record_model_metrics(test_metrics, "test")

        self.metrics.stop_timer("model_evaluation")

        if not skip_validation:
            self.metrics.start_timer("model_validation")
            valid, failed = self.model_trainer.validate_model(test_metrics)
            self.metrics.stop_timer("model_validation")

            if not valid:
                logger.warning(f"Model validation failed: {failed}")

        return {"validation": val_metrics, "test": test_metrics}

    def _finalize(self, model, metrics, features):
        self.metrics.start_timer("mlflow_logging")

        self.mlflow_manager.log_model(model, features["train"])

        try:
            version = self.mlflow_manager.register_model(
                metrics=metrics.get("test", {}),
                stage="staging",
            )
            self.metrics.record_metric("model_version", version)
        except Exception:
            logger.warning("Model registration failed", exc_info=True)

        self.metrics.stop_timer("mlflow_logging")

        self._print_metrics_summary()

        self.metrics.record_pipeline_completion(success=True)

        self.metrics.stop_timer("total_ml_pipeline")

        self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

    def _build_success_result(self, metrics: Dict, start: datetime) -> Dict:
        end = datetime.now()
        return {
            "success": True,
            "run_id": self.mlflow_manager.run_id,
            "metrics": metrics,
            "pipeline_metrics": self.metrics.export_metrics_to_dict(),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
        }

    def _persist_failure_metrics(self):
        try:
            self.metrics.save_metrics_to_file(
                os.path.join(
                    self.checkpoint_manager.checkpoint_location,
                    f"ml_metrics_failed_{datetime.now():%Y%m%d_%H%M%S}.json",
                )
            )
        except Exception:
            pass

    def _print_metrics_summary(self):
        logger.info("ML METRICS SUMMARY")

        metrics = self.metrics.get_all_metrics()

        logger.info("\nClassification Metrics:")
        logger.info(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision:    {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:       {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 Score:     {metrics.get('f1', 0):.4f}")

        logger.info("\nROC Metrics:")
        logger.info(f"  AUC-ROC:      {metrics.get('areaUnderROC', 0):.4f}")
        logger.info(f"  AUC-PR:       {metrics.get('areaUnderPR', 0):.4f}")

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            logger.info("\nConfusion Matrix:")
            logger.info("                Predicted")
            logger.info("                0        1")
            logger.info(f"  Actual 0    {cm['TN']:5d}    {cm['FP']:5d}")
            logger.info(f"  Actual 1    {cm['FN']:5d}    {cm['TP']:5d}")

    def predict(self, df):
        if self.model_trainer.model is None:
            raise ValueError("Model not trained. Run pipeline first.")

        self.metrics.start_timer("prediction")

        df, _, _ = self.feature_engineer.prepare_features(df, is_training=False)

        predictions = self.model_trainer.predict(df)

        prediction_time = self.metrics.stop_timer("prediction")
        logger.info(f"Predictions completed in {prediction_time:.2f} seconds")

        return predictions

    def load_production_model(self):
        model_name = self.config["mlflow"]["registry"]["model_name"]

        model = self.mlflow_manager.load_model(
            model_name=model_name, stage="Production"
        )

        logger.info(f"Loaded production model: {model_name}")

        return model

    def get_pipeline_metrics(self) -> Dict:
        return self.metrics.export_metrics_to_dict()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ML Pipeline for Corporate Profit Prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml_config.yaml",
        help="Path to ML configuration file",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only train model without evaluation"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip model validation checks"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )

    args = parser.parse_args()

    pipeline = MLPipeline(args.config)

    results = pipeline.run(
        train_only=args.train_only,
        skip_validation=args.skip_validation,
        resume_from_checkpoint=args.resume,
    )

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
