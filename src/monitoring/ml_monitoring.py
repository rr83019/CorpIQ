import time
import json
from typing import Dict, Optional, Any
from datetime import datetime
from loguru import logger


class MLMetrics:
    """
    Collects and manages ML pipeline execution metrics
    """

    def __init__(self, config: Dict):
        self.config = config
        self.metrics: Dict[str, any] = {}
        self.timers: Dict[str, float] = {}
        self.timer_results: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict] = {}
        self.feature_stats: Dict[str, any] = {}

        if config.get("monitoring", {}).get("metrics_enabled", False):
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        try:
            from prometheus_client import (
                Counter,
                Gauge,
                Histogram,
                Summary,
                start_http_server,
            )

            self.training_records = Gauge(
                "ml_pipeline_training_records", "Number of training records processed"
            )

            self.validation_records = Gauge(
                "ml_pipeline_validation_records",
                "Number of validation records processed",
            )

            self.test_records = Gauge(
                "ml_pipeline_test_records", "Number of test records processed"
            )

            # Model performance metrics
            self.model_accuracy = Gauge(
                "ml_pipeline_model_accuracy", "Model accuracy on test set"
            )

            self.model_precision = Gauge(
                "ml_pipeline_model_precision", "Model precision on test set"
            )

            self.model_recall = Gauge(
                "ml_pipeline_model_recall", "Model recall on test set"
            )

            self.model_f1_score = Gauge(
                "ml_pipeline_model_f1_score", "Model F1 score on test set"
            )

            self.model_auc_roc = Gauge(
                "ml_pipeline_model_auc_roc", "Model AUC-ROC on test set"
            )

            # Feature metrics
            self.num_features = Gauge(
                "ml_pipeline_num_features", "Number of features used in model"
            )

            self.feature_engineering_time = Histogram(
                "ml_pipeline_feature_engineering_seconds",
                "Time spent on feature engineering",
            )

            # Pipeline execution metrics
            self.execution_time = Histogram(
                "ml_pipeline_execution_seconds",
                "Pipeline stage execution time in seconds",
                ["stage"],
            )

            self.pipeline_runs_total = Counter(
                "ml_pipeline_runs_total", "Total number of pipeline runs", ["status"]
            )

            self.model_training_time = Summary(
                "ml_pipeline_model_training_seconds", "Model training time in seconds"
            )

            # Data quality metrics
            self.data_quality_checks = Counter(
                "ml_pipeline_data_quality_checks_total",
                "Total number of data quality checks",
                ["check_type", "status"],
            )

            self.missing_values_ratio = Gauge(
                "ml_pipeline_missing_values_ratio", "Ratio of missing values in dataset"
            )

            self.model_validation_checks = Counter(
                "ml_pipeline_model_validation_checks_total",
                "Total number of model validation checks",
                ["check_type", "status"],
            )

            port = int(self.config.get("monitoring", {}).get("prometheus_port", 8001))
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")

        except ImportError:
            logger.warning("prometheus_client not installed. Metrics disabled.")
        except Exception as e:
            logger.warning(f"Could not initialize Prometheus metrics: {str(e)}")

    def start_timer(self, timer_name: str):
        self.timers[timer_name] = time.time()
        logger.debug(f"Started timer: {timer_name}")

    def stop_timer(self, timer_name: str) -> float:
        if timer_name not in self.timers:
            logger.warning(f"Timer {timer_name} was not started")
            return 0.0

        duration = time.time() - self.timers[timer_name]
        self.timer_results[timer_name] = duration

        logger.debug(f"Stopped timer: {timer_name} (Duration: {duration:.2f}s)")

        if hasattr(self, "execution_time"):
            self.execution_time.labels(stage=timer_name).observe(duration)

        return duration

    def record_metric(self, metric_name: str, value: Any):
        self.metrics[metric_name] = value
        logger.debug(f"Recorded metric: {metric_name} = {value}")

        if hasattr(self, "training_records") and metric_name == "training_records":
            self.training_records.set(value)
        elif (
            hasattr(self, "validation_records") and metric_name == "validation_records"
        ):
            self.validation_records.set(value)
        elif hasattr(self, "test_records") and metric_name == "test_records":
            self.test_records.set(value)
        elif hasattr(self, "num_features") and metric_name == "num_features":
            self.num_features.set(value)
        elif (
            hasattr(self, "missing_values_ratio")
            and metric_name == "missing_values_ratio"
        ):
            self.missing_values_ratio.set(value)

    def record_model_metrics(self, metrics: Dict, stage: str = "test"):
        self.model_metrics[stage] = metrics
        logger.info(f"Recorded {stage} metrics: {metrics}")

        if stage == "test" and hasattr(self, "model_accuracy"):
            if "accuracy" in metrics:
                self.model_accuracy.set(metrics["accuracy"])
            if "precision" in metrics:
                self.model_precision.set(metrics["precision"])
            if "recall" in metrics:
                self.model_recall.set(metrics["recall"])
            if "f1" in metrics:
                self.model_f1_score.set(metrics["f1"])
            if "areaUnderROC" in metrics:
                self.model_auc_roc.set(metrics["areaUnderROC"])

    def record_feature_stats(self, stats: Dict):
        self.feature_stats = stats
        logger.debug(f"Recorded feature stats: {stats}")

    def record_data_quality_check(self, check_type: str, passed: bool):
        status = "passed" if passed else "failed"

        if hasattr(self, "data_quality_checks"):
            self.data_quality_checks.labels(check_type=check_type, status=status).inc()

        logger.info(f"Data quality check '{check_type}': {status}")

    def record_model_validation_check(self, check_type: str, passed: bool):
        status = "passed" if passed else "failed"

        if hasattr(self, "model_validation_checks"):
            self.model_validation_checks.labels(
                check_type=check_type, status=status
            ).inc()

        logger.info(f"Model validation check '{check_type}': {status}")

    def record_pipeline_completion(self, success: bool):
        status = "success" if success else "failure"

        if hasattr(self, "pipeline_runs_total"):
            self.pipeline_runs_total.labels(status=status).inc()

    def get_metric(self, metric_name: str) -> Optional[any]:
        return self.metrics.get(metric_name)

    def get_all_metrics(self) -> Dict[str, any]:
        return self.metrics.copy()

    def get_all_timers(self) -> Dict[str, float]:
        return self.timer_results.copy()

    def get_model_metrics(self, stage: str = "test") -> Optional[Dict]:
        return self.model_metrics.get(stage)

    def export_metrics_to_dict(self) -> Dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.get_all_metrics(),
            "timers": self.get_all_timers(),
            "model_metrics": self.model_metrics,
            "feature_stats": self.feature_stats,
            "total_duration": sum(self.timer_results.values()),
        }

    def save_metrics_to_file(self, filepath: str):
        try:
            metrics_data = self.export_metrics_to_dict()

            with open(filepath, "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)

            logger.info(f"Metrics saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving metrics to file: {str(e)}")
