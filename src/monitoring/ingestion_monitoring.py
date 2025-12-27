import time
import json
from typing import Dict, Optional, Any
from datetime import datetime
from loguru import logger


class IngestionMetrics:
    """
    Collects and manages Data pipeline execution metrics
    """

    def __init__(self, config: Dict):
        self.config = config
        self.metrics: Dict[str, any] = {}
        self.timers: Dict[str, float] = {}
        self.timer_results: Dict[str, float] = {}

        if config.get("monitoring", {}).get("metrics_enabled", False):
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        try:
            from prometheus_client import Counter, Gauge, Histogram, start_http_server

            self.records_processed = Counter(
                "pipeline_records_processed_total",
                "Total number of records processed",
                ["source"],
            )

            self.entities_resolved = Gauge(
                "pipeline_unique_entities", "Number of unique entities identified"
            )

            self.execution_time = Histogram(
                "pipeline_execution_seconds",
                "Pipeline execution time in seconds",
                ["stage"],
            )

            self.adverse_media_found = Counter(
                "pipeline_adverse_media_total", "Total number of adverse media findings"
            )

            port = int(self.config.get("monitoring", {}).get("prometheus_port", 8000))
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

        if metric_name == "source1_records" and hasattr(self, "records_processed"):
            self.records_processed.labels(source="source1").inc(value)
        elif metric_name == "source2_records" and hasattr(self, "records_processed"):
            self.records_processed.labels(source="source2").inc(value)
        elif metric_name == "unique_entities" and hasattr(self, "entities_resolved"):
            self.entities_resolved.set(value)
        elif metric_name == "adverse_media_found" and hasattr(
            self, "adverse_media_found"
        ):
            self.adverse_media_found.inc(value)

    def get_metric(self, metric_name: str) -> Optional[any]:
        return self.metrics.get(metric_name)

    def get_all_metrics(self) -> Dict[str, any]:
        return self.metrics.copy()

    def get_all_timers(self) -> Dict[str, float]:
        return self.timer_results.copy()

    def export_metrics_to_dict(self) -> Dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.get_all_metrics(),
            "timers": self.get_all_timers(),
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
