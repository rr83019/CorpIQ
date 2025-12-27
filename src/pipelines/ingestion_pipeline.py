import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from pyspark.sql import functions as F, DataFrame

from src.ingestion.entity_resolution import EntityMatcher
from src.ingestion.adverse_media import AdverseMediaAnalyzer
from src.storage.iceberg_manager import IcebergTableManager
from src.storage.data_loader import DataLoader
from src.ingestion.data_quality import DataQualityValidator
from src.monitoring.ingestion_monitoring import IngestionMetrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.helper import load_config, setup_logging, create_spark_session


class IngestionPipeline:
    """
    Main orchestrator for the data pipeline
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        setup_logging(self.config)

        self.spark = create_spark_session(self.config)

        self.data_loader = DataLoader(self.config, self.spark)
        self.entity_matcher = EntityMatcher(self.config, self.spark)
        self.adverse_media_analyzer = AdverseMediaAnalyzer(self.config, self.spark)
        self.iceberg_manager = IcebergTableManager(self.config, self.spark)
        self.data_quality_validator = DataQualityValidator(self.config, self.spark)
        self.metrics = IngestionMetrics(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)

        logger.info("Data Pipeline initialized")

    def run(self, run_adverse_media: bool = True, optimize_table: bool = False):
        try:
            start = datetime.now()
            self.metrics.start_timer("total_data_pipeline")
            logger.info("Starting Corporate Data Harmonization Pipeline")

            self._init_iceberg()
            source1_df, source2_df, counts = self._load_sources()

            if not self._validate_sources(source1_df, source2_df):
                return False

            harmonized_df, entity_stats = self._resolve_entities(source1_df, source2_df)

            harmonized_df = self._run_adverse_media(harmonized_df, False)

            self._upsert_to_iceberg(harmonized_df, optimize_table)

            results = self._finalize_pipeline(
                start,
                counts,
                entity_stats,
                run_adverse_media,
                optimize_table,
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "pipeline_metrics": self.metrics.export_metrics_to_dict(),
            }

    def _init_iceberg(self):
        self.metrics.start_timer("iceberg_init")
        self.iceberg_manager.initialize_catalog_and_database()
        self.iceberg_manager.create_table_if_not_exists()
        self.metrics.stop_timer("iceberg_init")

    def _load_sources(self):
        self.metrics.start_timer("data_loading")

        s1 = self.data_loader.load_source1()
        s2 = self.data_loader.load_source2()

        counts = {
            "source1": s1.count(),
            "source2": s2.count(),
        }

        self.metrics.record_metric("source1_records", counts["source1"])
        self.metrics.record_metric("source2_records", counts["source2"])

        self.metrics.stop_timer("data_loading")

        self.checkpoint_manager.save_checkpoint("data_loaded", counts)

        return s1, s2, counts

    def _validate_sources(self, s1: DataFrame, s2: DataFrame) -> bool:
        self.metrics.start_timer("data_quality")

        valid1 = self.data_quality_validator.validate(s1, "source1")
        valid2 = self.data_quality_validator.validate(s2, "source2")

        self.metrics.stop_timer("data_quality")

        if not (valid1 and valid2):
            logger.error("Data quality validation failed")
            return False

        return True

    def _resolve_entities(self, s1: DataFrame, s2: DataFrame):
        self.metrics.start_timer("entity_resolution")

        harmonized = self.entity_matcher.resolve_entities(s1, s2)

        if self.config["performance"]["cache_intermediate_results"]:
            harmonized = harmonized.cache()

        stats = {
            "harmonized_count": harmonized.count(),
            "unique_entities": harmonized.select("corporate_id").distinct().count(),
        }

        self.metrics.record_metric("unique_entities", stats["unique_entities"])
        self.metrics.stop_timer("entity_resolution")

        self.checkpoint_manager.save_checkpoint("entity_resolution", stats)

        return harmonized, stats

    def _run_adverse_media(
        self, harmonized: DataFrame, run_adverse_media: bool
    ) -> DataFrame:
        self.metrics.start_timer("adverse_media")

        try:
            if not run_adverse_media or not bool(
                self.config["llm"]["adverse_media"]["search_enabled"]
            ):
                logger.info("Adverse media analysis skipped")

                harmonized = (
                    harmonized.withColumn(
                        "has_adverse_media",
                        F.lit(False),
                    )
                    .withColumn(
                        "adverse_media_summary",
                        F.lit(None),
                    )
                    .withColumn(
                        "risk_score",
                        F.lit(0.0),
                    )
                    .withColumn(
                        "keywords_found",
                        F.lit(None),
                    )
                    .withColumn(
                        "adverse_media_last_checked",
                        F.lit(None),
                    )
                )

                return harmonized

            sample_size = min(harmonized.count(), 20)
            sample_df = harmonized.limit(sample_size)

            results = self.adverse_media_analyzer.analyze_adverse_media_batch(sample_df)

            harmonized = harmonized.join(results, on="corporate_id", how="left")

            adverse_count = results.filter(F.col("has_adverse_media")).count()

            self.metrics.record_metric("adverse_media_found", adverse_count)

            self.checkpoint_manager.save_checkpoint(
                "adverse_media",
                {
                    "analyzed": sample_size,
                    "adverse_found": adverse_count,
                },
            )

        except Exception:
            logger.error("Adverse media stage failed", exc_info=True)

        self.metrics.stop_timer("adverse_media")
        return harmonized

    def _upsert_to_iceberg(self, df: DataFrame, optimize: bool):
        self.metrics.start_timer("iceberg_upsert")
        self.iceberg_manager.upsert_data(df)
        self.metrics.stop_timer("iceberg_upsert")

        if optimize:
            self.metrics.start_timer("table_optimization")
            self.iceberg_manager.optimize_table()
            self.metrics.stop_timer("table_optimization")

    def _finalize_pipeline(
        self,
        start_time: datetime,
        source_counts: dict,
        entity_stats: dict,
        run_adverse_media: bool,
        optimize_table: bool,
    ):
        end = datetime.now()
        duration = (end - start_time).total_seconds()

        table_stats = self.iceberg_manager.get_table_statistics()

        self._print_metrics_summary()

        self.metrics.save_metrics_to_file(
            os.path.join(
                self.checkpoint_manager.checkpoint_location,
                f"ingestion_metrics_{end:%Y%m%d_%H%M%S}.json",
            )
        )

        self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

        self.metrics.stop_timer("total_data_pipeline")

        return {
            "success": True,
            "start_time": start_time.isoformat(),
            "end_time": end.isoformat(),
            "duration_seconds": duration,
            "table_statistics": table_stats,
            "adverse_media_analyzed": run_adverse_media,
            "optimize_table": optimize_table,
            "counts": {
                **source_counts,
                **entity_stats,
                "adverse_media_found": self.metrics.get_metric("adverse_media_found"),
            },
            "pipeline_metrics": self.metrics.export_metrics_to_dict(),
        }

    def _print_metrics_summary(self):
        logger.info("DATA METRICS SUMMARY")

        metrics = self.metrics.get_all_metrics()

        logger.info("\nRecord Counts:")
        logger.info(f"  Source 1 Records: {metrics.get('source1_records', 0):,}")
        logger.info(f"  Source 2 Records: {metrics.get('source2_records', 0):,}")
        logger.info(f"  Unique Entities: {metrics.get('unique_entities', 0):,}")
        logger.info(f"  Adverse Media Found: {metrics.get('adverse_media_found', 0):,}")

        logger.info("\nExecution Times:")
        timers = self.metrics.get_all_timers()
        for timer_name, duration in timers.items():
            logger.info(f"  {timer_name}: {duration:.2f}s")

    def query_registry(self, filter_condition: Optional[str] = None, limit: int = 100):
        try:
            logger.info("Querying corporate registry...")

            result = self.iceberg_manager.query_table(
                filter_condition=filter_condition, limit=limit
            )

            logger.info(f"Query returned {result.count()} records")

            return result

        except Exception as e:
            logger.error(f"Error querying registry: {str(e)}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Corporate Data Harmonization Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-adverse-media", action="store_true", help="Skip adverse media analysis"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Run table optimization after upsert"
    )
    parser.add_argument(
        "--query", type=str, help="Query the registry instead of running pipeline"
    )

    args = parser.parse_args()

    pipeline = IngestionPipeline(args.config)

    if args.query:
        result = pipeline.query_registry(filter_condition=args.query)
        result.show(truncate=False)
        return

    success = pipeline.run(
        run_adverse_media=not args.skip_adverse_media, optimize_table=args.optimize
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
