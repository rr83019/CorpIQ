import sys
import os
import re

import yaml
from loguru import logger
from pyspark.sql import SparkSession


from dotenv import load_dotenv

load_dotenv()


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)

    pattern = re.compile(r"\$\{([^}]+)\}")

    def resolve(v):
        if isinstance(v, str):
            for match in pattern.findall(v):
                var, *default = match.split(":-")
                v = v.replace(
                    f"${{{match}}}", os.getenv(var, default[0] if default else "")
                )
        return v

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(i) for i in obj]
        return resolve(obj)

    return walk(config)


def setup_logging(config):
    log_config = config.get("logging", {})

    logger.remove()

    logger.add(
        sys.stderr,
        format=log_config.get("format", "{time} | {level} | {message}"),
        level=log_config.get("level", "INFO"),
    )

    if log_config.get("log_to_file", False):
        log_file = log_config.get("log_file_path", "logs/pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger.add(
            log_file,
            format=log_config.get("format", "{time} | {level} | {message}"),
            level=log_config.get("level", "INFO"),
            rotation=log_config.get("rotation", "500 MB"),
            retention=log_config.get("retention", "30 days"),
        )


def create_spark_session(config) -> SparkSession:
    logger.info("Creating Spark session...")

    spark_config = config["spark"]

    builder = SparkSession.builder.appName(spark_config["app_name"])

    for key, value in spark_config["configs"].items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setCheckpointDir("tmp/spark-checkpoints")

    logger.info(f"Spark session created: {spark.version}")

    return spark
