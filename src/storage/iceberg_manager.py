from typing import Dict, List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from loguru import logger
from datetime import datetime, timedelta


class IcebergTableManager:
    """
    Manages Apache Iceberg tables including creation, upserts, and maintenance
    """

    HARMONIZED_COLUMNS = [
        # ---- Primary Key ----
        "corporate_id",
        # ---- Identity / Names ----
        "corporate_name_S1",
        "corporate_name_S2",
        "corporate_name_clean",
        # ---- Address ----
        "address",
        "address_clean",
        # ---- Geography ----
        "country",
        "country_clean",
        "postal_code",
        "postal_code_clean",
        # ---- Business attributes ----
        "industry_sector",
        # ---- Arrays ----
        "activity_places",
        "top_suppliers",
        "main_customers",
        # ---- Financials ----
        "revenue",
        "profit",
        "fiscal_year",
        # ---- Adverse Media ----
        "adverse_media_summary",
        "adverse_media_last_checked",
        "risk_score",
        "has_adverse_media",
        "keywords_found",
        # ---- Lineage / Audit ----
        "_source_system",
        "sources",
        "_load_timestamp",
        "blocking_keys",
        # ---- Traceability ----
        "temp_id_S1_list",
        "temp_id_S2_list",
        # ---- Metadata ----
        "created_at",
        "updated_at",
    ]

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.catalog_name = config["iceberg"]["catalog_name"]
        self.database = config["iceberg"]["database"]
        self.table_name = config["iceberg"]["table_name"]
        self.full_table_name = f"{self.catalog_name}.{self.database}.{self.table_name}"

    def initialize_catalog_and_database(self):
        try:
            self.spark.sql(
                f"CREATE DATABASE IF NOT EXISTS {self.catalog_name}.{self.database}"
            )
            logger.info(f"Database {self.catalog_name}.{self.database} ready")

            tables = self.spark.sql(
                f"SHOW TABLES IN {self.catalog_name}.{self.database}"
            ).collect()
            logger.info(
                f"Existing tables in {self.database}: {[t.tableName for t in tables]}"
            )

        except Exception as e:
            logger.error(f"Error initializing catalog/database: {str(e)}")
            raise

    def create_table_if_not_exists(self, schema: str = None):
        try:
            table_exists = self._table_exists()

            if table_exists:
                logger.info(f"Table {self.full_table_name} already exists")
                return

            logger.info(f"Creating Iceberg table: {self.full_table_name}")

            if schema is None:
                schema = self._get_default_schema()

            partition_spec = self.config["iceberg"].get("partition_spec", [])
            partition_clause = ""

            if partition_spec:
                partition_fields = [spec["field"] for spec in partition_spec]
                partition_clause = f"PARTITIONED BY ({', '.join(partition_fields)})"

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                {schema}
            )
            USING iceberg
            {partition_clause}
            TBLPROPERTIES (
                'format-version' = '2',
                'write.format.default' = 'parquet',
                'write.parquet.compression-codec' = 'snappy',
                'write.metadata.compression-codec' = 'gzip',
                'write.delete.mode' = 'merge-on-read',
                'write.update.mode' = 'merge-on-read',
                'write.merge.mode' = 'merge-on-read'
            )
            """

            self.spark.sql(create_table_sql)
            logger.info(f"Successfully created table {self.full_table_name}")

            sort_order = self.config["iceberg"].get("sort_order", [])
            if sort_order:
                self._create_sort_order(sort_order)

        except Exception as e:
            logger.error(f"Error creating Iceberg table: {str(e)}")
            raise

    def _get_default_schema(self) -> str:
        return """
            -- ---- Primary Key ----
            corporate_id STRING NOT NULL,

            -- ---- Identity / Names ----
            corporate_name_S1 STRING,
            corporate_name_S2 STRING,
            corporate_name_clean STRING,

            -- ---- Address ----
            address STRING,
            address_clean STRING,

            -- ---- Geography ----
            country STRING,
            country_clean STRING,
            postal_code STRING,
            postal_code_clean STRING,

            -- ---- Business attributes ----
            industry_sector STRING,

            -- ---- Arrays ----
            activity_places ARRAY<STRING>,
            top_suppliers ARRAY<STRING>,
            main_customers ARRAY<STRING>,

            -- ---- Financials ----
            revenue DOUBLE,
            profit DOUBLE,
            fiscal_year INT,
            
            -- ---- Adverse Media ----
            adverse_media_summary STRING,
            adverse_media_last_checked TIMESTAMP,
            risk_score DOUBLE,
            has_adverse_media BOOLEAN,
            keywords_found ARRAY<STRING>,

            -- ---- Lineage / Audit ----
            _source_system ARRAY<STRING>,
            sources ARRAY<STRING>,
            _load_timestamp TIMESTAMP,
            blocking_keys ARRAY<STRING>,

            -- ---- Traceability ----
            temp_id_S1_list ARRAY<STRING>,
            temp_id_S2_list ARRAY<STRING>,

            -- ---- Metadata ----
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        """

    def _table_exists(self) -> bool:
        try:
            self.spark.table(self.full_table_name)
            return True
        except Exception:
            return False

    def _create_sort_order(self, sort_order: List[Dict]):
        try:
            sort_fields = [
                f"{field['field']} {field.get('direction', 'asc').upper()}"
                for field in sort_order
            ]

            alter_sql = f"""
            ALTER TABLE {self.full_table_name}
            WRITE ORDERED BY {", ".join(sort_fields)}
            """

            self.spark.sql(alter_sql)
            logger.info(f"Created sort order: {', '.join(sort_fields)}")
        except Exception as e:
            logger.warning(f"Could not create sort order: {str(e)}")

    def upsert_data(self, source_df: DataFrame, merge_key: str = "corporate_id"):
        try:
            logger.info(f"Starting upsert operation on {self.full_table_name}")
            logger.info(f"Source records: {source_df.count()}")

            if not self._table_exists():
                logger.warning("Target table doesn't exist. Creating it first...")
                self.create_table_if_not_exists()

                # For first load, just write the data
                logger.info("Performing initial data load...")
                source_df.writeTo(self.full_table_name).append()
                logger.info("Initial load complete")
                return

            target_table = self.spark.table(self.full_table_name)
            existing_count = target_table.count()
            logger.info(f"Existing records in target: {existing_count}")

            source_view = f"source_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            source_df.createOrReplaceTempView(source_view)

            source_df = source_df.withColumn(
                "updated_at", F.current_timestamp()
            ).withColumn("created_at", F.current_timestamp())
            source_df.createOrReplaceTempView(source_view)

            merge_sql = self._build_merge_statement(source_view, merge_key)

            logger.info("Executing MERGE INTO operation...")
            logger.debug(f"Merge SQL: {merge_sql}")

            self.spark.sql(merge_sql)

            new_count = self.spark.table(self.full_table_name).count()
            logger.info(f"Upsert complete. Records after merge: {new_count}")
            logger.info(f"Net new records: {new_count - existing_count}")

        except Exception as e:
            logger.error(f"Error during upsert operation: {str(e)}")
            raise

    def _build_merge_statement(self, source_view: str, merge_key: str) -> str:
        all_columns = self.HARMONIZED_COLUMNS

        update_columns = [
            col for col in all_columns if col not in {merge_key, "created_at"}
        ]

        update_set_clause = ",\n ".join(
            f"target.{col} = COALESCE(source.{col}, target.{col})"
            for col in update_columns
        )

        insert_columns = ", ".join(all_columns)
        insert_values = ", ".join(f"source.{col}" for col in all_columns)

        merge_sql = f"""
        MERGE INTO {self.full_table_name} AS target
        USING {source_view} AS source
        ON target.{merge_key} = source.{merge_key}

        WHEN MATCHED THEN
            UPDATE SET
                {update_set_clause}

        WHEN NOT MATCHED THEN
            INSERT ({insert_columns})
            VALUES ({insert_values})
        """

        return merge_sql.strip()

    def update_adverse_media(self, adverse_media_df: DataFrame):
        try:
            logger.info("Updating adverse media information...")

            view_name = f"adverse_media_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            adverse_media_df.createOrReplaceTempView(view_name)

            update_sql = f"""
            MERGE INTO {self.full_table_name} AS target
            USING {view_name} AS source
            ON target.corporate_id = source.corporate_id
            WHEN MATCHED THEN
                UPDATE SET
                    target.adverse_media_summary = source.summary,
                    target.adverse_media_last_checked = source.checked_at,
                    target.risk_score = source.risk_score,
                    target.updated_at = current_timestamp()
            """

            self.spark.sql(update_sql)
            logger.info("Adverse media information updated successfully")

        except Exception as e:
            logger.error(f"Error updating adverse media: {str(e)}")
            raise

    def optimize_table(self):
        try:
            logger.info(f"Optimizing table {self.full_table_name}...")

            self.spark.sql(
                f"CALL {self.catalog_name}.system.rewrite_data_files('{self.database}.{self.table_name}')"
            )

            expire_timestamp = (datetime.now() - timedelta(days=7)).timestamp() * 1000
            self.spark.sql(f"""
                CALL {self.catalog_name}.system.expire_snapshots(
                    table => '{self.database}.{self.table_name}',
                    older_than => TIMESTAMP '{expire_timestamp}'
                )
            """)

            logger.info("Table optimization complete")

        except Exception as e:
            logger.warning(f"Error during table optimization: {str(e)}")

    def get_table_statistics(self) -> Dict:
        try:
            stats = {}

            stats["row_count"] = self.spark.table(self.full_table_name).count()

            snapshots = self.spark.sql(f"""
                SELECT * FROM {self.catalog_name}.{self.database}.{self.table_name}.snapshots
                ORDER BY committed_at DESC
                LIMIT 5
            """).collect()

            stats["snapshot_count"] = len(snapshots)
            stats["latest_snapshot"] = snapshots[0].asDict() if snapshots else None

            files = self.spark.sql(f"""
                SELECT COUNT(*) as file_count, 
                       SUM(file_size_in_bytes) as total_size_bytes
                FROM {self.catalog_name}.{self.database}.{self.table_name}.files
            """).first()

            stats["file_count"] = files["file_count"]
            stats["total_size_mb"] = (
                files["total_size_bytes"] / (1024 * 1024)
                if files["total_size_bytes"]
                else 0
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting table statistics: {str(e)}")
            return {}

    def query_table(
        self, filter_condition: Optional[str] = None, limit: Optional[int] = None
    ) -> DataFrame:
        try:
            sql = f"SELECT * FROM {self.full_table_name}"

            if filter_condition:
                sql += f" WHERE {filter_condition}"

            if limit:
                sql += f" LIMIT {limit}"

            return self.spark.sql(sql)

        except Exception as e:
            logger.error(f"Error querying table: {str(e)}")
            raise

    def time_travel_query(
        self, snapshot_id: Optional[int] = None, timestamp: Optional[str] = None
    ) -> DataFrame:
        try:
            if snapshot_id:
                sql = (
                    f"SELECT * FROM {self.full_table_name} VERSION AS OF {snapshot_id}"
                )
            elif timestamp:
                sql = f"SELECT * FROM {self.full_table_name} TIMESTAMP AS OF '{timestamp}'"
            else:
                raise ValueError("Either snapshot_id or timestamp must be provided")

            return self.spark.sql(sql)

        except Exception as e:
            logger.error(f"Error in time travel query: {str(e)}")
            raise
