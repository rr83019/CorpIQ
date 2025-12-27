import re
from typing import Dict
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame
import jellyfish
from loguru import logger


class EntityMatcher:
    """
    Stateless entity resolution engine.
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.matching_threshold = config["entity_resolution"]["matching_threshold"]
        self.fuzzy_fields = config["entity_resolution"]["fuzzy_match_fields"]

        self.clean_text_udf = self._build_clean_text_udf()
        self.jaro_udf = self._build_jaro_udf()
        self.token_udf = self._build_token_udf()

    def resolve_entities(
        self, source1_df: DataFrame, source2_df: DataFrame
    ) -> DataFrame:
        logger.info("Starting stateless entity resolution (hash-ID mode)")

        s1 = self._prepare_source(source1_df, "S1").persist()
        s2 = self._prepare_source(source2_df, "S2").persist()

        candidates = self._generate_candidate_pairs(s1, s2).persist()
        scored = self._score_candidates(candidates)
        matches = self._apply_threshold(scored).persist()

        id_mapping = self._assign_hash_ids(matches, s1, s2)
        harmonized = self._merge_sources_with_ids(s1, s2, id_mapping)

        # Avoid forcing a job unless explicitly needed
        logger.info("Entity resolution complete")

        return harmonized

    def _prepare_source(self, df: DataFrame, source: str) -> DataFrame:
        name_col = f"corporate_name_{source}"

        df = (
            df.withColumn(f"temp_id_{source}", F.monotonically_increasing_id())
            .withColumn("corporate_name_clean", self.clean_text_udf(F.col(name_col)))
            .withColumn(
                "address_clean",
                self.clean_text_udf(F.col("address"))
                if "address" in df.columns
                else F.lit(None),
            )
            .withColumn(
                "country_clean",
                F.upper(F.trim("country")) if "country" in df.columns else F.lit(None),
            )
            .withColumn(
                "postal_code_clean",
                F.regexp_replace("postal_code", r"[^0-9A-Z]", "")
                if "postal_code" in df.columns
                else F.lit(None),
            )
            .withColumn("source", F.lit(source))
        )

        return self._add_blocking_keys(df)

    @staticmethod
    def _build_clean_text_udf():
        def clean(text: str) -> str:
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r"\b(inc|corp|ltd|llc|plc|co|company|limited)\b", "", text)
            text = re.sub(r"[^a-z0-9\s]", "", text)
            return re.sub(r"\s+", " ", text).strip()

        return F.udf(clean)

    @staticmethod
    def _build_jaro_udf():
        return F.udf(
            lambda a, b: jellyfish.jaro_winkler_similarity(a, b) if a and b else 0.0
        )

    @staticmethod
    def _build_token_udf():
        def score(a, b):
            if not a or not b:
                return 0.0
            from fuzzywuzzy import fuzz

            return fuzz.token_sort_ratio(a, b) / 100.0

        return F.udf(score)

    def _add_blocking_keys(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "blocking_key",
            F.concat_ws(
                "||",
                F.coalesce(F.col("country_clean"), F.lit("UNK")),
                F.substring(F.col("corporate_name_clean"), 1, 3),
            ),
        )

    def _generate_candidate_pairs(self, s1: DataFrame, s2: DataFrame) -> DataFrame:
        return (
            s1.alias("a")
            .join(s2.alias("b"), "blocking_key", "inner")
            .select(
                F.col("a.temp_id_S1").alias("id_s1"),
                F.col("b.temp_id_S2").alias("id_s2"),
                F.col("a.corporate_name_clean").alias("name_s1"),
                F.col("b.corporate_name_clean").alias("name_s2"),
                F.col("a.address_clean").alias("addr_s1"),
                F.col("b.address_clean").alias("addr_s2"),
                F.col("a.postal_code_clean").alias("post_s1"),
                F.col("b.postal_code_clean").alias("post_s2"),
                F.col("a.country_clean").alias("country_s1"),
                F.col("b.country_clean").alias("country_s2"),
            )
        )

    def _score_candidates(self, df: DataFrame) -> DataFrame:
        weights = {f["name"]: f["weight"] for f in self.fuzzy_fields}

        return df.withColumn(
            "score",
            self.jaro_udf(F.col("name_s1"), F.col("name_s2"))
            * weights.get("corporate_name", 0.6)
            + self.token_udf(F.col("addr_s1"), F.col("addr_s2"))
            * weights.get("address", 0.3)
            + F.when(F.col("post_s1") == F.col("post_s2"), 1.0).otherwise(0.0)
            * weights.get("postal_code", 0.1),
        )

    def _apply_threshold(self, df: DataFrame) -> DataFrame:
        return df.filter(F.col("score") >= self.matching_threshold).select(
            "id_s1", "id_s2"
        )

    def _assign_hash_ids(
        self, matches: DataFrame, s1: DataFrame, s2: DataFrame
    ) -> Dict[str, DataFrame]:
        logger.info("Assigning deterministic hash-based corporate IDs")

        v1 = s1.select(
            F.col("temp_id_S1").cast("string").alias("id"),
            "corporate_name_clean",
            "country_clean",
            "postal_code_clean",
        )

        v2 = s2.select(
            F.col("temp_id_S2").cast("string").alias("id"),
            "corporate_name_clean",
            "country_clean",
            "postal_code_clean",
        )

        vertices = v1.unionByName(v2).distinct()

        edges = matches.select(
            F.col("id_s1").cast("string").alias("src"),
            F.col("id_s2").cast("string").alias("dst"),
        )

        graph = GraphFrame(vertices.select("id"), edges)
        components = graph.connectedComponents()

        components = components.join(vertices, "id", "left")

        fingerprints = components.groupBy("component").agg(
            F.sha2(
                F.concat_ws(
                    "|",
                    F.sort_array(F.collect_set("corporate_name_clean")),
                    F.sort_array(F.collect_set("country_clean")),
                    F.sort_array(F.collect_set("postal_code_clean")),
                ),
                256,
            ).alias("fingerprint")
        )

        resolved = components.join(fingerprints, "component").withColumn(
            "corporate_id", F.concat(F.lit("CORP-"), F.substring("fingerprint", 1, 16))
        )

        return {
            "source1": resolved.join(
                s1, resolved.id == s1.temp_id_S1.cast("string"), "inner"
            ).select("temp_id_S1", "corporate_id"),
            "source2": resolved.join(
                s2, resolved.id == s2.temp_id_S2.cast("string"), "inner"
            ).select("temp_id_S2", "corporate_id"),
        }

    def _merge_sources_with_ids(
        self, s1: DataFrame, s2: DataFrame, mapping: Dict[str, DataFrame]
    ) -> DataFrame:
        s1 = s1.join(mapping["source1"], "temp_id_S1", "left")
        s2 = s2.join(mapping["source2"], "temp_id_S2", "left")

        s1 = s1.withColumn("_source_system", F.lit("Source1"))
        s2 = s2.withColumn("_source_system", F.lit("Source2"))

        combined = s1.unionByName(s2, allowMissingColumns=True)

        harmonized = combined.groupBy("corporate_id").agg(
            F.first("corporate_name_S1", True).alias("corporate_name_S1"),
            F.first("corporate_name_S2", True).alias("corporate_name_S2"),
            F.first("corporate_name_clean", True).alias("corporate_name_clean"),
            F.first("address", True).alias("address"),
            F.first("address_clean", True).alias("address_clean"),
            F.max("country").alias("country"),
            F.max("country_clean").alias("country_clean"),
            F.max("postal_code").alias("postal_code"),
            F.max("postal_code_clean").alias("postal_code_clean"),
            F.first("industry_sector", True).alias("industry_sector"),
            F.array_distinct(F.flatten(F.collect_list("activity_places"))).alias(
                "activity_places"
            ),
            F.array_distinct(F.flatten(F.collect_list("top_suppliers"))).alias(
                "top_suppliers"
            ),
            F.array_distinct(F.flatten(F.collect_list("main_customers"))).alias(
                "main_customers"
            ),
            F.max("revenue").alias("revenue"),
            F.max("profit").alias("profit"),
            F.max("fiscal_year").alias("fiscal_year"),
            F.collect_set("_source_system").alias("_source_system"),
            F.collect_set("source").alias("sources"),
            F.max("_load_timestamp").alias("_load_timestamp"),
            F.collect_set("blocking_key").alias("blocking_keys"),
            F.collect_set("temp_id_S1").alias("temp_id_S1_list"),
            F.collect_set("temp_id_S2").alias("temp_id_S2_list"),
        )

        return harmonized
