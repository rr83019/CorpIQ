import time
import json
from itertools import islice
from typing import List, Dict, Optional, Iterator
from datetime import datetime

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pyspark.sql.functions import col
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from openai import OpenAI
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    BooleanType,
    TimestampType,
    ArrayType,
)


class AdverseMediaAnalyzer:
    """
    LLM-powered adverse media detection and analysis
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.llm_config = config["llm"]

        api_key = self.llm_config["api_key_env"]
        if not api_key:
            raise ValueError(
                f"API key not found in environment: {self.llm_config['api_key_env']}"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = self.llm_config["model"]

        self.max_calls_per_minute = int(
            self.llm_config["rate_limit"]["max_calls_per_minute"]
        )
        self.call_timestamps: List[float] = []

    def analyze_adverse_media_batch(
        self, corporate_df: DataFrame, batch_size: Optional[int] = None
    ) -> DataFrame:
        batch_size = int(batch_size or self.llm_config["adverse_media"]["batch_size"])

        record_iter = self._iter_corporate_records(corporate_df)
        results: List[Dict] = []

        for batch_idx, batch in enumerate(
            self._batched(record_iter, batch_size), start=1
        ):
            logger.info(f"Processing batch {batch_idx} ({len(batch)} records)")

            for record in batch:
                results.append(self._safe_analyze_record(record))

            time.sleep(1)

        results_df = self._build_results_dataframe(results)

        return results_df

    def _iter_corporate_records(self, df: DataFrame) -> Iterator[Dict]:
        cols = [
            "corporate_id",
            "corporate_name_clean",
            "country",
            "industry_sector",
        ]

        def row_mapper(partition):
            for row in partition:
                yield {
                    "corporate_id": row["corporate_id"],
                    "corporate_name_clean": row["corporate_name_clean"],
                    "country": row["country"],
                    "industry_sector": row["industry_sector"],
                }

        rdd = df.select(*cols).rdd
        for record in rdd.mapPartitions(row_mapper).toLocalIterator():
            yield record

    @staticmethod
    def _batched(iterable: Iterator, size: int):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                return
            yield batch

    def _safe_analyze_record(self, record: Dict) -> Dict:
        try:
            return self._analyze_single_corporate(
                corporate_id=record["corporate_id"],
                corporate_name=record["corporate_name_clean"],
                country=record["country"],
                industry_sector=record["industry_sector"],
            )
        except Exception as e:
            logger.error(f"Error analyzing {record['corporate_name_clean']}: {e}")
            return self._error_result(record, str(e))

    def _error_result(self, record: Dict, error: str) -> Dict:
        return {
            "corporate_id": record["corporate_id"],
            "corporate_name": record["corporate_name_clean"],
            "has_adverse_media": False,
            "summary": None,
            "risk_score": 0.0,
            "keywords_found": [],
            "checked_at": datetime.utcnow(),
        }

    def _analyze_single_corporate(
        self,
        corporate_id: str,
        corporate_name: str,
        country: Optional[str] = None,
        industry_sector: Optional[str] = None,
    ) -> Dict:
        self._rate_limit_check()

        search_context = self._build_search_context(
            corporate_name, country, industry_sector
        )

        analysis = self._call_llm_for_analysis(corporate_name, search_context)

        return {
            "corporate_id": corporate_id,
            "corporate_name": corporate_name,
            "has_adverse_media": analysis["has_adverse_media"],
            "summary": analysis["summary"],
            "risk_score": analysis["risk_score"],
            "keywords_found": analysis["keywords_found"],
            "checked_at": datetime.utcnow(),
        }

    def _build_search_context(
        self,
        corporate_name: str,
        country: Optional[str],
        industry_sector: Optional[str],
    ) -> str:
        keywords = self.llm_config["adverse_media"]["keywords"]
        days_back = self.llm_config["adverse_media"]["search_days_back"]

        context = f"""
        Search Query: "{corporate_name}" + adverse media keywords
        Time Range: Last {days_back} days
        Country: {country or "Unknown"}
        Industry: {industry_sector or "Unknown"}

        Keywords searched: {", ".join(keywords)}

        [SIMULATED SEARCH RESULTS]
        Note: In production, this would contain actual web search results from news sources,
        regulatory filings, court records, and other public sources.

        For demonstration purposes, the LLM will analyze based on the company name pattern
        and provide a risk assessment.
        """

        return context

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm_for_analysis(self, corporate_name: str, search_context: str) -> Dict:
        prompt = """You are an expert risk analyst specializing in corporate due diligence and adverse media detection.

        Analyze the following information about a corporation and determine if there are any adverse media findings:
        
        Your task:
        1. Determine if there are any adverse media findings (fraud, lawsuits, regulatory violations, scandals, etc.)
        2. Provide a brief summary of any findings (2-3 sentences max)
        3. Assign a risk score from 0.0 (no risk) to 1.0 (severe risk)
        4. List any relevant keywords found
        
        Respond ONLY with a valid JSON object in this exact format:
        {
        "has_adverse_media": true or false,
        "summary": "Brief summary of findings or null if none",
        "risk_score": 0.0 to 1.0,
        "keywords_found": ["keyword1", "keyword2"] or []
        }
        
        Do not include any markdown formatting, code blocks, or additional text. Only return the JSON object."""

        try:
            system_msg: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": prompt,
            }

            user_msg: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": search_context,
            }

            messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=int(self.llm_config["max_tokens"]),
                temperature=float(self.llm_config["temperature"]),
                messages=messages,
            )

            response_text = message.choices[0].message.content.strip()

            # Remove potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            required_keys = [
                "has_adverse_media",
                "summary",
                "risk_score",
                "keywords_found",
            ]
            if not all(key in result for key in required_keys):
                raise ValueError(
                    f"Invalid response structure. Missing keys: {[k for k in required_keys if k not in result]}"
                )

            result["risk_score"] = max(0.0, min(1.0, float(result["risk_score"])))

            return result

        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            return {
                "has_adverse_media": False,
                "summary": None,
                "risk_score": 0.0,
                "keywords_found": [],
            }
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise

    def _rate_limit_check(self):
        current_time = time.time()

        self.call_timestamps = [
            ts for ts in self.call_timestamps if current_time - ts < 60
        ]

        if len(self.call_timestamps) >= self.max_calls_per_minute:
            oldest_call = min(self.call_timestamps)
            sleep_time = 60 - (current_time - oldest_call) + 1

            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds"
                )
                time.sleep(sleep_time)
                self.call_timestamps = []

        self.call_timestamps.append(current_time)

    def _build_results_dataframe(self, results: List[Dict]) -> DataFrame:
        schema = StructType(
            [
                StructField("corporate_id", StringType(), False),
                StructField("corporate_name", StringType(), False),
                StructField("has_adverse_media", BooleanType(), False),
                StructField("summary", StringType(), True),
                StructField("risk_score", FloatType(), False),
                StructField("keywords_found", ArrayType(StringType()), True),
                StructField("checked_at", TimestampType(), False),
            ]
        )

        rows = [
            (
                r["corporate_id"],
                r["corporate_name"],
                r["has_adverse_media"],
                r["summary"],
                r["risk_score"],
                r["keywords_found"],
                r["checked_at"],
            )
            for r in results
        ]

        df = self.spark.createDataFrame(rows, schema)

        return df.select(
            col("corporate_id").alias("corporate_id"),
            col("corporate_name").alias("corporate_name"),
            col("has_adverse_media").alias("has_adverse_media"),
            col("summary").alias("adverse_media_summary"),
            col("risk_score").alias("risk_score"),
            col("keywords_found").alias("keywords_found"),
            col("checked_at").alias("adverse_media_last_checked"),
        )
