from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame, SparkSession, Column
from pyspark.sql import functions as F
from loguru import logger


class DataQualityValidator:
    """
    Validates data quality based on configured rules
    """

    def __init__(self, config: Dict, spark: SparkSession):
        self.spark = spark
        self.rules = config.get("data_quality", {}).get("validation_rules", {})
        self.error_threshold = config.get("data_quality", {}).get(
            "error_threshold", 0.05
        )
        self.default_approx_distinct = config.get("data_quality", {}).get(
            "approx_distinct", True
        )

    def validate(self, df: DataFrame, source_name: str) -> bool:
        logger.info(f"Starting data quality validation for {source_name}")

        rules = self._select_valid_rules(df, self.rules)
        if not rules:
            logger.error("No applicable validation rules found")
            return False

        agg_expression, rule_plan = self._build_rule_expressions(rules)
        agg_row = self._execute_aggregation(df, agg_expression)

        results = self._evaluate_results(agg_row, rule_plan)
        return self._log_and_decide(results, agg_row["__total_records"])

    def _select_valid_rules(
        self, df: DataFrame, rules: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        valid_columns = set(df.columns)
        selected = {}

        for field, rule_list in rules.items():
            if field in valid_columns:
                selected[field] = rule_list
            else:
                logger.warning(f"Column {field} not found; skipping")

        return selected

    def _build_rule_expressions(
        self, rules: Dict[str, List[Dict]]
    ) -> Tuple[List[Column], List[Dict[str, Any]]]:
        agg_expression = [F.count(F.lit(1)).alias("__total_records")]
        rule_plan = []

        for field, rule_list in rules.items():
            col = F.col(field)
            non_null_alias = f"{field}__non_null"

            agg_expression.append(
                F.sum(F.when(col.isNotNull(), 1).otherwise(0)).alias(non_null_alias)
            )

            for idx, rule in enumerate(rule_list):
                rule_type = rule["type"]

                if rule_type == "unique":
                    approx = rule.get("approx", self.default_approx_distinct)
                    distinct_alias = f"{field}__distinct__{idx}"

                    expr = (
                        F.approx_count_distinct(col) if approx else F.countDistinct(col)
                    ).alias(distinct_alias)

                    agg_expression.append(expr)

                    rule_plan.append(
                        {
                            "field": field,
                            "type": "unique",
                            "non_null": non_null_alias,
                            "distinct": distinct_alias,
                            "rule": rule,
                        }
                    )
                else:
                    alias = f"{field}__{rule_type}__{idx}"
                    condition = self._build_condition(col, rule)

                    agg_expression.append(
                        F.sum(F.when(condition, 1).otherwise(0)).alias(alias)
                    )

                    rule_plan.append(
                        {
                            "field": field,
                            "type": rule_type,
                            "alias": alias,
                            "rule": rule,
                        }
                    )

        return agg_expression, rule_plan

    def _build_condition(self, col: Column, rule: Dict) -> Column:
        rtype = rule["type"]

        if rtype == "not_null":
            return col.isNull()

        if rtype == "min_length":
            return F.length(col.cast("string")) < F.lit(rule.get("value", 1))

        if rtype == "max_length":
            return F.length(col.cast("string")) > F.lit(rule.get("value", 1000))

        if rtype == "numeric":
            return col.isNotNull() & col.cast("double").isNull()

        if rtype == "positive":
            return col.isNotNull() & (col < 0)

        if rtype == "range":
            conditions = []
            if "min" in rule:
                conditions.append(col < rule["min"])
            if "max" in rule:
                conditions.append(col > rule["max"])

            if not conditions:
                return F.lit(False)

            from functools import reduce

            return col.isNotNull() & reduce(lambda a, b: a | b, conditions)

        if rtype == "pattern":
            return col.isNotNull() & ~col.rlike(rule.get("pattern", ".*"))

        logger.warning(f"Unknown rule type: {rtype}")
        return F.lit(False)

    def _execute_aggregation(self, df: DataFrame, agg_expression: List[Column]):
        row = df.agg(*agg_expression).first()
        if row["__total_records"] == 0:
            raise ValueError("No records found")
        return row

    def _evaluate_results(self, agg_row, rule_plan: List[Dict]) -> List[Dict]:
        results = []

        for plan in rule_plan:
            if plan["type"] == "unique":
                non_null = int(agg_row[plan["non_null"]] or 0)
                distinct = int(agg_row[plan["distinct"]] or 0)
                errors = max(0, non_null - distinct)
            else:
                errors = int(agg_row[plan["alias"]] or 0)

            results.append(
                {
                    "field": plan["field"],
                    "rule_type": plan["type"],
                    "error_count": errors,
                    "rule": plan["rule"],
                }
            )

        return results

    def _log_and_decide(self, results: List[Dict], total_records: int) -> bool:
        total_errors = sum(r["error_count"] for r in results)
        total_checks = total_records * len(results)
        overall_error_rate = total_errors / total_checks if total_checks else 0

        for r in results:
            rate = r["error_count"] / total_records
            status = "PASS" if rate <= float(self.error_threshold) else "FAIL"
            logger.info(
                f"[{status}] {r['field']} - {r['rule_type']}: "
                f"{r['error_count']} ({rate:.2%})"
            )

        logger.info(f"Overall error rate: {overall_error_rate:.2%}")
        return overall_error_rate <= float(self.error_threshold)
