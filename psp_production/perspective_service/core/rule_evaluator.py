"""
Rule Evaluator - Converts rule criteria into Polars expressions for data filtering.
"""

import json
from typing import Dict, List, Any, Optional

import polars as pl

from perspective_service.utils.constants import INT_NULL


class RuleEvaluator:
    """Converts rule criteria into Polars expressions for data filtering."""

    @classmethod
    def evaluate(cls,
                 criteria: Dict[str, Any],
                 perspective_id: Optional[int] = None,
                 precomputed_values: Dict[str, List[Any]] = None,
                 rule_expr: Optional[pl.Expr] = None) -> pl.Expr:
        """
        Convert rule criteria into a Polars expression.

        Args:
            criteria: Dictionary defining the filter criteria
            perspective_id: ID of the current perspective (for variable substitution)
            precomputed_values: Pre-computed values for nested criteria
            rule_expr: Expression representing rule evaluation results (for rule_result references)

        Returns:
            Polars expression representing the criteria
        """
        if not criteria:
            return pl.lit(True)

        # Handle logical operators
        if "and" in criteria:
            return cls._evaluate_and(criteria["and"], perspective_id, precomputed_values, rule_expr)
        if "or" in criteria:
            return cls._evaluate_or(criteria["or"], perspective_id, precomputed_values, rule_expr)
        if "not" in criteria:
            return ~cls.evaluate(criteria["not"], perspective_id, precomputed_values, rule_expr)

        # Handle simple criteria
        return cls._evaluate_simple_criteria(criteria, perspective_id, precomputed_values, rule_expr)

    @classmethod
    def _evaluate_and(cls, subcriteria: List[Dict], perspective_id: int,
                      precomputed_values: Dict, rule_expr: Optional[pl.Expr] = None) -> pl.Expr:
        """Combine multiple criteria with AND logic using vectorized horizontal operation."""
        if not subcriteria:
            return pl.lit(True)

        exprs = [cls.evaluate(crit, perspective_id, precomputed_values, rule_expr)
                 for crit in subcriteria]
        return pl.all_horizontal(exprs)

    @classmethod
    def _evaluate_or(cls, subcriteria: List[Dict], perspective_id: int,
                     precomputed_values: Dict, rule_expr: Optional[pl.Expr] = None) -> pl.Expr:
        """Combine multiple criteria with OR logic using vectorized horizontal operation."""
        if not subcriteria:
            return pl.lit(False)

        exprs = [cls.evaluate(crit, perspective_id, precomputed_values, rule_expr)
                 for crit in subcriteria]
        return pl.any_horizontal(exprs)

    @classmethod
    def _evaluate_simple_criteria(cls, criteria: Dict, perspective_id: int,
                                  precomputed_values: Dict,
                                  rule_expr: Optional[pl.Expr] = None) -> pl.Expr:
        """Evaluate a simple column-operator-value criteria."""
        column = criteria.get("column", "").lower() if criteria.get("column") else None
        operator = criteria.get("operator_type")
        value = criteria.get("value")

        if not column or not operator:
            return pl.lit(True)

        # Substitute perspective_id in value if needed
        if perspective_id and isinstance(value, str) and 'perspective_id' in value:
            value = value.replace('perspective_id', str(perspective_id))

        # Handle nested criteria (In/NotIn with dict value)
        if operator in ["In", "NotIn"] and isinstance(value, dict):
            # Check if this is a rule_result reference
            if value.get("table_name", "").lower() == "rule_result":
                # =================================================================
                # rule_result handling for PostProcessing modifiers (e.g., exclude_trade_cash)
                # =================================================================
                # The criteria: simulated_trade_id IN (select simulated_trade_id where rules passed)
                # Means: "save this position if ANY position with the same simulated_trade_id passes"
                #
                # Example: exclude_trade_cash modifier
                # - Position A: simulated_trade_id=100, passes rules -> kept
                # - Position B: simulated_trade_id=100, fails rules -> ALSO kept (because A passed)
                # - Position C: simulated_trade_id=200, fails rules -> removed (no 200 passed)
                #
                # Implementation:
                # - rule_expr = boolean expression "does this row pass the rules?"
                # - rule_expr.any().over(column) = "does ANY row with my column value pass?"
                # - Equivalent to: column IN (select column from rows where rule_expr is True)
                # - We also exclude NULLs to match original behavior
                # =================================================================
                if rule_expr is not None:
                    if operator == "In":
                        return rule_expr.any().over(column) & pl.col(column).is_not_null()
                    else:  # NotIn
                        return ~rule_expr.any().over(column) | pl.col(column).is_null()
                # No rule_expr context available - fall back to True
                return pl.lit(True)

            # =================================================================
            # Precomputed nested criteria (non-rule_result)
            # =================================================================
            # For nested criteria from DB perspective rules (not rule_result references),
            # values are precomputed in engine._precompute_nested_criteria() for efficiency.
            #
            # Why precompute instead of .any().over()?
            # - is_in(list) uses O(1) hash lookup per row
            # - .any().over() requires grouping + aggregation (more expensive)
            # - Precomputed values are reused if same criteria appears multiple times
            #
            # The precomputation:
            # 1. Evaluates inner criteria to find matching rows
            # 2. Extracts unique values of the outer column from those rows
            # 3. Caches as Python list keyed by JSON of the criteria
            # =================================================================
            if precomputed_values:
                criteria_key = json.dumps(value, sort_keys=True)
                matching_values = precomputed_values.get(criteria_key, [])
                if operator == "In":
                    return pl.col(column).is_in(matching_values)
                return ~pl.col(column).is_in(matching_values)
            return pl.lit(True)

        # Parse and apply the operator
        parsed_value = cls._parse_value(value, operator)
        return cls._apply_operator(operator, column, parsed_value)

    @classmethod
    def _apply_operator(cls, operator: str, column: str, value: Any) -> pl.Expr:
        """Apply a comparison operator to create a Polars expression."""
        # Use idiomatic Polars methods for null-safe comparisons.
        # Original fills NULLs with sentinel (-2147483648) for ints, '' for strings.
        # Positive operators (==, >, >=, In, Between, Like): NULL → False (doesn't match)
        # Negative operators (!=, <, <=, NotIn, NotBetween, NotLike): NULL → True (matches original sentinel behavior)
        operators = {
            "=": lambda c, v: pl.col(c).eq_missing(v),
            "==": lambda c, v: pl.col(c).eq_missing(v),
            "!=": lambda c, v: pl.col(c).ne_missing(v),
            
            ">": lambda c, v: pl.col(c).gt(v).fill_null(False),
            "<": lambda c, v: pl.col(c).lt(v).fill_null(True),
            ">=": lambda c, v: pl.col(c).ge(v).fill_null(False),
            "<=": lambda c, v: pl.col(c).le(v).fill_null(True),
            
            "In": lambda c, v: pl.col(c).is_in(v).fill_null(False),
            "NotIn": lambda c, v: pl.col(c).is_in(v).not_().fill_null(True),
            
            "IsNull": lambda c, v: pl.col(c).is_null() | pl.col(c).eq(INT_NULL),
            "IsNotNull": lambda c, v: pl.col(c).is_not_null() & pl.col(c).ne(INT_NULL),
            
            "Between": lambda c, v: pl.col(c).is_between(v[0], v[1], closed="both").fill_null(False),
            "NotBetween": lambda c, v: pl.col(c).is_between(v[0], v[1], closed="both").not_().fill_null(True),
            
            "Like": lambda c, v: cls._build_like_expr(c, v, False).fill_null(False),
            "NotLike": lambda c, v: cls._build_like_expr(c, v, True).fill_null(True),
        }

        return operators.get(operator, lambda c, v: pl.lit(True))(column, value)

    @classmethod
    def _build_like_expr(cls, column: str, pattern: str, negate: bool) -> pl.Expr:
        """Build a LIKE expression for pattern matching."""
        # Remove surrounding quotes if present
        clean_pattern = pattern.strip("'\"").lower()
        expr = pl.col(column).str.to_lowercase()

        if clean_pattern.startswith("%") and clean_pattern.endswith("%"):
            # %pattern% → contains (use literal=True for exact string match)
            expr = expr.str.contains(clean_pattern[1:-1], literal=True)
        elif clean_pattern.endswith("%"):
            # pattern% → starts_with
            expr = expr.str.starts_with(clean_pattern[:-1])
        elif clean_pattern.startswith("%"):
            # %pattern → ends_with
            expr = expr.str.ends_with(clean_pattern[1:])
        else:
            # Exact match - use eq_missing for null safety
            expr = expr.eq_missing(clean_pattern)

        return expr.not_() if negate else expr

    @staticmethod
    def _parse_value(value: Any, operator: str) -> Any:
        """Parse value based on operator requirements."""
        if operator in ["IsNull", "IsNotNull"]:
            return None

        if operator in ["In", "NotIn"]:
            if isinstance(value, str):
                # Strip brackets, parentheses, and quotes to handle formats like "('USD','EUR')" or "[4,8,9]"
                items = [item.strip().strip("'\"") for item in value.strip("[]()").split(",")]
                return [int(x) if x.lstrip('-').isdigit() else x for x in items]
            return value if isinstance(value, list) else [value]

        if operator in ["Between", "NotBetween"]:
            if isinstance(value, str) and 'fncriteria:' in value:
                try:
                    parts = value.replace('fncriteria:', '').split(':')
                    return [float(p) if p.replace('.', '', 1).isdigit() else p for p in parts]
                except ValueError:
                    return [0, 0]
            return value if isinstance(value, list) and len(value) == 2 else [0, 0]

        # Strip embedded quotes from string values (DB stores values like 'USD')
        if isinstance(value, str):
            return value.strip("'\"")
        return value
