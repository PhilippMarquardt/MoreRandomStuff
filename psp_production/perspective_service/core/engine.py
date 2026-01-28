"""
Perspective Engine - Main orchestrator for the perspective processing pipeline.

Implements the 9-step flow:
1. Load Perspective
2. Load Modifiers
3. Parse Criteria & Required Data
4. Load Reference Data
5. Build DataFrames (Position + Lookthrough)
6. Precompute Nested Criteria
7. Build Plan (keep/scale expressions)
8. Collect All (materialize)
9. Format Output
"""

import json
from typing import Dict, List, Optional, Any

import polars as pl

from .configuration_manager import ConfigurationManager
from .data_ingestion import DataIngestion
from .perspective_processor import PerspectiveProcessor
from .output_formatter import OutputFormatter
from .rule_evaluator import RuleEvaluator
from ..database.loaders.database_loader import DatabaseLoader
from ..models.rule import Rule
from ..utils.constants import INT_NULL


class PerspectiveEngine:
    """Main orchestrator for perspective processing."""

    def __init__(self,
                 connection_string: Optional[str] = None,
                 system_version_timestamp: Optional[str] = None):
        """
        Initialize the PerspectiveEngine.

        Args:
            connection_string: ODBC connection string for database access
            system_version_timestamp: Optional timestamp for temporal queries
        """
        self.system_version_timestamp = system_version_timestamp

        # Create DatabaseLoader if connection string provided
        if connection_string:
            self.db_loader = DatabaseLoader(connection_string)
        else:
            self.db_loader = None

        # Step 1 & 2: Load Perspectives and Modifiers
        self.config = ConfigurationManager(self.db_loader, system_version_timestamp)

    def process(self,
                input_json: Dict,
                perspective_configs: Dict[str, Dict[str, List[str]]],
                position_weights: List[str],
                lookthrough_weights: List[str],
                verbose: bool = False,
                flatten_response: bool = False) -> Dict:
        """
        Process input data through perspective rules.

        Args:
            input_json: Raw input data containing positions and lookthroughs
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            position_weights: List of weight column names for positions
            lookthrough_weights: List of weight column names for lookthroughs
            verbose: Whether to include removal summary in output
            flatten_response: Whether to flatten output to columnar format

        Returns:
            Formatted output dictionary
        """
        self._parse_custom_perspectives(input_json)

        weight_labels_map, all_pos_weights, all_lt_weights = DataIngestion.get_weight_labels(input_json)

        if not position_weights:
            position_weights = all_pos_weights
        if not lookthrough_weights:
            lookthrough_weights = all_lt_weights

        # Step 3: Determine required tables based on perspectives and modifiers
        required_tables = self._determine_required_tables(perspective_configs)

        # Step 4 & 5: Load reference data and build dataframes
        positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(
            input_json,
            required_tables,
            position_weights + lookthrough_weights,
            self.db_loader
        )

        if positions_lf.collect_schema().names() == []:
            return {"perspective_configurations": {}}

        # Get original containers before any filtering (for empty container handling)
        original_containers = positions_lf.select("container").unique().collect().to_series().to_list()

        # Step 6: Precompute nested criteria values
        precomputed_values = self._precompute_nested_criteria(
            positions_lf, lookthroughs_lf, perspective_configs
        )

        # Step 7: Build perspective plan (keep/scale expressions)
        processor = PerspectiveProcessor(self.config)
        positions_lf, lookthroughs_lf, metadata_map = processor.build_perspective_plan(
            positions_lf,
            lookthroughs_lf,
            perspective_configs,
            position_weights,
            lookthrough_weights,
            precomputed_values,
            weight_labels_map
        )

        # Step 8: Calculate
        if lookthroughs_lf is not None:
            positions_df, lookthroughs_df = pl.collect_all([
                positions_lf,
                lookthroughs_lf
            ])
        else:
            positions_df = positions_lf.collect()
            lookthroughs_df = pl.DataFrame()

        # Step 9: Format output
        return OutputFormatter.format_output(
            positions_df,
            lookthroughs_df,
            metadata_map,
            position_weights,
            lookthrough_weights,
            verbose,
            flatten_response,
            weight_labels_map,
            perspective_configs,
            original_containers
        )

    def _determine_required_tables(self,
                                   perspective_configs: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Determine which database tables are needed based on perspective rules and modifiers.

        Returns:
            Dict of {table_name: [column_names]} including position_data requirements
        """
        required_tables: Dict[str, List[str]] = {}

        # Collect all perspective IDs being used
        perspective_ids = set()
        all_modifier_names = set()

        for config_name, perspective_map in perspective_configs.items():
            for perspective_id, modifier_names in perspective_map.items():
                perspective_ids.add(int(perspective_id))
                if modifier_names:
                    all_modifier_names.update(modifier_names)

        # Add default modifiers
        all_modifier_names.update(self.config.default_modifiers)

        # Get required columns from perspectives (already normalized by ConfigurationManager)
        for pid in perspective_ids:
            if pid in self.config.required_columns_by_perspective:
                for table, columns in self.config.required_columns_by_perspective[pid].items():
                    table_lower = table.lower()
                    # Legacy alias: InstrumentInput -> position_data
                    if table_lower == 'instrumentinput':
                        table_lower = 'position_data'
                    if table_lower not in required_tables:
                        required_tables[table_lower] = []
                    for col in columns:
                        col_lower = col.lower()
                        if col_lower not in required_tables[table_lower]:
                            required_tables[table_lower].append(col_lower)

        # Get required columns from modifiers
        modifier_columns = self.config.get_modifier_required_columns(list(all_modifier_names))
        for table, columns in modifier_columns.items():
            table_lower = table.lower()
            # Legacy alias: InstrumentInput -> position_data
            if table_lower == 'instrumentinput':
                table_lower = 'position_data'
            if table_lower not in required_tables:
                required_tables[table_lower] = []
            for col in columns:
                col_lower = col.lower()
                if col_lower not in required_tables[table_lower]:
                    required_tables[table_lower].append(col_lower)

        # Handle edge cases: add join key requirements to position_data
        self._handle_requirements_edge_cases(required_tables)

        # Always require sub_portfolio_id in position_data
        if 'position_data' not in required_tables:
            required_tables['position_data'] = []
        if 'sub_portfolio_id' not in required_tables['position_data']:
            required_tables['position_data'].append('sub_portfolio_id')

        return required_tables

    def _handle_requirements_edge_cases(self, requirements: Dict[str, List[str]]) -> None:
        """
        Handle edge cases where reference table join keys must come from position_data.

        - PARENT_INSTRUMENT.instrument_id -> position_data.parent_instrument_id
        - ASSET_ALLOCATION_ANALYTICS_CATEGORY_V.analytics_category_id -> position_data.asset_allocation_id
        """
        parent_key = 'parent_instrument'
        asset_key = 'asset_allocation_analytics_category_v'
        pos_key = 'position_data'

        if parent_key in requirements or asset_key in requirements:
            if pos_key not in requirements:
                requirements[pos_key] = []

            # PARENT_INSTRUMENT: join key is instrument_id -> parent_instrument_id
            if parent_key in requirements:
                if 'instrument_id' in requirements[parent_key]:
                    if 'parent_instrument_id' not in requirements[pos_key]:
                        requirements[pos_key].append('parent_instrument_id')
                    requirements[parent_key].remove('instrument_id')
                if not requirements[parent_key]:
                    del requirements[parent_key]

            # ASSET_ALLOCATION_ANALYTICS_CATEGORY_V: join key is analytics_category_id -> asset_allocation_id
            if asset_key in requirements:
                if 'analytics_category_id' in requirements[asset_key]:
                    if 'asset_allocation_id' not in requirements[pos_key]:
                        requirements[pos_key].append('asset_allocation_id')
                    requirements[asset_key].remove('analytics_category_id')
                if not requirements[asset_key]:
                    del requirements[asset_key]

    def _precompute_nested_criteria(self,
                                    positions_lf: pl.LazyFrame,
                                    lookthroughs_lf: pl.LazyFrame,
                                    perspective_configs: Dict) -> Dict[str, Any]:
        """
        Precompute values for nested criteria (criteria within criteria).

        When an In/NotIn operator has a dict value (nested criteria), we:
        1. Evaluate the inner criteria to filter rows
        2. Extract the outer column's values from matching rows
        3. Cache the results for use during rule evaluation

        Returns:
            Dict of precomputed values keyed by json.dumps of the nested criteria
        """
        nested_queries = {}

        def find_nested_criteria(criteria):
            """Recursively find and queue nested criteria for precomputation."""
            if not criteria:
                return

            if "and" in criteria:
                for c in criteria["and"]:
                    find_nested_criteria(c)
            elif "or" in criteria:
                for c in criteria["or"]:
                    find_nested_criteria(c)
            elif "not" in criteria:
                find_nested_criteria(criteria["not"])
            else:
                value = criteria.get("value")
                operator = criteria.get("operator_type")

                # Check for nested criteria in In/NotIn operators
                if operator in ["In", "NotIn"] and isinstance(value, dict):
                    # Skip rule_result references - they must be handled at evaluation time
                    # because rule_result depends on the outcome of rule evaluation
                    if value.get("table_name", "").lower() == "rule_result":
                        return  # Skip precomputation for rule_result references

                    key = json.dumps(value, sort_keys=True)
                    target_column = criteria.get("column")

                    # Build query for nested criteria - evaluate the inner criteria
                    # and extract the outer column's values from matching rows
                    inner_expr = RuleEvaluator.evaluate(value, None, None)
                    query = (
                        positions_lf.filter(inner_expr)
                        .select(pl.col(target_column))
                        .drop_nulls()
                        .unique()
                    )
                    nested_queries[key] = query

        # Search all rules and modifiers for nested criteria
        for perspective_map in perspective_configs.values():
            for perspective_id in perspective_map.keys():
                # Check perspective rules
                for rule in self.config.perspectives.get(int(perspective_id), []):
                    if rule.criteria:
                        find_nested_criteria(rule.criteria)

                # Check modifiers for this perspective
                modifier_names = perspective_map[perspective_id] or []
                all_modifiers = list(set(modifier_names + self.config.default_modifiers))
                for modifier_name in all_modifiers:
                    if modifier_name in self.config.modifiers:
                        modifier = self.config.modifiers[modifier_name]
                        if modifier.criteria:
                            find_nested_criteria(modifier.criteria)

        # Execute all nested queries in one batch for efficiency
        if not nested_queries:
            return {}

        keys = list(nested_queries.keys())
        results = pl.collect_all(list(nested_queries.values()))

        return {
            key: result.to_series().to_list()
            for key, result in zip(keys, results)
        }

    def _parse_custom_perspectives(self, input_json: Dict) -> None:
        """
        Parse custom perspective rules from input JSON.

        Custom perspective IDs MUST be negative to distinguish them from
        database-loaded perspectives.

        Args:
            input_json: Raw input data that may contain 'custom_perspective_rules'
        """
        custom_rules = input_json.get('custom_perspective_rules', {})
        if not custom_rules:
            return

        # Validate all IDs are negative
        for pid in custom_rules.keys():
            if int(pid) > 0:
                raise ValueError(
                    "Custom Perspective Rule IDs MUST be negative to separate them from real Perspective IDs"
                )

        # Add each custom perspective
        for pid_str, perspective_data in custom_rules.items():
            pid = int(pid_str)

            # rules is REQUIRED
            if 'rules' not in perspective_data:
                raise ValueError(f"Custom perspective {pid} is missing required 'rules' field")
            rules = perspective_data['rules']

            if not rules:
                continue

            # Track required columns for this custom perspective
            required_columns: Dict[str, List[str]] = {}

            # Convert to internal Rule format
            internal_rules = []
            for i, rule in enumerate(rules):
                # criteria is REQUIRED
                if 'criteria' not in rule:
                    raise ValueError(f"Custom perspective {pid} rule {i} is missing required 'criteria' field")
                criteria = rule['criteria']

                # apply_to is REQUIRED
                if 'apply_to' not in rule:
                    raise ValueError(f"Custom perspective {pid} rule {i} is missing required 'apply_to' field")

                # Extract required_columns for tracking, then remove from criteria
                # Normalize table and column names to lowercase
                if 'required_columns' in criteria:
                    for table, columns in criteria['required_columns'].items():
                        table_lower = table.lower()
                        if table_lower not in required_columns:
                            required_columns[table_lower] = []
                        for col in columns:
                            col_lower = col.lower()
                            if col_lower not in required_columns[table_lower]:
                                required_columns[table_lower].append(col_lower)

                # Remove required_columns metadata from criteria (not needed for evaluation)
                clean_criteria = {k: v for k, v in criteria.items()
                                  if k != 'required_columns'}

                is_scaling_rule = rule.get('is_scaling_rule', False)
                if is_scaling_rule and 'scale_factor' not in rule:
                    raise ValueError(f"Custom perspective {pid} rule {i} is a scaling rule but missing 'scale_factor'")

                internal_rules.append(Rule(
                    name=f"custom_rule_{pid}_{i}",
                    apply_to=rule['apply_to'],
                    criteria=clean_criteria,
                    condition_for_next_rule=rule.get('condition_for_next_rule'),
                    is_scaling_rule=is_scaling_rule,
                    scale_factor=(rule['scale_factor'] / 100) if is_scaling_rule else 1.0
                ))

            self.config.perspectives[pid] = internal_rules

            # Track required columns for this custom perspective
            if required_columns:
                self.config.required_columns_by_perspective[pid] = required_columns
