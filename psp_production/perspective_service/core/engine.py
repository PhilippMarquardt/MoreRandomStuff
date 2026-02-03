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
from typing import Dict, List, Optional, Any, Tuple

import polars as pl

from perspective_service.core.configuration_manager import ConfigurationManager
from perspective_service.core.data_ingestion import DataIngestion
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.core.output_formatter import OutputFormatter
from perspective_service.core.rule_evaluator import RuleEvaluator
from perspective_service.database.loaders.database_loader import DatabaseLoader
from perspective_service.models.rule import Rule
from perspective_service.models.enums import ApplyTo, LogicalOperator
from perspective_service.utils.constants import INT_NULL


class PerspectiveEngine:
    """Main orchestrator for perspective processing."""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the PerspectiveEngine.

        Args:
            connection_string: ODBC connection string for database access
        """
        # Create DatabaseLoader if connection string provided
        if connection_string:
            self.db_loader = DatabaseLoader(connection_string)
        else:
            self.db_loader = None

        # Load Perspectives and Modifiers
        self.config = ConfigurationManager(self.db_loader)

    def process(self,
                input_json: Dict,
                return_raw_dataframes: bool = False) -> Dict:
        """
        Process input data through perspective rules (JSON input path).

        Args:
            input_json: Raw input data containing positions, lookthroughs, and perspective_configurations
            return_raw_dataframes: If True, return raw DataFrames instead of formatted output

        Returns:
            Formatted output dictionary, or raw DataFrames dict if return_raw_dataframes=True
        """
        perspective_configs = input_json.get('perspective_configurations', {})
        weight_labels_map = DataIngestion.get_weight_labels(input_json)

        # Build dataframes from JSON
        positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(
            input_json,
            weight_labels_map
        )

        if positions_lf.collect_schema().names() == []:
            return {"perspective_configurations": {}}

        return self.process_dataframes(
            positions_lf=positions_lf,
            weight_labels_map=weight_labels_map,
            perspective_configs=perspective_configs,
            lookthroughs_lf=lookthroughs_lf,
            effective_date=input_json.get('ed'),
            system_version_timestamp=input_json.get('system_version_timestamp'),
            custom_perspective_rules=input_json.get('custom_perspective_rules'),
            return_raw_dataframes=return_raw_dataframes
        )

    def get_requirements(self, input_json: Dict) -> Dict[str, List[str]]:
        """
        Get required database tables/columns for the given request (thread-safe).

        Call this BEFORE process() to know what position data columns to fetch.
        This method does not modify engine state.

        Args:
            input_json: Request JSON (needs perspective_configurations, optionally custom_perspective_rules)

        Returns:
            Dict of {table_name: [column_names]}
        """
        custom_required_cols: Dict[int, Dict[str, List[str]]] = {}
        if input_json.get('custom_perspective_rules'):
            _, custom_required_cols = self._parse_custom_rules(input_json['custom_perspective_rules'])

        perspective_configs = input_json.get('perspective_configurations', {})
        return self._determine_required_tables(perspective_configs, custom_required_cols)

    def process_dataframes(
        self,
        positions_lf: pl.LazyFrame,
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]],
        perspective_configs: Dict[str, Dict[str, List[str]]],
        lookthroughs_lf: Optional[pl.LazyFrame] = None,
        effective_date: Optional[str] = None,
        system_version_timestamp: Optional[str] = None,
        custom_perspective_rules: Optional[Dict] = None,
        return_raw_dataframes: bool = False
    ) -> Dict:
        """
        Process pre-loaded DataFrames through perspective rules.

        Args:
            positions_lf: LazyFrame of positions (will be normalized)
            weight_labels_map: {container_name: (pos_weights, lt_weights)}
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            lookthroughs_lf: Optional LazyFrame of lookthroughs
            effective_date: For DB reference joins (required if DB connected and rules need ref data)
            system_version_timestamp: Optional timestamp for temporal reference data queries
            custom_perspective_rules: Optional {pid: {rules: [...]}} dict
            return_raw_dataframes: If True, return raw DataFrames instead of formatted output

        Returns:
            Formatted output dictionary, or raw DataFrames dict if return_raw_dataframes=True
        """
        # Parse custom perspectives (thread-safe: returns dicts, doesn't store on self)
        custom_perspectives: Dict[int, List[Rule]] = {}
        custom_required_columns: Dict[int, Dict[str, List[str]]] = {}
        if custom_perspective_rules:
            custom_perspectives, custom_required_columns = self._parse_custom_rules(custom_perspective_rules)

        # Normalize provided DataFrames
        all_pos, all_lt = DataIngestion.get_all_weights(weight_labels_map)
        all_weights = list(set(all_pos + all_lt))
        positions_lf, lookthroughs_lf = DataIngestion.normalize_dataframes(
            positions_lf, lookthroughs_lf, all_weights
        )

        if positions_lf.collect_schema().names() == []:
            return {"perspective_configurations": {}}

        # Join reference data if effective_date provided and DB connected
        if effective_date and self.db_loader:
            required_tables = self._determine_required_tables(perspective_configs, custom_required_columns)
            if required_tables:
                positions_lf, lookthroughs_lf = DataIngestion.join_reference_data(
                    positions_lf,
                    lookthroughs_lf,
                    required_tables,
                    self.db_loader,
                    system_version_timestamp,
                    effective_date
                )

        if positions_lf.collect_schema().names() == []:
            return {"perspective_configurations": {}}

        return self._process_core(
            positions_lf, lookthroughs_lf, perspective_configs,
            weight_labels_map, return_raw_dataframes, custom_perspectives
        )

    def _process_core(
        self,
        positions_lf: pl.LazyFrame,
        lookthroughs_lf: Optional[pl.LazyFrame],
        perspective_configs: Dict[str, Dict[str, List[str]]],
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]],
        return_raw_dataframes: bool = False,
        custom_perspectives: Optional[Dict[int, List[Rule]]] = None
    ) -> Dict:
        """
        Common processing core - precompute, build plan, collect, format.

        Called by both process() and process_dataframes() after data preparation.
        """
        # Build perspective plan (keep/scale expressions)
        processor = PerspectiveProcessor(self.config, custom_perspectives or {})
        positions_lf, lookthroughs_lf, scale_factors_lf = processor.build_perspective_plan(
            positions_lf,
            lookthroughs_lf,
            perspective_configs,
            weight_labels_map
        )

        # Return raw DataFrames if requested
        if return_raw_dataframes:
            return {
                "positions": positions_lf,
                "lookthroughs": lookthroughs_lf,
                "scale_factors": scale_factors_lf,
            }
        
        # Collect all
        to_collect = [positions_lf]
        has_lt = lookthroughs_lf is not None
        has_sf = scale_factors_lf is not None

        if has_lt:
            to_collect.append(lookthroughs_lf)
        if has_sf:
            to_collect.append(scale_factors_lf)

        collected = pl.collect_all(to_collect)

        positions_df = collected[0]
        lookthroughs_df = collected[1] if has_lt else pl.DataFrame()
        scale_factors_df = collected[-1] if has_sf else None


        # Format output
        return OutputFormatter.format_output(
            positions_df,
            lookthroughs_df,
            perspective_configs,
            weight_labels_map,
            scale_factors_df
        )

    def _determine_required_tables(
        self,
        perspective_configs: Dict[str, Dict[str, List[str]]],
        custom_required_columns: Optional[Dict[int, Dict[str, List[str]]]] = None
    ) -> Dict[str, List[str]]:
        """
        Determine which database tables are needed based on perspective rules and modifiers.

        Args:
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            custom_required_columns: Optional custom perspective required columns (thread-safe)

        Returns:
            Dict of {table_name: [column_names]} including position_data requirements
        """
        required_tables: Dict[str, List[str]] = {}
        custom_required_columns = custom_required_columns or {}

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

        # Get required columns from DB perspectives (already normalized by ConfigurationManager)
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

        # Get required columns from custom perspectives (passed as parameter, thread-safe)
        for pid in perspective_ids:
            if pid in custom_required_columns:
                for table, columns in custom_required_columns[pid].items():
                    table_lower = table.lower()
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

    def _parse_custom_rules(
        self,
        custom_perspective_rules: Dict
    ) -> Tuple[Dict[int, List[Rule]], Dict[int, Dict[str, List[str]]]]:
        """
        Parse custom perspective rules (thread-safe: returns dicts, doesn't store on self).

        Custom perspective IDs MUST be negative to distinguish them from
        database-loaded perspectives.

        Args:
            custom_perspective_rules: Dict of {pid_str: {rules: [...]}}

        Returns:
            Tuple of (perspectives_dict, required_columns_dict)
        """
        perspectives: Dict[int, List[Rule]] = {}
        all_required_columns: Dict[int, Dict[str, List[str]]] = {}

        if not custom_perspective_rules:
            return perspectives, all_required_columns

        # Validate all IDs are negative
        for pid in custom_perspective_rules.keys():
            if int(pid) > 0:
                raise ValueError(
                    "Custom Perspective Rule IDs MUST be negative to separate them from real Perspective IDs"
                )

        # Parse each custom perspective
        for pid_str, perspective_data in custom_perspective_rules.items():
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
                    apply_to=ApplyTo(rule['apply_to']),
                    criteria=clean_criteria,
                    condition_for_next_rule=LogicalOperator(rule['condition_for_next_rule'].lower()) if rule.get('condition_for_next_rule') else None,
                    is_scaling_rule=is_scaling_rule,
                    scale_factor=(rule['scale_factor'] / 100) if is_scaling_rule else 1.0
                ))

            perspectives[pid] = internal_rules

            # Track required columns for this custom perspective
            if required_columns:
                all_required_columns[pid] = required_columns

        return perspectives, all_required_columns
