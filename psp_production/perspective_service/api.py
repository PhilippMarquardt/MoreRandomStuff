"""
High-level API for the Perspective Service.
"""

from typing import Dict, List, Optional, Tuple

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.config import DatabaseConfig


class PerspectiveService:
    """
    Main API for perspective processing.

    Usage:
        service = PerspectiveService(connection_string="Driver={...};Server=...;...")

        result = service.process(request_json)

    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        config: Optional[DatabaseConfig] = None
    ):
        """
        Initialize the PerspectiveService.

        Args:
            connection_string: ODBC connection string for database access
            config: DatabaseConfig object (alternative to connection_string)
        """
        if connection_string:
            self._connection_string = connection_string
        elif config:
            self._connection_string = config.get_odbc_connection_string()
        else:
            self._connection_string = None

    def process(
        self,
        request: Dict,
        return_raw_dataframes: bool = False
    ) -> Dict:
        """
        Process request through perspective rules.

        All parameters are extracted from the request JSON:
        - perspective_configurations: {config_name: {perspective_id: [modifier_names]}}
        - position_weight_labels / lookthrough_weight_labels: extracted per-container from JSON
        - system_version_timestamp: Optional timestamp for temporal DB queries

        Args:
            request: Full request JSON containing positions, configs, and parameters
            return_raw_dataframes: If True, return raw DataFrames instead of formatted output

        Returns:
            Dict with perspective_configurations, or raw DataFrames dict if return_raw_dataframes=True
        """
        perspective_configs = request.get('perspective_configurations', {})
        system_version_timestamp = request.get('system_version_timestamp')

        engine = PerspectiveEngine(
            self._connection_string,
            system_version_timestamp
        )
        return engine.process(
            input_json=request,
            perspective_configs=perspective_configs,
            return_raw_dataframes=return_raw_dataframes
        )

    def process_dataframes(
        self,
        positions: pl.LazyFrame,
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]],
        perspective_configs: Dict[str, Dict[str, List[str]]],
        lookthroughs: Optional[pl.LazyFrame] = None,
        effective_date: Optional[str] = None,
        custom_perspective_rules: Optional[Dict] = None,
        system_version_timestamp: Optional[str] = None,
        return_raw_dataframes: bool = False
    ) -> Dict:
        """
        Process pre-loaded DataFrames through perspective rules.

        Use this when you have DataFrames from an external source (not JSON).

        Args:
            positions: LazyFrame of positions (will be normalized)
            weight_labels_map: {container_name: (pos_weight_labels, lt_weight_labels)}
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            lookthroughs: Optional LazyFrame of lookthroughs
            effective_date: For DB reference joins (if rules need reference data)
            custom_perspective_rules: Optional {pid: {rules: [...]}} dict
            system_version_timestamp: Optional timestamp for temporal DB queries
            return_raw_dataframes: If True, return raw DataFrames instead of formatted output

        Returns:
            Dict with perspective_configurations, or raw DataFrames dict if return_raw_dataframes=True
        """
        engine = PerspectiveEngine(
            self._connection_string,
            system_version_timestamp
        )
        return engine.process_dataframes(
            positions_lf=positions,
            weight_labels_map=weight_labels_map,
            perspective_configs=perspective_configs,
            lookthroughs_lf=lookthroughs,
            effective_date=effective_date,
            custom_perspective_rules=custom_perspective_rules,
            return_raw_dataframes=return_raw_dataframes
        )

    def get_requirements(self, request: Dict) -> Dict[str, list]:
        """
        Get required database tables and columns for the given request.

        Args:
            request: Request JSON (needs perspective_configurations and optionally custom_perspective_rules)

        Returns:
            Dict of {table_name: [column_names]} that will be queried
        """
        perspective_configs = request.get('perspective_configurations', {})
        system_version_timestamp = request.get('system_version_timestamp')

        engine = PerspectiveEngine(
            self._connection_string,
            system_version_timestamp
        )
        engine._parse_custom_perspectives(request)
        return engine._determine_required_tables(perspective_configs)
