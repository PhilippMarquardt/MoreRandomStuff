"""
High-level API for the Perspective Service.
"""

from typing import Dict, List, Optional

from .core.engine import PerspectiveEngine
from .config import DatabaseConfig, load_config


class PerspectiveService:
    """
    Main API for perspective processing.

    Usage:
        # With connection string
        service = PerspectiveService(connection_string="Driver={...};Server=...;...")
        result = service.process(input_json, perspective_configs, position_weights)

        # With DatabaseConfig
        config = DatabaseConfig(server="localhost", database="mydb")
        service = PerspectiveService(config=config)

        # Without database (for testing with custom perspectives only)
        service = PerspectiveService()
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        config: Optional[DatabaseConfig] = None,
        system_version_timestamp: Optional[str] = None
    ):
        """
        Initialize the PerspectiveService.

        Args:
            connection_string: ODBC connection string for database access
            config: DatabaseConfig object (alternative to connection_string)
            system_version_timestamp: Optional timestamp for temporal DB queries
        """
        if connection_string:
            self._connection_string = connection_string
        elif config:
            self._connection_string = config.get_odbc_connection_string()
        else:
            self._connection_string = None

        self._system_version_timestamp = system_version_timestamp

    def process(
        self,
        input_json: Dict,
        perspective_configs: Dict[str, Dict[str, List[str]]],
        position_weights: List[str],
        lookthrough_weights: Optional[List[str]] = None,
        verbose: bool = False,
        flatten_response: bool = False
    ) -> Dict:
        """
        Process input data through perspective rules.

        Args:
            input_json: Input data containing positions and lookthroughs
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            position_weights: Weight column names for positions
            lookthrough_weights: Weight column names for lookthroughs (defaults to position_weights)
            verbose: Whether to include extra metadata in output
            flatten_response: Whether to flatten output to columnar format

        Returns:
            Dict with perspective_configurations containing filtered/scaled weights
        """
        if lookthrough_weights is None:
            lookthrough_weights = position_weights

        engine = PerspectiveEngine(
            self._connection_string,
            self._system_version_timestamp
        )
        return engine.process(
            input_json,
            perspective_configs,
            position_weights,
            lookthrough_weights,
            verbose,
            flatten_response
        )

    def get_requirements(
        self,
        input_json: Dict,
        perspective_configs: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        """
        Get required database tables and columns for the given perspectives.

        Use this to understand what reference data will be fetched from the database.

        Args:
            input_json: Input data (needed for custom_perspective_rules)
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}

        Returns:
            Dict of {table_name: [column_names]} that will be queried
        """
        engine = PerspectiveEngine(
            self._connection_string,
            self._system_version_timestamp
        )
        # Parse custom perspectives first (they may have required_columns)
        engine._parse_custom_perspectives(input_json)
        return engine._determine_required_tables(perspective_configs)
