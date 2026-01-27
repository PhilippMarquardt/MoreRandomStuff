"""
High-level API for the Perspective Service.
"""

from typing import Dict, Optional

from .core.engine import PerspectiveEngine
from .config import DatabaseConfig


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
        verbose: bool = False,
        flatten_response: bool = False
    ) -> Dict:
        """
        Process request through perspective rules.

        All parameters are extracted from the request JSON:
        - perspective_configurations: {config_name: {perspective_id: [modifier_names]}}
        - position_weight_labels: Weight column names for positions (default: ['weight'])
        - lookthrough_weight_labels: Weight column names for lookthroughs (default: position_weight_labels)
        - system_version_timestamp: Optional timestamp for temporal DB queries

        Args:
            request: Full request JSON containing positions, configs, and parameters
            verbose: Whether to include extra metadata in output
            flatten_response: Whether to flatten output to columnar format

        Returns:
            Dict with perspective_configurations containing filtered/scaled weights
        """
        # Extract parameters from request JSON
        perspective_configs = request.get('perspective_configurations', {})
        position_weights = request.get('position_weight_labels', ['weight'])
        lookthrough_weights = request.get('lookthrough_weight_labels', position_weights)
        system_version_timestamp = request.get('system_version_timestamp')

        engine = PerspectiveEngine(
            self._connection_string,
            system_version_timestamp
        )
        return engine.process(
            input_json=request,
            perspective_configs=perspective_configs,
            position_weights=position_weights,
            lookthrough_weights=lookthrough_weights,
            verbose=verbose,
            flatten_response=flatten_response
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
