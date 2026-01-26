"""
Perspective Service - A Polars-based perspective processing engine.

Usage:
    from perspective_service import PerspectiveService

    service = PerspectiveService(connection_string="...")
    result = service.process(input_json, perspective_configs, weights)
"""

from .api import PerspectiveService
from .config import DatabaseConfig, load_config
from .core.engine import PerspectiveEngine

__all__ = [
    'PerspectiveService',
    'PerspectiveEngine',
    'DatabaseConfig',
    'load_config',
]

__version__ = "0.1.0"
