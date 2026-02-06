"""
Perspective Service - A Polars-based perspective processing engine.

Usage:
    from perspective_service import PerspectiveEngine

    engine = PerspectiveEngine(connection_string="...")
    result = engine.process(input_json)
"""

from perspective_service.config import CONNECTION_STRING
from perspective_service.core.engine import PerspectiveEngine

__all__ = [
    'PerspectiveEngine',
    'CONNECTION_STRING',
]

__version__ = "0.1.0"
