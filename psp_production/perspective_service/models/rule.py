"""
Rule dataclass for perspective rules.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from perspective_service.models.enums import ApplyTo, LogicalOperator


@dataclass
class Rule:
    """Represents a single filtering or scaling rule."""
    name: str
    apply_to: ApplyTo
    criteria: Optional[Dict[str, Any]] = None
    condition_for_next_rule: Optional[LogicalOperator] = None
    is_scaling_rule: bool = False
    scale_factor: float = 1.0
