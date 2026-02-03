"""
Modifier dataclass for rule modifiers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from perspective_service.models.enums import ApplyTo, ModifierType, LogicalOperator


@dataclass
class Modifier:
    """Represents a rule modifier that can adjust rule behavior."""
    name: str
    apply_to: ApplyTo
    modifier_type: ModifierType
    criteria: Optional[Dict[str, Any]] = None
    rule_result_operator: Optional[LogicalOperator] = None
    required_columns: Dict[str, List[str]] = field(default_factory=dict)
    override_modifiers: List[str] = field(default_factory=list)
