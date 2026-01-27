"""
Modifier dataclass for rule modifiers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# TODO: Should use enum
@dataclass
class Modifier:
    """Represents a rule modifier that can adjust rule behavior."""
    name: str
    apply_to: str  # 'position', 'lookthrough', or 'both'
    modifier_type: str  # 'PreProcessing', 'PostProcessing', or 'Scaling'
    criteria: Optional[Dict[str, Any]] = None
    rule_result_operator: Optional[str] = None  # 'and' or 'or'
    required_columns: Dict[str, List[str]] = field(default_factory=dict)
    override_modifiers: List[str] = field(default_factory=list)
