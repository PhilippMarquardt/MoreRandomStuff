"""
Rule dataclass for perspective rules.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Rule:
    """Represents a single filtering or scaling rule."""
    name: str
    apply_to: str  # 'position', 'lookthrough', or 'both'
    criteria: Optional[Dict[str, Any]] = None
    condition_for_next_rule: Optional[str] = None  # 'And' or 'Or'
    is_scaling_rule: bool = False
    scale_factor: float = 1.0
