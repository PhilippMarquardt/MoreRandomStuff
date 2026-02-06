"""
Configuration Manager - Manages rules, modifiers, and perspective configurations.
"""

import json
from typing import Dict, List, Optional, Tuple

from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier
from perspective_service.models.enums import ApplyTo, ModifierType, LogicalOperator
from perspective_service.utils.supported_modifiers import SUPPORTED_MODIFIERS, DEFAULT_MODIFIERS
from perspective_service.utils.ttl_cache import ttl_cache
from perspective_service.database.loaders.database_loader import DatabaseLoader, DatabaseLoadError


class ConfigurationManager:
    """Manages rules, modifiers, and perspective configurations."""

    def __init__(self, db_loader: Optional[DatabaseLoader] = None):
        """
        Initialize ConfigurationManager.

        Args:
            db_loader: DatabaseLoader instance for loading perspectives
        """
        self._db_loader = db_loader

        # Non-cached (hardcoded, never changes)
        self.modifiers: Dict[str, Modifier] = {}
        self.default_modifiers: List[str] = list(DEFAULT_MODIFIERS)
        self.modifier_overrides: Dict[str, List[str]] = {}

        self._load_hardcoded_modifiers()

    @ttl_cache(ttl=300)
    def _load_perspectives(self) -> Tuple[Dict[int, List[Rule]], Dict[int, Dict[str, List[str]]]]:
        """Load and parse perspectives from DB with TTL caching."""
        if self._db_loader is None:
            return {}, {}

        db_perspectives = self._db_loader.load_perspectives()
        return self._parse_db_perspectives(db_perspectives)

    @property
    def perspectives(self) -> Dict[int, List[Rule]]:
        """Get perspectives dict (TTL-cached, refreshes automatically)."""
        return self._load_perspectives()[0]

    @property
    def required_columns_by_perspective(self) -> Dict[int, Dict[str, List[str]]]:
        """Get required columns by perspective (TTL-cached, refreshes automatically)."""
        return self._load_perspectives()[1]

    def _parse_db_perspectives(
        self, db_perspectives: Dict[int, Dict]
    ) -> Tuple[Dict[int, List[Rule]], Dict[int, Dict[str, List[str]]]]:
        """Parse perspectives from database format. Returns (perspectives, required_columns)."""
        perspectives: Dict[int, List[Rule]] = {}
        all_required_columns: Dict[int, Dict[str, List[str]]] = {}

        for perspective_id, p_def in db_perspectives.items():
            if not p_def.get('is_active', True):
                continue
            if not p_def.get('is_supported', True):
                continue

            rules = []
            required_columns = {}

            for idx, rule_def in enumerate(p_def.get('rules', [])):
                criteria = self._parse_criteria(rule_def.get('criteria', {}))

                # These are the required tabe/columns to get the criteria running
                if 'required_columns' in criteria:
                    req_cols = criteria['required_columns']
                    if isinstance(req_cols, str):
                        req_cols = json.loads(req_cols)
                    self._update_required_columns(required_columns, req_cols)

                cond = (rule_def.get("condition_for_next_rule") or "").lower() or None
                rule = Rule(
                    name=f"rule_{idx}",
                    apply_to=ApplyTo(rule_def.get("apply_to", "both")),
                    criteria=self._clean_criteria(criteria),
                    condition_for_next_rule=LogicalOperator(cond) if cond else None,
                    is_scaling_rule=bool(rule_def.get("is_scaling_rule", False)),
                    scale_factor=rule_def.get("scale_factor", 100.0) / 100.0
                )
                rules.append(rule)

            perspectives[perspective_id] = rules
            if required_columns:
                all_required_columns[perspective_id] = required_columns

        return perspectives, all_required_columns

    def _load_hardcoded_modifiers(self) -> None:
        """Load modifiers from hardcoded SUPPORTED_MODIFIERS dict."""
        for name, mod_def in SUPPORTED_MODIFIERS.items():
            raw_op = mod_def.get('rule_result_operator')
            op = str(raw_op).lower() if raw_op else None
            criteria = mod_def.get('criteria')
            required_columns = mod_def.get('required_columns', {})
            override_modifiers = mod_def.get('override_modifiers', [])
            modifier = Modifier(
                name=name,
                apply_to=ApplyTo(str(mod_def.get('apply_to', 'both'))),
                modifier_type=ModifierType(str(mod_def.get('type', 'PreProcessing'))),
                criteria=dict(criteria) if isinstance(criteria, dict) else None,
                rule_result_operator=LogicalOperator(op) if op else None,
                required_columns={str(k): list(map(str, v)) for k, v in required_columns.items()} if isinstance(required_columns, dict) else {},  # type: ignore[union-attr]
                override_modifiers=list(override_modifiers) if isinstance(override_modifiers, list) else []
            )
            self.modifiers[name] = modifier

            # Build override map
            if modifier.override_modifiers:
                self.modifier_overrides[name] = modifier.override_modifiers

    def _parse_criteria(self, criteria):
        """Parse criteria from string or dict."""
        if isinstance(criteria, str):
            return json.loads(criteria)
        return criteria

    def _clean_criteria(self, criteria):
        """Remove metadata from criteria."""
        if not isinstance(criteria, dict):
            return criteria
        return {k: v for k, v in criteria.items() if k != 'required_columns'}

    def _update_required_columns(self, required_columns: Dict, new_columns: Dict):
        """Update required columns dictionary. Normalizes table and column names to lowercase."""
        for table, columns in new_columns.items():
            table_lower = table.lower()
            if table_lower not in required_columns:
                required_columns[table_lower] = []
            for col in columns:
                col_lower = col.lower()
                if col_lower not in required_columns[table_lower]:
                    required_columns[table_lower].append(col_lower)

    def get_modifier_required_columns(self, modifier_names: List[str]) -> Dict[str, List[str]]:
        """
        Get required columns for a list of modifiers.
        Normalizes table and column names to lowercase.

        Args:
            modifier_names: List of modifier names

        Returns:
            Dict of {table_name: [column_names]} with lowercase keys
        """
        required = {}
        for name in modifier_names:
            if name in self.modifiers:
                modifier = self.modifiers[name]
                for table, columns in modifier.required_columns.items():
                    table_lower = table.lower()
                    if table_lower not in required:
                        required[table_lower] = []
                    for col in columns:
                        col_lower = col.lower()
                        if col_lower not in required[table_lower]:
                            required[table_lower].append(col_lower)
        return required
