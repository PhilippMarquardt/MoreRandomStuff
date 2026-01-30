"""
Perspective Processor - Processes data through perspective rules and modifiers.
"""

from typing import Dict, List, Tuple, Optional

import polars as pl

from perspective_service.core.configuration_manager import ConfigurationManager
from perspective_service.core.rule_evaluator import RuleEvaluator
from perspective_service.models.enums import Container, RecordType, ApplyTo


class PerspectiveProcessor:
    """Processes data through perspective rules and modifiers."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager

    def build_perspective_plan(self,
                               positions_lf: pl.LazyFrame,
                               lookthroughs_lf: pl.LazyFrame,
                               perspective_configs: Dict,
                               precomputed_values: Dict,
                               weight_labels_map: Dict[str, Tuple[List[str], List[str]]]) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame], Dict, Optional[pl.LazyFrame]]:
        """
        Build execution plan for all perspectives.

        Returns:
            Tuple of (processed_positions, processed_lookthroughs, metadata_map, scale_factors_lf)
        """
        # Initialize collections for expressions
        factor_expressions_pos = []
        factor_expressions_lt = []

        metadata_map = {}
        has_lookthroughs = lookthroughs_lf is not None

        # Process each perspective configuration
        for config_name, perspective_map in perspective_configs.items():
            metadata_map[config_name] = {}
            perspective_ids = sorted([int(k) for k in perspective_map.keys()])

            for perspective_id in perspective_ids:
                # Create unique factor column name for this perspective
                column_name = f"f_{config_name}_{perspective_id}"
                metadata_map[config_name][perspective_id] = column_name

                # Get modifiers for this perspective
                modifier_names = perspective_map.get(str(perspective_id)) or []
                active_modifiers = self._filter_overridden_modifiers(modifier_names)

                # Build expressions - same expression works for both positions and lookthroughs
                # because it uses position_type column to determine applicability
                keep_expr = self._build_keep_expression(
                    perspective_id, active_modifiers, precomputed_values
                )
                scale_expr = self._build_scale_expression(
                    perspective_id, precomputed_values
                )
                factor_expressions_pos.append(
                    pl.when(keep_expr)
                    .then(scale_expr)
                    .otherwise(pl.lit(None))
                    .alias(column_name)
                )

                # Use same expressions for lookthroughs - they check position_type internally
                if has_lookthroughs:
                    factor_expressions_lt.append(
                        pl.when(keep_expr)
                        .then(scale_expr)
                        .otherwise(pl.lit(None))
                        .alias(column_name)
                    )

        # Apply factor expressions
        positions_lf = positions_lf.with_columns(factor_expressions_pos)
        if has_lookthroughs:
            lookthroughs_lf = lookthroughs_lf.with_columns(factor_expressions_lt)

            # Synchronize lookthroughs with parent positions -> Remove children if parent dead
            all_columns = [c for m in metadata_map.values() for c in m.values()]
            lookthroughs_lf = self._synchronize_lookthroughs(
                lookthroughs_lf, positions_lf, all_columns
            )

        # Handle rescaling if needed
        positions_lf, lookthroughs_lf = self._apply_rescaling(
            positions_lf,
            lookthroughs_lf,
            perspective_configs,
            metadata_map,
            has_lookthroughs,
            precomputed_values,
            weight_labels_map
        )

        # Build weight columns (extracted to helper)
        positions_lf, lookthroughs_lf = self._build_weight_columns(
            positions_lf,
            lookthroughs_lf,
            metadata_map,
            weight_labels_map
        )

        # Build scale factors
        scale_factors_lf = self._build_scale_factors(
            positions_lf,
            lookthroughs_lf,
            perspective_configs,
            metadata_map,
            weight_labels_map
        )

        return positions_lf, lookthroughs_lf if has_lookthroughs else None, metadata_map, scale_factors_lf

    def _build_keep_expression(self,
                               perspective_id: int,
                               modifier_names: List[str],
                               precomputed_values: Dict) -> pl.Expr:
        """Build expression to determine if a row should be kept."""
        # Start with preprocessing modifiers
        expr = pl.lit(True)
        for modifier_name in modifier_names:
            modifier = self.config.modifiers.get(modifier_name)
            if modifier and modifier.modifier_type == "PreProcessing":
                applicable_expr = self._get_applicable_expr(modifier.apply_to)
                criteria_expr = RuleEvaluator.evaluate(
                    modifier.criteria, perspective_id, precomputed_values
                )
                # If applicable, EXCLUDE matching rows; if not applicable, keep all (True)
                expr &= pl.when(applicable_expr).then(~criteria_expr).otherwise(pl.lit(True))

        # Apply perspective rules
        rule_expr = self._build_rule_expression(perspective_id, precomputed_values)

        # Apply postprocessing modifiers
        # PostProcessing modifiers may have rule_result references that need rule_expr context
        for modifier_name in modifier_names:
            modifier = self.config.modifiers.get(modifier_name)
            if modifier and modifier.modifier_type == "PostProcessing":
                applicable_expr = self._get_applicable_expr(modifier.apply_to)
                # Pass preproc_keep & rule_expr so rule_result can't see rows removed by preprocessing
                # (Legacy physically removed these rows; we match that by including expr in the context)
                savior_expr = RuleEvaluator.evaluate(
                    modifier.criteria, perspective_id, precomputed_values,
                    rule_expr=expr & rule_expr  # expr is preproc_keep
                )
                # If applicable, apply savior logic; if not applicable, don't change anything (False for or, True for and)
                savior_combined = pl.when(applicable_expr).then(savior_expr).otherwise(
                    pl.lit(False) if modifier.rule_result_operator == "or" else pl.lit(True)
                )
                if modifier.rule_result_operator == "or":
                    rule_expr = rule_expr | savior_combined
                else:
                    rule_expr = rule_expr & savior_combined

        return expr & rule_expr

    def _build_rule_expression(self,
                               perspective_id: int,
                               precomputed_values: Dict) -> pl.Expr:
        """Build expression from perspective rules."""
        rules = self.config.perspectives.get(perspective_id, [])
        rule_expr = None

        for idx, rule in enumerate(rules):
            if rule.is_scaling_rule:
                continue

            applicable_expr = self._get_applicable_expr(rule.apply_to)
            criteria_expr = RuleEvaluator.evaluate(
                rule.criteria, perspective_id, precomputed_values
            )

            # If applicable, use criteria; if not applicable, pass (True)
            current_expr = pl.when(applicable_expr).then(criteria_expr).otherwise(pl.lit(True))

            if rule_expr is None:
                rule_expr = current_expr
            else:
                previous_rule = rules[idx - 1]
                if previous_rule.condition_for_next_rule == "or":
                    rule_expr = rule_expr | current_expr
                else:
                    rule_expr = rule_expr & current_expr

        return rule_expr if rule_expr is not None else pl.lit(True)

    def _build_scale_expression(self,
                                perspective_id: int,
                                precomputed_values: Dict) -> pl.Expr:
        """Build scaling factor expression."""
        scale_factor = pl.lit(1.0)

        for rule in self.config.perspectives.get(perspective_id, []):
            if rule.is_scaling_rule:
                applicable_expr = self._get_applicable_expr(rule.apply_to)
                criteria_expr = RuleEvaluator.evaluate(
                    rule.criteria, perspective_id, precomputed_values
                )
                # Apply scale factor only when applicable AND criteria matches
                scale_factor = pl.when(applicable_expr & criteria_expr).then(
                    scale_factor * rule.scale_factor
                ).otherwise(scale_factor)

        return scale_factor

    def _synchronize_lookthroughs(self,
                                  lookthroughs_lf: pl.LazyFrame,
                                  positions_lf: pl.LazyFrame,
                                  factor_columns: List[str]) -> pl.LazyFrame:
        """Synchronize lookthrough factors with parent position factors."""
        # Get parent factors - aggregate by (instrument_id, sub_portfolio_id)
        # Legacy behavior: if ANY parent with same (instrument_id, sub_portfolio_id) fails,
        # the lookthrough should be removed. So we check if ANY factor is NULL.
        parent_factors = positions_lf.group_by(["instrument_id", "sub_portfolio_id"]).agg([
            pl.when(pl.col(col).is_null().any())
            .then(pl.lit(None))  # ANY parent failed → NULL (remove lookthrough)
            .otherwise(pl.col(col).first())  # ALL parents passed → keep factor
            .alias(col)
            for col in factor_columns
        ])

        # Rename columns for joining
        rename_map = {col: f"parent_{col}" for col in factor_columns}
        parent_factors = parent_factors.rename(rename_map)

        # Join with lookthroughs
        synchronized = lookthroughs_lf.join(
            parent_factors,
            left_on=["parent_instrument_id", "sub_portfolio_id"],
            right_on=["instrument_id", "sub_portfolio_id"],
            how="left"
        )

        # If parent removed, remove ALL children
        final_expressions = [
            pl.when(pl.col(f"parent_{col}").is_null())
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col)
            for col in factor_columns
        ]

        result = synchronized.with_columns(final_expressions)

        return result

    def _apply_rescaling(
        self,
        positions_lf: pl.LazyFrame,
        lookthroughs_lf: Optional[pl.LazyFrame],
        perspective_configs: Dict,
        metadata_map: Dict,
        has_lookthroughs: bool,
        precomputed_values: Dict,
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]],
    ) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]:
        """
        Rescale (normalize) weights for the kept set.

        IMPORTANT: rescaling is done per CONTAINER (NOT per sub_portfolio_id).

        Positions:
        - Per container, per perspective fcol, per weight w:
            denom_w = sum_positions(w * fcol) + sum_essential_LT(w * fcol)  (if w exists on LT)
            factor_w = fcol / denom_w  (only when row is kept and denom_w != 0)
        - Stored as struct column: f"{fcol}_factors" with one field per weight label,
            because OutputFormatter reads factors via:
            pl.col(f"{fcol}_factors").struct.field(w)

        Lookthroughs:
        - Per (container, parent_instrument_id, record_type), per fcol, per weight:
            denom_w = sum_children(w * fcol) over matching parents only (and kept rows only)
            factor_w = fcol / denom_w (only if row kept AND parent matches AND denom_w != 0)
        """

        # -------------------------
        # 1) Determine rescale cols
        # -------------------------
        pos_rescale_cols: List[str] = []
        lt_rescale_cols: List[str] = []

        for config_name, pmap in perspective_configs.items():
            for pid in self._get_rescale_perspectives(pmap, "scale_holdings_to_100_percent"):
                pos_rescale_cols.append(metadata_map[config_name][pid])
            for pid in self._get_rescale_perspectives(pmap, "scale_lookthroughs_to_100_percent"):
                lt_rescale_cols.append(metadata_map[config_name][pid])

        if not pos_rescale_cols and not lt_rescale_cols:
            return positions_lf, lookthroughs_lf

        # -------------------------
        # 2) Resolve weight labels from weight_labels_map
        # -------------------------
        pos_schema = set(positions_lf.collect_schema().names())
        lt_schema = (
            set(lookthroughs_lf.collect_schema().names())
            if has_lookthroughs and lookthroughs_lf is not None
            else set()
        )

        # Union of all weight labels from weight_labels_map
        all_pos_weights = set()
        all_lt_weights = set()
        for _, (pw, lw) in weight_labels_map.items():
            all_pos_weights.update(pw or [])
            all_lt_weights.update(lw or [])

        pos_weights_valid = [w for w in all_pos_weights if w in pos_schema]
        lt_weights_valid = [w for w in all_lt_weights if w in lt_schema]

        # ===================
        # POSITION RESCALING (per container)
        # ===================
        if pos_rescale_cols and pos_weights_valid:
            group_keys = ["container"]  # TODO: Do we want subportfolioid here? Legacy does not do it
            pos_weights_in_lt = [w for w in pos_weights_valid if w in lt_schema]

            # 1) Position denoms: sum(w * fcol) per container
            pos_denom_exprs: List[pl.Expr] = []
            for fcol in pos_rescale_cols:
                for w in pos_weights_valid:
                    pos_denom_exprs.append(
                        (pl.col(w) * pl.col(fcol)).sum().alias(f"__pden__{fcol}__{w}")
                    )

            pos_denoms = positions_lf.group_by(group_keys).agg(pos_denom_exprs)

            # 2) Essential lookthrough denoms (only for weights that exist on LT)
            # LT's fcol is already null if parent failed (due to _synchronize_lookthroughs),
            # so filtering on fcol.is_not_null() naturally excludes orphaned lookthroughs.
            if has_lookthroughs and lookthroughs_lf is not None and pos_weights_in_lt:
                lt_denom_exprs: List[pl.Expr] = []
                for fcol in pos_rescale_cols:
                    for w in pos_weights_in_lt:
                        lt_denom_exprs.append(
                            (pl.col(w) * pl.col(fcol))  # Weight * factor (consistent with positions)
                            .filter(
                                (pl.col("record_type") == RecordType.ESSENTIAL_LOOKTHROUGHS)
                                & pl.col(fcol).is_not_null()  # LT's fcol is null if parent failed (due to sync)
                            )
                            .sum()
                            .alias(f"__ltden__{fcol}__{w}")
                        )

                lt_denoms = lookthroughs_lf.group_by(group_keys).agg(lt_denom_exprs)
                pos_denoms = pos_denoms.join(lt_denoms, on=group_keys, how="left")

            # 3) Join denoms to positions (by container only)
            positions_lf = positions_lf.join(pos_denoms, on=group_keys, how="left")

            # 4) Build struct factors per fcol
            struct_exprs: List[pl.Expr] = []
            for fcol in pos_rescale_cols:
                factor_fields: Dict[str, pl.Expr] = {}
                for w in pos_weights_valid:
                    denom = pl.col(f"__pden__{fcol}__{w}").fill_null(0.0)

                    # Add essential LT denom only if that weight exists on LT
                    if has_lookthroughs and w in lt_schema:
                        denom = denom + pl.col(f"__ltden__{fcol}__{w}").fill_null(0.0)

                    factor_fields[w] = (
                        pl.when(pl.col(fcol).is_not_null() & (denom != 0.0))
                        .then(pl.col(fcol) / denom)
                        .otherwise(pl.col(fcol))
                    )

                struct_exprs.append(pl.struct(**factor_fields).alias(f"{fcol}_factors"))

            positions_lf = positions_lf.with_columns(struct_exprs)

            # 5) Drop temp denom columns
            drop_cols = [f"__pden__{fcol}__{w}" for fcol in pos_rescale_cols for w in pos_weights_valid]
            if has_lookthroughs and lookthroughs_lf is not None and pos_weights_in_lt:
                drop_cols += [f"__ltden__{fcol}__{w}" for fcol in pos_rescale_cols for w in pos_weights_in_lt]
            positions_lf = positions_lf.drop(drop_cols)

        # ======================
        # LOOKTHROUGH RESCALING (per container + parent + record_type + sub_portfolio_id if exists)
        # ======================
        if has_lookthroughs and lookthroughs_lf is not None and lt_rescale_cols and lt_weights_valid:
            group_cols = ["container", "parent_instrument_id", "record_type", "sub_portfolio_id"]

            modifier = self.config.modifiers.get("scale_lookthroughs_to_100_percent")
            has_criteria = bool(modifier and modifier.criteria)

            # -------------------------
            # 1) Build match flags ONCE per parent_instrument_id
            #    Columns: __m_{fcol} (bool)
            # -------------------------
            fcols_for_matching: List[str] = []
            match_exprs: List[pl.Expr] = []

            if has_criteria:
                for config_name, pmap in perspective_configs.items():
                    for pid in self._get_rescale_perspectives(pmap, "scale_lookthroughs_to_100_percent"):
                        fcol = metadata_map[config_name][pid]
                        if fcol not in lt_schema:
                            continue
                        fcols_for_matching.append(fcol)
                        crit = RuleEvaluator.evaluate(modifier.criteria, pid, precomputed_values)
                        match_exprs.append(crit.cast(pl.Boolean).alias(f"__m_{fcol}"))

                if match_exprs:
                    matching_parents = (
                        positions_lf
                        .select([pl.col("instrument_id").alias("parent_instrument_id")] + match_exprs)
                        .group_by("parent_instrument_id")
                        .agg([pl.col(f"__m_{fcol}").any().alias(f"__m_{fcol}") for fcol in fcols_for_matching])
                    )

                    lookthroughs_lf = lookthroughs_lf.join(
                        matching_parents, on="parent_instrument_id", how="left"
                    ).with_columns([
                        pl.col(f"__m_{fcol}").fill_null(False) for fcol in fcols_for_matching
                    ])
            else:
                fcols_for_matching = list(lt_rescale_cols)
                lookthroughs_lf = lookthroughs_lf.with_columns([
                    pl.lit(True).alias(f"__m_{fcol}") for fcol in fcols_for_matching
                ])

            # -------------------------
            # 2) Compute ALL denoms with ONE group_by (no windows)
            #    denom = sum(w * fcol) over rows where parent matches AND row is kept
            # -------------------------
            denom_exprs: List[pl.Expr] = []
            for fcol in lt_rescale_cols:
                mcol = f"__m_{fcol}"
                for w in lt_weights_valid:
                    cond = pl.col(mcol) & pl.col(fcol).is_not_null()
                    denom_exprs.append(
                        (pl.col(w) * pl.col(fcol))
                        .filter(cond)
                        .sum()
                        .alias(f"__den__{fcol}__{w}")
                    )

            lt_denoms = lookthroughs_lf.group_by(group_cols).agg(denom_exprs)
            lookthroughs_lf = lookthroughs_lf.join(lt_denoms, on=group_cols, how="left")

            # -------------------------
            # 3) Build struct factors per fcol
            # -------------------------
            struct_exprs: List[pl.Expr] = []
            for fcol in lt_rescale_cols:
                mcol = f"__m_{fcol}"
                factor_fields: Dict[str, pl.Expr] = {}
                for w in lt_weights_valid:
                    denom = pl.col(f"__den__{fcol}__{w}").fill_null(0.0)
                    factor_fields[w] = (
                        pl.when(pl.col(mcol) & pl.col(fcol).is_not_null() & (denom != 0.0))
                        .then(pl.col(fcol) / denom)
                        .otherwise(pl.col(fcol))
                    )
                struct_exprs.append(pl.struct(**factor_fields).alias(f"{fcol}_factors"))

            lookthroughs_lf = lookthroughs_lf.with_columns(struct_exprs)

            # -------------------------
            # 4) Drop temp columns
            # -------------------------
            drop_cols = (
                [f"__m_{fcol}" for fcol in fcols_for_matching] +
                [f"__den__{fcol}__{w}" for fcol in lt_rescale_cols for w in lt_weights_valid]
            )
            lookthroughs_lf = lookthroughs_lf.drop(drop_cols)

        return positions_lf, lookthroughs_lf



    def _get_rescale_perspectives(self, perspective_map: Dict, modifier_key: str) -> List[int]:
        """Get perspective IDs that have a specific modifier."""
        result = []
        for perspective_id, modifiers in perspective_map.items():
            filtered_modifiers = self._filter_overridden_modifiers(modifiers or [])
            if modifier_key in filtered_modifiers:
                result.append(int(perspective_id))
        return result

    def _get_criteria_expr(self, modifier_name: str) -> Optional[pl.Expr]:
        """Get the raw criteria expression for a modifier."""
        modifier = self.config.modifiers.get(modifier_name)
        if modifier and modifier.criteria:
            return RuleEvaluator.evaluate(modifier.criteria)
        return None

    def _filter_overridden_modifiers(self, modifiers: List[str]) -> List[str]:
        """Filter out overridden modifiers."""
        final_set = set(modifiers + self.config.default_modifiers)

        for modifier in list(final_set):
            if modifier in self.config.modifier_overrides:
                for override in self.config.modifier_overrides[modifier]:
                    final_set.discard(override)

        return list(final_set)

    def _get_applicable_expr(self, apply_to: str) -> pl.Expr:
        """
        Return expression that checks if rule applies based on container type.

        Logic:
        - 'both' -> applies to all rows
        - 'holding' -> only applies to holding container
        - 'reference' (or any other value) -> applies to non-holding containers
        """
        apply_to = apply_to.lower()
        if apply_to == ApplyTo.BOTH:
            return pl.lit(True)
        if apply_to == ApplyTo.HOLDING:
            return pl.col("container") == Container.HOLDING
        # 'reference' or any other value = not holding
        return pl.col("container") != Container.HOLDING

    def _build_weight_columns(
        self,
        positions_lf: pl.LazyFrame,
        lookthroughs_lf: Optional[pl.LazyFrame],
        metadata_map: Dict,
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]]
    ) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]:
        """
        Build final weight columns: {w}_{config}_{pid} = w * factor_used

        factor_used = fcol (no rescale) OR f"{fcol}_factors".struct.field(w) (rescaled)
        """
        has_lookthroughs = lookthroughs_lf is not None

        # Factor columns from metadata_map (f_{config}_{pid})
        factor_cols = [c for pmap in metadata_map.values() for c in pmap.values()]

        # Resolve valid weight labels from weight_labels_map
        pos_schema = set(positions_lf.collect_schema().names())

        pos_weights_set = set()
        ordered_pos_weights = []

        lt_weights_set = set()
        ordered_lt_weights = []

        for _, (pw, lw) in weight_labels_map.items():
            for w in (pw or []):
                if w not in pos_weights_set:
                    ordered_pos_weights.append(w)
                    pos_weights_set.add(w)
            for w in (lw or []):
                if w not in lt_weights_set:
                    ordered_lt_weights.append(w)
                    lt_weights_set.add(w)

        # Filter by schema membership
        pos_weights_valid = [w for w in ordered_pos_weights if w in pos_schema]

        # Build expressions for positions
        pos_weight_exprs: List[pl.Expr] = []
        for fcol in factor_cols:
            if fcol not in pos_schema:
                continue

            suffix = fcol[2:]  # "config_pid"
            factors_col = f"{fcol}_factors"
            has_struct = factors_col in pos_schema

            for w in pos_weights_valid:
                if has_struct:
                    factor_used = pl.col(factors_col).struct.field(w)
                else:
                    factor_used = pl.col(fcol)

                pos_weight_exprs.append(
                    pl.when(pl.col(fcol).is_not_null())
                    .then(pl.col(w) * factor_used)
                    .otherwise(pl.lit(None))
                    .alias(f"{w}_{suffix}")
                )

        if pos_weight_exprs:
            positions_lf = positions_lf.with_columns(pos_weight_exprs)

        # Lookthroughs (if present)
        if has_lookthroughs and lookthroughs_lf is not None:
            lt_schema = set(lookthroughs_lf.collect_schema().names())
            lt_weights_valid = [w for w in ordered_lt_weights if w in lt_schema]

            lt_weight_exprs: List[pl.Expr] = []
            for fcol in factor_cols:
                if fcol not in lt_schema:
                    continue

                suffix = fcol[2:]
                factors_col = f"{fcol}_factors"
                has_struct = factors_col in lt_schema

                for w in lt_weights_valid:
                    if has_struct:
                        factor_used = pl.col(factors_col).struct.field(w)
                    else:
                        factor_used = pl.col(fcol)

                    lt_weight_exprs.append(
                        pl.when(pl.col(fcol).is_not_null())
                        .then(pl.col(w) * factor_used)
                        .otherwise(pl.lit(None))
                        .alias(f"{w}_{suffix}")
                    )

            if lt_weight_exprs:
                lookthroughs_lf = lookthroughs_lf.with_columns(lt_weight_exprs)

        return positions_lf, lookthroughs_lf

    def _build_scale_factors(
        self,
        positions_lf: pl.LazyFrame,
        lookthroughs_lf: Optional[pl.LazyFrame],
        perspective_configs: Dict,
        metadata_map: Dict,
        weight_labels_map: Dict[str, Tuple[List[str], List[str]]]
    ) -> Optional[pl.LazyFrame]:
        """
        Build scale factors LazyFrame for perspectives with scale_holdings_to_100_percent.

        Returns None if no rescaling enabled for any perspective.

        Formula:
            numerator = sum(w * fcol) for kept positions + kept essential LTs
            denominator = sum(w) for ALL positions + ALL essential LTs
            scale_factor = numerator / denominator
        """
        # Find perspectives with scale_holdings_to_100_percent enabled
        rescale_info = []  # List of (config_name, perspective_id, fcol)
        for config_name, pmap in perspective_configs.items():
            for pid in self._get_rescale_perspectives(pmap, "scale_holdings_to_100_percent"):
                fcol = metadata_map[config_name][pid]
                rescale_info.append((config_name, pid, fcol))

        if not rescale_info:
            return None

        # Resolve valid weight labels from weight_labels_map
        pos_schema = set(positions_lf.collect_schema().names())
        lt_schema = (
            set(lookthroughs_lf.collect_schema().names())
            if lookthroughs_lf is not None
            else set()
        )

        all_pos_weights = set()
        for _, (pw, _) in weight_labels_map.items():
            all_pos_weights.update(pw or [])

        valid_weights = [w for w in all_pos_weights if w in pos_schema]
        if not valid_weights:
            return None

        # Weights that exist on essential lookthroughs
        elt_weights = [w for w in valid_weights if w in lt_schema]

        # Compute total denominators per container (positions + essential LT)
        tot_pos = (
            positions_lf
            .group_by("container")
            .agg([pl.col(w).sum().alias(f"__totpos__{w}") for w in valid_weights])
        )

        if elt_weights and lookthroughs_lf is not None:
            tot_elt = (
                lookthroughs_lf
                .filter(pl.col("record_type") == RecordType.ESSENTIAL_LOOKTHROUGHS)
                .group_by("container")
                .agg([pl.col(w).sum().alias(f"__totelt__{w}") for w in elt_weights])
            )
            total_denoms = tot_pos.join(tot_elt, on="container", how="left")
            # Combine: total = positions + essential LT
            combine_exprs = []
            for w in valid_weights:
                if w in elt_weights:
                    combine_exprs.append(
                        (pl.col(f"__totpos__{w}") + pl.col(f"__totelt__{w}").fill_null(0.0)).alias(f"__tot__{w}")
                    )
                else:
                    combine_exprs.append(pl.col(f"__totpos__{w}").alias(f"__tot__{w}"))
            total_denoms = total_denoms.with_columns(combine_exprs)
        else:
            total_denoms = tot_pos.with_columns([
                pl.col(f"__totpos__{w}").alias(f"__tot__{w}") for w in valid_weights
            ])

        # Build scale factors for each rescaling perspective
        scale_factor_frames = []

        for config_name, pid, fcol in rescale_info:
            if fcol not in pos_schema:
                continue

            # Kept positions numerator: sum(w * fcol) for kept rows
            kept_pos = positions_lf.filter(pl.col(fcol).is_not_null())
            keep_pos_nums = (
                kept_pos
                .group_by("container")
                .agg([(pl.col(w) * pl.col(fcol)).sum().alias(f"__keeppos__{w}") for w in valid_weights])
            )

            # Include kept essential LT if fcol exists on lookthroughs
            if elt_weights and lookthroughs_lf is not None and fcol in lt_schema:
                kept_elt = (
                    lookthroughs_lf
                    .filter(
                        (pl.col("record_type") == RecordType.ESSENTIAL_LOOKTHROUGHS)
                        & pl.col(fcol).is_not_null()
                    )
                )
                keep_elt_nums = (
                    kept_elt
                    .group_by("container")
                    .agg([(pl.col(w) * pl.col(fcol)).sum().alias(f"__keepelt__{w}") for w in elt_weights])
                )
                keep_nums = keep_pos_nums.join(keep_elt_nums, on="container", how="left")
                # Combine numerators
                combine_exprs = []
                for w in valid_weights:
                    if w in elt_weights:
                        combine_exprs.append(
                            (pl.col(f"__keeppos__{w}") + pl.col(f"__keepelt__{w}").fill_null(0.0)).alias(f"__keep__{w}")
                        )
                    else:
                        combine_exprs.append(pl.col(f"__keeppos__{w}").alias(f"__keep__{w}"))
                keep_nums = keep_nums.with_columns(combine_exprs)
            else:
                keep_nums = keep_pos_nums.with_columns([
                    pl.col(f"__keeppos__{w}").alias(f"__keep__{w}") for w in valid_weights
                ])

            # Join totals and compute scale_factor = keep / total
            joined = keep_nums.join(total_denoms, on="container", how="left")

            # Build flat rows: one per (container, weight_label)
            for w in valid_weights:
                sf_frame = (
                    joined
                    .select([
                        pl.lit(config_name).alias("config"),
                        pl.lit(pid).alias("perspective_id"),
                        pl.col("container"),
                        pl.lit(w).alias("weight_label"),
                        pl.when(pl.col(f"__tot__{w}") > 0)
                        .then(pl.col(f"__keep__{w}").fill_null(0.0) / pl.col(f"__tot__{w}"))
                        .otherwise(pl.lit(None))
                        .alias("scale_factor")
                    ])
                )
                scale_factor_frames.append(sf_frame)

        if not scale_factor_frames:
            return None

        # Union all scale factor frames
        result = pl.concat(scale_factor_frames)
        return result
