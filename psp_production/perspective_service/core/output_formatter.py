"""
Output Formatter - Formats the processed data into the final output structure.
"""

from typing import Dict, List, Optional, Tuple

import polars as pl


class OutputFormatter:
    """Formats the processed data into the final output structure."""

    @staticmethod
    def format_output(positions_df: pl.DataFrame,
                      lookthroughs_df: pl.DataFrame,
                      metadata_map: Dict,
                      position_weights: List[str],
                      lookthrough_weights: List[str],
                      verbose: bool,
                      flatten_response: bool = False,
                      weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
                      perspective_configs: Optional[Dict] = None,
                      original_containers: Optional[List[str]] = None) -> Dict:
        """Format processed dataframes into structured output.

        Args:
            weight_labels_map: Optional per-container weight labels mapping
                {container_name: (pos_weight_labels, lt_weight_labels)}
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
                Used to determine which perspectives have rescaling enabled.
            original_containers: List of container names from input (for empty container handling)
        """
        if not metadata_map:
            return {"perspective_configurations": {}}

        results = OutputFormatter._initialize_results(metadata_map)

        # Get factor columns once
        factor_columns = [
            col for pmap in metadata_map.values()
            for col in pmap.values()
        ]

        # Process positions
        if not positions_df.is_empty():
            OutputFormatter._process_dataframe_batch(
                positions_df,
                "positions",
                metadata_map,
                factor_columns,
                position_weights,
                results,
                "identifier",
                weight_labels_map
            )

        # Process lookthroughs
        if not lookthroughs_df.is_empty():
            OutputFormatter._process_dataframe_batch(
                lookthroughs_df,
                "lookthrough",
                metadata_map,
                factor_columns,
                lookthrough_weights,
                results,
                "identifier",
                weight_labels_map
            )

        # Scale factors are ALWAYS computed
        OutputFormatter._add_scale_factors(
            positions_df,
            lookthroughs_df,
            metadata_map,
            factor_columns,
            position_weights,
            results,
            weight_labels_map,
            perspective_configs
        )

        # =============================================================================
        # LEGACY COMPATIBILITY: Remove this block when caller is adjusted
        # =============================================================================
        # When all positions are filtered out, legacy expects:
        # - Container key still present in output
        # - "positions": None (explicitly null, not missing)
        # - "scale_factors": already set to 1.0 by _add_scale_factors (no impact since no positions)
        #
        # TODO: Remove this block once callers are updated to handle missing containers.
        # =============================================================================
        if original_containers:
            for config_name, perspective_map in results.items():
                for perspective_id in perspective_map:
                    for container in original_containers:
                        container_data = results[config_name][perspective_id].get(container, {})
                        if "positions" not in container_data:
                            if container not in results[config_name][perspective_id]:
                                results[config_name][perspective_id][container] = {}
                            results[config_name][perspective_id][container]["positions"] = None

        # Flatten response if requested
        if flatten_response:
            OutputFormatter._flatten_results(results)

        return {"perspective_configurations": results}

    @staticmethod
    def _initialize_results(metadata_map: Dict) -> Dict:
        """Initialize the results structure."""
        return {
            config_name: {pid: {} for pid in perspective_map}
            for config_name, perspective_map in metadata_map.items()
            if perspective_map
        }

    @staticmethod
    def _process_dataframe_batch(df: pl.DataFrame,
                                 mode: str,
                                 metadata_map: Dict,
                                 factor_columns: List[str],
                                 weights: List[str],
                                 results: Dict,
                                 id_column: str,
                                 weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]] = None):
        """Process a batch of data and update results."""
        available_factors = [c for c in factor_columns if c in df.columns]
        if not available_factors:
            return

        # Base columns
        base_cols = [id_column, "container"]
        if mode == "lookthrough":
            base_cols.append("record_type")

        # IMPORTANT: Build candidate_weights from SAME universe as Processor
        candidate_weights_set = set(weights)
        ordered_candidate_weights = list(weights) 

        if weight_labels_map:
            for _, (pw, lw) in weight_labels_map.items():
                extra = (pw or []) if mode == "positions" else (lw or [])
                for w in extra:
                    if w not in candidate_weights_set:
                        ordered_candidate_weights.append(w)
                        candidate_weights_set.add(w)

        # Preselect: base + factor cols + ALL computed weight columns
        computed_cols = []
        for fc in available_factors:
            suffix = fc[2:]  # Remove 'f_' prefix
            for w in candidate_weights_set:  
                c = f"{w}_{suffix}"
                if c in df.columns:
                    computed_cols.append(c)

        # If none of the computed cols exist, nothing to output
        if not computed_cols:
            return

        select_cols = base_cols + available_factors + computed_cols
        df_slim = df.select(select_cols)

        # Process each perspective
        for config_name, perspective_map in metadata_map.items():
            for perspective_id, col_name in perspective_map.items():
                if col_name not in df_slim.columns:
                    continue

                OutputFormatter._process_single_perspective(
                    df_slim,
                    mode,
                    config_name,
                    perspective_id,
                    col_name,
                    base_cols,
                    ordered_candidate_weights,  
                    id_column,
                    results,
                    weight_labels_map
                )

    @staticmethod
    def _process_single_perspective(df: pl.DataFrame,
                                    mode: str,
                                    config_name: str,
                                    perspective_id: int,
                                    factor_col: str,
                                    base_cols: List[str],
                                    ordered_candidate_weights: List[str],
                                    id_column: str,
                                    results: Dict,
                                    weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]] = None):
        """Process a single perspective's data."""
        # Filter to non-null factors (kept positions)
        filtered = df.filter(pl.col(factor_col).is_not_null())
        if filtered.is_empty():
            return

        suffix = factor_col[2:]  # "config_pid"

        # Build weight_col_names from ordered_candidate_weights (deterministic)
        weight_col_names = [f"{w}_{suffix}" for w in ordered_candidate_weights]
        available_weight_cols = [c for c in weight_col_names if c in filtered.columns]
        if not available_weight_cols:
            return

        weighted = filtered.select(base_cols + available_weight_cols)

        # Rename back to original label names (preserving order)
        rename_map = {f"{w}_{suffix}": w for w in ordered_candidate_weights if f"{w}_{suffix}" in weighted.columns}
        weighted = weighted.rename(rename_map)

        # Partition by container (+ record_type for lookthroughs)
        group_cols = ["container"]
        if mode == "lookthrough":
            group_cols.append("record_type")

        partitions = weighted.partition_by(group_cols, as_dict=True, maintain_order=False)

        for group_key, group_df in partitions.items():
            if isinstance(group_key, tuple):
                container = group_key[0]
                record_type = group_key[1] if len(group_key) > 1 else None
            else:
                container = group_key
                record_type = None

            # Output weights: use container-specific list order if available,
            # otherwise use ordered_candidate_weights (preserves legacy ordering TODO: Whether that is needed is highly questionable)
            if weight_labels_map and container in weight_labels_map:
                container_pos_weights, container_lt_weights = weight_labels_map[container]
                if mode == "positions":
                    # Use container's order, filter by presence in columns
                    output_weights = [w for w in (container_pos_weights or []) if w in group_df.columns]
                else:
                    output_weights = [w for w in (container_lt_weights or []) if w in group_df.columns]
            else:
                output_weights = [w for w in ordered_candidate_weights if w in group_df.columns]

            formatted = OutputFormatter._df_to_id_dict(group_df, id_column, output_weights)

            perspective_target = results[config_name][perspective_id]
            if container not in perspective_target:
                perspective_target[container] = {}
            target = perspective_target[container]

            if mode == "positions":
                target.setdefault("positions", {}).update(formatted)
            else:
                key = record_type or "lookthrough"
                target.setdefault(key, {}).update(formatted)

    @staticmethod
    def _df_to_id_dict(df: pl.DataFrame, id_column: str, value_columns: List[str]) -> Dict:
        """Convert dataframe to {id: {col: val, ...}} dict efficiently."""
        if df.is_empty():
            return {}

        ids = df[id_column].to_list()

        if len(value_columns) == 1:
            # Single column: avoid struct overhead
            col = value_columns[0]
            values = df[col].to_list()
            return {id_val: {col: val} for id_val, val in zip(ids, values)}

        # Multiple columns: use struct
        structs = df.select(pl.struct(value_columns).alias("_s"))["_s"].to_list()
        return dict(zip(ids, structs))

    @staticmethod
    def _add_scale_factors(positions_df: pl.DataFrame,
                           lookthroughs_df: pl.DataFrame,
                           metadata_map: Dict,
                           factor_columns: List[str],
                           position_weights: List[str],
                           results: Dict,
                           weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
                           perspective_configs: Optional[Dict] = None):
        """
        Compute portfolio "kept fraction" per container, per perspective, per weight label.

        This is the value you want for TNA attribution:
            notional_i = TNA * output_weight_i * kept_fraction

        kept_fraction is ALWAYS correct (normalizes even if raw weights don't sum to 1):
            kept_fraction = (sum of kept positions + sum of kept essential LT) /
                           (sum of all positions + sum of all essential LT)

        IMPORTANT:
        - Includes essential lookthroughs in the universe (for exposure labels where they contribute).
        - Uses RAW/UNCHANGED weight columns.
        - Scale factor is 1.0 when scale_holdings_to_100_percent is NOT enabled for the perspective.
        """
        if positions_df.is_empty():
            return

        available_factors = [c for c in factor_columns if c in positions_df.columns]
        if not available_factors:
            return

        # Build full set of possible position weight labels (global + per-container)
        all_pos_weights = set(position_weights)
        if weight_labels_map:
            for _, (pw, _) in weight_labels_map.items():
                all_pos_weights.update(pw or [])

        valid_weights = [w for w in all_pos_weights if w in positions_df.columns]
        if not valid_weights:
            return

        # Extract essential lookthroughs (if any)
        elt_df = pl.DataFrame()
        if lookthroughs_df is not None and not lookthroughs_df.is_empty() and "record_type" in lookthroughs_df.columns:
            elt_df = lookthroughs_df.filter(pl.col("record_type") == "essential_lookthroughs")

        # Determine which weights exist on essential LT
        elt_weights = [w for w in valid_weights if not elt_df.is_empty() and w in elt_df.columns]

        # -------------------------
        # Compute TOTAL denominators per container (positions + essential LT)
        # -------------------------
        tot_pos = (
            positions_df
            .group_by("container")
            .agg([pl.col(w).sum().alias(f"__totpos__{w}") for w in valid_weights])
        )

        if elt_weights and not elt_df.is_empty():
            tot_elt = (
                elt_df
                .group_by("container")
                .agg([pl.col(w).sum().alias(f"__totelt__{w}") for w in elt_weights])
            )
            total_denoms = tot_pos.join(tot_elt, on="container", how="left")
            # Combine: total = positions + essential LT (for weights that exist on both)
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

        # Process each perspective
        for config_name, perspective_map in metadata_map.items():
            for perspective_id, col_name in perspective_map.items():
                if col_name not in positions_df.columns:
                    continue

                # Check if scale_holdings_to_100_percent is enabled for this perspective
                # If not, SF = 1.0 for all weights
                rescaling_enabled = False
                if perspective_configs:
                    # perspective_configs uses string keys for perspective IDs
                    modifiers = perspective_configs.get(config_name, {}).get(str(perspective_id), []) or []
                    rescaling_enabled = "scale_holdings_to_100_percent" in modifiers

                if not rescaling_enabled:
                    # No rescaling: SF = 1.0 for all weight labels
                    # Get unique containers from positions
                    containers = positions_df.select("container").unique().to_series().to_list()
                    for container in containers:
                        # Container-specific weights if provided
                        if weight_labels_map and container in weight_labels_map:
                            container_weights, _ = weight_labels_map[container]
                            output_weights = [w for w in (container_weights or []) if w in valid_weights]
                        else:
                            output_weights = valid_weights

                        kept_fraction = {w: 1.0 for w in output_weights}
                        if kept_fraction:
                            perspective_target = results[config_name][perspective_id]
                            if container not in perspective_target:
                                perspective_target[container] = {}
                            perspective_target[container]["scale_factors"] = kept_fraction
                    continue

                # -------------------------
                # Compute KEPT numerators using w * fcol (World A: scaling affects economic mass)
                # This must match how rescaling denom is computed for consistency
                # -------------------------
                kept_pos = positions_df.filter(pl.col(col_name).is_not_null())
                if kept_pos.is_empty():
                    continue

                # Use w * fcol for kept mass (not raw w)
                keep_pos_nums = (
                    kept_pos
                    .group_by("container")
                    .agg([(pl.col(w) * pl.col(col_name)).sum().alias(f"__keeppos__{w}") for w in valid_weights])
                )

                # Include kept essential LT if col_name exists on lookthroughs
                if elt_weights and not elt_df.is_empty() and col_name in elt_df.columns:
                    kept_elt = elt_df.filter(pl.col(col_name).is_not_null())
                    if not kept_elt.is_empty():
                        # Use w * fcol for ELT too (ELT has its own factor, not inherited from parent)
                        keep_elt_nums = (
                            kept_elt
                            .group_by("container")
                            .agg([(pl.col(w) * pl.col(col_name)).sum().alias(f"__keepelt__{w}") for w in elt_weights])
                        )
                        keep_nums = keep_pos_nums.join(keep_elt_nums, on="container", how="left")
                        # Combine: kept = positions + essential LT
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
                else:
                    keep_nums = keep_pos_nums.with_columns([
                        pl.col(f"__keeppos__{w}").alias(f"__keep__{w}") for w in valid_weights
                    ])

                # Join totals and compute kept_fraction = keep / total (guard total=0)
                joined = keep_nums.join(total_denoms, on="container", how="left")

                for row in joined.iter_rows(named=True):
                    container = row["container"]

                    # Container-specific weights if provided
                    if weight_labels_map and container in weight_labels_map:
                        container_weights, _ = weight_labels_map[container]
                        output_weights = [w for w in (container_weights or []) if w in valid_weights]
                    else:
                        output_weights = valid_weights

                    kept_fraction = {}
                    for w in output_weights:
                        num = row.get(f"__keep__{w}")
                        den = row.get(f"__tot__{w}")

                        if den is None or den == 0:
                            continue
                        if num is None:
                            num = 0.0

                        kept_fraction[w] = num / den

                    if kept_fraction:
                        perspective_target = results[config_name][perspective_id]
                        if container not in perspective_target:
                            perspective_target[container] = {}
                        # keep the key name you already use downstream
                        perspective_target[container]["scale_factors"] = kept_fraction

    @staticmethod
    def _flatten_results(results: Dict) -> None:
        """
        Flatten positions and lookthroughs from row-based to columnar format.

        Converts from:
            {"positions": {"123": {"weight": 0.5}, "456": {"weight": 0.2}}}
        To:
            {"positions": {"identifier": [123, 456], "weight": [0.5, 0.2]}}
        """
        for config_name, perspectives in results.items():
            for perspective_id, containers in perspectives.items():
                for container_name, container_data in containers.items():
                    # Flatten positions
                    if "positions" in container_data:
                        container_data["positions"] = OutputFormatter._flatten_entries(
                            container_data["positions"]
                        )

                    # Flatten all lookthrough types
                    for key in list(container_data.keys()):
                        if "lookthrough" in key:
                            container_data[key] = OutputFormatter._flatten_entries(
                                container_data[key]
                            )

    @staticmethod
    def _flatten_entries(entries: Dict) -> Dict:
        """
        Convert row-based dict to columnar format.

        Input:  {"123": {"weight": 0.5, "exposure": 0.3}, "456": {"weight": 0.2, "exposure": 0.1}}
        Output: {"identifier": [123, 456], "weight": [0.5, 0.2], "exposure": [0.3, 0.1]}
        """
        if not entries:
            return {"identifier": []}

        processed = {"identifier": []}

        for entry_id, entry_data in entries.items():
            # Try to convert identifier to int, otherwise keep as string
            try:
                processed["identifier"].append(int(entry_id))
            except (ValueError, TypeError):
                processed["identifier"].append(entry_id)

            # Add each property value to its column
            for key, value in entry_data.items():
                if key not in processed:
                    processed[key] = []
                # Round floats to 13 decimal places (matching original)
                if isinstance(value, float):
                    value = round(value, 13)
                processed[key].append(value)

        return processed
