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
                      perspective_configs: Dict,
                      weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
                      scale_factors_df: Optional[pl.DataFrame] = None) -> Dict:
        """Format processed dataframes into structured output.

        Args:
            perspective_configs: {config_name: {perspective_id: [modifier_names]}}
            weight_labels_map: Per-container weight labels mapping
                {container_name: (pos_weight_labels, lt_weight_labels)}
            scale_factors_df: DataFrame with scale factors from Processor (or None if no rescaling)
        """
        if not perspective_configs:
            return {"perspective_configurations": {}}

        # Build factor_map from perspective_configs (factor column = f_{config}_{pid})
        factor_map = OutputFormatter._build_factor_map(perspective_configs)

        results = OutputFormatter._initialize_results(factor_map)

        # Compute union of weights from weight_labels_map
        position_weights = []
        lookthrough_weights = []
        if weight_labels_map:
            pos_set = set()
            lt_set = set()
            for _, (pw, lw) in weight_labels_map.items():
                for w in (pw or []):
                    if w not in pos_set:
                        position_weights.append(w)
                        pos_set.add(w)
                for w in (lw or []):
                    if w not in lt_set:
                        lookthrough_weights.append(w)
                        lt_set.add(w)

        # Get factor columns once
        factor_columns = [
            col for pmap in factor_map.values()
            for col in pmap.values()
        ]

        # Process positions
        if not positions_df.is_empty():
            OutputFormatter._process_dataframe_batch(
                positions_df,
                "positions",
                factor_map,
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
                factor_map,
                factor_columns,
                lookthrough_weights,
                results,
                "identifier",
                weight_labels_map
            )

        # Populate scale factors from DataFrame or default to 1.0
        OutputFormatter._populate_scale_factors_from_df(
            scale_factors_df,
            results,
            weight_labels_map,
            factor_map,
            positions_df,
            position_weights
        )

        return {"perspective_configurations": results}

    @staticmethod
    def _build_factor_map(perspective_configs: Dict) -> Dict:
        """Build factor column map from perspective_configs.

        Factor columns follow the pattern: f_{config_name}_{perspective_id}
        """
        factor_map = {}
        for config_name, pmap in perspective_configs.items():
            factor_map[config_name] = {}
            for pid_str in pmap:
                pid = int(pid_str)
                factor_map[config_name][pid] = f"f_{config_name}_{pid}"
        return factor_map

    @staticmethod
    def _initialize_results(factor_map: Dict) -> Dict:
        """Initialize the results structure."""
        return {
            config_name: {pid: {} for pid in perspective_map}
            for config_name, perspective_map in factor_map.items()
            if perspective_map
        }

    @staticmethod
    def _process_dataframe_batch(df: pl.DataFrame,
                                 mode: str,
                                 factor_map: Dict,
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
        for config_name, perspective_map in factor_map.items():
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
    def _populate_scale_factors_from_df(
        scale_factors_df: Optional[pl.DataFrame],
        results: Dict,
        weight_labels_map: Optional[Dict[str, Tuple[List[str], List[str]]]],
        factor_map: Dict,
        positions_df: pl.DataFrame,
        position_weights: List[str]
    ):
        """
        Populate scale_factors in results from DataFrame or default to 1.0.

        When scale_factors_df is None (no rescaling enabled):
            scale_factors = {w: 1.0 for w in container_weights}
        When scale_factors_df is provided:
            Extract values from DataFrame rows
        """
        if positions_df.is_empty():
            return

        # Build full set of possible position weight labels (global + per-container)
        all_pos_weights = set(position_weights)
        if weight_labels_map:
            for _, (pw, _) in weight_labels_map.items():
                all_pos_weights.update(pw or [])

        valid_weights = [w for w in all_pos_weights if w in positions_df.columns]
        if not valid_weights:
            return

        # Get unique containers
        containers = positions_df.select("container").unique().to_series().to_list()

        # Build lookup from scale_factors_df if provided
        sf_lookup = {}  # (config, pid, container, weight_label) -> scale_factor
        if scale_factors_df is not None and not scale_factors_df.is_empty():
            for row in scale_factors_df.iter_rows(named=True):
                key = (row["config"], row["perspective_id"], str(row["container"]), row["weight_label"])
                sf_lookup[key] = row["scale_factor"]

        # Populate scale_factors for each config/perspective/container
        for config_name, perspective_map in factor_map.items():
            for perspective_id in perspective_map:
                for container in containers:
                    # Determine which weights to include
                    if weight_labels_map and container in weight_labels_map:
                        container_weights, _ = weight_labels_map[container]
                        output_weights = [w for w in (container_weights or []) if w in valid_weights]
                    else:
                        output_weights = valid_weights

                    # Build scale_factors dict
                    kept_fraction = {}
                    for w in output_weights:
                        key = (config_name, perspective_id, str(container), w)
                        if key in sf_lookup:
                            sf = sf_lookup[key]
                            if sf is not None:
                                kept_fraction[w] = sf
                        else:
                            # Default to 1.0 when no rescaling
                            kept_fraction[w] = 1.0

                    if kept_fraction:
                        perspective_target = results[config_name][perspective_id]
                        if container not in perspective_target:
                            perspective_target[container] = {}
                        perspective_target[container]["scale_factors"] = kept_fraction
