"""
Test: Compute scale_factors AFTER collect instead of as LazyFrame.
"""

import time
import random
from typing import Dict, Any, List, Tuple

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.core.data_ingestion import DataIngestion
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.models.rule import Rule


def generate_data(num_positions: int, num_lt_per_pos: int = 1) -> Dict[str, Any]:
    random.seed(42)
    positions = {}
    essential_lts = {}

    for i in range(num_positions):
        instrument_id = 1000 + i
        positions[f"pos_{i}"] = {
            "instrument_id": instrument_id,
            "parent_instrument_id": None,
            "sub_portfolio_id": 1,
            "initial_weight": random.uniform(0.0001, 0.01),
            "resulting_weight": random.uniform(0.0001, 0.01),
        }
        for j in range(num_lt_per_pos):
            essential_lts[f"lt_{i}_{j}"] = {
                "instrument_id": 100000 + i * 100 + j,
                "parent_instrument_id": instrument_id,
                "sub_portfolio_id": 1,
                "initial_weight": random.uniform(0.00001, 0.001),
                "resulting_weight": random.uniform(0.00001, 0.001),
            }

    return {
        "ed": "2024-01-15",
        "position_weight_labels": ["initial_weight", "resulting_weight"],
        "lookthrough_weight_labels": ["initial_weight", "resulting_weight"],
        "holding": {
            "positions": positions,
            "essential_lookthroughs": essential_lts,
            "complete_lookthroughs": {},
        },
    }


def compute_scale_factors_after_collect(
    positions_df: pl.DataFrame,
    lookthroughs_df: pl.DataFrame,
    rescale_fcols: List[Tuple[str, int, str]],  # (config, pid, fcol)
    weights: List[str],
) -> pl.DataFrame:
    """
    Compute scale factors on collected DataFrames using Polars (vectorized).
    """
    if not rescale_fcols or not weights:
        return None

    pos_cols = set(positions_df.columns)
    lt_cols = set(lookthroughs_df.columns) if not lookthroughs_df.is_empty() else set()
    valid_weights = [w for w in weights if w in pos_cols]
    elt_weights = [w for w in valid_weights if w in lt_cols]

    # 1) Total denominators per container
    tot_pos = positions_df.group_by("container").agg([
        pl.col(w).sum().alias(f"tot__{w}") for w in valid_weights
    ])

    if not lookthroughs_df.is_empty() and elt_weights:
        tot_elt = (
            lookthroughs_df
            .filter(pl.col("record_type") == "essential_lookthroughs")
            .group_by("container")
            .agg([pl.col(w).sum().alias(f"totelt__{w}") for w in elt_weights])
        )
        tot = tot_pos.join(tot_elt, on="container", how="left")
        for w in valid_weights:
            if w in elt_weights:
                tot = tot.with_columns([
                    (pl.col(f"tot__{w}") + pl.col(f"totelt__{w}").fill_null(0.0)).alias(f"den__{w}")
                ])
            else:
                tot = tot.with_columns([pl.col(f"tot__{w}").alias(f"den__{w}")])
    else:
        tot = tot_pos
        for w in valid_weights:
            tot = tot.with_columns([pl.col(f"tot__{w}").alias(f"den__{w}")])

    # 2) Kept numerators - one agg with all expressions
    keep_exprs = []
    for config, pid, fcol in rescale_fcols:
        if fcol not in pos_cols:
            continue
        for w in valid_weights:
            keep_exprs.append(
                pl.when(pl.col(fcol).is_not_null())
                .then(pl.col(w) * pl.col(fcol))
                .otherwise(0.0)
                .sum()
                .alias(f"keep__{config}__{pid}__{w}")
            )

    kept_pos = positions_df.group_by("container").agg(keep_exprs)

    # Add ELT contributions
    if not lookthroughs_df.is_empty() and elt_weights:
        lt_keep_exprs = []
        for config, pid, fcol in rescale_fcols:
            if fcol not in lt_cols:
                continue
            for w in elt_weights:
                lt_keep_exprs.append(
                    pl.when(
                        (pl.col("record_type") == "essential_lookthroughs") &
                        pl.col(fcol).is_not_null()
                    )
                    .then(pl.col(w) * pl.col(fcol))
                    .otherwise(0.0)
                    .sum()
                    .alias(f"keepelt__{config}__{pid}__{w}")
                )
        if lt_keep_exprs:
            kept_elt = lookthroughs_df.group_by("container").agg(lt_keep_exprs)
            kept = kept_pos.join(kept_elt, on="container", how="left")
        else:
            kept = kept_pos
    else:
        kept = kept_pos

    # 3) Combine into num__ columns
    kept_cols = set(kept.columns)
    combine_exprs = []
    for config, pid, fcol in rescale_fcols:
        for w in valid_weights:
            keep_col = f"keep__{config}__{pid}__{w}"
            keepelt_col = f"keepelt__{config}__{pid}__{w}"
            if keep_col not in kept_cols:
                continue
            base = pl.col(keep_col).fill_null(0.0)
            if w in elt_weights and keepelt_col in kept_cols:
                base = base + pl.col(keepelt_col).fill_null(0.0)
            combine_exprs.append(base.alias(f"num__{config}__{pid}__{w}"))

    kept = kept.with_columns(combine_exprs)

    # 4) Join and compute SF
    wide = kept.join(tot, on="container", how="left")

    # Build result rows
    results = []
    for config, pid, fcol in rescale_fcols:
        for w in valid_weights:
            num_col = f"num__{config}__{pid}__{w}"
            den_col = f"den__{w}"
            if num_col not in wide.columns:
                continue

            sf_df = wide.select([
                pl.lit(config).alias("config"),
                pl.lit(pid).alias("perspective_id"),
                pl.col("container"),
                pl.lit(w).alias("weight_label"),
                pl.when(pl.col(den_col).is_null() | (pl.col(den_col) == 0.0))
                .then(pl.lit(None))
                .when(pl.col(num_col).is_null() | (pl.col(num_col) == 0.0))
                .then(pl.lit(1.0))
                .otherwise(pl.col(num_col) / pl.col(den_col))
                .alias("scale_factor")
            ])
            results.append(sf_df)

    if not results:
        return None

    return pl.concat(results)


def profile_comparison(num_positions: int, num_lt_per_pos: int, num_perspectives: int):
    """Compare lazy SF vs after-collect SF."""

    print(f"\n{'='*80}")
    print(f"COMPARISON: {num_positions} pos, {num_lt_per_pos} LT/pos, {num_perspectives} perspectives")
    print(f"{'='*80}")

    # Setup
    input_json = generate_data(num_positions, num_lt_per_pos)

    engine = PerspectiveEngine()
    engine.config.default_modifiers = []
    for i in range(num_perspectives):
        pid = i + 1
        engine.config.perspectives[pid] = [
            Rule(name=f"filter_{pid}", apply_to="both",
                 criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 5)},
                 is_scaling_rule=False)
        ]

    perspective_configs = {"config": {}}
    for i in range(num_perspectives):
        perspective_configs["config"][str(i + 1)] = ["scale_holdings_to_100_percent"]

    weight_labels_map = DataIngestion.get_weight_labels(input_json)
    positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(input_json, weight_labels_map)

    processor = PerspectiveProcessor(engine.config)

    # ============================================
    # OPTION A: Current approach (optimized lazy SF via build_perspective_plan)
    # ============================================
    t0 = time.perf_counter()
    positions_lf, lookthroughs_lf, scale_factors_lf = processor.build_perspective_plan(
        positions_lf, lookthroughs_lf, perspective_configs, {}, weight_labels_map
    )
    t_build_plan = time.perf_counter() - t0

    t0 = time.perf_counter()
    collected_with_sf = pl.collect_all([positions_lf, lookthroughs_lf, scale_factors_lf])
    t_collect_with_sf = time.perf_counter() - t0

    total_lazy = t_build_plan + t_collect_with_sf

    # For after-collect comparison, we need to rebuild the plan without collecting
    positions_lf2, lookthroughs_lf2 = DataIngestion.build_dataframes(input_json, weight_labels_map)
    positions_lf2, lookthroughs_lf2, _ = processor.build_perspective_plan(
        positions_lf2, lookthroughs_lf2, perspective_configs, {}, weight_labels_map
    )

    # ============================================
    # OPTION B: After-collect approach
    # ============================================
    t0 = time.perf_counter()
    collected_no_sf = pl.collect_all([positions_lf2, lookthroughs_lf2])
    t_collect_no_sf = time.perf_counter() - t0

    positions_df = collected_no_sf[0]
    lookthroughs_df = collected_no_sf[1]

    # Build rescale_fcols list (factor column pattern: f_{config}_{pid})
    rescale_fcols = []
    for config, pmap in perspective_configs.items():
        for pid_str, modifiers in pmap.items():
            if "scale_holdings_to_100_percent" in (modifiers or []):
                pid = int(pid_str)
                fcol = f"f_{config}_{pid}"
                rescale_fcols.append((config, pid, fcol))

    weights = ["initial_weight", "resulting_weight"]

    t0 = time.perf_counter()
    sf_df_after = compute_scale_factors_after_collect(
        positions_df, lookthroughs_df, rescale_fcols, weights
    )
    t_compute_sf_after = time.perf_counter() - t0

    total_after = t_collect_no_sf + t_compute_sf_after

    # ============================================
    # Results
    # ============================================
    print(f"\nOPTION A (Optimized Lazy SF):")
    print(f"  Build plan:     {t_build_plan:.3f}s")
    print(f"  Collect all:    {t_collect_with_sf:.3f}s")
    print(f"  TOTAL:          {total_lazy:.3f}s")

    print(f"\nOPTION B (After Collect):")
    print(f"  Collect (no SF): {t_collect_no_sf:.3f}s")
    print(f"  Compute SF:      {t_compute_sf_after:.3f}s")
    print(f"  TOTAL:           {total_after:.3f}s")

    speedup = total_lazy / total_after if total_after > 0 else 0
    print(f"\n  SPEEDUP: {speedup:.2f}x faster")

    # Verify results match
    sf_lazy = collected_with_sf[2].sort(["config", "perspective_id", "container", "weight_label"])
    sf_after = sf_df_after.sort(["config", "perspective_id", "container", "weight_label"])

    # Quick check
    if sf_lazy.shape == sf_after.shape:
        print(f"  SF shapes match: {sf_lazy.shape}")
    else:
        print(f"  WARNING: SF shapes differ! Lazy={sf_lazy.shape}, After={sf_after.shape}")


if __name__ == "__main__":
    profile_comparison(1000, 1, 10)
    profile_comparison(1000, 1, 50)
    profile_comparison(1000, 1, 100)
    profile_comparison(1000, 1, 500)
