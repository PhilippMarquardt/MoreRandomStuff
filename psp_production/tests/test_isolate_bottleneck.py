"""
Isolate whether rescaling or scale_factors is the bottleneck.
"""

import time
import random
from typing import Dict, Any

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


def profile_isolated(num_positions: int, num_lt_per_pos: int, num_perspectives: int):
    """Profile with isolated timing for rescaling vs scale_factors."""

    print(f"\n{'='*80}")
    print(f"ISOLATED PROFILE: {num_positions} pos, {num_positions*num_lt_per_pos} LT, {num_perspectives} perspectives")
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

    # All perspectives have rescaling
    perspective_configs = {"config": {}}
    for i in range(num_perspectives):
        perspective_configs["config"][str(i + 1)] = ["scale_holdings_to_100_percent"]

    weight_labels_map = DataIngestion.get_weight_labels(input_json)
    positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(input_json, weight_labels_map)

    processor = PerspectiveProcessor(engine.config)

    # ========================================
    # Step 1: Build factor expressions (baseline)
    # ========================================
    t0 = time.perf_counter()

    factor_expressions_pos = []
    factor_expressions_lt = []
    factor_map = {"config": {}}

    for config_name, perspective_map in perspective_configs.items():
        perspective_ids = sorted([int(k) for k in perspective_map.keys()])
        for perspective_id in perspective_ids:
            column_name = f"f_{config_name}_{perspective_id}"
            factor_map[config_name][perspective_id] = column_name

            modifier_names = perspective_map.get(str(perspective_id)) or []
            active_modifiers = processor._filter_overridden_modifiers(modifier_names)

            keep_expr = processor._build_keep_expression(perspective_id, active_modifiers, {})
            scale_expr = processor._build_scale_expression(perspective_id, {})

            factor_expressions_pos.append(
                pl.when(keep_expr).then(scale_expr).otherwise(pl.lit(None)).alias(column_name)
            )
            factor_expressions_lt.append(
                pl.when(keep_expr).then(scale_expr).otherwise(pl.lit(None)).alias(column_name)
            )

    positions_lf = positions_lf.with_columns(factor_expressions_pos)
    lookthroughs_lf = lookthroughs_lf.with_columns(factor_expressions_lt)

    # Synchronize
    all_columns = [c for m in factor_map.values() for c in m.values()]
    lookthroughs_lf = processor._synchronize_lookthroughs(lookthroughs_lf, positions_lf, all_columns)

    t_baseline = time.perf_counter() - t0
    print(f"\n1. Build expressions + sync: {t_baseline:.3f}s")

    # ========================================
    # Step 2: Apply rescaling (_apply_rescaling)
    # ========================================
    t0 = time.perf_counter()

    positions_lf_rescaled, lookthroughs_lf_rescaled, sf_data = processor._apply_rescaling(
        positions_lf, lookthroughs_lf, perspective_configs, factor_map,
        True, {}, weight_labels_map
    )

    t_rescaling = time.perf_counter() - t0
    print(f"2. _apply_rescaling: {t_rescaling:.3f}s")

    # ========================================
    # Step 3: Build weight columns (_build_weight_columns)
    # ========================================
    t0 = time.perf_counter()

    positions_lf_weighted, lookthroughs_lf_weighted = processor._build_weight_columns(
        positions_lf_rescaled, lookthroughs_lf_rescaled, factor_map, weight_labels_map
    )

    t_weight_cols = time.perf_counter() - t0
    print(f"3. _build_weight_columns: {t_weight_cols:.3f}s")

    # ========================================
    # Step 4: Build scale factors (_build_scale_factors)
    # ========================================
    t0 = time.perf_counter()

    scale_factors_lf = processor._build_scale_factors(
        sf_data, perspective_configs, factor_map, weight_labels_map
    )

    t_scale_factors = time.perf_counter() - t0
    print(f"4. _build_scale_factors: {t_scale_factors:.3f}s")

    # ========================================
    # Step 5: Collect all
    # ========================================
    t0 = time.perf_counter()

    to_collect = [positions_lf_weighted, lookthroughs_lf_weighted]
    if scale_factors_lf is not None:
        to_collect.append(scale_factors_lf)

    collected = pl.collect_all(to_collect)

    t_collect = time.perf_counter() - t0
    print(f"5. Collect all: {t_collect:.3f}s")

    # ========================================
    # Compare: Collect WITHOUT scale_factors
    # ========================================
    t0 = time.perf_counter()

    collected_no_sf = pl.collect_all([positions_lf_weighted, lookthroughs_lf_weighted])

    t_collect_no_sf = time.perf_counter() - t0
    print(f"5b. Collect WITHOUT scale_factors: {t_collect_no_sf:.3f}s")

    # Summary
    total = t_baseline + t_rescaling + t_weight_cols + t_scale_factors + t_collect
    print(f"\n{'-'*40}")
    print(f"TOTAL: {total:.3f}s")
    print(f"\nBreakdown:")
    print(f"  Expressions + sync:  {t_baseline:.3f}s ({t_baseline/total*100:.1f}%)")
    print(f"  _apply_rescaling:    {t_rescaling:.3f}s ({t_rescaling/total*100:.1f}%)")
    print(f"  _build_weight_cols:  {t_weight_cols:.3f}s ({t_weight_cols/total*100:.1f}%)")
    print(f"  _build_scale_factors:{t_scale_factors:.3f}s ({t_scale_factors/total*100:.1f}%)")
    print(f"  Collect all:         {t_collect:.3f}s ({t_collect/total*100:.1f}%)")

    print(f"\nScale factors impact on collect: +{t_collect - t_collect_no_sf:.3f}s")

    positions_df = collected[0]
    lt_df = collected[1]
    sf_df = collected[2] if len(collected) > 2 else None

    print(f"\nShapes:")
    print(f"  Positions: {positions_df.shape}")
    print(f"  Lookthroughs: {lt_df.shape}")
    if sf_df is not None:
        print(f"  Scale factors: {sf_df.shape}")


if __name__ == "__main__":
    profile_isolated(1000, 1, 10)
    profile_isolated(1000, 1, 50)
    profile_isolated(1000, 1, 100)
    profile_isolated(1000, 1, 500)
