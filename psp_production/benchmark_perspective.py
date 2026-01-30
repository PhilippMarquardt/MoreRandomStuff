"""
Benchmark script for perspective processing.
Tests the PerspectiveProcessor with synthetic data to identify bottlenecks.
"""
import time
import random
import polars as pl
from typing import Dict, List, Tuple

# Add path for imports
import sys
sys.path.insert(0, '.')

from perspective_service.core.configuration_manager import ConfigurationManager
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.models.rule import Rule


def create_test_data(num_positions: int = 100000, num_lookthroughs: int = 200000) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create synthetic position and lookthrough data."""
    random.seed(42)  # Reproducible

    # Positions
    positions_data = {
        "instrument_id": list(range(num_positions)),
        "sub_portfolio_id": [random.randint(1, 100) for _ in range(num_positions)],
        "container": [random.choice(["container_a", "container_b"]) for _ in range(num_positions)],
        "position_type": ["position"] * num_positions,
        "weight": [random.uniform(0.001, 0.05) for _ in range(num_positions)],
        "currency_id": [random.choice([1, 2, 3]) for _ in range(num_positions)],
        "instrument_subtype_id": [random.choice([27, 38, 66, 81, 84, 1, 2, 3, 4, 5]) for _ in range(num_positions)],  # Include values for scale_lookthroughs criteria
        "liquidity_type_id": [random.choice([1, 2, 3, 4, 5, 6]) for _ in range(num_positions)],
        "position_source_type_id": [random.choice([1, 8, 9, 10, 11]) for _ in range(num_positions)],
        "simulated_trade_id": [random.randint(1, 1000) if random.random() > 0.8 else None for _ in range(num_positions)],
        "perspective_id": [None] * num_positions,
        "is_class_position": [random.choice([True, False]) for _ in range(num_positions)],
        "trade_type_id": [random.choice([1, 2, 3, 4]) for _ in range(num_positions)],
        "upcoming_trade_date": [random.choice([True, False]) for _ in range(num_positions)],
        "red_flag_exclusion_type_id": [random.randint(1, 5) if random.random() > 0.9 else None for _ in range(num_positions)],
        "position_blocking_type_id": [random.randint(1, 3) if random.random() > 0.95 else None for _ in range(num_positions)],
        "parent_instrument_subtype_id": [random.randint(1, 100) for _ in range(num_positions)],
    }

    # Lookthroughs
    lookthroughs_data = {
        "parent_instrument_id": [random.randint(0, num_positions - 1) for _ in range(num_lookthroughs)],
        "instrument_id": [num_positions + i for i in range(num_lookthroughs)],
        "sub_portfolio_id": [random.randint(1, 100) for _ in range(num_lookthroughs)],
        "container": [random.choice(["container_a", "container_b"]) for _ in range(num_lookthroughs)],
        "position_type": ["lookthrough"] * num_lookthroughs,
        "record_type": ["essential_lookthroughs"] * num_lookthroughs,
        "weight": [random.uniform(0.001, 0.05) for _ in range(num_lookthroughs)],
        "currency_id": [random.choice([1, 2, 3]) for _ in range(num_lookthroughs)],
        "instrument_subtype_id": [random.randint(1, 100) for _ in range(num_lookthroughs)],
        "liquidity_type_id": [random.choice([1, 2, 3, 4, 5, 6]) for _ in range(num_lookthroughs)],
        "position_source_type_id": [random.choice([1, 8, 9, 10, 11]) for _ in range(num_lookthroughs)],
        "simulated_trade_id": [None] * num_lookthroughs,
        "perspective_id": [None] * num_lookthroughs,
        "is_class_position": [False] * num_lookthroughs,
        "trade_type_id": [1] * num_lookthroughs,
        "upcoming_trade_date": [False] * num_lookthroughs,
        "red_flag_exclusion_type_id": [None] * num_lookthroughs,
        "position_blocking_type_id": [None] * num_lookthroughs,
        "parent_instrument_subtype_id": [random.randint(1, 100) for _ in range(num_lookthroughs)],
    }

    positions_lf = pl.LazyFrame(positions_data)
    lookthroughs_lf = pl.LazyFrame(lookthroughs_data)

    return positions_lf, lookthroughs_lf


def create_perspective_configs(num_perspectives: int = 10) -> Dict:
    """Create perspective configurations with various modifiers."""
    perspective_configs = {
        "config_1": {}
    }

    # Include scale_lookthroughs_to_100_percent to test join-based instrument matching
    modifiers_options = [
        [],
        ["exclude_other_net_assets"],
        ["exclude_simulated_cash"],
        ["exclude_simulated_trades"],
        ["scale_lookthroughs_to_100_percent"],  # Tests join-based matching (criteria: instrument_subtype_id IN [27,38,66,81,84])
    ]

    for pid in range(1, num_perspectives + 1):
        modifiers = modifiers_options[pid % len(modifiers_options)]
        perspective_configs["config_1"][str(pid)] = modifiers

    return perspective_configs


def setup_config_manager() -> ConfigurationManager:
    """Create ConfigurationManager without database."""
    config = ConfigurationManager(db_loader=None)

    # Add 700 test perspectives with rules
    for pid in range(1, 10000):
        config.perspectives[pid] = [
            Rule(
                name="rule_0",
                apply_to="both",
                criteria={"column": "currency_id", "operator_type": "In", "value": [1, 2, 3]},
                condition_for_next_rule=None,
                is_scaling_rule=False,
                scale_factor=1.0
            )
        ]

    return config


def run_benchmark(num_positions: int = 100000, num_perspectives: int = 10):
    """Run the benchmark."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {num_positions:,} positions, {num_perspectives} perspectives")
    print(f"{'='*60}")

    timings = {}

    # Create test data
    print("\n[1] Creating test data...")
    t0 = time.perf_counter()
    positions_lf, lookthroughs_lf = create_test_data(num_positions, num_positions)
    timings["create_test_data"] = time.perf_counter() - t0
    print(f"    Done: {timings['create_test_data']:.3f}s")

    # Create configs
    print("\n[2] Creating configurations...")
    t0 = time.perf_counter()
    config_manager = setup_config_manager()
    perspective_configs = create_perspective_configs(num_perspectives)
    timings["create_configs"] = time.perf_counter() - t0
    print(f"    Done: {timings['create_configs']:.3f}s")

    # Create processor
    processor = PerspectiveProcessor(config_manager)

    # Run perspective processing with internal timing
    print("\n[3] Running perspective processing...")
    t0 = time.perf_counter()

    # Expression building phase
    t_expr = time.perf_counter()
    factor_expressions_pos = []
    factor_expressions_lt = []
    metadata_map = {}
    has_lookthroughs = True

    for config_name, perspective_map in perspective_configs.items():
        metadata_map[config_name] = {}
        perspective_ids = sorted([int(k) for k in perspective_map.keys()])

        for perspective_id in perspective_ids:
            column_name = f"f_{config_name}_{perspective_id}"
            metadata_map[config_name][perspective_id] = column_name

            modifier_names = perspective_map.get(str(perspective_id)) or []
            active_modifiers = processor._filter_overridden_modifiers(modifier_names)

            keep_expr = processor._build_keep_expression(
                perspective_id, active_modifiers, {}
            )
            scale_expr = processor._build_scale_expression(perspective_id, {})

            factor_expr = (
                pl.when(keep_expr)
                .then(scale_expr)
                .otherwise(pl.lit(None))
                .alias(column_name)
            )
            factor_expressions_pos.append(factor_expr)
            factor_expressions_lt.append(factor_expr)

    timings["expression_building"] = time.perf_counter() - t_expr
    print(f"    Expression building: {timings['expression_building']:.3f}s")

    # with_columns phase
    t_apply = time.perf_counter()
    positions_lf = positions_lf.with_columns(factor_expressions_pos)
    lookthroughs_lf = lookthroughs_lf.with_columns(factor_expressions_lt)
    timings["with_columns_apply"] = time.perf_counter() - t_apply
    print(f"    with_columns apply: {timings['with_columns_apply']:.3f}s")

    # Synchronize lookthroughs
    t_sync = time.perf_counter()
    all_columns = [c for m in metadata_map.values() for c in m.values()]
    lookthroughs_lf = processor._synchronize_lookthroughs(
        lookthroughs_lf, positions_lf, all_columns
    )
    timings["lookthrough_sync"] = time.perf_counter() - t_sync
    print(f"    Lookthrough sync: {timings['lookthrough_sync']:.3f}s")

    # Rescaling
    t_rescale = time.perf_counter()
    positions_lf, lookthroughs_lf, sf_data = processor._apply_rescaling(
        positions_lf,
        lookthroughs_lf,
        perspective_configs,
        metadata_map,
        has_lookthroughs,
        {},
        weight_labels_map
    )
    timings["rescaling"] = time.perf_counter() - t_rescale
    print(f"    Rescaling: {timings['rescaling']:.3f}s")

    timings["total_build_plan"] = time.perf_counter() - t0
    print(f"    Total build plan: {timings['total_build_plan']:.3f}s")

    # Analyze query plan BEFORE collect
    print("\n[4] Analyzing query plan...")
    print("\n--- POSITIONS QUERY PLAN ---")
    query_plan = positions_lf.explain()
    # Handle unicode characters for Windows console
    print(query_plan.encode('ascii', 'replace').decode('ascii'))

    print("\n--- POSITIONS STREAMING PLAN ---")
    try:
        streaming_plan = positions_lf.explain(streaming=True)
        print(streaming_plan.encode('ascii', 'replace').decode('ascii'))
        if "STREAMING" in streaming_plan.upper():
            print("\n>>> Streaming IS available <<<")
        else:
            print("\n>>> Streaming NOT available <<<")
    except Exception as e:
        print(f"Error getting streaming plan: {e}")

    # Save full plans to files for detailed analysis
    with open("query_plan.txt", "w", encoding="utf-8") as f:
        f.write("=== POSITIONS QUERY PLAN ===\n")
        f.write(query_plan)
        f.write("\n\n=== STREAMING PLAN ===\n")
        f.write(streaming_plan)

    # Final collect (regular)
    print("\n[5] Collecting results (regular)...")
    t_collect = time.perf_counter()
    positions_df = positions_lf.collect()
    lookthroughs_df = lookthroughs_lf.collect()
    timings["final_collect"] = time.perf_counter() - t_collect
    print(f"    Done: {timings['final_collect']:.3f}s")
    print(f"    Positions rows: {len(positions_df):,}")
    print(f"    Lookthroughs rows: {len(lookthroughs_df):,}")

    # Final collect (streaming)
    print("\n[6] Collecting results (streaming)...")
    t_stream = time.perf_counter()
    positions_df_s = positions_lf.collect(streaming=True)
    lookthroughs_df_s = lookthroughs_lf.collect(streaming=True)
    timings["streaming_collect"] = time.perf_counter() - t_stream
    print(f"    Done: {timings['streaming_collect']:.3f}s")

    # Summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    for phase, duration in timings.items():
        pct = (duration / timings.get("final_collect", duration)) * 100 if "final_collect" in timings else 0
        print(f"  {phase:25s}: {duration:8.3f}s")
    print(f"{'='*60}")

    return timings


if __name__ == "__main__":
    # Run benchmark with 50k positions, 50k lookthroughs, 700 perspectives
    run_benchmark(num_positions=500000, num_perspectives=1500)
