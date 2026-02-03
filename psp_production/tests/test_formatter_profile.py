"""
Detailed profiling of OutputFormatter bottlenecks.
"""

import time
import random
from typing import Dict, List, Any

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.core.data_ingestion import DataIngestion
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.core.output_formatter import OutputFormatter
from perspective_service.models.rule import Rule
from perspective_service.models.enums import ApplyTo


def generate_data(num_positions: int) -> Dict[str, Any]:
    random.seed(42)
    positions = {}
    for i in range(num_positions):
        positions[f"pos_{i}"] = {
            "instrument_id": 1000 + i,
            "parent_instrument_id": None,
            "sub_portfolio_id": 1,
            "initial_weight": random.uniform(0.00001, 0.0001),
            "resulting_weight": random.uniform(0.00001, 0.0001),
        }
    return {
        "ed": "2024-01-15",
        "position_weight_labels": ["initial_weight", "resulting_weight"],
        "lookthrough_weight_labels": ["initial_weight", "resulting_weight"],
        "holding": {"positions": positions, "essential_lookthroughs": {}, "complete_lookthroughs": {}},
    }


def setup_engine(num_perspectives: int):
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []
    for i in range(num_perspectives):
        pid = i + 1
        engine.config.perspectives[pid] = [
            Rule(name=f"filter_{pid}", apply_to=ApplyTo.BOTH,
                 criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 10)},
                 is_scaling_rule=False)
        ]
    return engine


def profile_formatter(num_positions: int, num_perspectives: int):
    print(f"\n{'='*80}")
    print(f"FORMATTER PROFILE: {num_positions:,} positions, {num_perspectives} perspectives")
    print(f"{'='*80}")

    # Setup
    input_json = generate_data(num_positions)
    engine = setup_engine(num_perspectives)

    perspective_configs = {"test_config": {}}
    for i in range(num_perspectives):
        pid = i + 1
        perspective_configs["test_config"][str(pid)] = ["scale_holdings_to_100_percent"] if i % 2 == 0 else []

    weight_labels_map = DataIngestion.get_weight_labels(input_json)
    positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(input_json, weight_labels_map)

    processor = PerspectiveProcessor(engine.config)
    positions_lf, lookthroughs_lf, scale_factors_lf = processor.build_perspective_plan(
        positions_lf, lookthroughs_lf, perspective_configs, {}, weight_labels_map
    )

    # Build factor_map locally (for profiling - follows same pattern as processor)
    factor_map = {}
    for config_name, pmap in perspective_configs.items():
        factor_map[config_name] = {}
        for pid_str in pmap:
            pid = int(pid_str)
            factor_map[config_name][pid] = f"f_{config_name}_{pid}"

    # Collect
    to_collect = [positions_lf]
    if scale_factors_lf is not None:
        to_collect.append(scale_factors_lf)
    collected = pl.collect_all(to_collect)
    positions_df = collected[0]
    scale_factors_df = collected[1] if len(collected) > 1 else None
    lookthroughs_df = pl.DataFrame()
    original_containers = ["holding"]

    print(f"Positions shape: {positions_df.shape}")
    print(f"Scale factors shape: {scale_factors_df.shape if scale_factors_df is not None else None}")

    # Profile format_output in detail
    print("\nProfiling format_output internals...")

    # Simulate _process_dataframe_batch internals
    factor_columns = [col for pmap in factor_map.values() for col in pmap.values()]
    position_weights = ["initial_weight", "resulting_weight"]

    # Time: filter + select per perspective
    start = time.perf_counter()
    for config_name, perspective_map in factor_map.items():
        for perspective_id, col_name in perspective_map.items():
            # This is what _process_single_perspective does
            filtered = positions_df.filter(pl.col(col_name).is_not_null())
    filter_time = time.perf_counter() - start
    print(f"  filter() x {num_perspectives}: {filter_time:.3f}s")

    # Time: partition_by per perspective
    start = time.perf_counter()
    for config_name, perspective_map in factor_map.items():
        for perspective_id, col_name in perspective_map.items():
            filtered = positions_df.filter(pl.col(col_name).is_not_null())
            if not filtered.is_empty():
                partitions = filtered.partition_by(["container"], as_dict=True, maintain_order=False)
    partition_time = time.perf_counter() - start
    print(f"  filter() + partition_by() x {num_perspectives}: {partition_time:.3f}s")

    # Time: to_list per perspective
    start = time.perf_counter()
    for config_name, perspective_map in factor_map.items():
        for perspective_id, col_name in perspective_map.items():
            filtered = positions_df.filter(pl.col(col_name).is_not_null())
            if not filtered.is_empty():
                ids = filtered["identifier"].to_list()
                structs = filtered.select(pl.struct(["initial_weight", "resulting_weight"]).alias("_s"))["_s"].to_list()
    to_list_time = time.perf_counter() - start
    print(f"  filter() + to_list() x {num_perspectives}: {to_list_time:.3f}s")

    # Full format_output
    start = time.perf_counter()
    output = OutputFormatter.format_output(
        positions_df, lookthroughs_df, perspective_configs, weight_labels_map,
        original_containers, scale_factors_df
    )
    format_time = time.perf_counter() - start
    print(f"  FULL format_output(): {format_time:.3f}s")

    # Compute per-perspective cost
    print(f"\n  Per-perspective cost: {format_time/num_perspectives*1000:.2f}ms")


if __name__ == "__main__":
    profile_formatter(5000, 10)
    profile_formatter(5000, 50)
    profile_formatter(5000, 100)
    profile_formatter(10000, 100)
