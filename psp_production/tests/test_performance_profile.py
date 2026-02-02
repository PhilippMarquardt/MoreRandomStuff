"""
Performance profiling test to identify bottlenecks.

Tests with up to 100k positions and 2000 perspectives.
"""

import time
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.core.data_ingestion import DataIngestion
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.models.rule import Rule


@dataclass
class TimingResult:
    name: str
    elapsed: float

    def __str__(self):
        return f"{self.name}: {self.elapsed:.3f}s"


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str, results: List[TimingResult]):
        self.name = name
        self.results = results

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.results.append(TimingResult(self.name, elapsed))


def generate_large_dataset(
    num_positions: int,
    num_lookthroughs_per_position: int = 0,
) -> Dict[str, Any]:
    """Generate large test dataset."""
    random.seed(42)

    positions = {}
    essential_lts = {}
    complete_lts = {}

    for i in range(num_positions):
        instrument_id = 1000 + i
        pos_id = f"pos_{i}"

        positions[pos_id] = {
            "instrument_id": instrument_id,
            "parent_instrument_id": None,
            "sub_portfolio_id": 1,
            "initial_weight": random.uniform(0.00001, 0.0001),
            "resulting_weight": random.uniform(0.00001, 0.0001),
        }

        for j in range(num_lookthroughs_per_position):
            lt_id = f"lt_{i}_{j}"
            lt_data = {
                "instrument_id": 100000 + i * 100 + j,
                "parent_instrument_id": instrument_id,
                "sub_portfolio_id": 1,
                "initial_weight": random.uniform(0.000001, 0.00001),
                "resulting_weight": random.uniform(0.000001, 0.00001),
            }
            if j == 0:
                essential_lts[lt_id] = lt_data
            else:
                complete_lts[lt_id] = lt_data

    return {
        "ed": "2024-01-15",
        "position_weight_labels": ["initial_weight", "resulting_weight"],
        "lookthrough_weight_labels": ["initial_weight", "resulting_weight"],
        "holding": {
            "positions": positions,
            "essential_lookthroughs": essential_lts,
            "complete_lookthroughs": complete_lts,
        }
    }


def generate_perspectives(num_perspectives: int) -> Dict[str, Dict[str, List[str]]]:
    """Generate perspective configs."""
    perspective_configs = {"test_config": {}}
    for i in range(num_perspectives):
        pid = i + 1
        # Alternate between with/without rescaling
        if i % 2 == 0:
            perspective_configs["test_config"][str(pid)] = ["scale_holdings_to_100_percent"]
        else:
            perspective_configs["test_config"][str(pid)] = []
    return perspective_configs


def setup_engine_with_perspectives(engine: PerspectiveEngine, num_perspectives: int):
    """Setup engine with many perspective rules."""
    for i in range(num_perspectives):
        pid = i + 1
        # Simple filter rule - keep positions where instrument_id % (pid+1) != 0
        # This creates varying filter rates per perspective
        engine.config.perspectives[pid] = [
            Rule(
                name=f"filter_{pid}",
                apply_to="both",
                criteria={
                    "column": "instrument_id",
                    "operator_type": ">=",
                    "value": 1000 + (i * 10)  # Different threshold per perspective
                },
                is_scaling_rule=False
            )
        ]


def profile_full_pipeline(
    num_positions: int,
    num_perspectives: int,
    num_lt_per_pos: int = 0,
) -> List[TimingResult]:
    """Profile the full pipeline with detailed timing."""
    results = []

    print(f"\n{'='*80}")
    print(f"PROFILING: {num_positions:,} positions, {num_perspectives:,} perspectives, {num_lt_per_pos} LT/pos")
    print(f"{'='*80}")

    # Generate data
    with Timer("1. Generate test data", results):
        input_json = generate_large_dataset(num_positions, num_lt_per_pos)

    # Setup engine
    with Timer("2. Setup engine & perspectives", results):
        engine = PerspectiveEngine()
        engine.config.default_modifiers = []
        setup_engine_with_perspectives(engine, num_perspectives)
        perspective_configs = generate_perspectives(num_perspectives)

    # Get weight labels
    with Timer("3. Get weight labels", results):
        weight_labels_map = DataIngestion.get_weight_labels(input_json)

    # Build dataframes
    with Timer("4. Build dataframes (extract + normalize)", results):
        positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(
            input_json, weight_labels_map
        )

    # Determine required tables
    with Timer("5. Determine required tables", results):
        required_tables = engine._determine_required_tables(perspective_configs)

    # Get original containers
    with Timer("6. Get original containers", results):
        original_containers = positions_lf.select("container").unique().collect().to_series().to_list()

    # Precompute nested criteria
    with Timer("7. Precompute nested criteria", results):
        precomputed_values = engine._precompute_nested_criteria(
            positions_lf, lookthroughs_lf, perspective_configs
        )

    # Build perspective plan
    with Timer("8. Build perspective plan (expressions)", results):
        processor = PerspectiveProcessor(engine.config)
        positions_lf, lookthroughs_lf, scale_factors_lf = processor.build_perspective_plan(
            positions_lf,
            lookthroughs_lf,
            perspective_configs,
            precomputed_values,
            weight_labels_map
        )

    # Collect all
    with Timer("9. Collect all (materialize)", results):
        to_collect = [positions_lf]
        has_lt = lookthroughs_lf is not None
        has_sf = scale_factors_lf is not None

        if has_lt:
            to_collect.append(lookthroughs_lf)
        if has_sf:
            to_collect.append(scale_factors_lf)

        collected = pl.collect_all(to_collect)

        positions_df = collected[0]
        lookthroughs_df = collected[1] if has_lt else pl.DataFrame()
        scale_factors_df = collected[-1] if has_sf else None

    # Format output
    with Timer("10. Format output", results):
        from perspective_service.core.output_formatter import OutputFormatter
        output = OutputFormatter.format_output(
            positions_df,
            lookthroughs_df,
            perspective_configs,
            weight_labels_map,
            original_containers,
            scale_factors_df
        )

    # Print results
    print("\nTiming breakdown:")
    total = 0
    for r in results:
        print(f"  {r}")
        total += r.elapsed
    print(f"  {'-'*40}")
    print(f"  TOTAL: {total:.3f}s")

    # Print data stats
    print(f"\nData stats:")
    print(f"  Positions rows: {len(positions_df):,}")
    print(f"  Positions cols: {len(positions_df.columns)}")
    print(f"  LT rows: {len(lookthroughs_df):,}")
    if scale_factors_df is not None:
        print(f"  Scale factors rows: {len(scale_factors_df):,}")

    return results


def profile_build_perspective_plan_detail(
    num_positions: int,
    num_perspectives: int,
) -> List[TimingResult]:
    """Profile build_perspective_plan in detail."""
    results = []

    print(f"\n{'='*80}")
    print(f"DETAILED PLAN PROFILING: {num_positions:,} positions, {num_perspectives:,} perspectives")
    print(f"{'='*80}")

    # Setup
    input_json = generate_large_dataset(num_positions, 0)
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []
    setup_engine_with_perspectives(engine, num_perspectives)
    perspective_configs = generate_perspectives(num_perspectives)

    weight_labels_map = DataIngestion.get_weight_labels(input_json)
    positions_lf, lookthroughs_lf = DataIngestion.build_dataframes(input_json, weight_labels_map)
    precomputed_values = {}

    processor = PerspectiveProcessor(engine.config)

    # Profile each part of build_perspective_plan
    print("\nProfiling build_perspective_plan internals...")

    # Part 1: Build factor expressions
    with Timer("8a. Build factor expressions loop", results):
        factor_expressions_pos = []
        factor_map = {}

        for config_name, perspective_map in perspective_configs.items():
            factor_map[config_name] = {}
            perspective_ids = sorted([int(k) for k in perspective_map.keys()])

            for perspective_id in perspective_ids:
                column_name = f"f_{config_name}_{perspective_id}"
                factor_map[config_name][perspective_id] = column_name

                modifier_names = perspective_map.get(str(perspective_id)) or []
                active_modifiers = processor._filter_overridden_modifiers(modifier_names)

                keep_expr = processor._build_keep_expression(
                    perspective_id, active_modifiers, precomputed_values
                )
                scale_expr = processor._build_scale_expression(
                    perspective_id, precomputed_values
                )
                factor_expressions_pos.append(
                    pl.when(keep_expr)
                    .then(scale_expr)
                    .otherwise(pl.lit(None))
                    .alias(column_name)
                )

    print(f"  Built {len(factor_expressions_pos)} factor expressions")

    # Part 2: Apply factor expressions
    with Timer("8b. Apply factor expressions (with_columns)", results):
        positions_lf = positions_lf.with_columns(factor_expressions_pos)

    # Part 2b: Apply rescaling (now includes sf_data computation)
    with Timer("8b2. Apply rescaling", results):
        positions_lf, _, sf_data = processor._apply_rescaling(
            positions_lf, None, perspective_configs, factor_map,
            False, precomputed_values, weight_labels_map
        )

    # Part 3: Build weight columns
    with Timer("8c. Build weight columns", results):
        positions_lf, _ = processor._build_weight_columns(
            positions_lf, None, factor_map, weight_labels_map
        )

    # Part 4: Build scale factors (uses precomputed sf_data)
    with Timer("8d. Build scale factors", results):
        scale_factors_lf = processor._build_scale_factors(
            sf_data, perspective_configs, factor_map, weight_labels_map
        )

    # Part 5: Collect
    with Timer("9. Collect all", results):
        to_collect = [positions_lf]
        if scale_factors_lf is not None:
            to_collect.append(scale_factors_lf)
        collected = pl.collect_all(to_collect)

    print("\nTiming breakdown:")
    total = 0
    for r in results:
        print(f"  {r}")
        total += r.elapsed
    print(f"  {'-'*40}")
    print(f"  TOTAL: {total:.3f}s")

    print(f"\nCollected positions shape: {collected[0].shape}")
    if len(collected) > 1:
        print(f"Collected scale_factors shape: {collected[1].shape}")

    return results


def run_scaling_tests():
    """Run tests at different scales to identify scaling bottlenecks."""
    print("\n" + "="*80)
    print("SCALING TESTS - Varying positions with fixed perspectives (10)")
    print("="*80)

    for num_positions in [1000, 5000, 10000]:
        profile_full_pipeline(num_positions, num_perspectives=10, num_lt_per_pos=0)

    print("\n" + "="*80)
    print("SCALING TESTS - Varying perspectives with fixed positions (5000)")
    print("="*80)

    for num_perspectives in [10, 25, 50, 75, 100]:
        profile_full_pipeline(num_positions=5000, num_perspectives=num_perspectives, num_lt_per_pos=0)

    print("\n" + "="*80)
    print("STRESS TEST - 10k positions, 100 perspectives")
    print("="*80)

    profile_full_pipeline(num_positions=10000, num_perspectives=100, num_lt_per_pos=0)

    print("\n" + "="*80)
    print("DETAILED PLAN PROFILE - 5k positions, 50 perspectives")
    print("="*80)

    profile_build_perspective_plan_detail(num_positions=5000, num_perspectives=50)


if __name__ == "__main__":
    run_scaling_tests()
