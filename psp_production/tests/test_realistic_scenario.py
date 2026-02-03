"""
Realistic scenario test: 1k positions, 1k lookthroughs.
Compare with and without scale factors.
"""

import time
import random
from typing import Dict, Any

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule
from perspective_service.models.enums import ApplyTo


def generate_realistic_data(num_positions: int, num_lt_per_position: int = 1) -> Dict[str, Any]:
    """Generate realistic test data with positions and lookthroughs."""
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
            "sub_portfolio_id": random.randint(1, 10),  # 10 sub-portfolios
            "initial_weight": random.uniform(0.0001, 0.01),
            "resulting_weight": random.uniform(0.0001, 0.01),
        }

        # Create lookthroughs
        for j in range(num_lt_per_position):
            lt_id = f"lt_{i}_{j}"
            lt_data = {
                "instrument_id": 100000 + i * 100 + j,
                "parent_instrument_id": instrument_id,
                "sub_portfolio_id": positions[pos_id]["sub_portfolio_id"],
                "initial_weight": random.uniform(0.00001, 0.001),
                "resulting_weight": random.uniform(0.00001, 0.001),
            }
            # Split between essential and complete
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
        },
    }


def setup_engine_realistic(num_perspectives: int) -> PerspectiveEngine:
    """Setup engine with realistic perspective rules."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    for i in range(num_perspectives):
        pid = i + 1
        # Use instrument_id filter (exists on both positions and lookthroughs)
        engine.config.perspectives[pid] = [
            Rule(
                name=f"filter_instrument_{pid}",
                apply_to=ApplyTo.BOTH,
                criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 5)},
                is_scaling_rule=False
            )
        ]

    return engine


def profile_scenario(
    num_positions: int,
    num_lt_per_position: int,
    num_perspectives: int,
    with_rescaling: bool,
    return_raw: bool = True,
) -> Dict:
    """Profile a specific scenario."""

    # Generate data
    t0 = time.perf_counter()
    input_json = generate_realistic_data(num_positions, num_lt_per_position)
    t_gen = time.perf_counter() - t0

    # Setup engine
    t0 = time.perf_counter()
    engine = setup_engine_realistic(num_perspectives)

    # Create perspective configs
    perspective_configs = {"config": {}}
    for i in range(num_perspectives):
        pid = i + 1
        if with_rescaling:
            perspective_configs["config"][str(pid)] = ["scale_holdings_to_100_percent"]
        else:
            perspective_configs["config"][str(pid)] = []
    t_setup = time.perf_counter() - t0

    # Process
    t0 = time.perf_counter()
    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        return_raw_dataframes=return_raw
    )
    t_process = time.perf_counter() - t0

    if return_raw:
        # Collect manually
        t0 = time.perf_counter()
        to_collect = [result["positions"]]
        if result["lookthroughs"] is not None:
            to_collect.append(result["lookthroughs"])
        if result["scale_factors"] is not None:
            to_collect.append(result["scale_factors"])

        collected = pl.collect_all(to_collect)
        t_collect = time.perf_counter() - t0

        positions_df = collected[0]
        lt_idx = 1 if result["lookthroughs"] is not None else -1
        lookthroughs_df = collected[1] if result["lookthroughs"] is not None else None
        scale_factors_df = collected[-1] if result["scale_factors"] is not None else None

        return {
            "generate": t_gen,
            "setup": t_setup,
            "process": t_process,
            "collect": t_collect,
            "total": t_gen + t_setup + t_process + t_collect,
            "positions_shape": positions_df.shape,
            "lookthroughs_shape": lookthroughs_df.shape if lookthroughs_df is not None else None,
            "scale_factors_shape": scale_factors_df.shape if scale_factors_df is not None else None,
        }
    else:
        return {
            "generate": t_gen,
            "setup": t_setup,
            "process": t_process,
            "collect": 0,
            "total": t_gen + t_setup + t_process,
        }


def run_comparison():
    """Run comparison tests."""

    print("="*80)
    print("REALISTIC SCENARIO: 1k positions, 1k lookthroughs")
    print("="*80)

    scenarios = [
        # (positions, lt_per_pos, perspectives, with_rescaling)
        (1000, 1, 10, False),
        (1000, 1, 10, True),
        (1000, 1, 50, False),
        (1000, 1, 50, True),
        (1000, 1, 100, False),
        (1000, 1, 100, True),
    ]

    print("\n" + "-"*80)
    print(f"{'Positions':<10} {'LTs':<8} {'Persp':<8} {'Rescale':<10} {'Process':<10} {'Collect':<10} {'Total':<10} {'Cols':<8}")
    print("-"*80)

    for num_pos, num_lt, num_persp, rescale in scenarios:
        result = profile_scenario(num_pos, num_lt, num_persp, rescale, return_raw=True)
        cols = result["positions_shape"][1] if result["positions_shape"] else 0
        print(f"{num_pos:<10} {num_pos*num_lt:<8} {num_persp:<8} {'Yes' if rescale else 'No':<10} "
              f"{result['process']:<10.3f} {result['collect']:<10.3f} {result['total']:<10.3f} {cols:<8}")

    print("\n" + "="*80)
    print("SCALING WITH MORE PERSPECTIVES")
    print("="*80)

    scenarios2 = [
        (1000, 1, 100, True),
        (1000, 1, 200, True),
        (1000, 1, 500, True),
        (1000, 1, 1000, True),
    ]

    print("\n" + "-"*80)
    print(f"{'Positions':<10} {'LTs':<8} {'Persp':<8} {'Process':<10} {'Collect':<10} {'Total':<10} {'Pos Cols':<10} {'SF Rows':<10}")
    print("-"*80)

    for num_pos, num_lt, num_persp, rescale in scenarios2:
        result = profile_scenario(num_pos, num_lt, num_persp, rescale, return_raw=True)
        pos_cols = result["positions_shape"][1] if result["positions_shape"] else 0
        sf_rows = result["scale_factors_shape"][0] if result["scale_factors_shape"] else 0
        print(f"{num_pos:<10} {num_pos*num_lt:<8} {num_persp:<8} "
              f"{result['process']:<10.3f} {result['collect']:<10.3f} {result['total']:<10.3f} "
              f"{pos_cols:<10} {sf_rows:<10}")

    print("\n" + "="*80)
    print("DETAILED BREAKDOWN: 1k positions, 1k LTs, 100 perspectives, WITH rescaling")
    print("="*80)

    result = profile_scenario(1000, 1, 100, True, return_raw=True)
    print(f"\n  Generate data:  {result['generate']:.3f}s")
    print(f"  Setup engine:   {result['setup']:.3f}s")
    print(f"  Process:        {result['process']:.3f}s")
    print(f"  Collect:        {result['collect']:.3f}s")
    print(f"  ----------------------------------------")
    print(f"  TOTAL:          {result['total']:.3f}s")
    print(f"\n  Positions shape:     {result['positions_shape']}")
    print(f"  Lookthroughs shape:  {result['lookthroughs_shape']}")
    print(f"  Scale factors shape: {result['scale_factors_shape']}")


def generate_production_data(num_positions: int, num_lookthroughs: int, num_weight_labels: int = 2) -> Dict[str, Any]:
    """Generate production-scale test data with specified positions and lookthroughs."""
    random.seed(42)

    positions = {}
    essential_lts = {}

    # Generate weight label names
    weight_labels = [f"weight_{i}" for i in range(num_weight_labels)]

    # Generate positions
    for i in range(num_positions):
        instrument_id = 1000 + i
        pos_id = f"pos_{i}"
        pos_data = {
            "instrument_id": instrument_id,
            "parent_instrument_id": None,
            "sub_portfolio_id": random.randint(1, 10),
        }
        for w in weight_labels:
            pos_data[w] = random.uniform(0.0001, 0.01)
        positions[pos_id] = pos_data

    # Generate lookthroughs (spread across positions)
    for j in range(num_lookthroughs):
        parent_pos_idx = j % num_positions
        parent_instrument_id = 1000 + parent_pos_idx
        lt_id = f"lt_{j}"
        lt_data = {
            "instrument_id": 100000 + j,
            "parent_instrument_id": parent_instrument_id,
            "sub_portfolio_id": positions[f"pos_{parent_pos_idx}"]["sub_portfolio_id"],
        }
        for w in weight_labels:
            lt_data[w] = random.uniform(0.00001, 0.001)
        essential_lts[lt_id] = lt_data

    return {
        "ed": "2024-01-15",
        "position_weight_labels": weight_labels,
        "lookthrough_weight_labels": weight_labels,
        "holding": {
            "positions": positions,
            "essential_lookthroughs": essential_lts,
            "complete_lookthroughs": {},
        },
    }


def run_production_scenario():
    """Run production-scale scenario: 70k positions, 10k lookthroughs, few perspectives."""

    print("\n" + "="*80)
    print("PRODUCTION SCENARIO: 70k positions, 10k lookthroughs")
    print("="*80)

    num_positions = 70000
    num_lookthroughs = 10000

    # Test with different perspective and weight label counts
    # Target: Build ~0.02s, Collect ~0.35s
    scenarios = [
        (3, 2),    # baseline
        (3, 20),   # many weight labels
        (10, 20),  # more perspectives + weights
        (20, 20),  # even more
        (50, 10),  # many perspectives
        (50, 20),  # many perspectives + weights
    ]

    for num_perspectives, num_weights in scenarios:
        print(f"\n--- {num_perspectives} perspectives, {num_weights} weight labels ---")

        print(f"Generating data: {num_positions:,} positions, {num_lookthroughs:,} lookthroughs...")
        t0 = time.perf_counter()
        input_json = generate_production_data(num_positions, num_lookthroughs, num_weights)
        t_gen = time.perf_counter() - t0
        print(f"  Data generation: {t_gen:.3f}s")

        # Setup engine
        engine = PerspectiveEngine()
        engine.config.default_modifiers = []
        for i in range(num_perspectives):
            pid = i + 1
            engine.config.perspectives[pid] = [
                Rule(
                    name=f"filter_{pid}",
                    apply_to=ApplyTo.BOTH,
                    criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 1000)},
                    is_scaling_rule=False
                )
            ]

        perspective_configs = {"config": {}}
        for i in range(num_perspectives):
            perspective_configs["config"][str(i + 1)] = ["scale_holdings_to_100_percent"]

        # Process (build plan)
        t0 = time.perf_counter()
        result = engine.process(
            input_json=input_json,
            perspective_configs=perspective_configs,
            return_raw_dataframes=True
        )
        t_process = time.perf_counter() - t0

        # Collect
        t0 = time.perf_counter()
        to_collect = [result["positions"]]
        if result["lookthroughs"] is not None:
            to_collect.append(result["lookthroughs"])
        if result["scale_factors"] is not None:
            to_collect.append(result["scale_factors"])

        collected = pl.collect_all(to_collect)
        t_collect = time.perf_counter() - t0

        positions_df = collected[0]

        print(f"  Build: {t_process:.3f}s | Collect: {t_collect:.3f}s | Total: {t_process + t_collect:.3f}s | Cols: {positions_df.shape[1]}")

    print(f"\n" + "-"*60)
    print(f"Target from production: Build ~0.02s, Collect ~0.35s")
    print(f"Closest match: ~20-50 perspectives with ~20 weight labels")
    print(f"Note: Production may have additional columns from reference data")

    # Test streaming mode
    print(f"\n" + "="*80)
    print("STREAMING MODE COMPARISON")
    print("="*80)

    num_perspectives = 20
    num_weights = 20
    print(f"\nScenario: {num_positions:,} pos, {num_lookthroughs:,} LT, {num_perspectives} persp, {num_weights} weights")

    input_json = generate_production_data(num_positions, num_lookthroughs, num_weights)

    engine = PerspectiveEngine()
    engine.config.default_modifiers = []
    for i in range(num_perspectives):
        pid = i + 1
        engine.config.perspectives[pid] = [
            Rule(
                name=f"filter_{pid}",
                apply_to=ApplyTo.BOTH,
                criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 1000)},
                is_scaling_rule=False
            )
        ]

    perspective_configs = {"config": {}}
    for i in range(num_perspectives):
        perspective_configs["config"][str(i + 1)] = ["scale_holdings_to_100_percent"]

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        return_raw_dataframes=True
    )

    to_collect = [result["positions"]]
    if result["lookthroughs"] is not None:
        to_collect.append(result["lookthroughs"])
    if result["scale_factors"] is not None:
        to_collect.append(result["scale_factors"])

    # Regular collect
    t0 = time.perf_counter()
    collected_regular = pl.collect_all(to_collect)
    t_regular = time.perf_counter() - t0

    # Streaming collect - need to rebuild LazyFrames since they were consumed
    result2 = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        return_raw_dataframes=True
    )
    to_collect2 = [result2["positions"]]
    if result2["lookthroughs"] is not None:
        to_collect2.append(result2["lookthroughs"])
    if result2["scale_factors"] is not None:
        to_collect2.append(result2["scale_factors"])

    t0 = time.perf_counter()
    collected_streaming = pl.collect_all(to_collect2, streaming=True)
    t_streaming = time.perf_counter() - t0

    print(f"\n  Regular collect:   {t_regular:.3f}s")
    print(f"  Streaming collect: {t_streaming:.3f}s")
    print(f"  Difference:        {t_streaming - t_regular:+.3f}s ({(t_streaming/t_regular - 1)*100:+.1f}%)")

    # Test with larger scenario
    print(f"\n--- Larger scenario: 50 persp, 20 weights ---")
    num_perspectives = 50
    num_weights = 20

    engine2 = PerspectiveEngine()
    engine2.config.default_modifiers = []
    for i in range(num_perspectives):
        pid = i + 1
        engine2.config.perspectives[pid] = [
            Rule(
                name=f"filter_{pid}",
                apply_to=ApplyTo.BOTH,
                criteria={"column": "instrument_id", "operator_type": ">=", "value": 1000 + (i * 1000)},
                is_scaling_rule=False
            )
        ]

    perspective_configs2 = {"config": {}}
    for i in range(num_perspectives):
        perspective_configs2["config"][str(i + 1)] = ["scale_holdings_to_100_percent"]

    input_json2 = generate_production_data(num_positions, num_lookthroughs, num_weights)

    # Regular
    result3 = engine2.process(input_json=input_json2, perspective_configs=perspective_configs2, return_raw_dataframes=True)
    to_collect3 = [result3["positions"]]
    if result3["lookthroughs"] is not None:
        to_collect3.append(result3["lookthroughs"])
    if result3["scale_factors"] is not None:
        to_collect3.append(result3["scale_factors"])

    t0 = time.perf_counter()
    pl.collect_all(to_collect3)
    t_regular2 = time.perf_counter() - t0

    # Streaming
    result4 = engine2.process(input_json=input_json2, perspective_configs=perspective_configs2, return_raw_dataframes=True)
    to_collect4 = [result4["positions"]]
    if result4["lookthroughs"] is not None:
        to_collect4.append(result4["lookthroughs"])
    if result4["scale_factors"] is not None:
        to_collect4.append(result4["scale_factors"])

    t0 = time.perf_counter()
    pl.collect_all(to_collect4, streaming=True)
    t_streaming2 = time.perf_counter() - t0

    print(f"  Regular collect:   {t_regular2:.3f}s")
    print(f"  Streaming collect: {t_streaming2:.3f}s")
    print(f"  Difference:        {t_streaming2 - t_regular2:+.3f}s ({(t_streaming2/t_regular2 - 1)*100:+.1f}%)")
    print(f"\nConclusion: Streaming is for memory efficiency, not speed.")


if __name__ == "__main__":
    run_comparison()
    run_production_scenario()
