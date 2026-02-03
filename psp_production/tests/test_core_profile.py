"""
Profile core processing (excluding formatter) with raw dataframes.
"""

import time
import random
from typing import Dict, Any

import polars as pl

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.core.data_ingestion import DataIngestion
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


def profile_core(num_positions: int, num_perspectives: int):
    print(f"\n{'='*80}")
    print(f"CORE PROFILE: {num_positions:,} positions, {num_perspectives:,} perspectives")
    print(f"{'='*80}")

    # Generate data
    t0 = time.perf_counter()
    input_json = generate_data(num_positions)
    t_gen = time.perf_counter() - t0
    print(f"1. Generate data: {t_gen:.3f}s")

    # Setup engine
    t0 = time.perf_counter()
    engine = setup_engine(num_perspectives)
    perspective_configs = {"test_config": {}}
    for i in range(num_perspectives):
        pid = i + 1
        perspective_configs["test_config"][str(pid)] = ["scale_holdings_to_100_percent"] if i % 2 == 0 else []
    t_setup = time.perf_counter() - t0
    print(f"2. Setup engine: {t_setup:.3f}s")

    # Process with raw dataframes (skip formatter)
    t0 = time.perf_counter()
    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        return_raw_dataframes=True
    )
    t_process = time.perf_counter() - t0
    print(f"3. Process (raw dataframes): {t_process:.3f}s")

    # Extract LazyFrames
    positions_lf = result["positions"]
    lookthroughs_lf = result["lookthroughs"]
    scale_factors_lf = result["scale_factors"]

    # Collect manually
    t0 = time.perf_counter()
    to_collect = [positions_lf]
    if lookthroughs_lf is not None:
        to_collect.append(lookthroughs_lf)
    if scale_factors_lf is not None:
        to_collect.append(scale_factors_lf)

    collected = pl.collect_all(to_collect)
    t_collect = time.perf_counter() - t0
    print(f"4. Collect all: {t_collect:.3f}s")

    positions_df = collected[0]
    scale_factors_df = collected[-1] if scale_factors_lf is not None else None

    print(f"\nResults:")
    print(f"  Positions shape: {positions_df.shape}")
    print(f"  Positions columns: {len(positions_df.columns)}")
    if scale_factors_df is not None:
        print(f"  Scale factors shape: {scale_factors_df.shape}")

    total = t_gen + t_setup + t_process + t_collect
    print(f"\n  TOTAL: {total:.3f}s")

    return {
        "generate": t_gen,
        "setup": t_setup,
        "process": t_process,
        "collect": t_collect,
        "total": total,
        "positions_shape": positions_df.shape,
    }


if __name__ == "__main__":
    # Warmup
    print("Warmup run...")
    profile_core(1000, 10)

    # Push the limits
    print("\n" + "="*80)
    print("STRESS TESTS - Finding breaking points")
    print("="*80)

    tests = [
        (100000, 2000),
        (100000, 3000),
        (100000, 5000),
        (200000, 1000),
        (200000, 2000),
    ]

    for n_pos, n_persp in tests:
        try:
            profile_core(n_pos, n_persp)
        except Exception as e:
            print(f"FAILED at {n_pos:,} positions, {n_persp:,} perspectives: {e}")
