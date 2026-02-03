"""
Speed test for perspective service.

Tests performance with 3000 positions and 9000 lookthroughs.
"""

import time
import random
from typing import Dict, List, Any

import pytest

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule
from perspective_service.models.enums import ApplyTo


# Test configuration
NUM_POSITIONS = 3000
LOOKTHROUGHS_PER_POSITION = 3
PERSPECTIVE_ID = 1
CONTAINER = "holding"


def generate_test_data(
    num_positions: int,
    lookthroughs_per_position: int,
) -> Dict[str, Any]:
    """Generate test input JSON with positions and lookthroughs."""
    random.seed(42)  # Reproducible results

    # Format: {id: {attributes}} - dict of dicts, not list
    positions = {}
    essential_lts = {}
    complete_lts = {}

    for i in range(num_positions):
        instrument_id = 1000 + i
        pos_id = f"pos_{i}"

        # Create position
        positions[pos_id] = {
            "instrument_id": instrument_id,
            "parent_instrument_id": None,
            "sub_portfolio_id": 1,
            "initial_weight": random.uniform(0.001, 0.01),
            "resulting_weight": random.uniform(0.001, 0.01),
        }

        # Create lookthroughs for this position
        for j in range(lookthroughs_per_position):
            lt_instrument_id = 100000 + i * 100 + j
            lt_id = f"lt_{i}_{j}"
            lt_data = {
                "instrument_id": lt_instrument_id,
                "parent_instrument_id": instrument_id,
                "sub_portfolio_id": 1,
                "initial_weight": random.uniform(0.0001, 0.001),
                "resulting_weight": random.uniform(0.0001, 0.001),
            }
            if j == 0:
                essential_lts[lt_id] = lt_data
            else:
                complete_lts[lt_id] = lt_data

    return {
        "ed": "2024-01-15",
        CONTAINER: {
            "positions": positions,
            "essential_lookthroughs": essential_lts,
            "complete_lookthroughs": complete_lts,
        }
    }


def test_speed_with_rescaling():
    """
    Speed test: 3000 positions + 9000 lookthroughs with rescaling enabled.

    Target: < 2 seconds
    """
    print(f"\n{'='*80}")
    print(f"SPEED TEST: {NUM_POSITIONS} positions, {NUM_POSITIONS * LOOKTHROUGHS_PER_POSITION} lookthroughs")
    print(f"{'='*80}")

    # Setup engine
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    # Add a filter rule that keeps ~50% of positions (instrument_id < 2500)
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_by_instrument",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "instrument_id", "operator_type": "<", "value": 1000 + NUM_POSITIONS // 2},
            is_scaling_rule=False
        )
    ]

    # Generate test data
    print("Generating test data...")
    input_json = generate_test_data(NUM_POSITIONS, LOOKTHROUGHS_PER_POSITION)

    # Add weight labels
    input_json["position_weight_labels"] = ["initial_weight", "resulting_weight"]
    input_json["lookthrough_weight_labels"] = ["initial_weight", "resulting_weight"]

    pos_count = len(input_json[CONTAINER]["positions"])
    elt_count = len(input_json[CONTAINER]["essential_lookthroughs"])
    clt_count = len(input_json[CONTAINER]["complete_lookthroughs"])
    print(f"  Positions: {pos_count}")
    print(f"  Essential LTs: {elt_count}")
    print(f"  Complete LTs: {clt_count}")
    print(f"  Total rows: {pos_count + elt_count + clt_count}")

    # Run with rescaling enabled
    modifiers = ["scale_holdings_to_100_percent"]
    input_json["perspective_configurations"] = {"test_config": {str(PERSPECTIVE_ID): modifiers}}

    print("\nRunning perspective engine with rescaling...")
    start = time.perf_counter()

    output = engine.process(input_json)

    elapsed = time.perf_counter() - start
    print(f"\n  Execution time: {elapsed:.3f} seconds")

    # Verify output structure
    container_data = (
        output
        .get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get(CONTAINER, {})
    )

    positions_out = container_data.get("positions") or {}
    scale_factors = container_data.get("scale_factors", {})

    print(f"  Positions kept: {len(positions_out)}")
    print(f"  Scale factors: {scale_factors}")

    # Assert performance
    assert elapsed < 2.0, f"Execution took {elapsed:.3f}s, expected < 2.0s"
    print(f"\n  [PASS] Performance test passed ({elapsed:.3f}s < 2.0s)")


def test_speed_without_rescaling():
    """
    Speed test: 3000 positions + 9000 lookthroughs without rescaling.

    Target: < 1 second
    """
    print(f"\n{'='*80}")
    print(f"SPEED TEST (no rescale): {NUM_POSITIONS} positions, {NUM_POSITIONS * LOOKTHROUGHS_PER_POSITION} lookthroughs")
    print(f"{'='*80}")

    # Setup engine
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    # Add a filter rule that keeps ~50% of positions
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_by_instrument",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "instrument_id", "operator_type": "<", "value": 1000 + NUM_POSITIONS // 2},
            is_scaling_rule=False
        )
    ]

    # Generate test data
    input_json = generate_test_data(NUM_POSITIONS, LOOKTHROUGHS_PER_POSITION)

    # Add weight labels
    input_json["position_weight_labels"] = ["initial_weight", "resulting_weight"]
    input_json["lookthrough_weight_labels"] = ["initial_weight", "resulting_weight"]

    # Run without rescaling
    modifiers = []
    input_json["perspective_configurations"] = {"test_config": {str(PERSPECTIVE_ID): modifiers}}

    print("Running perspective engine without rescaling...")
    start = time.perf_counter()

    output = engine.process(input_json)

    elapsed = time.perf_counter() - start
    print(f"\n  Execution time: {elapsed:.3f} seconds")

    # Verify output
    container_data = (
        output
        .get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get(CONTAINER, {})
    )

    positions_out = container_data.get("positions") or {}
    print(f"  Positions kept: {len(positions_out)}")

    # Assert performance
    assert elapsed < 1.0, f"Execution took {elapsed:.3f}s, expected < 1.0s"
    print(f"\n  [PASS] Performance test passed ({elapsed:.3f}s < 1.0s)")


if __name__ == "__main__":
    test_speed_with_rescaling()
    test_speed_without_rescaling()
