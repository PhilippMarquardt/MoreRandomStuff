"""
Test Per-Label Factor Normalization

This test verifies that each weight label is normalized independently.
Uses scale_holdings_to_100_percent modifier.

Test Data (4 weight labels):
  pos_1 (kept):     w1=0.5, w2=0.5, w3=0.6, w4=0.7
  pos_2 (removed):  w1=0.3, w2=0.3, w3=0.2, w4=0.1
  pos_3 (kept):     w1=0.2, w2=0.2, w3=0.2, w4=0.2

Expected (per-label normalization):
  Kept sums: w1_sum=0.7, w2_sum=0.7, w3_sum=0.8, w4_sum=0.9

  pos_1: w1=0.5/0.7≈0.714, w2=0.5/0.7≈0.714, w3=0.6/0.8=0.75, w4=0.7/0.9≈0.778
  pos_3: w1=0.2/0.7≈0.286, w2=0.2/0.7≈0.286, w3=0.2/0.8=0.25, w4=0.2/0.9≈0.222

  scale_factors: w1=0.7, w2=0.7, w3=0.8, w4=0.9

If BUG (single denominator for all):
  All weights would use same factor → w3 and w4 results would be wrong.
"""

import sys
import io
import json

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule


def test_per_label_normalization():
    print("=" * 80)
    print("TEST: Per-Label Factor Normalization")
    print("=" * 80)

    # Input with 4 weight labels
    input_json = {
        "position_weight_labels": ["w1", "w2", "w3", "w4"],
        "lookthrough_weight_labels": ["w1", "w2", "w3", "w4"],
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {
                    "instrument_id": 100,
                    "sub_portfolio_id": 1,
                    "w1": 0.5, "w2": 0.5, "w3": 0.6, "w4": 0.7,
                    "liquidity_type_id": 1  # Kept
                },
                "pos_2": {
                    "instrument_id": 200,
                    "sub_portfolio_id": 1,
                    "w1": 0.3, "w2": 0.3, "w3": 0.2, "w4": 0.1,
                    "liquidity_type_id": 2  # Removed
                },
                "pos_3": {
                    "instrument_id": 300,
                    "sub_portfolio_id": 1,
                    "w1": 0.2, "w2": 0.2, "w3": 0.2, "w4": 0.2,
                    "liquidity_type_id": 1  # Kept
                }
            }
        }
    }

    # Create engine
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Perspective rule: keep liquidity_type_id == 1
    rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[100] = [rule]

    # Use scale_holdings_to_100_percent modifier
    perspective_configs = {"config_test": {100: ["scale_holdings_to_100_percent"]}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["w1", "w2", "w3", "w4"],
        lookthrough_weights=["w1", "w2", "w3", "w4"],
        verbose=True
    )

    print("\n" + "=" * 80)
    print("OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))

    # Extract results
    container = result["perspective_configurations"]["config_test"][100]["portfolio"]
    positions = container.get("positions", {})
    scale_factors = container.get("scale_factors", {})

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    # Expected values (per-label normalization)
    expected = {
        "pos_1": {"w1": 0.5/0.7, "w2": 0.5/0.7, "w3": 0.6/0.8, "w4": 0.7/0.9},
        "pos_3": {"w1": 0.2/0.7, "w2": 0.2/0.7, "w3": 0.2/0.8, "w4": 0.2/0.9},
        "scale_factors": {"w1": 0.7, "w2": 0.7, "w3": 0.8, "w4": 0.9}
    }

    print("\nExpected values (per-label normalization):")
    print(f"  Kept sums: w1=0.7, w2=0.7, w3=0.8, w4=0.9")
    print(f"  pos_1: w1={expected['pos_1']['w1']:.4f}, w2={expected['pos_1']['w2']:.4f}, w3={expected['pos_1']['w3']:.4f}, w4={expected['pos_1']['w4']:.4f}")
    print(f"  pos_3: w1={expected['pos_3']['w1']:.4f}, w2={expected['pos_3']['w2']:.4f}, w3={expected['pos_3']['w3']:.4f}, w4={expected['pos_3']['w4']:.4f}")
    print(f"  scale_factors: w1={expected['scale_factors']['w1']}, w2={expected['scale_factors']['w2']}, w3={expected['scale_factors']['w3']}, w4={expected['scale_factors']['w4']}")

    print("\nActual values:")
    if "pos_1" in positions:
        p1 = positions["pos_1"]
        print(f"  pos_1: w1={p1.get('w1', 'N/A')}, w2={p1.get('w2', 'N/A')}, w3={p1.get('w3', 'N/A')}, w4={p1.get('w4', 'N/A')}")
    if "pos_3" in positions:
        p3 = positions["pos_3"]
        print(f"  pos_3: w1={p3.get('w1', 'N/A')}, w2={p3.get('w2', 'N/A')}, w3={p3.get('w3', 'N/A')}, w4={p3.get('w4', 'N/A')}")
    print(f"  scale_factors: {scale_factors}")

    # Check if per-label normalization is working
    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    all_pass = True
    tolerance = 0.0001

    for pos_id in ["pos_1", "pos_3"]:
        if pos_id not in positions:
            print(f"  [FAIL] {pos_id} missing from output")
            all_pass = False
            continue

        for w in ["w1", "w2", "w3", "w4"]:
            actual = positions[pos_id].get(w)
            exp = expected[pos_id][w]
            if actual is None:
                print(f"  [FAIL] {pos_id}.{w} is None")
                all_pass = False
            elif abs(actual - exp) < tolerance:
                print(f"  [PASS] {pos_id}.{w}: expected={exp:.4f}, actual={actual:.4f}")
            else:
                print(f"  [FAIL] {pos_id}.{w}: expected={exp:.4f}, actual={actual:.4f}, diff={abs(actual-exp):.6f}")
                all_pass = False

    # Check scale_factors
    for w in ["w1", "w2", "w3", "w4"]:
        actual = scale_factors.get(w)
        exp = expected["scale_factors"][w]
        if actual is None:
            print(f"  [FAIL] scale_factors.{w} is None")
            all_pass = False
        elif abs(actual - exp) < tolerance:
            print(f"  [PASS] scale_factors.{w}: expected={exp}, actual={actual}")
        else:
            print(f"  [FAIL] scale_factors.{w}: expected={exp}, actual={actual}")
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("RESULT: ALL TESTS PASSED - Per-label normalization is working correctly!")
    else:
        print("RESULT: TESTS FAILED - Per-label normalization is NOT working correctly!")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    success = test_per_label_normalization()
    sys.exit(0 if success else 1)
