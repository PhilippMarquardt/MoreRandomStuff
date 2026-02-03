"""
Advanced Scenarios Test - Tests for complex edge cases and regressions.

Test Categories:
1. rule_result + PreProcessing + Group Savior (Critical Regression)
2. rule_result with NotIn operator
3. PreProcessing Propagation to Lookthroughs (Sync Semantics)
4. apply_to Scoping Tests
5. Real-World Multi-Weight-Label Test

These tests verify the fix: rule_expr=expr & rule_expr in PostProcessing,
ensuring rows removed by PreProcessing can't be "seen" by rule_result references.
"""

import sys
import os
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Note: sys.stdout encoding fix removed as it causes pytest issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier
from perspective_service.models.enums import ApplyTo, ModifierType, LogicalOperator


PERSPECTIVE_ID = 100
TNA = 1_000_000.0


# =============================================================================
# Category 1: rule_result + PreProcessing + Group Savior (Critical Regression)
# =============================================================================

def test_rule_result_preprocess_group_savior():
    """
    CRITICAL REGRESSION TEST

    Scenario:
    - Two rows share simulated_trade_id = 100
    - Row A: exclude_me=True (preproc removes), filter_a=True (would pass rule)
    - Row B: exclude_me=False, filter_a=False (fails rule)
    - PostProcessing modifier with rule_result_operator='or' and criteria:
      {"column": "simulated_trade_id", "operator_type": "In",
       "value": {"table_name": "rule_result", "column": "simulated_trade_id"}}

    Expected (Legacy behavior):
    - A removed before rules -> group has no passer -> B NOT saved

    If the fix (rule_expr=expr & rule_expr) is NOT applied:
    - B would be incorrectly saved because A's rule result would still be visible
    """
    print("\n" + "=" * 90)
    print("TEST: rule_result + PreProcessing + Group Savior (Critical Regression)")
    print("=" * 90)

    # Create test data
    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_a": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.40,
                    "simulated_trade_id": 100,  # Same group as B
                    "exclude_me": True,          # PreProc removes this
                    "filter_a": True,            # Would pass filter
                },
                "pos_b": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.30,
                    "simulated_trade_id": 100,  # Same group as A
                    "exclude_me": False,         # Not excluded by PreProc
                    "filter_a": False,           # Fails filter
                },
                "pos_c": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.30,
                    "simulated_trade_id": 200,  # Different group
                    "exclude_me": False,
                    "filter_a": True,            # Passes filter
                },
            },
        }
    }

    # Setup engine
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # PreProcessing: exclude where exclude_me=True
    engine.config.modifiers["exclude_flagged"] = Modifier(
        name="exclude_flagged",
        modifier_type=ModifierType.PRE_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={"column": "exclude_me", "operator_type": "==", "value": True},
        rule_result_operator=None,
        override_modifiers=[],
    )

    # PostProcessing: savior based on rule_result (simulated_trade_id In rule_result)
    # This should save any position whose simulated_trade_id has ANY position that passed rules
    engine.config.modifiers["trade_savior"] = Modifier(
        name="trade_savior",
        modifier_type=ModifierType.POST_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={
            "column": "simulated_trade_id",
            "operator_type": "In",
            "value": {
                "table_name": "rule_result",
                "column": "simulated_trade_id"
            }
        },
        rule_result_operator=LogicalOperator.OR,
        override_modifiers=[],
    )

    # Filtering rule: keep where filter_a=True
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    modifiers = ["exclude_flagged", "trade_savior"]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): modifiers}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify results
    container_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )
    positions = container_data.get("positions", {})
    kept = set(positions.keys())

    # Expected: pos_a removed by PreProc, pos_b fails filter and NOT saved (group has no passer after PreProc)
    # pos_c passes filter and kept
    expected_kept = {"pos_c"}

    print(f"\n  Kept positions: {sorted(kept)}")
    print(f"  Expected kept:  {sorted(expected_kept)}")

    if kept == expected_kept:
        print("\n  [PASS] rule_result + PreProcessing correctly handled")
        return True
    else:
        if "pos_b" in kept:
            print("\n  [FAIL] pos_b was incorrectly saved!")
            print("         This indicates rule_result saw pos_a's result even though PreProc removed it")
            print("         The fix (rule_expr=expr & rule_expr) may not be applied")
        else:
            missing = expected_kept - kept
            extra = kept - expected_kept
            if missing:
                print(f"\n  [FAIL] Missing positions: {missing}")
            if extra:
                print(f"\n  [FAIL] Extra positions: {extra}")
        return False


def test_rule_result_preprocess_group_savior_multiple_groups():
    """
    Extended test with multiple groups to ensure correctness.

    Groups:
    - simulated_trade_id=100: A(preproc removed, would pass), B(kept, fails) -> B NOT saved
    - simulated_trade_id=200: C(kept, passes) -> C kept
    - simulated_trade_id=300: D(kept, fails), E(kept, fails) -> D,E NOT saved
    - simulated_trade_id=400: F(kept, passes), G(kept, fails) -> F,G both saved
    """
    print("\n" + "=" * 90)
    print("TEST: rule_result + PreProcessing with Multiple Groups")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                # Group 100: A removed by preproc, B fails -> B NOT saved
                "pos_a": {"instrument_id": 1, "sub_portfolio_id": 100, "initial_weight": 0.10,
                          "simulated_trade_id": 100, "exclude_me": True, "filter_a": True},
                "pos_b": {"instrument_id": 2, "sub_portfolio_id": 100, "initial_weight": 0.10,
                          "simulated_trade_id": 100, "exclude_me": False, "filter_a": False},
                # Group 200: C passes -> C kept
                "pos_c": {"instrument_id": 3, "sub_portfolio_id": 100, "initial_weight": 0.20,
                          "simulated_trade_id": 200, "exclude_me": False, "filter_a": True},
                # Group 300: D,E both fail -> neither saved
                "pos_d": {"instrument_id": 4, "sub_portfolio_id": 100, "initial_weight": 0.15,
                          "simulated_trade_id": 300, "exclude_me": False, "filter_a": False},
                "pos_e": {"instrument_id": 5, "sub_portfolio_id": 100, "initial_weight": 0.15,
                          "simulated_trade_id": 300, "exclude_me": False, "filter_a": False},
                # Group 400: F passes, G fails -> both saved by savior
                "pos_f": {"instrument_id": 6, "sub_portfolio_id": 100, "initial_weight": 0.15,
                          "simulated_trade_id": 400, "exclude_me": False, "filter_a": True},
                "pos_g": {"instrument_id": 7, "sub_portfolio_id": 100, "initial_weight": 0.15,
                          "simulated_trade_id": 400, "exclude_me": False, "filter_a": False},
            },
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.modifiers["exclude_flagged"] = Modifier(
        name="exclude_flagged",
        modifier_type=ModifierType.PRE_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={"column": "exclude_me", "operator_type": "==", "value": True},
        rule_result_operator=None,
        override_modifiers=[],
    )

    engine.config.modifiers["trade_savior"] = Modifier(
        name="trade_savior",
        modifier_type=ModifierType.POST_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={
            "column": "simulated_trade_id",
            "operator_type": "In",
            "value": {"table_name": "rule_result", "column": "simulated_trade_id"}
        },
        rule_result_operator=LogicalOperator.OR,
        override_modifiers=[],
    )

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["exclude_flagged", "trade_savior"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    container_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )
    kept = set(container_data.get("positions", {}).keys())

    # Group 100: A removed by preproc, B fails, no passer in group -> B NOT saved
    # Group 200: C passes -> kept
    # Group 300: D,E both fail, no passer -> NOT saved
    # Group 400: F passes, G fails, F is passer -> both saved
    expected_kept = {"pos_c", "pos_f", "pos_g"}

    print(f"\n  Kept positions: {sorted(kept)}")
    print(f"  Expected kept:  {sorted(expected_kept)}")

    errors = []
    if "pos_b" in kept:
        errors.append("pos_b saved but group 100's only passer (A) was removed by preproc")
    if "pos_d" in kept or "pos_e" in kept:
        errors.append("pos_d/pos_e saved but group 300 has no passers")
    if "pos_f" not in kept:
        errors.append("pos_f should be kept (passes filter)")
    if "pos_g" not in kept:
        errors.append("pos_g should be saved by savior (group 400 has passer F)")
    if "pos_c" not in kept:
        errors.append("pos_c should be kept (passes filter)")

    if not errors:
        print("\n  [PASS] Multiple groups handled correctly")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


# =============================================================================
# Category 2: rule_result with NotIn operator
# =============================================================================

def test_rule_result_notin_or_savior():
    """
    Test NotIn with OR savior logic (more realistic use case).

    OR logic: final = rule_expr OR savior_expr
    Savior with NotIn: "Save position if its group has NO passers"

    Scenario:
    - pos_a: simulated_trade_id=100, passes filter -> kept by rule
    - pos_b: simulated_trade_id=100, fails filter, group has passer -> NOT saved
    - pos_c: simulated_trade_id=200, fails filter, group has no passer -> SAVED by NotIn
    - pos_d: simulated_trade_id=NULL, fails filter -> SAVED (NULL handling)
    """
    print("\n" + "=" * 90)
    print("TEST: rule_result with NotIn + OR savior logic")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_a": {"instrument_id": 1, "sub_portfolio_id": 100, "initial_weight": 0.25,
                          "simulated_trade_id": 100, "filter_a": True},
                "pos_b": {"instrument_id": 2, "sub_portfolio_id": 100, "initial_weight": 0.25,
                          "simulated_trade_id": 100, "filter_a": False},
                "pos_c": {"instrument_id": 3, "sub_portfolio_id": 100, "initial_weight": 0.25,
                          "simulated_trade_id": 200, "filter_a": False},
                "pos_d": {"instrument_id": 4, "sub_portfolio_id": 100, "initial_weight": 0.25,
                          "simulated_trade_id": None, "filter_a": False},
            },
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # PostProcessing with NotIn + OR: Save if group has no passers
    engine.config.modifiers["save_orphan_groups"] = Modifier(
        name="save_orphan_groups",
        modifier_type=ModifierType.POST_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={
            "column": "simulated_trade_id",
            "operator_type": "NotIn",
            "value": {"table_name": "rule_result", "column": "simulated_trade_id"}
        },
        rule_result_operator=LogicalOperator.OR,  # Keep if (filter OR NotIn group that passes)
        override_modifiers=[],
    )

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["save_orphan_groups"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    container_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )
    kept = set(container_data.get("positions", {}).keys())

    # pos_a: filter=True -> kept by rule
    # pos_b: filter=False, NotIn(100 which has passer)=False -> (False OR False) = False -> removed
    # pos_c: filter=False, NotIn(200 which has no passer)=True -> (False OR True) = True -> saved
    # pos_d: filter=False, NotIn(NULL)=True (NULL handling) -> (False OR True) = True -> saved
    expected_kept = {"pos_a", "pos_c", "pos_d"}

    print(f"\n  Kept positions: {sorted(kept)}")
    print(f"  Expected kept:  {sorted(expected_kept)}")

    if kept == expected_kept:
        print("\n  [PASS] NotIn with OR savior logic works correctly")
        return True
    else:
        errors = []
        if "pos_b" in kept:
            errors.append("pos_b should be removed (group 100 has passer)")
        if "pos_c" not in kept:
            errors.append("pos_c should be saved (group 200 has no passers)")
        if "pos_d" not in kept:
            errors.append("pos_d should be saved (NULL simulated_trade_id)")
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


# =============================================================================
# Category 3: PreProcessing Propagation to Lookthroughs (Sync Semantics)
# =============================================================================

def test_preprocess_propagation_to_lookthroughs():
    """
    Test that PreProcessing removal propagates to lookthroughs.

    When a position is removed by PreProcessing, its child lookthroughs
    should also be removed (factor becomes NULL).

    Scenario:
    - pos_1: instrument_id=1, exclude_me=True -> removed by preproc
    - pos_2: instrument_id=2, exclude_me=False -> kept
    - elt_1: parent_instrument_id=1 -> should be removed (parent removed)
    - elt_2: parent_instrument_id=2 -> should be kept (parent kept)
    """
    print("\n" + "=" * 90)
    print("TEST: PreProcessing Propagation to Lookthroughs")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": True,   # PreProc removes
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": False,  # Kept
                },
            },
            "essential_lookthroughs": {
                "elt_1": {
                    "instrument_id": 101,
                    "parent_instrument_id": 1,  # Parent removed
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "weight": 0.30,
                    "exclude_me": False,  # Column must exist for schema consistency
                },
                "elt_2": {
                    "instrument_id": 102,
                    "parent_instrument_id": 2,  # Parent kept
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "weight": 0.70,
                    "exclude_me": False,  # Column must exist for schema consistency
                },
            },
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.modifiers["exclude_flagged"] = Modifier(
        name="exclude_flagged",
        modifier_type=ModifierType.PRE_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={"column": "exclude_me", "operator_type": "==", "value": True},
        rule_result_operator=None,
        override_modifiers=[],
    )

    # No filtering rules - just pass all
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["exclude_flagged"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    container_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    positions = container_data.get("positions", {})
    essential_lts = container_data.get("essential_lookthroughs", {})

    kept_positions = set(positions.keys())
    kept_lts = set(essential_lts.keys())

    print(f"\n  Kept positions: {sorted(kept_positions)}")
    print(f"  Kept essential LTs: {sorted(kept_lts)}")

    expected_positions = {"pos_2"}
    expected_lts = {"elt_2"}

    errors = []
    if kept_positions != expected_positions:
        errors.append(f"Position mismatch: expected {expected_positions}, got {kept_positions}")
    if kept_lts != expected_lts:
        errors.append(f"LT mismatch: expected {expected_lts}, got {kept_lts}")

    if not errors:
        print("\n  [PASS] PreProcessing correctly propagates to lookthroughs")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


def test_preprocess_shared_parent():
    """
    Test when same instrument_id exists via multiple positions.

    If a lookthrough's parent is removed by preproc, but another position
    with the same instrument_id exists and is NOT removed, the lookthrough
    should still be kept.

    Scenario:
    - pos_1: instrument_id=1, sub_portfolio_id=100, exclude_me=True -> removed
    - pos_2: instrument_id=1, sub_portfolio_id=200, exclude_me=False -> kept
    - elt_1: parent_instrument_id=1, sub_portfolio_id=100 -> removed (parent in same sub_portfolio removed)
    - elt_2: parent_instrument_id=1, sub_portfolio_id=200 -> kept (parent in same sub_portfolio kept)
    """
    print("\n" + "=" * 90)
    print("TEST: PreProcessing with Shared Parent (sub_portfolio scoping)")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.25,
                    "exclude_me": True,   # Removed
                },
                "pos_2": {
                    "instrument_id": 1,       # Same instrument_id
                    "sub_portfolio_id": 200,  # Different sub_portfolio
                    "initial_weight": 0.25,
                    "exclude_me": False,  # Kept
                },
                "pos_3": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": False,
                },
            },
            "essential_lookthroughs": {
                "elt_1": {
                    "instrument_id": 101,
                    "parent_instrument_id": 1,
                    "sub_portfolio_id": 100,  # Same sub_portfolio as removed pos_1
                    "initial_weight": 0.0,
                    "weight": 0.50,
                    "exclude_me": False,  # Column must exist for schema consistency
                },
                "elt_2": {
                    "instrument_id": 102,
                    "parent_instrument_id": 1,
                    "sub_portfolio_id": 200,  # Same sub_portfolio as kept pos_2
                    "initial_weight": 0.0,
                    "weight": 0.50,
                    "exclude_me": False,  # Column must exist for schema consistency
                },
            },
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.modifiers["exclude_flagged"] = Modifier(
        name="exclude_flagged",
        modifier_type=ModifierType.PRE_PROCESSING,
        apply_to=ApplyTo.BOTH,
        criteria={"column": "exclude_me", "operator_type": "==", "value": True},
        rule_result_operator=None,
        override_modifiers=[],
    )

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["exclude_flagged"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    container_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    positions = container_data.get("positions", {})
    essential_lts = container_data.get("essential_lookthroughs", {})

    kept_positions = set(positions.keys())
    kept_lts = set(essential_lts.keys())

    print(f"\n  Kept positions: {sorted(kept_positions)}")
    print(f"  Kept essential LTs: {sorted(kept_lts)}")

    # pos_1 removed (exclude_me=True)
    # pos_2, pos_3 kept
    expected_positions = {"pos_2", "pos_3"}

    # elt_1: parent=1, sub_portfolio=100 -> parent (pos_1) removed -> elt_1 removed
    # elt_2: parent=1, sub_portfolio=200 -> parent (pos_2) kept -> elt_2 kept
    expected_lts = {"elt_2"}

    errors = []
    if kept_positions != expected_positions:
        errors.append(f"Position mismatch: expected {expected_positions}, got {kept_positions}")
    if kept_lts != expected_lts:
        errors.append(f"LT mismatch: expected {expected_lts}, got {kept_lts}")

    if not errors:
        print("\n  [PASS] sub_portfolio_id scoping works correctly")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


# =============================================================================
# Category 4: apply_to Scoping Tests
# =============================================================================

def test_filter_positions_only():
    """
    Test apply_to scoping with container types.

    NOTE: In PSP, apply_to compares against the container name:
    - 'holding' -> only applies to container named 'holding'
    - 'reference' -> applies to all non-holding containers

    This test verifies that a filter with apply_to='holding' affects only
    the holding container, while a reference container is unaffected.

    Scenario:
    - holding: filter applies -> pos_1 fails, pos_2 passes
    - reference: filter doesn't apply -> both kept
    """
    print("\n" + "=" * 90)
    print("TEST: apply_to='holding' - Filter holding container only")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": False,
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": True,
                },
            },
            "essential_lookthroughs": {
                "elt_1": {
                    "instrument_id": 101,
                    "parent_instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "weight": 0.50,
                    "filter_a": False,
                },
                "elt_2": {
                    "instrument_id": 102,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "weight": 0.50,
                    "filter_a": True,
                },
            },
        },
        "reference": {
            "positions": {
                "bench_1": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": False,  # Would fail if filter applied
                },
                "bench_2": {
                    "instrument_id": 4,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": True,
                },
            },
        },
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Filter with apply_to='holding' - only affects holding containers
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_holding_only",
            apply_to=ApplyTo.HOLDING,  # Only affects container named 'holding'
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): []}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    holding_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    bench_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("reference", {})
    )

    kept_positions = set(holding_data.get("positions", {}).keys())
    kept_lts = set(holding_data.get("essential_lookthroughs", {}).keys())
    kept_bench = set(bench_data.get("positions", {}).keys())

    print(f"\n  Holding positions: {sorted(kept_positions)}")
    print(f"  Holding essential LTs: {sorted(kept_lts)}")
    print(f"  Benchmark positions: {sorted(kept_bench)}")

    # Holding container: filter applies
    # pos_1 fails filter -> removed
    # pos_2 passes filter -> kept
    # elt_1 parent (pos_1) removed -> elt_1 removed
    # elt_2 parent (pos_2) kept -> elt_2 kept
    expected_positions = {"pos_2"}
    expected_lts = {"elt_2"}

    # Benchmark container: filter doesn't apply (apply_to='holding')
    expected_bench = {"bench_1", "bench_2"}

    errors = []
    if kept_positions != expected_positions:
        errors.append(f"Holding position mismatch: expected {expected_positions}, got {kept_positions}")
    if kept_lts != expected_lts:
        errors.append(f"Holding LT mismatch: expected {expected_lts}, got {kept_lts}")
    if kept_bench != expected_bench:
        errors.append(f"Benchmark mismatch: expected {expected_bench}, got {kept_bench}")

    if not errors:
        print("\n  [PASS] apply_to='holding' correctly scopes rule")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


def test_filter_lookthroughs_only():
    """
    Test apply_to='reference' scoping (filters non-holding containers).

    In PSP, apply_to compares against the container name:
    - 'holding' -> only applies to container named 'holding'
    - 'reference' (or any other value) -> applies to all non-holding containers

    This test verifies that a filter with apply_to='reference' affects only
    non-holding containers, while the 'holding' container passes through.

    Scenario:
    - holding: filter doesn't apply -> all kept
    - selected_reference: filter applies -> filtered
    """
    print("\n" + "=" * 90)
    print("TEST: apply_to='reference' - Filter non-holding containers")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": False,
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": True,
                },
            },
        },
        "selected_reference": {
            "positions": {
                "lt_pos_1": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": False,
                },
                "lt_pos_2": {
                    "instrument_id": 4,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "filter_a": True,
                },
            },
        },
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_reference_only",
            apply_to=ApplyTo.REFERENCE,  # Only affects non-holding containers
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): []}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    holding_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    lt_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("selected_reference", {})
    )

    holding_kept = set(holding_data.get("positions", {}).keys())
    lt_kept = set(lt_data.get("positions", {}).keys())

    print(f"\n  Holding container positions: {sorted(holding_kept)}")
    print(f"  Lookthrough container positions: {sorted(lt_kept)}")

    # Holding container: apply_to='lookthrough' doesn't match 'holding' -> all pass
    expected_holding = {"pos_1", "pos_2"}

    # Lookthrough container: apply_to='lookthrough' matches -> filter applied
    # lt_pos_1 fails, lt_pos_2 passes
    expected_lt = {"lt_pos_2"}

    errors = []
    if holding_kept != expected_holding:
        errors.append(f"Holding mismatch: expected {expected_holding}, got {holding_kept}")
    if lt_kept != expected_lt:
        errors.append(f"Lookthrough mismatch: expected {expected_lt}, got {lt_kept}")

    if not errors:
        print("\n  [PASS] apply_to='lookthrough' correctly scopes to lookthrough containers")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


def test_preprocess_positions_only():
    """
    Test PreProcessing modifier with apply_to='position'.
    """
    print("\n" + "=" * 90)
    print("TEST: PreProcessing with apply_to='position'")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": True,
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": False,
                },
            },
        },
        "reference": {
            "positions": {
                "bench_1": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": True,  # Would be excluded if apply_to matched
                },
                "bench_2": {
                    "instrument_id": 4,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "exclude_me": False,
                },
            },
        },
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.modifiers["exclude_flagged"] = Modifier(
        name="exclude_flagged",
        modifier_type=ModifierType.PRE_PROCESSING,
        apply_to=ApplyTo.HOLDING,  # Only affects holding containers
        criteria={"column": "exclude_me", "operator_type": "==", "value": True},
        rule_result_operator=None,
        override_modifiers=[],
    )

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule=None,
            is_scaling_rule=False,
            scale_factor=1.0,
        )
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["exclude_flagged"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    holding_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    bench_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("reference", {})
    )

    holding_kept = set(holding_data.get("positions", {}).keys())
    bench_kept = set(bench_data.get("positions", {}).keys())

    print(f"\n  Holding container positions: {sorted(holding_kept)}")
    print(f"  Benchmark container positions: {sorted(bench_kept)}")

    # Holding: preproc applies -> pos_1 excluded
    expected_holding = {"pos_2"}
    # Benchmark: preproc doesn't apply (apply_to='holding') -> all kept
    expected_bench = {"bench_1", "bench_2"}

    errors = []
    if holding_kept != expected_holding:
        errors.append(f"Holding mismatch: expected {expected_holding}, got {holding_kept}")
    if bench_kept != expected_bench:
        errors.append(f"Benchmark mismatch: expected {expected_bench}, got {bench_kept}")

    if not errors:
        print("\n  [PASS] PreProcessing with apply_to='holding' works correctly")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


def test_scale_positions_only():
    """
    Test scaling rule with apply_to='holding' - only scales holding positions.
    """
    print("\n" + "=" * 90)
    print("TEST: Scaling rule with apply_to='holding'")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "scale_me": True,
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "scale_me": False,
                },
            },
        },
        "reference": {
            "positions": {
                "bench_1": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "scale_me": True,  # Would be scaled if apply_to matched
                },
                "bench_2": {
                    "instrument_id": 4,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.50,
                    "scale_me": False,
                },
            },
        },
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule="and",
            is_scaling_rule=False,
            scale_factor=1.0,
        ),
        Rule(
            name="scale_holding",
            apply_to=ApplyTo.HOLDING,  # Only scales holding containers
            criteria={"column": "scale_me", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=True,
            scale_factor=0.5,
        ),
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): []}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    holding_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    bench_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("reference", {})
    )

    holding_positions = holding_data.get("positions", {})
    bench_positions = bench_data.get("positions", {})

    print(f"\n  Holding positions:")
    for pid, data in holding_positions.items():
        print(f"    {pid}: {data.get('initial_weight')}")

    print(f"  Benchmark positions:")
    for pid, data in bench_positions.items():
        print(f"    {pid}: {data.get('initial_weight')}")

    # Holding: pos_1 scaled by 0.5 -> 0.25, pos_2 not scaled -> 0.50
    # Benchmark: scale rule doesn't apply -> bench_1 = 0.50, bench_2 = 0.50

    errors = []
    pos_1_w = holding_positions.get("pos_1", {}).get("initial_weight")
    pos_2_w = holding_positions.get("pos_2", {}).get("initial_weight")
    bench_1_w = bench_positions.get("bench_1", {}).get("initial_weight")
    bench_2_w = bench_positions.get("bench_2", {}).get("initial_weight")

    if pos_1_w is None or abs(pos_1_w - 0.25) > 1e-6:
        errors.append(f"pos_1 should be 0.25 (scaled), got {pos_1_w}")
    if pos_2_w is None or abs(pos_2_w - 0.50) > 1e-6:
        errors.append(f"pos_2 should be 0.50, got {pos_2_w}")
    if bench_1_w is None or abs(bench_1_w - 0.50) > 1e-6:
        errors.append(f"bench_1 should be 0.50 (not scaled), got {bench_1_w}")
    if bench_2_w is None or abs(bench_2_w - 0.50) > 1e-6:
        errors.append(f"bench_2 should be 0.50, got {bench_2_w}")

    if not errors:
        print("\n  [PASS] Scaling with apply_to='holding' works correctly")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


# =============================================================================
# Category 5: Real-World Multi-Weight-Label Test
# =============================================================================

def test_multi_weight_labels():
    """
    Real-world test with multiple weight labels.

    Holdings container with 4 position weight labels:
    - initial_weight, resulting_weight, initial_exposure_weight, resulting_exposure_weight

    Essential LT with position weight labels (contributed to holdings universe) + LT weight label:
    - initial_weight, resulting_weight, initial_exposure_weight, resulting_exposure_weight (all 0.0)
    - weight (the actual LT contribution)

    Complete LT with same structure.

    Reference container with single weight label:
    - weight

    Test:
    - Apply perspective with filtering + scaling + rescaling
    - Verify all weight labels are handled correctly
    """
    print("\n" + "=" * 90)
    print("TEST: Real-World Multi-Weight-Label Test")
    print("=" * 90)

    input_json = {
        "position_weight_labels": ["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.40,
                    "resulting_weight": 0.40,
                    "initial_exposure_weight": 0.20,
                    "resulting_exposure_weight": 0.20,
                    "filter_out": True,  # Will be filtered
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.30,
                    "resulting_weight": 0.30,
                    "initial_exposure_weight": 0.15,
                    "resulting_exposure_weight": 0.15,
                    "filter_out": False,
                    "apply_scale": True,  # Will be scaled by 0.5
                },
                "pos_3": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.30,
                    "resulting_weight": 0.30,
                    "initial_exposure_weight": 0.15,
                    "resulting_exposure_weight": 0.15,
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
            "essential_lookthroughs": {
                "elt_1": {
                    "instrument_id": 101,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    # Position weight labels (0 since ELT doesn't contribute to position universe)
                    "initial_weight": 0.0,
                    "resulting_weight": 0.0,
                    "initial_exposure_weight": 0.10,
                    "resulting_exposure_weight": 0.10,
                    # LT weight label
                    "weight": 0.50,
                    "filter_out": False,
                    "apply_scale": False,
                },
                "elt_2": {
                    "instrument_id": 102,
                    "parent_instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "resulting_weight": 0.0,
                    "initial_exposure_weight": 0.05,
                    "resulting_exposure_weight": 0.05,
                    "weight": 0.50,
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
            "complete_lookthroughs": {
                "clt_1": {
                    "instrument_id": 201,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_weight": 0.0,
                    "resulting_weight": 0.0,
                    "initial_exposure_weight": 0.02,
                    "resulting_exposure_weight": 0.02,
                    "weight": 0.10,
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
        },
        "selected_reference": {
            "positions": {
                "ref_1": {
                    "instrument_id": 10,
                    "sub_portfolio_id": 100,
                    "weight": 0.50,
                    "filter_out": False,
                    "apply_scale": False,
                },
                "ref_2": {
                    "instrument_id": 11,
                    "sub_portfolio_id": 100,
                    "weight": 0.50,
                    "filter_out": True,  # Will be filtered
                    "apply_scale": False,
                },
            },
        },
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Rescaling modifier
    engine.config.modifiers["scale_holdings_to_100_percent"] = Modifier(
        name="scale_holdings_to_100_percent",
        modifier_type=ModifierType.SCALING,
        apply_to=ApplyTo.BOTH,
        criteria=None,
        rule_result_operator=None,
        override_modifiers=[],
    )

    # Filtering rule (keep where filter_out != True)
    # Scaling rule (scale where apply_scale == True by 0.5)
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_rule",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_out", "operator_type": "!=", "value": True},
            condition_for_next_rule="and",
            is_scaling_rule=False,
            scale_factor=1.0,
        ),
        Rule(
            name="scale_rule",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "apply_scale", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=True,
            scale_factor=0.5,
        ),
    ]

    try:
        output = engine.process(
            input_json=input_json,
            perspective_configs={"test_config": {str(PERSPECTIVE_ID): ["scale_holdings_to_100_percent"]}},
        )
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    holdings_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("holding", {})
    )

    ref_data = (
        output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get("selected_reference", {})
    )

    positions = holdings_data.get("positions", {})
    essential_lts = holdings_data.get("essential_lookthroughs", {})
    complete_lts = holdings_data.get("complete_lookthroughs", {})
    scale_factors = holdings_data.get("scale_factors", {})

    ref_positions = ref_data.get("positions", {})
    ref_scale_factors = ref_data.get("scale_factors", {})

    print(f"\n  Holdings positions: {list(positions.keys())}")
    print(f"  Holdings essential LTs: {list(essential_lts.keys())}")
    print(f"  Holdings complete LTs: {list(complete_lts.keys())}")
    print(f"  Holdings scale_factors: {scale_factors}")

    print(f"\n  Reference positions: {list(ref_positions.keys())}")
    print(f"  Reference scale_factors: {ref_scale_factors}")

    errors = []

    # Check kept positions
    expected_positions = {"pos_2", "pos_3"}  # pos_1 filtered out
    if set(positions.keys()) != expected_positions:
        errors.append(f"Position mismatch: expected {expected_positions}, got {set(positions.keys())}")

    # Check ELTs - elt_1 parent=pos_2 (kept), elt_2 parent=pos_3 (kept)
    expected_elts = {"elt_1", "elt_2"}
    if set(essential_lts.keys()) != expected_elts:
        errors.append(f"ELT mismatch: expected {expected_elts}, got {set(essential_lts.keys())}")

    # Check CLTs - clt_1 parent=pos_2 (kept)
    expected_clts = {"clt_1"}
    if set(complete_lts.keys()) != expected_clts:
        errors.append(f"CLT mismatch: expected {expected_clts}, got {set(complete_lts.keys())}")

    # Check reference - ref_1 kept, ref_2 filtered
    expected_refs = {"ref_1"}
    if set(ref_positions.keys()) != expected_refs:
        errors.append(f"Reference mismatch: expected {expected_refs}, got {set(ref_positions.keys())}")

    # Check scale factors exist for all weight labels
    expected_weight_labels = ["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight"]
    for w in expected_weight_labels:
        if w not in scale_factors:
            errors.append(f"Missing scale_factor for {w}")

    # Verify scale factor calculation (World A: kept mass / total mass)
    # pos_1 removed: 0.40 -> 0
    # pos_2 scaled by 0.5: 0.30 * 0.5 = 0.15
    # pos_3 not scaled: 0.30 * 1.0 = 0.30
    # elt_1 (initial_weight): 0.0 (doesn't contribute to position mass)
    # elt_2 (initial_weight): 0.0
    # Total universe (initial_weight): 0.40 + 0.30 + 0.30 = 1.0
    # Kept mass: 0.15 + 0.30 = 0.45
    # SF = 0.45 / 1.0 = 0.45

    expected_sf = 0.45
    actual_sf = scale_factors.get("initial_weight", 0)
    if abs(actual_sf - expected_sf) > 1e-4:
        errors.append(f"SF for initial_weight: expected {expected_sf}, got {actual_sf}")

    # For ELT position weight labels, they're all 0.0, so they don't affect the denom
    # But initial_exposure_weight is non-zero:
    # Total (initial_exposure): 0.20 + 0.15 + 0.15 + 0.10 + 0.05 = 0.65
    # Kept: pos_2(0.15*0.5=0.075) + pos_3(0.15) + elt_1(0.10) + elt_2(0.05) = 0.375
    # SF = 0.375 / 0.65 = 0.5769...

    expected_sf_exp = 0.375 / 0.65
    actual_sf_exp = scale_factors.get("initial_exposure_weight", 0)
    if abs(actual_sf_exp - expected_sf_exp) > 1e-4:
        errors.append(f"SF for initial_exposure_weight: expected {expected_sf_exp:.6f}, got {actual_sf_exp:.6f}")

    # Print position weights for verification
    print("\n  Position weights after processing:")
    for pid, data in positions.items():
        print(f"    {pid}: initial_weight={data.get('initial_weight')}")

    if not errors:
        print("\n  [PASS] Multi-weight-label test passed")
        return True
    else:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all advanced scenario tests."""
    tests = [
        # Category 1: rule_result + PreProcessing + Group Savior
        ("rule_result_preprocess_group_savior", test_rule_result_preprocess_group_savior),
        ("rule_result_preprocess_multiple_groups", test_rule_result_preprocess_group_savior_multiple_groups),

        # Category 2: rule_result with NotIn
        ("rule_result_notin_or_savior", test_rule_result_notin_or_savior),

        # Category 3: PreProcessing Propagation to Lookthroughs
        ("preprocess_propagation_to_lt", test_preprocess_propagation_to_lookthroughs),
        ("preprocess_shared_parent", test_preprocess_shared_parent),

        # Category 4: apply_to Scoping
        ("filter_positions_only", test_filter_positions_only),
        ("filter_lookthroughs_only", test_filter_lookthroughs_only),
        ("preprocess_positions_only", test_preprocess_positions_only),
        ("scale_positions_only", test_scale_positions_only),

        # Category 5: Multi-Weight-Label
        ("multi_weight_labels", test_multi_weight_labels),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    passed_count = 0
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        passed_count += int(passed)

    print(f"\n  Total: {passed_count}/{len(results)} passed")

    return passed_count == len(results)


if __name__ == "__main__":
    print("=" * 90)
    print("ADVANCED SCENARIOS TEST")
    print("=" * 90)

    success = run_all_tests()
    if success:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        raise SystemExit(1)
