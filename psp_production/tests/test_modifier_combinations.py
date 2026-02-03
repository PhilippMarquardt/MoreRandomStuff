"""
Comprehensive Modifier & Rule Combinations Test

Tests all combinations of:
- PreProcessing modifiers (EXCLUDE matching positions before rules run)
- PostProcessing modifiers (KEEP/savior positions after rules, with rule_result_operator)
- Multiple filtering rules (combined with AND/OR via condition_for_next_rule)
- Multiple scaling rules (multiplicative factors)
- apply_to scoping ('position' vs 'lookthrough' vs 'both')
- Complex criteria (AND/OR conditions)

Execution order:
1. PreProcessing modifiers -> EXCLUDE matching rows
2. Perspective filtering rules -> AND/OR combined
3. PostProcessing modifiers -> SAVE matching rows (savior logic)
4. Scaling rules -> Multiply factors for matching rows
"""

import sys
import os
import io
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import pytest

# Note: sys.stdout encoding fix removed as it causes pytest issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier
from perspective_service.models.enums import ApplyTo, ModifierType, LogicalOperator


# =============================================================================
# Test Data
# =============================================================================

CONTAINER = "holding"
PERSPECTIVE_ID = 100
TNA = 1_000_000.0

# Position raw weights and attributes for testing different scenarios
# Each position has attributes that can be used by different rules/modifiers
POS_DATA = {
    "pos_1": {
        "weight": 0.30,
        "exclude_me": True,   # PreProcessing: will exclude
        "save_me": False,     # PostProcessing: won't save
        "filter_a": True,     # Passes filter_a
        "filter_b": False,    # Fails filter_b
        "scale_a": True,      # Matches scale_a (factor 0.7)
        "scale_b": False,     # Doesn't match scale_b
        "position_type_flag": "holding",
    },
    "pos_2": {
        "weight": 0.25,
        "exclude_me": False,  # PreProcessing: won't exclude
        "save_me": True,      # PostProcessing: will save
        "filter_a": False,    # Fails filter_a
        "filter_b": True,     # Passes filter_b
        "scale_a": False,     # Doesn't match scale_a
        "scale_b": True,      # Matches scale_b (factor 0.8)
        "position_type_flag": "holding",
    },
    "pos_3": {
        "weight": 0.25,
        "exclude_me": False,
        "save_me": False,
        "filter_a": True,     # Passes filter_a
        "filter_b": True,     # Passes filter_b
        "scale_a": True,      # Matches scale_a
        "scale_b": True,      # Matches scale_b
        "position_type_flag": "holding",
    },
    "pos_4": {
        "weight": 0.20,
        "exclude_me": False,
        "save_me": False,
        "filter_a": False,    # Fails filter_a
        "filter_b": False,    # Fails filter_b
        "scale_a": False,
        "scale_b": False,
        "position_type_flag": "holding",
    },
}

# Essential lookthroughs with parent relationships
ELT_DATA = {
    "elt_1": {
        "weight": 0.50,
        "parent": "pos_2",
        "parent_instrument_id": 2,
        "exclude_me": False,
        "save_me": False,
        "filter_a": True,
        "filter_b": True,
        "scale_a": False,
        "scale_b": False,
    },
    "elt_2": {
        "weight": 0.50,
        "parent": "pos_3",
        "parent_instrument_id": 3,
        "exclude_me": False,
        "save_me": False,
        "filter_a": True,
        "filter_b": True,
        "scale_a": False,
        "scale_b": False,
    },
}

TOTAL_UNIVERSE = sum(p["weight"] for p in POS_DATA.values())  # 1.0

# Scale factors for scaling rules
SCALE_A_FACTOR = 0.7
SCALE_B_FACTOR = 0.8


def create_test_input_json() -> Dict:
    """Create test input JSON with all position attributes."""
    positions = {}
    for pos_id, data in POS_DATA.items():
        positions[pos_id] = {
            "instrument_id": int(pos_id.split("_")[1]),
            "sub_portfolio_id": 100,
            "initial_weight": data["weight"],
            "exclude_me": data["exclude_me"],
            "save_me": data["save_me"],
            "filter_a": data["filter_a"],
            "filter_b": data["filter_b"],
            "scale_a": data["scale_a"],
            "scale_b": data["scale_b"],
        }

    essential_lts = {}
    for elt_id, data in ELT_DATA.items():
        essential_lts[elt_id] = {
            "instrument_id": int(elt_id.split("_")[1]) + 100,
            "parent_instrument_id": data["parent_instrument_id"],
            "sub_portfolio_id": 100,
            "initial_weight": 0.0,  # ELT doesn't contribute to position weight
            "weight": data["weight"],
            "exclude_me": data["exclude_me"],
            "save_me": data["save_me"],
            "filter_a": data["filter_a"],
            "filter_b": data["filter_b"],
            "scale_a": data["scale_a"],
            "scale_b": data["scale_b"],
        }

    return {
        "position_weight_labels": ["initial_weight"],
        "lookthrough_weight_labels": ["weight"],
        CONTAINER: {
            "positions": positions,
            "essential_lookthroughs": essential_lts,
        }
    }


# =============================================================================
# Scenario Definition
# =============================================================================

@dataclass
class ModifierScenario:
    """Describes a test scenario with modifiers and rules."""
    name: str

    # PreProcessing modifier
    has_preprocess: bool = False          # Exclude where exclude_me=True

    # PostProcessing modifier
    has_postprocess_or: bool = False      # Save where save_me=True (OR logic)
    has_postprocess_and: bool = False     # Require save_me=True (AND logic)

    # Filtering rules
    has_filter_a: bool = False            # Keep where filter_a=True
    has_filter_b: bool = False            # Keep where filter_b=True
    filter_combine: str = "and"           # How to combine filter_a and filter_b

    # Scaling rules
    has_scale_a: bool = False             # Scale by 0.7 where scale_a=True
    has_scale_b: bool = False             # Scale by 0.8 where scale_b=True

    # Rescaling modifiers
    scale_holdings: bool = False

    def __str__(self) -> str:
        parts = []
        if self.has_preprocess:
            parts.append("preproc")
        if self.has_filter_a:
            parts.append("filterA")
        if self.has_filter_b:
            parts.append("filterB")
        if self.has_filter_a and self.has_filter_b:
            parts.append(f"({self.filter_combine})")
        if self.has_postprocess_or:
            parts.append("postproc_or")
        if self.has_postprocess_and:
            parts.append("postproc_and")
        if self.has_scale_a:
            parts.append("scaleA")
        if self.has_scale_b:
            parts.append("scaleB")
        if self.scale_holdings:
            parts.append("rescale")
        return "_".join(parts) if parts else "no_changes"


# =============================================================================
# Engine Setup
# =============================================================================

def setup_engine(scenario: ModifierScenario) -> PerspectiveEngine:
    """Configure engine with rules and modifiers based on scenario."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rules: List[Rule] = []
    modifiers_to_add: List[str] = []

    # --- PreProcessing modifier ---
    if scenario.has_preprocess:
        engine.config.modifiers["exclude_flagged"] = Modifier(
            name="exclude_flagged",
            modifier_type=ModifierType.PRE_PROCESSING,
            apply_to=ApplyTo.BOTH,
            criteria={"column": "exclude_me", "operator_type": "==", "value": True},
            rule_result_operator=None,
            override_modifiers=[],
        )
        modifiers_to_add.append("exclude_flagged")

    # --- Filtering rules ---
    if scenario.has_filter_a and scenario.has_filter_b:
        # Two filters combined
        rules.append(Rule(
            name="filter_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=scenario.filter_combine,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))
        rules.append(Rule(
            name="filter_b",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_b", "operator_type": "==", "value": True},
            condition_for_next_rule=LogicalOperator.AND if (scenario.has_scale_a or scenario.has_scale_b) else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))
    elif scenario.has_filter_a:
        rules.append(Rule(
            name="filter_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_a", "operator_type": "==", "value": True},
            condition_for_next_rule=LogicalOperator.AND if (scenario.has_scale_a or scenario.has_scale_b) else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))
    elif scenario.has_filter_b:
        rules.append(Rule(
            name="filter_b",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_b", "operator_type": "==", "value": True},
            condition_for_next_rule=LogicalOperator.AND if (scenario.has_scale_a or scenario.has_scale_b) else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))
    else:
        # No filtering - pass all
        rules.append(Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule=LogicalOperator.AND if (scenario.has_scale_a or scenario.has_scale_b) else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))

    # --- PostProcessing modifier (savior) ---
    if scenario.has_postprocess_or:
        engine.config.modifiers["savior_or"] = Modifier(
            name="savior_or",
            modifier_type=ModifierType.POST_PROCESSING,
            apply_to=ApplyTo.BOTH,
            criteria={"column": "save_me", "operator_type": "==", "value": True},
            rule_result_operator=LogicalOperator.OR,  # Keep if rule passes OR save_me=True
            override_modifiers=[],
        )
        modifiers_to_add.append("savior_or")

    if scenario.has_postprocess_and:
        engine.config.modifiers["restrict_and"] = Modifier(
            name="restrict_and",
            modifier_type=ModifierType.POST_PROCESSING,
            apply_to=ApplyTo.BOTH,
            criteria={"column": "save_me", "operator_type": "==", "value": True},
            rule_result_operator=LogicalOperator.AND,  # Keep only if rule passes AND save_me=True
            override_modifiers=[],
        )
        modifiers_to_add.append("restrict_and")

    # --- Scaling rules ---
    if scenario.has_scale_a:
        rules.append(Rule(
            name="scale_a",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "scale_a", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=True,
            scale_factor=SCALE_A_FACTOR,
        ))

    if scenario.has_scale_b:
        rules.append(Rule(
            name="scale_b",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "scale_b", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=True,
            scale_factor=SCALE_B_FACTOR,
        ))

    engine.config.perspectives[PERSPECTIVE_ID] = rules

    # --- Scaling modifiers (rescaling) ---
    if scenario.scale_holdings:
        engine.config.modifiers["scale_holdings_to_100_percent"] = Modifier(
            name="scale_holdings_to_100_percent",
            modifier_type=ModifierType.SCALING,
            apply_to=ApplyTo.BOTH,
            criteria=None,
            rule_result_operator=None,
            override_modifiers=[],
        )
        modifiers_to_add.append("scale_holdings_to_100_percent")

    return engine, modifiers_to_add


# =============================================================================
# Expected Value Calculation
# =============================================================================

def calculate_expected(scenario: ModifierScenario) -> Dict:
    """
    Calculate expected results based on scenario.

    Execution order:
    1. PreProcessing: exclude positions where exclude_me=True
    2. Filtering rules: combine with AND/OR
    3. PostProcessing: savior logic
    4. Scaling rules: multiply factors
    """
    kept_positions: Set[str] = set(POS_DATA.keys())
    position_factors: Dict[str, float] = {pid: 1.0 for pid in POS_DATA.keys()}

    # --- Step 1: PreProcessing (exclude) ---
    if scenario.has_preprocess:
        for pid, data in POS_DATA.items():
            if data["exclude_me"]:
                kept_positions.discard(pid)

    # --- Step 2: Filtering rules ---
    def passes_filter(pid: str) -> bool:
        data = POS_DATA[pid]
        if scenario.has_filter_a and scenario.has_filter_b:
            if scenario.filter_combine == "and":
                return data["filter_a"] and data["filter_b"]
            else:  # or
                return data["filter_a"] or data["filter_b"]
        elif scenario.has_filter_a:
            return data["filter_a"]
        elif scenario.has_filter_b:
            return data["filter_b"]
        return True  # No filter = pass all

    filter_passed: Set[str] = {pid for pid in kept_positions if passes_filter(pid)}

    # --- Step 3: PostProcessing (savior logic) ---
    if scenario.has_postprocess_or:
        # OR: keep if (filter passed) OR (save_me=True)
        for pid in kept_positions:
            if POS_DATA[pid]["save_me"]:
                filter_passed.add(pid)

    if scenario.has_postprocess_and:
        # AND: keep only if (filter passed) AND (save_me=True)
        filter_passed = {pid for pid in filter_passed if POS_DATA[pid]["save_me"]}

    kept_positions = kept_positions & filter_passed

    # --- Step 4: Scaling rules ---
    for pid in kept_positions:
        data = POS_DATA[pid]
        factor = 1.0
        if scenario.has_scale_a and data["scale_a"]:
            factor *= SCALE_A_FACTOR
        if scenario.has_scale_b and data["scale_b"]:
            factor *= SCALE_B_FACTOR
        position_factors[pid] = factor

    # --- Calculate kept mass ---
    kept_mass = sum(
        POS_DATA[pid]["weight"] * position_factors[pid]
        for pid in kept_positions
    )

    # --- Calculate SF ---
    # When all positions are removed, SF = 1.0 (no rescaling applied)
    # The value is irrelevant since there are no positions to multiply
    if kept_mass == 0:
        expected_sf = 1.0
    elif scenario.scale_holdings:
        expected_sf = kept_mass / TOTAL_UNIVERSE
    else:
        expected_sf = 1.0

    # --- Calculate output weights ---
    output_weights: Dict[str, float] = {}
    for pid in kept_positions:
        eff = POS_DATA[pid]["weight"] * position_factors[pid]
        if scenario.scale_holdings and kept_mass > 0:
            output_weights[pid] = eff / kept_mass
        else:
            output_weights[pid] = eff

    return {
        "kept_positions": kept_positions,
        "position_factors": position_factors,
        "kept_mass": kept_mass,
        "expected_sf": expected_sf,
        "output_weights": output_weights,
    }


# =============================================================================
# Verification
# =============================================================================

def verify_scenario(
    scenario: ModifierScenario,
    psp_output: Dict,
    expected: Dict,
) -> bool:
    """Verify PSP output matches expected results."""
    container_data = (
        psp_output.get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get(CONTAINER, {})
    )

    if not container_data and expected["kept_positions"]:
        print("  [FAIL] Missing container output")
        return False

    positions = container_data.get("positions") or {}  # Handle positions: None
    scale_factors = container_data.get("scale_factors", {})

    # Check scale factor
    actual_sf = scale_factors.get("initial_weight", 1.0)
    if abs(actual_sf - expected["expected_sf"]) > 1e-6:
        print(f"  [FAIL] SF mismatch: expected {expected['expected_sf']:.6f}, got {actual_sf:.6f}")
        return False
    print(f"  [OK] SF = {actual_sf:.6f}")

    # Check kept positions
    actual_kept = set(positions.keys())
    if actual_kept != expected["kept_positions"]:
        missing = expected["kept_positions"] - actual_kept
        extra = actual_kept - expected["kept_positions"]
        if missing:
            print(f"  [FAIL] Missing positions: {missing}")
        if extra:
            print(f"  [FAIL] Extra positions: {extra}")
        return False
    print(f"  [OK] Kept positions: {sorted(actual_kept)}")

    # Check output weights
    for pid, exp_w in expected["output_weights"].items():
        actual_w = positions.get(pid, {}).get("initial_weight")
        if actual_w is None:
            print(f"  [FAIL] Position {pid} missing weight")
            return False
        if abs(actual_w - exp_w) > 1e-6:
            print(f"  [FAIL] Position {pid} weight: expected {exp_w:.6f}, got {actual_w:.6f}")
            return False
    print("  [OK] Output weights correct")

    return True


# =============================================================================
# Test Runner
# =============================================================================

def run_scenario(scenario: ModifierScenario) -> bool:
    """Run a single test scenario (called by run_all_tests, not pytest directly)."""
    print(f"\n{'='*90}")
    print(f"TEST: {scenario.name}")
    print(f"{'='*90}")
    print(f"  PreProcess: {scenario.has_preprocess}")
    print(f"  FilterA: {scenario.has_filter_a}, FilterB: {scenario.has_filter_b}, Combine: {scenario.filter_combine}")
    print(f"  PostProcess OR: {scenario.has_postprocess_or}, AND: {scenario.has_postprocess_and}")
    print(f"  ScaleA: {scenario.has_scale_a}, ScaleB: {scenario.has_scale_b}")
    print(f"  Rescale: {scenario.scale_holdings}")

    input_json = create_test_input_json()
    engine, modifiers = setup_engine(scenario)
    input_json["perspective_configurations"] = {"test_config": {str(PERSPECTIVE_ID): modifiers}}

    try:
        output = engine.process(input_json)
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    expected = calculate_expected(scenario)
    print(f"\n  Expected kept: {sorted(expected['kept_positions'])}")
    print(f"  Expected SF: {expected['expected_sf']:.6f}")
    print(f"  Expected weights: {expected['output_weights']}")

    ok = verify_scenario(scenario, output, expected)
    print(f"\n  {'[PASS]' if ok else '[FAIL]'} {scenario.name}")
    return ok


# Module-level scenarios list for parametrized testing
MODIFIER_SCENARIOS = [
    # === No changes (baseline) ===
    ModifierScenario("baseline"),

    # === PreProcessing only ===
    ModifierScenario("preprocess_only", has_preprocess=True),

    # === Single filter ===
    ModifierScenario("filter_a_only", has_filter_a=True),
    ModifierScenario("filter_b_only", has_filter_b=True),

    # === Two filters combined ===
    ModifierScenario("filters_and", has_filter_a=True, has_filter_b=True, filter_combine="and"),
    ModifierScenario("filters_or", has_filter_a=True, has_filter_b=True, filter_combine="or"),

    # === PreProcessing + filter ===
    ModifierScenario("preprocess_filter_a", has_preprocess=True, has_filter_a=True),

    # === PostProcessing OR (savior) ===
    ModifierScenario("postprocess_or_with_filter", has_filter_a=True, has_postprocess_or=True),

    # === PostProcessing AND (restrictor) ===
    ModifierScenario("postprocess_and_with_filter", has_filter_a=True, has_postprocess_and=True),

    # === Full preprocess + filter + postprocess ===
    ModifierScenario("pre_filter_post_or", has_preprocess=True, has_filter_a=True, has_postprocess_or=True),
    ModifierScenario("pre_filter_post_and", has_preprocess=True, has_filter_a=True, has_postprocess_and=True),

    # === Single scaling rule ===
    ModifierScenario("scale_a_only", has_scale_a=True),
    ModifierScenario("scale_b_only", has_scale_b=True),

    # === Two scaling rules (multiplicative) ===
    ModifierScenario("scale_a_and_b", has_scale_a=True, has_scale_b=True),

    # === Filter + scaling ===
    ModifierScenario("filter_a_scale_a", has_filter_a=True, has_scale_a=True),

    # === Rescaling with filter ===
    ModifierScenario("filter_rescale", has_filter_a=True, scale_holdings=True),

    # === Full pipeline: preprocess + filter + postprocess + scale + rescale ===
    ModifierScenario(
        "full_pipeline",
        has_preprocess=True,
        has_filter_a=True,
        has_postprocess_or=True,
        has_scale_a=True,
        scale_holdings=True,
    ),

    # === Complex: two filters OR + scale_a + scale_b + rescale ===
    ModifierScenario(
        "complex_filters_or_scales",
        has_filter_a=True,
        has_filter_b=True,
        filter_combine="or",
        has_scale_a=True,
        has_scale_b=True,
        scale_holdings=True,
    ),

    # === Complex: preprocess + two filters AND + postprocess AND + scales ===
    ModifierScenario(
        "complex_restrictive",
        has_preprocess=True,
        has_filter_a=True,
        has_filter_b=True,
        filter_combine="and",
        has_postprocess_and=True,
        has_scale_a=True,
        has_scale_b=True,
        scale_holdings=True,
    ),
]


def run_all_tests() -> bool:
    """Run all test scenarios (for __main__ execution)."""
    results: List[Tuple[ModifierScenario, bool]] = []
    for s in MODIFIER_SCENARIOS:
        try:
            results.append((s, run_scenario(s)))
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((s, False))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    passed = 0
    for s, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {s.name}")
        passed += int(ok)
    print(f"\n  Total: {passed}/{len(results)} passed")

    return passed == len(results)


def test_all_positions_removed() -> bool:
    """
    LEGACY COMPATIBILITY TEST - Remove when caller is adjusted.

    Test that when ALL positions are removed, the container still appears with:
    - positions: None (not missing, explicitly None)
    - scale_factors: {weight: 1.0} (default, value is irrelevant since nothing to multiply)

    This matches legacy behavior. Once callers are updated to handle missing
    containers, this test and the corresponding code in output_formatter.py
    should be removed.
    """
    print("\n" + "=" * 90)
    print("TEST: all_positions_removed (legacy: positions=None)")
    print("=" * 90)

    # Create a rule that filters out ALL positions
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    # Custom perspective rule that matches NOTHING (instrument_id = -999 doesn't exist)
    engine.config.perspectives[PERSPECTIVE_ID] = [
        Rule(
            name="filter_all",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "instrument_id", "operator_type": "==", "value": -999},
            is_scaling_rule=False
        )
    ]

    input_json = create_test_input_json()
    input_json["perspective_configurations"] = {"test_config": {str(PERSPECTIVE_ID): []}}

    try:
        output = engine.process(input_json)
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get container data
    container_data = (
        output
        .get("perspective_configurations", {})
        .get("test_config", {})
        .get(PERSPECTIVE_ID, {})
        .get(CONTAINER, {})
    )

    # Check 1: Container must be present
    if not container_data:
        print("  [FAIL] Container missing from output")
        return False
    print("  [OK] Container present in output")

    # Check 2: positions key must exist and be None
    if "positions" not in container_data:
        print("  [FAIL] 'positions' key missing from container")
        return False
    if container_data["positions"] is not None:
        print(f"  [FAIL] 'positions' should be None, got: {container_data['positions']}")
        return False
    print("  [OK] positions = None")

    # Check 3: scale_factors must exist with value 0.0
    scale_factors = container_data.get("scale_factors", {})
    if not scale_factors:
        print("  [FAIL] 'scale_factors' missing from container")
        return False

    sf_value = scale_factors.get("initial_weight", None)
    if sf_value is None:
        print("  [FAIL] 'initial_weight' missing from scale_factors")
        return False

    # SF = 1.0 when all positions removed (value is irrelevant since nothing to multiply)
    if sf_value != 1.0:
        print(f"  [FAIL] scale_factor = {sf_value}, expected 1.0")
        return False
    print(f"  [OK] scale_factors = 1.0")

    print("\n  [PASS] all_positions_removed")
    return True


@pytest.mark.parametrize("scenario", MODIFIER_SCENARIOS, ids=lambda s: s.name)
def test_modifier_scenario(scenario: ModifierScenario):
    """Pytest parametrized test for each modifier scenario."""
    assert run_scenario(scenario), f"Scenario {scenario.name} failed"


if __name__ == "__main__":
    print("=" * 90)
    print("COMPREHENSIVE MODIFIER & RULE COMBINATIONS TEST")
    print("=" * 90)

    success = run_all_tests()

    # Run dedicated "all positions removed" test
    success = test_all_positions_removed() and success

    if success:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        raise SystemExit(1)
