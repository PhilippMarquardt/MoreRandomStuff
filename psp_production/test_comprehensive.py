"""
Comprehensive Test Suite for Perspective Service

Tests all combinations of:
- Containers (single/multiple)
- Lookthroughs (none/with)
- Weight labels (single/multiple)
- Perspectives (single/multiple)
- Scaling rules (none/with)
- Modifiers (none/scale_holdings/scale_lookthroughs/both)
- Rule conditions (single/AND/OR)
"""

import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def make_position(id_suffix, instrument_id, weights, **extra_fields):
    """Create a position dict with given weights and extra fields."""
    pos = {
        "instrument_id": instrument_id,
        "sub_portfolio_id": 1,
        **weights,
        **extra_fields
    }
    return f"pos_{id_suffix}", pos


def make_lookthrough(id_suffix, parent_id, instrument_id, weights, record_type="essential_lookthroughs", **extra):
    """Create a lookthrough dict."""
    lt = {
        "instrument_id": instrument_id,
        "parent_instrument_id": parent_id,
        "sub_portfolio_id": 1,
        **weights,
        **extra
    }
    return f"lt_{id_suffix}", lt, record_type


# =============================================================================
# TEST 1: BASIC - No Modifiers, No Scaling
# =============================================================================

def test_1_1_single_container_no_lt_single_weight():
    """Single container, no lookthroughs, single weight label."""
    print("\n" + "=" * 80)
    print("TEST 1.1: Single container, no LT, single weight")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "pos_1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 2},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep all (no filtering)
    rule = Rule(name="keep_all", apply_to="both", criteria=None)
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    # Verify: all positions kept with original weights (no rescaling)
    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    assert abs(positions["pos_1"]["w1"] - 0.6) < 0.0001, "pos_1.w1 should be 0.6"
    assert abs(positions["pos_2"]["w1"] - 0.4) < 0.0001, "pos_2.w1 should be 0.4"
    print("  [PASS] All positions kept with original weights")
    return True


def test_1_2_multiple_weights():
    """Single container, no LT, multiple weight labels."""
    print("\n" + "=" * 80)
    print("TEST 1.2: Single container, no LT, multiple weights")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1", "w2", "w3"],
        "lookthrough_weight_labels": ["w1", "w2", "w3"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "pos_1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.5, "w2": 0.6, "w3": 0.7, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "w2": 0.2, "w3": 0.1, "liquidity_type_id": 2},
                "pos_3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.2, "w2": 0.2, "w3": 0.2, "liquidity_type_id": 1},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep liquidity_type_id == 1
    rule = Rule(name="keep_liq_1", apply_to="both",
                criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1})
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1", "w2", "w3"],
        lookthrough_weights=["w1", "w2", "w3"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    assert "pos_1" in positions, "pos_1 should be kept"
    assert "pos_2" not in positions, "pos_2 should be removed"
    assert "pos_3" in positions, "pos_3 should be kept"
    print("  [PASS] Correct positions kept/removed")
    return True


def test_1_3_multiple_containers():
    """Multiple containers (portfolio + benchmark)."""
    print("\n" + "=" * 80)
    print("TEST 1.3: Multiple containers")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1", "w2"],
        "lookthrough_weight_labels": ["w1", "w2"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "w2": 0.5, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "w2": 0.5, "liquidity_type_id": 2},
            }
        },
        "benchmark": {
            "position_type": "benchmark",
            "positions": {
                "b1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.7, "w2": 0.6, "liquidity_type_id": 1},
                "b2": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.3, "w2": 0.4, "liquidity_type_id": 1},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rule = Rule(name="keep_liq_1", apply_to="both",
                criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1})
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1", "w2"],
        lookthrough_weights=["w1", "w2"],
        verbose=True
    )

    pf = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    bm = result["perspective_configurations"]["main"][100]["benchmark"]["positions"]

    assert "p1" in pf and "p2" not in pf, "Portfolio filtering correct"
    assert "b1" in bm and "b2" in bm, "Benchmark all kept"
    print("  [PASS] Multiple containers processed correctly")
    return True


def test_1_4_with_lookthroughs():
    """Single container with lookthroughs."""
    print("\n" + "=" * 80)
    print("TEST 1.4: With lookthroughs")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 1},
            },
            "essential_lookthroughs": {
                "lt1": {"instrument_id": 101, "parent_instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.3},
                "lt2": {"instrument_id": 102, "parent_instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.3},
                "lt3": {"instrument_id": 201, "parent_instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rule = Rule(name="keep_all", apply_to="both", criteria=None)
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    container = result["perspective_configurations"]["main"][100]["portfolio"]
    assert "positions" in container, "Positions present"
    assert "essential_lookthroughs" in container, "Lookthroughs present"
    print("  [PASS] Lookthroughs processed correctly")
    return True


# =============================================================================
# TEST 2: SCALING RULES
# =============================================================================

def test_2_1_scaling_rule_no_modifier():
    """Scaling rule without rescaling modifier."""
    print("\n" + "=" * 80)
    print("TEST 2.1: Scaling rule (factor=0.5) without rescaling")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 2},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep rule
    keep_rule = Rule(name="keep_all", apply_to="both", criteria=None, condition_for_next_rule="and")
    # Scaling rule: scale liquidity_type_id==2 by 0.5
    scale_rule = Rule(
        name="scale_liq_2", apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 2},
        is_scaling_rule=True, scale_factor=0.5
    )
    engine.config.perspectives[100] = [keep_rule, scale_rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},  # No rescaling modifier
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    # p1: w1 = 0.6 * 1.0 = 0.6
    # p2: w1 = 0.4 * 0.5 = 0.2
    assert abs(positions["p1"]["w1"] - 0.6) < 0.0001, f"p1.w1 should be 0.6, got {positions['p1']['w1']}"
    assert abs(positions["p2"]["w1"] - 0.2) < 0.0001, f"p2.w1 should be 0.2, got {positions['p2']['w1']}"
    print("  [PASS] Scaling rule applied correctly")
    return True


def test_2_2_scaling_rule_with_rescaling():
    """Scaling rule WITH scale_holdings_to_100_percent."""
    print("\n" + "=" * 80)
    print("TEST 2.2: Scaling rule WITH rescaling modifier")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 2},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    keep_rule = Rule(name="keep_all", apply_to="both", criteria=None, condition_for_next_rule="and")
    scale_rule = Rule(
        name="scale_liq_2", apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 2},
        is_scaling_rule=True, scale_factor=0.5
    )
    engine.config.perspectives[100] = [keep_rule, scale_rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: ["scale_holdings_to_100_percent"]}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    # Legacy behavior: rescale BEFORE applying exposure_factor
    # denom = Σ(kept weights) = 0.6 + 0.4 = 1.0
    # p1.w1 = 0.6 × (1.0 / 1.0) = 0.6
    # p2.w1 = 0.4 × (0.5 / 1.0) = 0.2
    # Note: sum = 0.8 (NOT 1.0) - this is correct legacy behavior
    assert abs(positions["p1"]["w1"] - 0.6) < 0.0001, f"p1.w1 should be 0.6, got {positions['p1']['w1']}"
    assert abs(positions["p2"]["w1"] - 0.2) < 0.0001, f"p2.w1 should be 0.2, got {positions['p2']['w1']}"
    print("  [PASS] Scaling + Rescaling correct (legacy behavior: sum = 0.8)")
    return True


def test_2_3_multiple_scale_factors():
    """Multiple rows with different scale factors + rescaling."""
    print("\n" + "=" * 80)
    print("TEST 2.3: Multiple scale factors with rescaling")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1", "w2"],
        "lookthrough_weight_labels": ["w1", "w2"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.4, "w2": 0.5, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "w2": 0.3, "liquidity_type_id": 2},
                "p3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.3, "w2": 0.2, "liquidity_type_id": 3},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    keep_rule = Rule(name="keep_all", apply_to="both", criteria=None, condition_for_next_rule="and")
    # Scale liq_2 by 0.5
    scale_rule_1 = Rule(
        name="scale_liq_2", apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 2},
        is_scaling_rule=True, scale_factor=0.5, condition_for_next_rule="and"
    )
    # Scale liq_3 by 0.75
    scale_rule_2 = Rule(
        name="scale_liq_3", apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 3},
        is_scaling_rule=True, scale_factor=0.75
    )
    engine.config.perspectives[100] = [keep_rule, scale_rule_1, scale_rule_2]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: ["scale_holdings_to_100_percent"]}},
        position_weights=["w1", "w2"],
        lookthrough_weights=["w1", "w2"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    # Legacy behavior: rescale BEFORE applying exposure_factor
    # denom = Σ(kept weights), NOT Σ(weight × factor)
    # w1_denom = 0.4 + 0.3 + 0.3 = 1.0
    # w2_denom = 0.5 + 0.3 + 0.2 = 1.0
    #
    # p1: w1 = 0.4 × (1.0 / 1.0) = 0.4, w2 = 0.5 × (1.0 / 1.0) = 0.5
    # p2: w1 = 0.3 × (0.5 / 1.0) = 0.15, w2 = 0.3 × (0.5 / 1.0) = 0.15
    # p3: w1 = 0.3 × (0.75 / 1.0) = 0.225, w2 = 0.2 × (0.75 / 1.0) = 0.15

    expected = {
        "p1": {"w1": 0.4, "w2": 0.5},
        "p2": {"w1": 0.15, "w2": 0.15},
        "p3": {"w1": 0.225, "w2": 0.15},
    }

    all_pass = True
    for pos_id, exp_weights in expected.items():
        for w, exp_val in exp_weights.items():
            actual = positions[pos_id][w]
            if abs(actual - exp_val) > 0.0001:
                print(f"  [FAIL] {pos_id}.{w}: expected={exp_val:.4f}, got={actual:.4f}")
                all_pass = False

    if all_pass:
        print("  [PASS] Multiple scale factors + per-label rescaling correct (legacy behavior)")
    return all_pass


# =============================================================================
# TEST 3: RESCALING MODIFIERS
# =============================================================================

def test_3_1_scale_holdings_no_lt():
    """scale_holdings_to_100_percent without lookthroughs."""
    print("\n" + "=" * 80)
    print("TEST 3.1: scale_holdings_to_100_percent, no LT")
    print("=" * 80)

    # This is essentially the same as test_per_label_normalization
    input_json = {
        "position_weight_labels": ["w1", "w2"],
        "lookthrough_weight_labels": ["w1", "w2"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.5, "w2": 0.6, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "w2": 0.2, "liquidity_type_id": 2},
                "p3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.2, "w2": 0.2, "liquidity_type_id": 1},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rule = Rule(name="keep_liq_1", apply_to="both",
                criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1})
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: ["scale_holdings_to_100_percent"]}},
        position_weights=["w1", "w2"],
        lookthrough_weights=["w1", "w2"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    # Kept: p1, p3 (liq_type=1)
    # w1: 0.5 + 0.2 = 0.7 -> p1=0.5/0.7, p3=0.2/0.7
    # w2: 0.6 + 0.2 = 0.8 -> p1=0.6/0.8, p3=0.2/0.8

    expected_p1_w1 = 0.5 / 0.7
    expected_p3_w1 = 0.2 / 0.7
    expected_p1_w2 = 0.6 / 0.8
    expected_p3_w2 = 0.2 / 0.8

    all_pass = True
    checks = [
        ("p1", "w1", expected_p1_w1),
        ("p1", "w2", expected_p1_w2),
        ("p3", "w1", expected_p3_w1),
        ("p3", "w2", expected_p3_w2),
    ]

    for pos, w, exp in checks:
        actual = positions[pos][w]
        if abs(actual - exp) < 0.0001:
            print(f"  [PASS] {pos}.{w}: {actual:.4f}")
        else:
            print(f"  [FAIL] {pos}.{w}: expected={exp:.4f}, got={actual:.4f}")
            all_pass = False

    return all_pass


def test_3_2_scale_holdings_with_essential_lt():
    """scale_holdings_to_100_percent with essential_lookthroughs in denominator."""
    print("\n" + "=" * 80)
    print("TEST 3.2: scale_holdings with essential_LT in denominator")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
            },
            "essential_lookthroughs": {
                "lt1": {"instrument_id": 101, "parent_instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.2},
                "lt2": {"instrument_id": 102, "parent_instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.2},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rule = Rule(name="keep_all", apply_to="both", criteria=None)
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: ["scale_holdings_to_100_percent"]}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    # Denominator = positions[w1] + essential_lt[w1] = 0.6 + 0.2 + 0.2 = 1.0
    # p1.w1 = 0.6 / 1.0 = 0.6

    expected = 0.6 / 1.0
    actual = positions["p1"]["w1"]

    if abs(actual - expected) < 0.0001:
        print(f"  [PASS] p1.w1 = {actual:.4f} (includes essential_LT in denom)")
        return True
    else:
        print(f"  [FAIL] p1.w1: expected={expected:.4f}, got={actual:.4f}")
        return False


# =============================================================================
# TEST 4: MULTIPLE PERSPECTIVES
# =============================================================================

def test_4_1_multiple_perspectives_different_rules():
    """Two perspectives with different filtering rules."""
    print("\n" + "=" * 80)
    print("TEST 4.1: Multiple perspectives, different rules")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 2},
                "p3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 1},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Perspective 100: Keep liq=1
    rule_100 = Rule(name="keep_liq_1", apply_to="both",
                    criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1})
    # Perspective 200: Keep liq=2
    rule_200 = Rule(name="keep_liq_2", apply_to="both",
                    criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 2})

    engine.config.perspectives[100] = [rule_100]
    engine.config.perspectives[200] = [rule_200]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: [], 200: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    p100 = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    p200 = result["perspective_configurations"]["main"][200]["portfolio"]["positions"]

    assert "p1" in p100 and "p3" in p100 and "p2" not in p100, "Perspective 100 correct"
    assert "p2" in p200 and "p1" not in p200 and "p3" not in p200, "Perspective 200 correct"
    print("  [PASS] Multiple perspectives with different rules")
    return True


# =============================================================================
# TEST 5: COMPLEX RULES (AND/OR)
# =============================================================================

def test_5_1_rules_with_and():
    """Two rules combined with AND."""
    print("\n" + "=" * 80)
    print("TEST 5.1: Rules with AND condition")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 1, "asset_class": "equity"},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 1, "asset_class": "bond"},
                "p3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 2, "asset_class": "equity"},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep liq=1 AND asset_class=equity -> only p1
    rule_1 = Rule(name="liq_1", apply_to="both",
                  criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
                  condition_for_next_rule="and")
    rule_2 = Rule(name="equity", apply_to="both",
                  criteria={"column": "asset_class", "operator_type": "==", "value": "equity"})

    engine.config.perspectives[100] = [rule_1, rule_2]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    assert "p1" in positions, "p1 should be kept (liq=1 AND equity)"
    assert "p2" not in positions, "p2 should be removed (liq=1 but bond)"
    assert "p3" not in positions, "p3 should be removed (equity but liq=2)"
    print("  [PASS] AND rules work correctly")
    return True


def test_5_2_rules_with_or():
    """Two rules combined with OR."""
    print("\n" + "=" * 80)
    print("TEST 5.2: Rules with OR condition")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 1, "asset_class": "equity"},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.3, "liquidity_type_id": 2, "asset_class": "bond"},
                "p3": {"instrument_id": 300, "sub_portfolio_id": 1, "w1": 0.2, "liquidity_type_id": 2, "asset_class": "equity"},
                "p4": {"instrument_id": 400, "sub_portfolio_id": 1, "w1": 0.2, "liquidity_type_id": 3, "asset_class": "cash"},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep liq=1 OR asset_class=equity -> p1, p3
    rule_1 = Rule(name="liq_1", apply_to="both",
                  criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
                  condition_for_next_rule="or")
    rule_2 = Rule(name="equity", apply_to="both",
                  criteria={"column": "asset_class", "operator_type": "==", "value": "equity"})

    engine.config.perspectives[100] = [rule_1, rule_2]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    positions = result["perspective_configurations"]["main"][100]["portfolio"]["positions"]
    assert "p1" in positions, "p1 kept (liq=1)"
    assert "p2" not in positions, "p2 removed (liq=2, bond)"
    assert "p3" in positions, "p3 kept (equity)"
    assert "p4" not in positions, "p4 removed (liq=3, cash)"
    print("  [PASS] OR rules work correctly")
    return True


# =============================================================================
# TEST 6: EDGE CASES
# =============================================================================

def test_6_1_all_removed():
    """All positions removed by filter."""
    print("\n" + "=" * 80)
    print("TEST 6.1: All positions removed")
    print("=" * 80)

    input_json = {
        "position_weight_labels": ["w1"],
        "lookthrough_weight_labels": ["w1"],
        "portfolio": {
            "position_type": "portfolio",
            "positions": {
                "p1": {"instrument_id": 100, "sub_portfolio_id": 1, "w1": 0.6, "liquidity_type_id": 1},
                "p2": {"instrument_id": 200, "sub_portfolio_id": 1, "w1": 0.4, "liquidity_type_id": 1},
            }
        }
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Keep liq=99 (none match)
    rule = Rule(name="keep_none", apply_to="both",
                criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 99})
    engine.config.perspectives[100] = [rule]

    result = engine.process(
        input_json=input_json,
        perspective_configs={"main": {100: []}},
        position_weights=["w1"],
        lookthrough_weights=["w1"],
        verbose=True
    )

    container = result["perspective_configurations"]["main"][100].get("portfolio", {})
    positions = container.get("positions", {})

    assert len(positions) == 0, "All positions should be removed"
    print("  [PASS] All positions removed correctly")
    return True


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all test cases."""
    print("=" * 80)
    print("COMPREHENSIVE PERSPECTIVE SERVICE TESTS")
    print("=" * 80)

    tests = [
        ("1.1", test_1_1_single_container_no_lt_single_weight),
        ("1.2", test_1_2_multiple_weights),
        ("1.3", test_1_3_multiple_containers),
        ("1.4", test_1_4_with_lookthroughs),
        ("2.1", test_2_1_scaling_rule_no_modifier),
        ("2.2", test_2_2_scaling_rule_with_rescaling),
        ("2.3", test_2_3_multiple_scale_factors),
        ("3.1", test_3_1_scale_holdings_no_lt),
        ("3.2", test_3_2_scale_holdings_with_essential_lt),
        ("4.1", test_4_1_multiple_perspectives_different_rules),
        ("5.1", test_5_1_rules_with_and),
        ("5.2", test_5_2_rules_with_or),
        ("6.1", test_6_1_all_removed),
    ]

    passed = 0
    failed = 0

    for test_id, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] Test {test_id} raised: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
