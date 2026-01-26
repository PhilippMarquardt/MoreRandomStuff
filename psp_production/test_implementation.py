"""
Test Implementation - Comprehensive testing of the Perspective Service.

This file tests the perspective service with multiple configurations:
- With and without lookthroughs
- Different containers
- Multiple perspectives
- Output structure validation

Usage:
    python test_implementation.py
"""

import sys
import io
import json
from pprint import pprint
from typing import Dict, Any, List

# Fix Windows console encoding for Polars output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add the current directory to path
sys.path.insert(0, '.')

import polars as pl

from perspective_service.utils.constants import INT_NULL, FLOAT_NULL
from perspective_service.utils.supported_modifiers import SUPPORTED_MODIFIERS, DEFAULT_MODIFIERS
from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier
from perspective_service.core.rule_evaluator import RuleEvaluator
from perspective_service.core.configuration_manager import ConfigurationManager
from perspective_service.core.data_ingestion import DataIngestion
from perspective_service.core.perspective_processor import PerspectiveProcessor
from perspective_service.core.output_formatter import OutputFormatter
from perspective_service.core.engine import PerspectiveEngine


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, message: str):
        self.passed += 1
        print(f"  [OK] {message}")

    def fail(self, message: str):
        self.failed += 1
        self.errors.append(message)
        print(f"  [FAIL] {message}")

    def assert_equal(self, actual, expected, message: str):
        if actual == expected:
            self.ok(message)
        else:
            self.fail(f"{message}: expected {expected}, got {actual}")

    def assert_true(self, condition: bool, message: str):
        if condition:
            self.ok(message)
        else:
            self.fail(message)

    def assert_in(self, item, container, message: str):
        if item in container:
            self.ok(message)
        else:
            self.fail(f"{message}: {item} not in {container}")

    def assert_not_in(self, item, container, message: str):
        if item not in container:
            self.ok(message)
        else:
            self.fail(f"{message}: {item} found in {container}")

    def summary(self):
        print(f"\n  Results: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("  Errors:")
            for err in self.errors:
                print(f"    - {err}")
        return self.failed == 0


def validate_output_structure(result: Dict, test: TestResult,
                              expected_configs: List[str],
                              expected_perspectives: Dict[str, List[int]],
                              expected_containers: Dict[str, Dict[int, List[str]]]):
    """Validate the output structure matches expected format."""

    # Check top-level structure
    test.assert_in("perspective_configurations", result, "Has perspective_configurations key")

    configs = result.get("perspective_configurations", {})

    for config_name in expected_configs:
        test.assert_in(config_name, configs, f"Config '{config_name}' exists")

        config_data = configs.get(config_name, {})
        expected_pids = expected_perspectives.get(config_name, [])

        for pid in expected_pids:
            test.assert_in(pid, config_data, f"Perspective {pid} in config '{config_name}'")

            perspective_data = config_data.get(pid, {})
            expected_cont = expected_containers.get(config_name, {}).get(pid, [])

            for container in expected_cont:
                test.assert_in(container, perspective_data,
                              f"Container '{container}' in perspective {pid}")

                container_data = perspective_data.get(container, {})

                # Each container should have 'positions' if there are positions
                if "positions" in container_data:
                    test.assert_true(isinstance(container_data["positions"], dict),
                                    f"Container '{container}' has positions dict")


# =============================================================================
# TEST 1: Without Lookthroughs - Single Container
# =============================================================================
def test_without_lookthroughs_single_container():
    """Test with positions only, no lookthroughs, single container."""
    print("\n" + "=" * 80)
    print("TEST 1: Without Lookthroughs - Single Container")
    print("=" * 80)

    test = TestResult()

    # Create input with no lookthroughs
    input_json = {
        "portfolio_1": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 0.5, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 0.3, "liquidity_type_id": 2},  # Should be excluded
                "pos_3": {"instrument_id": 300, "weight": 0.2, "liquidity_type_id": 1}
            }
        }
    }

    # Create engine
    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    # Add test perspective
    test_rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[100] = [test_rule]

    perspective_configs = {"config_a": {100: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    # Validate structure
    validate_output_structure(
        result, test,
        expected_configs=["config_a"],
        expected_perspectives={"config_a": [100]},
        expected_containers={"config_a": {100: ["portfolio_1"]}}
    )

    # Check positions
    container_data = result["perspective_configurations"]["config_a"][100]["portfolio_1"]
    positions = container_data.get("positions", {})

    test.assert_in("pos_1", positions, "pos_1 kept (liquidity_type_id=1)")
    test.assert_in("pos_3", positions, "pos_3 kept (liquidity_type_id=1)")
    test.assert_not_in("pos_2", positions, "pos_2 excluded (liquidity_type_id=2)")

    # Check actual weight values
    test.assert_equal(positions["pos_1"]["weight"], 0.5, "pos_1 weight = 0.5")
    test.assert_equal(positions["pos_3"]["weight"], 0.2, "pos_3 weight = 0.2")

    # Check scale_factors
    test.assert_in("scale_factors", container_data, "scale_factors present")
    scale_factors = container_data.get("scale_factors", {})
    test.assert_in("weight", scale_factors, "weight in scale_factors")
    test.assert_equal(scale_factors["weight"], 0.3, "scale_factors weight = 0.3 (removed pos_2)")

    # Check removed_positions_weight_summary
    test.assert_in("removed_positions_weight_summary", container_data, "removed_positions_weight_summary present")
    removed = container_data.get("removed_positions_weight_summary", {})
    test.assert_in("positions", removed, "removed has positions key")
    test.assert_in("pos_2", removed["positions"], "pos_2 in removed positions")

    # No lookthroughs should exist
    test.assert_not_in("essential_lookthroughs", container_data, "No essential_lookthroughs")
    test.assert_not_in("complete_lookthroughs", container_data, "No complete_lookthroughs")

    return test.summary()


# =============================================================================
# TEST 2: Without Lookthroughs - Multiple Containers
# =============================================================================
def test_without_lookthroughs_multiple_containers():
    """Test with positions only, no lookthroughs, multiple containers."""
    print("\n" + "=" * 80)
    print("TEST 2: Without Lookthroughs - Multiple Containers")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "holding": {
            "position_type": "benchmark",
            "positions": {
                "h_pos_1": {"instrument_id": 100, "weight": 0.4, "liquidity_type_id": 1},
                "h_pos_2": {"instrument_id": 200, "weight": 0.3, "liquidity_type_id": 2}  # Excluded
            }
        },
        "reference": {
            "position_type": "fund",
            "positions": {
                "r_pos_1": {"instrument_id": 300, "weight": 0.5, "liquidity_type_id": 1},
                "r_pos_2": {"instrument_id": 400, "weight": 0.5, "liquidity_type_id": 1}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    test_rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[200] = [test_rule]

    perspective_configs = {"config_b": {200: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    # Validate structure
    validate_output_structure(
        result, test,
        expected_configs=["config_b"],
        expected_perspectives={"config_b": [200]},
        expected_containers={"config_b": {200: ["holding", "reference"]}}
    )

    # Check holding container
    holding = result["perspective_configurations"]["config_b"][200]["holding"]
    test.assert_in("h_pos_1", holding.get("positions", {}), "h_pos_1 kept in holding")
    test.assert_not_in("h_pos_2", holding.get("positions", {}), "h_pos_2 excluded from holding")
    test.assert_in("scale_factors", holding, "holding has scale_factors")

    # Check holding weight values
    test.assert_equal(holding["positions"]["h_pos_1"]["weight"], 0.4, "h_pos_1 weight = 0.4")
    test.assert_equal(holding["scale_factors"]["weight"], 0.3, "holding scale_factors weight = 0.3")

    # Check reference container
    reference = result["perspective_configurations"]["config_b"][200]["reference"]
    test.assert_in("r_pos_1", reference.get("positions", {}), "r_pos_1 kept in reference")
    test.assert_in("r_pos_2", reference.get("positions", {}), "r_pos_2 kept in reference")
    # Reference should NOT have scale_factors (nothing removed)
    test.assert_not_in("scale_factors", reference, "reference has no scale_factors (nothing removed)")

    # Check reference weight values
    test.assert_equal(reference["positions"]["r_pos_1"]["weight"], 0.5, "r_pos_1 weight = 0.5")
    test.assert_equal(reference["positions"]["r_pos_2"]["weight"], 0.5, "r_pos_2 weight = 0.5")

    return test.summary()


# =============================================================================
# TEST 3: With Lookthroughs - Single Container
# =============================================================================
def test_with_lookthroughs_single_container():
    """Test with positions and lookthroughs, single container."""
    print("\n" + "=" * 80)
    print("TEST 3: With Lookthroughs - Single Container")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 0.6, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 0.4, "liquidity_type_id": 1}
            },
            "essential_lookthroughs": {
                "lt_1": {"instrument_id": 101, "parent_instrument_id": 100, "weight": 0.3, "liquidity_type_id": 1},
                "lt_2": {"instrument_id": 102, "parent_instrument_id": 100, "weight": 0.3, "liquidity_type_id": 2}  # Excluded
            },
            "complete_lookthroughs": {
                "lt_3": {"instrument_id": 201, "parent_instrument_id": 200, "weight": 0.4, "liquidity_type_id": 1}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    test_rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[300] = [test_rule]

    perspective_configs = {"config_c": {300: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    # Validate structure
    container = result["perspective_configurations"]["config_c"][300]["portfolio"]

    # Check positions
    positions = container.get("positions", {})
    test.assert_in("pos_1", positions, "pos_1 kept")
    test.assert_in("pos_2", positions, "pos_2 kept")

    # Check position weight values
    test.assert_equal(positions["pos_1"]["weight"], 0.6, "pos_1 weight = 0.6")
    test.assert_equal(positions["pos_2"]["weight"], 0.4, "pos_2 weight = 0.4")

    # Check essential_lookthroughs
    test.assert_in("essential_lookthroughs", container, "essential_lookthroughs present")
    essential = container.get("essential_lookthroughs", {})
    test.assert_in("lt_1", essential, "lt_1 kept (liquidity_type_id=1)")
    test.assert_not_in("lt_2", essential, "lt_2 excluded (liquidity_type_id=2)")

    # Check essential_lookthroughs weight values
    test.assert_equal(essential["lt_1"]["weight"], 0.3, "lt_1 weight = 0.3")

    # Check complete_lookthroughs
    test.assert_in("complete_lookthroughs", container, "complete_lookthroughs present")
    complete = container.get("complete_lookthroughs", {})
    test.assert_in("lt_3", complete, "lt_3 kept")

    # Check complete_lookthroughs weight values
    test.assert_equal(complete["lt_3"]["weight"], 0.4, "lt_3 weight = 0.4")

    return test.summary()


# =============================================================================
# TEST 4: With Lookthroughs - Multiple Containers
# =============================================================================
def test_with_lookthroughs_multiple_containers():
    """Test with positions and lookthroughs, multiple containers."""
    print("\n" + "=" * 80)
    print("TEST 4: With Lookthroughs - Multiple Containers")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "holding": {
            "position_type": "benchmark",
            "positions": {
                "h_pos_1": {"instrument_id": 100, "weight": 0.5, "liquidity_type_id": 1}
            },
            "essential_lookthroughs": {
                "h_lt_1": {"instrument_id": 101, "parent_instrument_id": 100, "weight": 0.25, "liquidity_type_id": 1}
            }
        },
        "selected_reference": {
            "position_type": "fund",
            "positions": {
                "s_pos_1": {"instrument_id": 200, "weight": 0.5, "liquidity_type_id": 1},
                "s_pos_2": {"instrument_id": 201, "weight": 0.3, "liquidity_type_id": 2}  # Excluded
            },
            "complete_lookthroughs": {
                "s_lt_1": {"instrument_id": 202, "parent_instrument_id": 200, "weight": 0.2, "liquidity_type_id": 1}
            }
        },
        "contractual_reference": {
            "position_type": "fund",
            "positions": {
                "c_pos_1": {"instrument_id": 300, "weight": 1.0, "liquidity_type_id": 1}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    test_rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[400] = [test_rule]

    perspective_configs = {"config_d": {400: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    perspective_data = result["perspective_configurations"]["config_d"][400]

    # Check all three containers exist
    test.assert_in("holding", perspective_data, "holding container exists")
    test.assert_in("selected_reference", perspective_data, "selected_reference container exists")
    test.assert_in("contractual_reference", perspective_data, "contractual_reference container exists")

    # Check holding
    holding = perspective_data["holding"]
    test.assert_in("h_pos_1", holding.get("positions", {}), "h_pos_1 in holding")
    test.assert_in("essential_lookthroughs", holding, "holding has essential_lookthroughs")
    test.assert_in("h_lt_1", holding.get("essential_lookthroughs", {}), "h_lt_1 in holding")

    # Check holding weight values
    test.assert_equal(holding["positions"]["h_pos_1"]["weight"], 0.5, "h_pos_1 weight = 0.5")
    test.assert_equal(holding["essential_lookthroughs"]["h_lt_1"]["weight"], 0.25, "h_lt_1 weight = 0.25")

    # Check selected_reference
    selected = perspective_data["selected_reference"]
    test.assert_in("s_pos_1", selected.get("positions", {}), "s_pos_1 in selected_reference")
    test.assert_not_in("s_pos_2", selected.get("positions", {}), "s_pos_2 excluded from selected_reference")
    test.assert_in("scale_factors", selected, "selected_reference has scale_factors")
    test.assert_in("complete_lookthroughs", selected, "selected_reference has complete_lookthroughs")

    # Check selected_reference weight values
    test.assert_equal(selected["positions"]["s_pos_1"]["weight"], 0.5, "s_pos_1 weight = 0.5")
    test.assert_equal(selected["scale_factors"]["weight"], 0.3, "selected scale_factors weight = 0.3")
    test.assert_equal(selected["complete_lookthroughs"]["s_lt_1"]["weight"], 0.2, "s_lt_1 weight = 0.2")

    # Check contractual_reference
    contractual = perspective_data["contractual_reference"]
    test.assert_in("c_pos_1", contractual.get("positions", {}), "c_pos_1 in contractual_reference")

    # Check contractual_reference weight values
    test.assert_equal(contractual["positions"]["c_pos_1"]["weight"], 1.0, "c_pos_1 weight = 1.0")

    return test.summary()


# =============================================================================
# TEST 5: Multiple Perspectives in Same Config
# =============================================================================
def test_multiple_perspectives():
    """Test with multiple perspectives in the same config."""
    print("\n" + "=" * 80)
    print("TEST 5: Multiple Perspectives in Same Config")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 0.25, "liquidity_type_id": 1, "position_source_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 0.25, "liquidity_type_id": 2, "position_source_type_id": 1},
                "pos_3": {"instrument_id": 300, "weight": 0.25, "liquidity_type_id": 1, "position_source_type_id": 2},
                "pos_4": {"instrument_id": 400, "weight": 0.25, "liquidity_type_id": 2, "position_source_type_id": 2}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    # Perspective 500: Keep liquidity_type_id == 1
    rule_500 = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[500] = [rule_500]

    # Perspective 501: Keep position_source_type_id == 1
    rule_501 = Rule(
        name="keep_source_1",
        apply_to="both",
        criteria={"column": "position_source_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[501] = [rule_501]

    # Perspective 502: Keep both conditions (AND)
    rule_502 = Rule(
        name="keep_both",
        apply_to="both",
        criteria={
            "and": [
                {"column": "liquidity_type_id", "operator_type": "==", "value": 1},
                {"column": "position_source_type_id", "operator_type": "==", "value": 1}
            ]
        },
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[502] = [rule_502]

    perspective_configs = {
        "multi_perspective_config": {
            500: [],
            501: [],
            502: []
        }
    }

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    config_data = result["perspective_configurations"]["multi_perspective_config"]

    # Check perspective 500 (liquidity_type_id == 1)
    p500 = config_data[500]["portfolio"]["positions"]
    test.assert_in("pos_1", p500, "P500: pos_1 kept (liquidity=1)")
    test.assert_not_in("pos_2", p500, "P500: pos_2 excluded (liquidity=2)")
    test.assert_in("pos_3", p500, "P500: pos_3 kept (liquidity=1)")
    test.assert_not_in("pos_4", p500, "P500: pos_4 excluded (liquidity=2)")

    # Check P500 weight values
    test.assert_equal(p500["pos_1"]["weight"], 0.25, "P500: pos_1 weight = 0.25")
    test.assert_equal(p500["pos_3"]["weight"], 0.25, "P500: pos_3 weight = 0.25")

    # Check perspective 501 (position_source_type_id == 1)
    p501 = config_data[501]["portfolio"]["positions"]
    test.assert_in("pos_1", p501, "P501: pos_1 kept (source=1)")
    test.assert_in("pos_2", p501, "P501: pos_2 kept (source=1)")
    test.assert_not_in("pos_3", p501, "P501: pos_3 excluded (source=2)")
    test.assert_not_in("pos_4", p501, "P501: pos_4 excluded (source=2)")

    # Check P501 weight values
    test.assert_equal(p501["pos_1"]["weight"], 0.25, "P501: pos_1 weight = 0.25")
    test.assert_equal(p501["pos_2"]["weight"], 0.25, "P501: pos_2 weight = 0.25")

    # Check perspective 502 (both conditions)
    p502 = config_data[502]["portfolio"]["positions"]
    test.assert_in("pos_1", p502, "P502: pos_1 kept (liquidity=1 AND source=1)")
    test.assert_not_in("pos_2", p502, "P502: pos_2 excluded")
    test.assert_not_in("pos_3", p502, "P502: pos_3 excluded")
    test.assert_not_in("pos_4", p502, "P502: pos_4 excluded")

    # Check P502 weight values
    test.assert_equal(p502["pos_1"]["weight"], 0.25, "P502: pos_1 weight = 0.25")

    return test.summary()


# =============================================================================
# TEST 6: Multiple Weight Labels
# =============================================================================
def test_multiple_weight_labels():
    """Test with multiple weight labels."""
    print("\n" + "=" * 80)
    print("TEST 6: Multiple Weight Labels")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 0.5, "exposure": 0.6, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 0.3, "exposure": 0.2, "liquidity_type_id": 2},  # Excluded
                "pos_3": {"instrument_id": 300, "weight": 0.2, "exposure": 0.2, "liquidity_type_id": 1}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    test_rule = Rule(
        name="keep_liquidity_1",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 1},
        condition_for_next_rule=None,
        is_scaling_rule=False,
        scale_factor=1.0
    )
    engine.config.perspectives[600] = [test_rule]

    perspective_configs = {"config_weights": {600: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight", "exposure"],
        lookthrough_weights=["weight", "exposure"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    container = result["perspective_configurations"]["config_weights"][600]["portfolio"]

    # Check positions have both weight labels
    positions = container.get("positions", {})
    test.assert_in("pos_1", positions, "pos_1 kept")
    test.assert_in("weight", positions["pos_1"], "pos_1 has weight")
    test.assert_in("exposure", positions["pos_1"], "pos_1 has exposure")
    test.assert_equal(positions["pos_1"]["weight"], 0.5, "pos_1 weight = 0.5")
    test.assert_equal(positions["pos_1"]["exposure"], 0.6, "pos_1 exposure = 0.6")

    # Check scale_factors has both weight labels
    scale_factors = container.get("scale_factors", {})
    test.assert_in("weight", scale_factors, "scale_factors has weight")
    test.assert_in("exposure", scale_factors, "scale_factors has exposure")
    test.assert_equal(scale_factors["weight"], 0.3, "scale_factors weight = 0.3")
    test.assert_equal(scale_factors["exposure"], 0.2, "scale_factors exposure = 0.2")

    # Check removed summary has both weight labels
    removed = container.get("removed_positions_weight_summary", {}).get("positions", {}).get("pos_2", {})
    test.assert_in("weight", removed, "removed pos_2 has weight")
    test.assert_in("exposure", removed, "removed pos_2 has exposure")

    return test.summary()


# =============================================================================
# TEST 7: Custom Perspective Rules
# =============================================================================
def test_custom_perspective_rules():
    """Test custom perspective rules from input JSON."""
    print("\n" + "=" * 80)
    print("TEST 7: Custom Perspective Rules")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "custom_perspective_rules": {
            "-1": {
                "id": -1,
                "name": "Custom Perspective 1",
                "requirements": ["holding"],
                "rules": [
                    {
                        "criteria": {
                            "column": "liquidity_type_id",
                            "operator_type": "!=",
                            "value": 3,
                            "required_columns": {"position_data": ["liquidity_type_id"]}
                        },
                        "apply_to": "both"
                    }
                ]
            },
            "-2": {
                "id": -2,
                "name": "Custom Perspective 2",
                "requirements": ["holding"],
                "rules": [
                    {
                        "criteria": {
                            "column": "position_source_type_id",
                            "operator_type": "==",
                            "value": 1,
                            "required_columns": {"position_data": ["position_source_type_id"]}
                        },
                        "apply_to": "both"
                    }
                ]
            }
        },
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 0.3, "liquidity_type_id": 1, "position_source_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 0.3, "liquidity_type_id": 3, "position_source_type_id": 1},  # Excluded by -1
                "pos_3": {"instrument_id": 300, "weight": 0.4, "liquidity_type_id": 1, "position_source_type_id": 2}   # Excluded by -2
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    perspective_configs = {
        "custom_config": {
            -1: [],
            -2: []
        }
    }

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=True
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    config_data = result["perspective_configurations"]["custom_config"]

    # Check perspective -1 (liquidity_type_id != 3)
    p_neg1 = config_data[-1]["portfolio"]["positions"]
    test.assert_in("pos_1", p_neg1, "P-1: pos_1 kept (liquidity != 3)")
    test.assert_not_in("pos_2", p_neg1, "P-1: pos_2 excluded (liquidity = 3)")
    test.assert_in("pos_3", p_neg1, "P-1: pos_3 kept (liquidity != 3)")

    # Check P-1 weight values
    test.assert_equal(p_neg1["pos_1"]["weight"], 0.3, "P-1: pos_1 weight = 0.3")
    test.assert_equal(p_neg1["pos_3"]["weight"], 0.4, "P-1: pos_3 weight = 0.4")

    # Check perspective -2 (position_source_type_id == 1)
    p_neg2 = config_data[-2]["portfolio"]["positions"]
    test.assert_in("pos_1", p_neg2, "P-2: pos_1 kept (source = 1)")
    test.assert_in("pos_2", p_neg2, "P-2: pos_2 kept (source = 1)")
    test.assert_not_in("pos_3", p_neg2, "P-2: pos_3 excluded (source = 2)")

    # Check P-2 weight values
    test.assert_equal(p_neg2["pos_1"]["weight"], 0.3, "P-2: pos_1 weight = 0.3")
    test.assert_equal(p_neg2["pos_2"]["weight"], 0.3, "P-2: pos_2 weight = 0.3")

    # Test validation: positive IDs should raise error
    print("\n  Testing validation: positive custom perspective IDs should raise error...")
    try:
        bad_input = {"custom_perspective_rules": {"1": {"id": 1, "name": "Bad", "rules": []}}}
        engine._parse_custom_perspectives(bad_input)
        test.fail("Should have raised ValueError for positive ID")
    except ValueError:
        test.ok("Correctly raised ValueError for positive ID")

    return test.summary()


# =============================================================================
# TEST 8: Scaling Rules
# =============================================================================
def test_scaling_rules():
    """Test perspective scaling rules."""
    print("\n" + "=" * 80)
    print("TEST 8: Scaling Rules")
    print("=" * 80)

    test = TestResult()

    input_json = {
        "portfolio": {
            "position_type": "benchmark",
            "positions": {
                "pos_1": {"instrument_id": 100, "weight": 1.0, "liquidity_type_id": 1},
                "pos_2": {"instrument_id": 200, "weight": 1.0, "liquidity_type_id": 2}
            }
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []

    # Scaling rule: Scale positions with liquidity_type_id == 2 by 50%
    scaling_rule = Rule(
        name="scale_liquidity_2",
        apply_to="both",
        criteria={"column": "liquidity_type_id", "operator_type": "==", "value": 2},
        condition_for_next_rule=None,
        is_scaling_rule=True,
        scale_factor=0.5
    )
    engine.config.perspectives[700] = [scaling_rule]

    perspective_configs = {"config_scale": {700: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=False
    )

    print("\nOutput:")
    print(json.dumps(result, indent=2, default=str))

    positions = result["perspective_configurations"]["config_scale"][700]["portfolio"]["positions"]

    # pos_1 should have weight = 1.0 (no scaling)
    test.assert_in("pos_1", positions, "pos_1 present")
    test.assert_equal(positions["pos_1"]["weight"], 1.0, "pos_1 weight = 1.0 (no scaling)")

    # pos_2 should have weight = 0.5 (scaled by 50%)
    test.assert_in("pos_2", positions, "pos_2 present")
    test.assert_equal(positions["pos_2"]["weight"], 0.5, "pos_2 weight = 0.5 (scaled)")

    return test.summary()


# =============================================================================
# TEST 9: Empty Input Handling
# =============================================================================
def test_empty_input():
    """Test handling of empty input."""
    print("\n" + "=" * 80)
    print("TEST 9: Empty Input Handling")
    print("=" * 80)

    test = TestResult()

    # Empty positions
    input_json = {
        "portfolio": {
            "position_type": "benchmark",
            "positions": {}
        }
    }

    engine = PerspectiveEngine(db_connection=None)
    engine.config.default_modifiers = []
    engine.config.perspectives[800] = []

    perspective_configs = {"empty_config": {800: []}}

    result = engine.process(
        input_json=input_json,
        perspective_configs=perspective_configs,
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=False
    )

    print("\nOutput for empty positions:")
    print(json.dumps(result, indent=2, default=str))

    # Should return empty perspective_configurations
    test.assert_in("perspective_configurations", result, "Has perspective_configurations")
    test.assert_equal(result["perspective_configurations"], {}, "perspective_configurations is empty for empty input")

    return test.summary()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("PERSPECTIVE SERVICE - COMPREHENSIVE TESTS")
    print("=" * 80)

    all_passed = True

    tests = [
        ("Test 1: Without Lookthroughs - Single Container", test_without_lookthroughs_single_container),
        ("Test 2: Without Lookthroughs - Multiple Containers", test_without_lookthroughs_multiple_containers),
        ("Test 3: With Lookthroughs - Single Container", test_with_lookthroughs_single_container),
        ("Test 4: With Lookthroughs - Multiple Containers", test_with_lookthroughs_multiple_containers),
        ("Test 5: Multiple Perspectives in Same Config", test_multiple_perspectives),
        ("Test 6: Multiple Weight Labels", test_multiple_weight_labels),
        ("Test 7: Custom Perspective Rules", test_custom_perspective_rules),
        ("Test 8: Scaling Rules", test_scaling_rules),
        ("Test 9: Empty Input Handling", test_empty_input),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n  [ERROR] {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if all_passed:
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  SOME TESTS FAILED!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
