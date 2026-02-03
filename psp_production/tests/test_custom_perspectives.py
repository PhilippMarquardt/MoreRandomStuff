"""
Tests for custom perspective rules functionality.

Tests verify that custom perspectives (negative IDs) work correctly:
- Basic filtering with criteria
- Scaling rules with scale_factor
- Multiple rules with AND/OR conditions
- apply_to scoping
- get_requirements() includes custom perspective required_columns
"""

import pytest

from perspective_service.core.engine import PerspectiveEngine


def test_custom_perspective_basic_filter():
    """Custom perspective with criteria filters positions correctly."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {"column": "keep_me", "operator_type": "==", "value": True}
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    result = engine.process(input_json)

    positions = result["perspective_configurations"]["cfg"][-1]["holding"]["positions"]
    assert "p1" in positions, "Position with keep_me=True should be kept"
    assert "p2" not in positions, "Position with keep_me=False should be filtered"


def test_custom_perspective_scaling_rule():
    """Custom perspective with scaling rule applies factor correctly."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {"column": "scale_me", "operator_type": "==", "value": True},
                    "is_scaling_rule": True,
                    "scale_factor": 50  # 50% = 0.5x
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 1.0, "scale_me": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 1.0, "scale_me": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    result = engine.process(input_json)

    positions = result["perspective_configurations"]["cfg"][-1]["holding"]["positions"]
    # p1 matches scaling rule, gets 50% factor
    assert positions["p1"]["weight"] == pytest.approx(0.5), "Scaled position should have 0.5 weight"
    # p2 doesn't match, keeps original weight
    assert positions["p2"]["weight"] == pytest.approx(1.0), "Unscaled position should have 1.0 weight"


def test_custom_perspective_and_condition():
    """Custom perspective with AND logic requires all rules to pass."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [
                    {
                        "apply_to": "both",
                        "criteria": {"column": "cond_a", "operator_type": "==", "value": True},
                        "condition_for_next_rule": None  # AND (default)
                    },
                    {
                        "apply_to": "both",
                        "criteria": {"column": "cond_b", "operator_type": "==", "value": True}
                    }
                ]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": True, "cond_b": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": True, "cond_b": False},
                "p3": {"instrument_id": 3, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": False, "cond_b": True},
                "p4": {"instrument_id": 4, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": False, "cond_b": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    result = engine.process(input_json)

    positions = result["perspective_configurations"]["cfg"][-1]["holding"]["positions"]
    # Only p1 has both conditions True
    assert "p1" in positions, "Position with both conditions True should be kept"
    assert "p2" not in positions, "Position with cond_a=True, cond_b=False should be filtered"
    assert "p3" not in positions, "Position with cond_a=False, cond_b=True should be filtered"
    assert "p4" not in positions, "Position with both conditions False should be filtered"


def test_custom_perspective_or_condition():
    """Custom perspective with OR logic keeps if any rule passes."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [
                    {
                        "apply_to": "both",
                        "criteria": {"column": "cond_a", "operator_type": "==", "value": True},
                        "condition_for_next_rule": "or"
                    },
                    {
                        "apply_to": "both",
                        "criteria": {"column": "cond_b", "operator_type": "==", "value": True}
                    }
                ]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": True, "cond_b": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": True, "cond_b": False},
                "p3": {"instrument_id": 3, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": False, "cond_b": True},
                "p4": {"instrument_id": 4, "sub_portfolio_id": 1, "weight": 0.25, "cond_a": False, "cond_b": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    result = engine.process(input_json)

    positions = result["perspective_configurations"]["cfg"][-1]["holding"]["positions"]
    # p1, p2, p3 have at least one condition True
    assert "p1" in positions, "Position with both conditions True should be kept"
    assert "p2" in positions, "Position with cond_a=True should be kept (OR)"
    assert "p3" in positions, "Position with cond_b=True should be kept (OR)"
    assert "p4" not in positions, "Position with both conditions False should be filtered"


def test_custom_perspective_apply_to_holding():
    """Custom perspective with apply_to='holding' only affects holding container."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "holding",  # Only affects holding container
                    "criteria": {"column": "keep_me", "operator_type": "==", "value": True}
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": False},
            }
        },
        "reference": {
            "positions": {
                "r1": {"instrument_id": 3, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": True},
                "r2": {"instrument_id": 4, "sub_portfolio_id": 1, "weight": 0.5, "keep_me": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    result = engine.process(input_json)

    holding_positions = result["perspective_configurations"]["cfg"][-1]["holding"]["positions"]
    reference_positions = result["perspective_configurations"]["cfg"][-1]["reference"]["positions"]

    # Holding container is filtered
    assert "p1" in holding_positions, "Holding position with keep_me=True should be kept"
    assert "p2" not in holding_positions, "Holding position with keep_me=False should be filtered"

    # Reference container is NOT filtered (apply_to='holding' doesn't match 'reference')
    assert "r1" in reference_positions, "Reference position should be kept (rule doesn't apply)"
    assert "r2" in reference_positions, "Reference position should be kept (rule doesn't apply)"


def test_get_requirements_includes_custom():
    """get_requirements() includes custom perspective required_columns."""
    engine = PerspectiveEngine()

    input_json = {
        "perspective_configurations": {"cfg": {"-1": []}},
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {
                        "column": "some_col",
                        "operator_type": "==",
                        "value": True,
                        "required_columns": {
                            "instrument": ["instrument_type_id", "currency_id"]
                        }
                    }
                }]
            }
        },
    }

    requirements = engine.get_requirements(input_json)

    # Should include the custom perspective's required_columns
    assert "instrument" in requirements, "Requirements should include 'instrument' table"
    assert "instrument_type_id" in requirements["instrument"], "Requirements should include instrument_type_id"
    assert "currency_id" in requirements["instrument"], "Requirements should include currency_id"


def test_custom_perspective_positive_id_raises():
    """Custom perspective with positive ID raises ValueError."""
    engine = PerspectiveEngine()
    engine.config.default_modifiers = []

    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {"cfg": {"1": []}},  # Positive ID
        "custom_perspective_rules": {
            "1": {  # Positive ID - should fail
                "rules": [{
                    "apply_to": "both",
                    "criteria": {"column": "x", "operator_type": "==", "value": True}
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 1.0, "x": True},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    with pytest.raises(ValueError, match="negative"):
        engine.process(input_json)


if __name__ == "__main__":
    test_custom_perspective_basic_filter()
    test_custom_perspective_scaling_rule()
    test_custom_perspective_and_condition()
    test_custom_perspective_or_condition()
    test_custom_perspective_apply_to_holding()
    test_get_requirements_includes_custom()
    test_custom_perspective_positive_id_raises()
    print("All tests passed!")
