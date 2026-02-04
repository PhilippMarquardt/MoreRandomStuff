"""Tests for Flask server endpoints."""

import os
import pytest

# Disable DB connection for tests
os.environ["PERSPECTIVE_SERVICE_NO_DB"] = "1"

from perspective_service.server import app, engine

# Disable default modifiers for simpler test data
engine.config.default_modifiers = []


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    """Health endpoint returns ok."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_process_empty_body(client):
    """Process rejects empty body."""
    response = client.post('/api/perspective/process')
    assert response.status_code == 400
    assert "error" in response.json


def test_process_empty_config(client):
    """Process handles empty perspective config."""
    response = client.post('/api/perspective/process', json={
        "perspective_configurations": {},
        "holding": {"positions": {}},
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    })
    assert response.status_code == 200
    assert response.json == {"perspective_configurations": {}}


def test_process_basic(client):
    """Process filters positions correctly."""
    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {
            "cfg": {"-1": []}
        },
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {
                        "column": "keep",
                        "operator_type": "==",
                        "value": True
                    }
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 0.6, "keep": True},
                "p2": {"instrument_id": 2, "sub_portfolio_id": 1, "weight": 0.4, "keep": False},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    response = client.post('/api/perspective/process', json=input_json)
    assert response.status_code == 200

    result = response.json
    # JSON keys are strings, so perspective ID -1 becomes "-1"
    positions = result["perspective_configurations"]["cfg"]["-1"]["holding"]["positions"]
    assert "p1" in positions
    assert "p2" not in positions


def test_process_scaling(client):
    """Process applies scaling rules."""
    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {
            "cfg": {"-1": []}
        },
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {
                        "column": "scale_me",
                        "operator_type": "==",
                        "value": True
                    },
                    "is_scaling_rule": True,
                    "scale_factor": 50  # 50% = 0.5x
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 1.0, "scale_me": True},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    response = client.post('/api/perspective/process', json=input_json)
    assert response.status_code == 200

    result = response.json
    # JSON keys are strings
    positions = result["perspective_configurations"]["cfg"]["-1"]["holding"]["positions"]
    assert positions["p1"]["weight"] == pytest.approx(0.5)


def test_process_invalid_custom_perspective_id(client):
    """Process rejects positive custom perspective ID."""
    input_json = {
        "ed": "2024-01-15",
        "perspective_configurations": {
            "cfg": {"1": []}  # Positive ID with custom rules should fail
        },
        "custom_perspective_rules": {
            "1": {  # Should be negative
                "rules": [{
                    "apply_to": "both",
                    "criteria": {"column": "x", "operator_type": "==", "value": 1}
                }]
            }
        },
        "holding": {
            "positions": {
                "p1": {"instrument_id": 1, "sub_portfolio_id": 1, "weight": 1.0, "x": 1},
            }
        },
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
    }

    response = client.post('/api/perspective/process', json=input_json)
    assert response.status_code == 400
    assert "negative" in response.json["error"].lower()


def test_requirements_empty_body(client):
    """Requirements rejects empty body."""
    response = client.post('/api/perspective/requirements')
    assert response.status_code == 400
    assert "error" in response.json


def test_requirements_basic(client):
    """Requirements returns required columns."""
    input_json = {
        "perspective_configurations": {
            "cfg": {"-1": []}
        },
        "custom_perspective_rules": {
            "-1": {
                "rules": [{
                    "apply_to": "both",
                    "criteria": {
                        "column": "some_col",
                        "operator_type": "==",
                        "value": True,
                        "required_columns": {
                            "instrument": ["asset_class", "currency"]
                        }
                    }
                }]
            }
        },
    }

    response = client.post('/api/perspective/requirements', json=input_json)
    assert response.status_code == 200

    result = response.json
    assert "instrument" in result
    assert "asset_class" in result["instrument"]
    assert "currency" in result["instrument"]


def test_requirements_empty_config(client):
    """Requirements handles empty config."""
    response = client.post('/api/perspective/requirements', json={
        "perspective_configurations": {},
    })
    assert response.status_code == 200
    # Should return at least position_data with sub_portfolio_id
    assert "position_data" in response.json
