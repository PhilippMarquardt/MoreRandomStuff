"""
Case Sensitivity Tests for Table/Column Name Normalization

Tests verify that table names and column names are normalized to lowercase
throughout the pipeline, preventing duplicate database queries and ensuring
consistent key lookups regardless of input casing.

Components tested:
- ConfigurationManager._update_required_columns
- ConfigurationManager.get_modifier_required_columns
- Engine._determine_required_tables
- DatabaseLoader._build_reference_query
- DatabaseLoader._build_stored_proc_call
- DatabaseLoader._asset_allocation_query
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspective_service.core.configuration_manager import ConfigurationManager
from perspective_service.core.engine import PerspectiveEngine
from perspective_service.database.loaders.database_loader import DatabaseLoader
from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier


# =============================================================================
# ConfigurationManager Tests
# =============================================================================

def test_update_required_columns_table_case_normalization():
    """Test that table names are normalized to lowercase."""
    cm = ConfigurationManager(db_loader=None)
    required = {}

    # Add same table with different cases - should merge to one key
    cm._update_required_columns(required, {'INSTRUMENT': ['col1']})
    cm._update_required_columns(required, {'instrument': ['col2']})
    cm._update_required_columns(required, {'Instrument': ['col3']})
    cm._update_required_columns(required, {'InStRuMeNt': ['col4']})

    assert len(required) == 1, f"Expected 1 table key, got {len(required)}: {list(required.keys())}"
    assert 'instrument' in required, f"Expected lowercase key 'instrument', got {list(required.keys())}"
    assert set(required['instrument']) == {'col1', 'col2', 'col3', 'col4'}, \
        f"Expected all columns merged, got {required['instrument']}"

    print("[PASS] test_update_required_columns_table_case_normalization")


def test_update_required_columns_column_case_normalization():
    """Test that column names are normalized to lowercase."""
    cm = ConfigurationManager(db_loader=None)
    required = {}

    # Add same column with different cases - should deduplicate
    cm._update_required_columns(required, {'table1': ['COLUMN_A', 'Column_B']})
    cm._update_required_columns(required, {'table1': ['column_a', 'COLUMN_C']})  # column_a is duplicate
    cm._update_required_columns(required, {'table1': ['Column_A', 'column_b']})  # both duplicates

    assert 'table1' in required
    assert set(required['table1']) == {'column_a', 'column_b', 'column_c'}, \
        f"Expected 3 unique lowercase columns, got {required['table1']}"

    print("[PASS] test_update_required_columns_column_case_normalization")


def test_update_required_columns_multiple_tables():
    """Test normalization across multiple tables."""
    cm = ConfigurationManager(db_loader=None)
    required = {}

    cm._update_required_columns(required, {
        'INSTRUMENT': ['col1'],
        'PARENT_INSTRUMENT': ['col2'],
        'position_data': ['col3']
    })
    cm._update_required_columns(required, {
        'instrument': ['col1', 'col4'],  # col1 duplicate
        'Parent_Instrument': ['col5'],
        'POSITION_DATA': ['col6']
    })

    assert len(required) == 3, f"Expected 3 tables, got {len(required)}"
    assert set(required.keys()) == {'instrument', 'parent_instrument', 'position_data'}
    assert set(required['instrument']) == {'col1', 'col4'}
    assert set(required['parent_instrument']) == {'col2', 'col5'}
    assert set(required['position_data']) == {'col3', 'col6'}

    print("[PASS] test_update_required_columns_multiple_tables")


def test_get_modifier_required_columns_case_normalization():
    """Test that get_modifier_required_columns returns lowercase keys."""
    cm = ConfigurationManager(db_loader=None)

    # Add modifiers with uppercase table names
    cm.modifiers['test_mod_1'] = Modifier(
        name='test_mod_1',
        modifier_type='PreProcessing',
        apply_to='both',
        criteria={'column': 'col', 'operator_type': '==', 'value': 1},
        required_columns={'INSTRUMENT': ['instrument_subtype_id']},
        override_modifiers=[]
    )
    cm.modifiers['test_mod_2'] = Modifier(
        name='test_mod_2',
        modifier_type='PreProcessing',
        apply_to='both',
        criteria={'column': 'col', 'operator_type': '==', 'value': 2},
        required_columns={'PARENT_INSTRUMENT': ['INSTRUMENT_SUBTYPE_ID']},
        override_modifiers=[]
    )

    result = cm.get_modifier_required_columns(['test_mod_1', 'test_mod_2'])

    # All keys should be lowercase
    for key in result.keys():
        assert key == key.lower(), f"Table key '{key}' is not lowercase"

    # All column names should be lowercase
    for table, columns in result.items():
        for col in columns:
            assert col == col.lower(), f"Column '{col}' in table '{table}' is not lowercase"

    assert 'instrument' in result
    assert 'parent_instrument' in result

    print("[PASS] test_get_modifier_required_columns_case_normalization")


# =============================================================================
# Engine Tests
# =============================================================================

def test_determine_required_tables_case_normalization():
    """Test that _determine_required_tables normalizes all keys to lowercase."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Add perspective with uppercase table names in required_columns
    engine.config.perspectives[1] = [
        Rule(
            name='rule1',
            apply_to='both',
            criteria={'column': 'test', 'operator_type': '==', 'value': 1},
            is_scaling_rule=False,
            scale_factor=1.0
        )
    ]
    engine.config.required_columns_by_perspective[1] = {
        'INSTRUMENT': ['col1', 'COL2'],
        'INSTRUMENT_CATEGORIZATION': ['col3']
    }

    # Add another perspective with lowercase
    engine.config.perspectives[2] = [
        Rule(
            name='rule2',
            apply_to='both',
            criteria={'column': 'test', 'operator_type': '==', 'value': 2},
            is_scaling_rule=False,
            scale_factor=1.0
        )
    ]
    engine.config.required_columns_by_perspective[2] = {
        'instrument': ['col1', 'col4'],  # col1 duplicate
        'Instrument_Categorization': ['COL5']
    }

    perspective_configs = {
        'config1': {
            '1': [],
            '2': []
        }
    }

    result = engine._determine_required_tables(perspective_configs)

    # All keys should be lowercase
    for key in result.keys():
        assert key == key.lower(), f"Table key '{key}' is not lowercase"

    # Should have merged tables
    assert 'instrument' in result
    assert 'instrument_categorization' in result

    # Columns should be merged and lowercase
    assert set(result['instrument']) == {'col1', 'col2', 'col4'}
    assert set(result['instrument_categorization']) == {'col3', 'col5'}

    print("[PASS] test_determine_required_tables_case_normalization")


def test_determine_required_tables_includes_position_data():
    """Test that position_data is included and normalized to lowercase."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Add modifier with POSITION_DATA (uppercase)
    engine.config.modifiers['test_mod'] = Modifier(
        name='test_mod',
        modifier_type='PreProcessing',
        apply_to='both',
        criteria={'column': 'col', 'operator_type': '==', 'value': 1},
        required_columns={
            'POSITION_DATA': ['liquidity_type_id'],
            'INSTRUMENT': ['name']
        },
        override_modifiers=[]
    )

    engine.config.perspectives[1] = []

    perspective_configs = {
        'config1': {'1': ['test_mod']}
    }

    result = engine._determine_required_tables(perspective_configs)

    # position_data should be included (caller needs to know what position columns to send)
    assert 'position_data' in result, f"position_data should be included, got {result.keys()}"
    assert 'liquidity_type_id' in result['position_data'], f"Expected liquidity_type_id in position_data"
    assert 'instrument' in result

    print("[PASS] test_determine_required_tables_includes_position_data")


def test_determine_required_tables_edge_cases():
    """Test edge case handling for asset_allocation and parent_instrument join keys."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Add perspective with ASSET_ALLOCATION_ANALYTICS_CATEGORY_V and PARENT_INSTRUMENT
    engine.config.perspectives[1] = []
    engine.config.required_columns_by_perspective[1] = {
        'ASSET_ALLOCATION_ANALYTICS_CATEGORY_V': ['analytics_category_id', 'category_name'],
        'PARENT_INSTRUMENT': ['instrument_id', 'parent_name'],
        'INSTRUMENT': ['name']
    }

    perspective_configs = {
        'config1': {'1': []}
    }

    result = engine._determine_required_tables(perspective_configs)

    # position_data should have asset_allocation_id and parent_instrument_id added
    assert 'position_data' in result, f"position_data should be created for join keys"
    assert 'asset_allocation_id' in result['position_data'], \
        f"asset_allocation_id should be added to position_data, got {result.get('position_data')}"
    assert 'parent_instrument_id' in result['position_data'], \
        f"parent_instrument_id should be added to position_data, got {result.get('position_data')}"

    # analytics_category_id should be removed from asset_allocation table (it's a join key)
    if 'asset_allocation_analytics_category_v' in result:
        assert 'analytics_category_id' not in result['asset_allocation_analytics_category_v'], \
            "analytics_category_id should be removed from asset_allocation table"
        # category_name should still be there
        assert 'category_name' in result['asset_allocation_analytics_category_v']

    # instrument_id should be removed from parent_instrument table (it's a join key)
    if 'parent_instrument' in result:
        assert 'instrument_id' not in result['parent_instrument'], \
            "instrument_id should be removed from parent_instrument table"
        # parent_name should still be there
        assert 'parent_name' in result['parent_instrument']

    # instrument table should still be there
    assert 'instrument' in result

    print("[PASS] test_determine_required_tables_edge_cases")


# =============================================================================
# DatabaseLoader Tests
# =============================================================================

def test_build_reference_query_instrument_case_variations():
    """Test _build_reference_query handles all case variations of 'instrument'."""
    loader = DatabaseLoader('dummy_connection')

    # All these should produce valid queries
    cases = ['instrument', 'INSTRUMENT', 'Instrument', 'InStRuMeNt']

    for case in cases:
        query = loader._build_reference_query(
            case, ['col1'], [1, 2, 3], [], [], None, None
        )
        assert query is not None, f"Query should not be None for table '{case}'"
        assert 'INSTRUMENT' in query, f"Query should reference INSTRUMENT table for '{case}'"

    print("[PASS] test_build_reference_query_instrument_case_variations")


def test_build_reference_query_parent_instrument_case_variations():
    """Test _build_reference_query handles all case variations of 'parent_instrument'."""
    loader = DatabaseLoader('dummy_connection')

    cases = ['parent_instrument', 'PARENT_INSTRUMENT', 'Parent_Instrument']

    for case in cases:
        query = loader._build_reference_query(
            case, ['col1'], [], [1, 2, 3], [], None, None
        )
        assert query is not None, f"Query should not be None for table '{case}'"
        # Parent instrument uses INSTRUMENT table with parent_instrument_ids
        assert 'INSTRUMENT' in query, f"Query should reference INSTRUMENT table for '{case}'"

    print("[PASS] test_build_reference_query_parent_instrument_case_variations")


def test_build_reference_query_instrument_categorization_case_variations():
    """Test _build_reference_query handles all case variations of 'instrument_categorization'."""
    loader = DatabaseLoader('dummy_connection')

    cases = ['instrument_categorization', 'INSTRUMENT_CATEGORIZATION', 'Instrument_Categorization']

    for case in cases:
        query = loader._build_reference_query(
            case, ['col1'], [1, 2, 3], [], [], None, None
        )
        assert query is not None, f"Query should not be None for table '{case}'"
        assert 'INSTRUMENT_CATEGORIZATION' in query, \
            f"Query should reference INSTRUMENT_CATEGORIZATION table for '{case}'"

    print("[PASS] test_build_reference_query_instrument_categorization_case_variations")


def test_build_reference_query_asset_allocation_case_variations():
    """Test _build_reference_query handles all case variations of asset allocation table."""
    loader = DatabaseLoader('dummy_connection')

    cases = [
        'asset_allocation_analytics_category_v',
        'ASSET_ALLOCATION_ANALYTICS_CATEGORY_V',
        'Asset_Allocation_Analytics_Category_V'
    ]

    for case in cases:
        query = loader._build_reference_query(
            case, ['col1'], [], [], [1, 2, 3], None, None
        )
        assert query is not None, f"Query should not be None for table '{case}'"
        assert 'ASSET_ALLOCATION_ANALYTICS_CATEGORY_V' in query, \
            f"Query should reference ASSET_ALLOCATION_ANALYTICS_CATEGORY_V table for '{case}'"

    print("[PASS] test_build_reference_query_asset_allocation_case_variations")


def test_build_stored_proc_call_filters_instrument_id_any_case():
    """Test that instrument_id is filtered out regardless of case."""
    loader = DatabaseLoader('dummy_connection')

    # Test with various cases of instrument_id
    test_cases = [
        (['instrument_id', 'col1'], 'col1'),
        (['INSTRUMENT_ID', 'col1'], 'col1'),
        (['Instrument_Id', 'col1'], 'col1'),
        (['INSTRUMENT_ID', 'COL1', 'col2'], 'COL1,col2'),
    ]

    for columns, expected_cols in test_cases:
        query = loader._build_stored_proc_call('INSTRUMENT', [1], columns, None, None)
        assert query is not None, f"Query should not be None for columns {columns}"
        assert f"@REQUIRED_COLUMNS = '{expected_cols}'" in query, \
            f"Expected columns '{expected_cols}' in query, got: {query}"

    print("[PASS] test_build_stored_proc_call_filters_instrument_id_any_case")


def test_build_stored_proc_call_only_instrument_id_returns_none():
    """Test that query returns None when only instrument_id is requested (any case)."""
    loader = DatabaseLoader('dummy_connection')

    cases = [['instrument_id'], ['INSTRUMENT_ID'], ['Instrument_Id']]

    for columns in cases:
        query = loader._build_stored_proc_call('INSTRUMENT', [1], columns, None, None)
        assert query is None, f"Query should be None when only instrument_id requested: {columns}"

    print("[PASS] test_build_stored_proc_call_only_instrument_id_returns_none")


def test_asset_allocation_query_analytics_category_id_any_case():
    """Test that analytics_category_id is added if missing (case-insensitive check)."""
    loader = DatabaseLoader('dummy_connection')

    # When analytics_category_id is missing, it should be added to SELECT
    query1 = loader._asset_allocation_query([1, 2], ['col1', 'col2'])
    assert 'analytics_category_id' in query1.lower(), \
        "analytics_category_id should be added when missing"
    # Should appear in SELECT clause
    select_part = query1.split('FROM')[0].lower()
    assert 'analytics_category_id' in select_part, \
        "analytics_category_id should be in SELECT clause"

    # When ANALYTICS_CATEGORY_ID is present (uppercase), should not duplicate in SELECT
    query2 = loader._asset_allocation_query([1, 2], ['ANALYTICS_CATEGORY_ID', 'col1'])
    # Extract SELECT clause and count occurrences there
    select_part2 = query2.split('FROM')[0].lower()
    count_in_select = select_part2.count('analytics_category_id')
    assert count_in_select == 1, \
        f"analytics_category_id should appear once in SELECT, found {count_in_select} times"

    # Also test with lowercase - should not duplicate
    query3 = loader._asset_allocation_query([1, 2], ['analytics_category_id', 'col1'])
    select_part3 = query3.split('FROM')[0].lower()
    count_in_select3 = select_part3.count('analytics_category_id')
    assert count_in_select3 == 1, \
        f"analytics_category_id should appear once in SELECT (lowercase input), found {count_in_select3}"

    print("[PASS] test_asset_allocation_query_analytics_category_id_any_case")


# =============================================================================
# Integration Test
# =============================================================================

def test_integration_mixed_case_custom_perspective():
    """Integration test: custom perspective with mixed-case required_columns."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Create input with custom perspective that has mixed-case required_columns
    input_json = {
        "ed": "2024-01-15",
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
        "custom_perspective_rules": {
            "-1": {
                "rules": [
                    {
                        "apply_to": "both",
                        "criteria": {
                            "column": "filter_col",
                            "operator_type": "==",
                            "value": True,
                            "required_columns": {
                                "INSTRUMENT": ["col1"],
                                "instrument": ["col2"],  # Same table, different case
                                "Instrument": ["COL1"],  # Duplicate column, different case
                            }
                        }
                    }
                ]
            }
        },
        "holding": {
            "position_type": "holding",
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "weight": 0.5,
                    "filter_col": True
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "weight": 0.5,
                    "filter_col": False
                }
            }
        }
    }

    # Parse custom perspective
    engine._parse_custom_perspectives(input_json)

    # Check that required_columns were normalized
    assert -1 in engine.config.required_columns_by_perspective
    req_cols = engine.config.required_columns_by_perspective[-1]

    # Should have single 'instrument' key (not multiple case variants)
    assert len(req_cols) == 1, f"Expected 1 table, got {len(req_cols)}: {list(req_cols.keys())}"
    assert 'instrument' in req_cols, f"Expected 'instrument' key, got {list(req_cols.keys())}"

    # Should have merged columns without duplicates
    assert set(req_cols['instrument']) == {'col1', 'col2'}, \
        f"Expected merged columns, got {req_cols['instrument']}"

    print("[PASS] test_integration_mixed_case_custom_perspective")


def test_integration_full_pipeline_no_db():
    """Integration test: full pipeline with mixed-case, no database."""
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    # Add perspective with rule
    engine.config.perspectives[1] = [
        Rule(
            name='filter_rule',
            apply_to='both',
            criteria={'column': 'keep_me', 'operator_type': '==', 'value': True},
            is_scaling_rule=False,
            scale_factor=1.0
        )
    ]

    # Add required columns with mixed case
    engine.config.required_columns_by_perspective[1] = {
        'INSTRUMENT': ['COL1'],
        'instrument': ['col2']  # Should merge with above
    }

    input_json = {
        "ed": "2024-01-15",
        "position_weight_labels": ["weight"],
        "lookthrough_weight_labels": ["weight"],
        "holding": {
            "position_type": "holding",
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "weight": 0.6,
                    "keep_me": True
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "weight": 0.4,
                    "keep_me": False
                }
            }
        }
    }

    # Process (no DB, so reference data won't be loaded, but normalization should work)
    output = engine.process(
        input_json=input_json,
        perspective_configs={"test": {"1": []}},
        position_weights=["weight"],
        lookthrough_weights=["weight"],
        verbose=False
    )

    # Verify output
    result = output["perspective_configurations"]["test"][1]
    positions = result["holding"]["positions"]

    assert "pos_1" in positions, "pos_1 should be kept"
    assert "pos_2" not in positions, "pos_2 should be filtered out"

    print("[PASS] test_integration_full_pipeline_no_db")


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("CASE SENSITIVITY TESTS")
    print("=" * 80)
    print()

    tests = [
        # ConfigurationManager tests
        test_update_required_columns_table_case_normalization,
        test_update_required_columns_column_case_normalization,
        test_update_required_columns_multiple_tables,
        test_get_modifier_required_columns_case_normalization,

        # Engine tests
        test_determine_required_tables_case_normalization,
        test_determine_required_tables_includes_position_data,
        test_determine_required_tables_edge_cases,

        # DatabaseLoader tests
        test_build_reference_query_instrument_case_variations,
        test_build_reference_query_parent_instrument_case_variations,
        test_build_reference_query_instrument_categorization_case_variations,
        test_build_reference_query_asset_allocation_case_variations,
        test_build_stored_proc_call_filters_instrument_id_any_case,
        test_build_stored_proc_call_only_instrument_id_returns_none,
        test_asset_allocation_query_analytics_category_id_any_case,

        # Integration tests
        test_integration_mixed_case_custom_perspective,
        test_integration_full_pipeline_no_db,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test in tests:
        print(f"  [{'PASS' if True else 'FAIL'}] {test.__name__}")
    print()
    print(f"  Total: {passed}/{len(tests)} passed")
    print()

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {failed} test(s)")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
