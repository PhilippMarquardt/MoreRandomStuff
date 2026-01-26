"""
Hardcoded modifier definitions.
These are constants - modifiers and their requirements.

Modifier Types:
- PreProcessing: Positions matching criteria are REMOVED
- PostProcessing: Positions matching criteria are KEPT (savior)
- Scaling: Just flags for scaling operations

required_columns format:
- 'position_data': columns already in input JSON
- 'PARENT_INSTRUMENT': columns from INSTRUMENT table using parent_instrument_id
- 'INSTRUMENT': columns from INSTRUMENT table using instrument_id
- 'INSTRUMENT_CATEGORIZATION': columns from INSTRUMENT_CATEGORIZATION table
"""

from .constants import INT_NULL

SUPPORTED_MODIFIERS = {
    # ==========================================================================
    # PreProcessing - positions matching criteria are REMOVED
    # ==========================================================================
    'exclude_other_net_assets': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 2},
        'required_columns': {'position_data': ['liquidity_type_id']},
        'override_modifiers': []
    },
    'exclude_class_positions': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'is_class_position', 'operator_type': '==', 'value': True},
        'required_columns': {'position_data': ['is_class_position']},
        'override_modifiers': []
    },
    'exclude_non_class_positions': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'is_class_position', 'operator_type': '==', 'value': False},
        'required_columns': {'position_data': ['is_class_position']},
        'override_modifiers': []
    },
    'exclude_future_in_kind_delivery': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'trade_type_id', 'operator_type': '==', 'value': 1},
        'required_columns': {'position_data': ['trade_type_id']},
        'override_modifiers': []
    },
    'exclude_future_trades': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'and': [
            {'column': 'trade_type_id', 'operator_type': '==', 'value': 4},
            {'column': 'upcoming_trade_date', 'operator_type': '==', 'value': True}
        ]},
        'required_columns': {'position_data': ['trade_type_id', 'upcoming_trade_date']},
        'override_modifiers': []
    },
    'exclude_future_flows': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'and': [
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 3},
            {'column': 'upcoming_trade_date', 'operator_type': '==', 'value': True}
        ]},
        'required_columns': {'position_data': ['liquidity_type_id', 'upcoming_trade_date']},
        'override_modifiers': []
    },
    'exclude_pending_stop_trades': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'trade_type_id', 'operator_type': '==', 'value': 2},
        'required_columns': {'position_data': ['trade_type_id']},
        'override_modifiers': []
    },
    'exclude_pending_limit_trades': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'trade_type_id', 'operator_type': '==', 'value': 3},
        'required_columns': {'position_data': ['trade_type_id']},
        'override_modifiers': []
    },
    'exclude_potential_red_flags': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'red_flag_exclusion_type_id', 'operator_type': 'IsNotNull'},
        'required_columns': {'position_data': ['red_flag_exclusion_type_id']},
        'override_modifiers': []
    },
    'exclude_simulated_trades': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'or': [
            {'and': [
                {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
                {'or': [
                    {'column': 'liquidity_type_id', 'operator_type': '==', 'value': INT_NULL},
                    {'column': 'liquidity_type_id', 'operator_type': '!=', 'value': 5}
                ]}
            ]},
            {'column': 'simulated_trade_id', 'operator_type': 'IsNotNull'}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id', 'simulated_trade_id']},
        'override_modifiers': ['include_all_trade_cash', 'include_trade_cash_within_perspective']
    },
    'exclude_simulated_cash': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'and': [
            {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 5}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id']},
        'override_modifiers': ['exclude_perspective_level_simulated_cash', 'include_simulated_cash']
    },
    'exclude_pending_trades': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'position_source_type_id', 'operator_type': '==', 'value': 9},
        'required_columns': {'position_data': ['position_source_type_id']},
        'override_modifiers': []
    },
    'exclude_current_flow': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'and': [
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 3},
            {'column': 'upcoming_trade_date', 'operator_type': '==', 'value': False}
        ]},
        'required_columns': {'position_data': ['liquidity_type_id', 'upcoming_trade_date']},
        'override_modifiers': []
    },
    'exclude_manual_corrections': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'position_source_type_id', 'operator_type': '==', 'value': 8},
        'required_columns': {'position_data': ['position_source_type_id']},
        'override_modifiers': []
    },
    'exclude_initial_positions': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'position_source_type_id', 'operator_type': 'In', 'value': [1, 9, 11]},
        'required_columns': {'position_data': ['position_source_type_id']},
        'override_modifiers': []
    },
    'exclude_blocked_positions': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'column': 'position_blocking_type_id', 'operator_type': 'IsNotNull'},
        'required_columns': {'position_data': ['position_blocking_type_id']},
        'override_modifiers': []
    },
    # This modifier needs PARENT_INSTRUMENT from DB
    'exclude_other_net_assets_excl_investment_grade_accrual': {
        'type': 'PreProcessing',
        'apply_to': 'both',
        'criteria': {'and': [
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 2},
            {'column': 'parent_instrument_subtype_id', 'operator_type': 'NotIn', 'value': [93, 94]}
        ]},
        'required_columns': {
            'position_data': ['liquidity_type_id', 'parent_instrument_id'],
            'parent_instrument': ['instrument_subtype_id']  # Will become parent_instrument_subtype_id
        },
        'override_modifiers': []
    },

    # ==========================================================================
    # PostProcessing - positions matching criteria are KEPT (savior)
    # ==========================================================================
    'include_all_trade_cash': {
        'type': 'PostProcessing',
        'apply_to': 'both',
        'rule_result_operator': 'or',
        'criteria': {'and': [
            {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 6}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id']},
        'override_modifiers': ['exclude_perspective_level_simulated_cash']
    },
    'include_trade_cash_within_perspective': {
        'type': 'PostProcessing',
        'apply_to': 'both',
        'rule_result_operator': 'or',
        'criteria': {'or': [
            {'and': [
                {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
                {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 6}
            ]},
            # Nested criteria: simulated_trade_id IN (values from positions where rule passed)
            {'column': 'simulated_trade_id', 'operator_type': 'In',
             'value': {'table_name': 'rule_result', 'column': 'result', 'operator_type': '==', 'value': True}}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id', 'simulated_trade_id']},
        'override_modifiers': ['exclude_perspective_level_simulated_cash']
    },
    'exclude_trade_cash': {
        'type': 'PostProcessing',
        'apply_to': 'both',
        'rule_result_operator': 'and',
        'criteria': {'not': {'and': [
            {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 6}
        ]}},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id']},
        'override_modifiers': ['exclude_perspective_level_simulated_cash']
    },
    'exclude_perspective_level_simulated_cash': {
        'type': 'PostProcessing',
        'apply_to': 'both',
        'rule_result_operator': 'or',
        'criteria': {'and': [
            {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 5},
            {'column': 'perspective_id', 'operator_type': 'In', 'value': f'({INT_NULL},perspective_id)'}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id', 'perspective_id']},
        'override_modifiers': []
    },
    'include_simulated_cash': {
        'type': 'PostProcessing',
        'apply_to': 'both',
        'rule_result_operator': 'or',
        'criteria': {'and': [
            {'column': 'position_source_type_id', 'operator_type': '==', 'value': 10},
            {'column': 'liquidity_type_id', 'operator_type': '==', 'value': 5}
        ]},
        'required_columns': {'position_data': ['position_source_type_id', 'liquidity_type_id']},
        'override_modifiers': []
    },

    # ==========================================================================
    # Scaling - flags for scaling operations
    # ==========================================================================
    'scale_holdings_to_100_percent': {
        'type': 'Scaling',
        'apply_to': 'both',
        'criteria': None,
        'required_columns': {},
        'override_modifiers': []
    },
    # This modifier needs INSTRUMENT from DB
    'scale_lookthroughs_to_100_percent': {
        'type': 'Scaling',
        'apply_to': 'both',
        'criteria': {'column': 'instrument_subtype_id', 'operator_type': 'In', 'value': [27, 38, 66, 81, 84]},
        'required_columns': {'instrument': ['instrument_subtype_id']},
        'override_modifiers': []
    },
}

# Default modifiers that are always applied (unless overridden)
DEFAULT_MODIFIERS = ['exclude_perspective_level_simulated_cash']
