def _parse_perspective_collections(
        self, perspective_info, perspective_config_id: str
    ):
        # loop over different collection levels and types to parse
        # perspective service response into SA data structure:
        perspective_data = {}
        scale_factors = {}

        # Default scale factors for each collection level
        DEFAULT_SCALE_FACTORS = {
            'holding': {
                'initial_weight': 0,
                'resulting_weight': 0,
                'initial_exposure_weight': 0,
                'resulting_exposure_weight': 0,
            },
            'contractual_reference': {
                'weight': 0,
            },
            'selected_reference': {
                'weight': 0,
            },
        }

        # Default empty perspective data structure
        def empty_perspective_data():
            return {
                'positions': [],
                'composite_info': {
                    'essential': create_composites_from_underlyings([]),
                    'complete': create_composites_from_underlyings([]),
                },
            }

        # All collection levels we need in the output
        ALL_COLLECTION_LEVELS = ('holding', 'contractual_reference', 'selected_reference')

        # Initialize all collection levels with defaults
        for collection_level in ALL_COLLECTION_LEVELS:
            perspective_data[collection_level] = empty_perspective_data()
            scale_factors[collection_level] = DEFAULT_SCALE_FACTORS.get(collection_level, {})

        # Now process what's actually in the response
        for collection_level, collection_types in perspective_info.items():
            if collection_level not in ALL_COLLECTION_LEVELS:
                continue

            # Update scale_factors if present in response
            if 'scale_factors' in collection_types:
                scale_factors[collection_level] = collection_types['scale_factors']

            insts = {}
            for collection_type, instruments in collection_types.items():
                if collection_type in (
                    'scale_factors',
                    'removed_positions_weight_summary',
                ):
                    continue

                if collection_type not in (
                    'positions',
                    'essential_lookthroughs',
                    'complete_lookthroughs',
                ):
                    continue

                # Skip if instruments is None or missing
                if instruments is None:
                    continue

                collections_to_be_parsed = {
                    'sent_positions': self.payload[collection_level][collection_type],
                    'parent_positions': self.payload[collection_level]['positions'],
                    'weight_properties': self.payload[collection_level][
                        'lookthrough_weight_labels'
                    ]
                    if collection_type.endswith('lookthroughs')
                    else self.payload[collection_level]['position_weight_labels'],
                    'received_positions': instruments or {},
                    'keep_parent_info': collection_type.endswith('lookthroughs'),
                }

                insts[collection_type] = parse_instrument_collections(
                    **collections_to_be_parsed
                )

            # Update perspective_data with parsed instruments (if any)
            if insts:
                perspective_data[collection_level] = {
                    'positions': insts.get('positions', []),
                    'composite_info': {
                        'essential': create_composites_from_underlyings(
                            insts.get('essential_lookthroughs', []),
                        ),
                        'complete': create_composites_from_underlyings(
                            insts.get('complete_lookthroughs', []),
                        ),
                    },
                }

        scale_holdings = (
            'scale_holdings_to_100_percent'
            in list(
                self.payload['perspective_configurations'][
                    perspective_config_id
                ].values()
            )[0]
        )

        # reformat perspective data - highly customized and idiosyncratic:
        return {
            'holdings': perspective_data['holding']['positions'],
            'benchmark': {
                'selected': perspective_data['selected_reference']['positions'],
                'contractual': perspective_data['contractual_reference']['positions'],
            },
            'composite_info': {
                'portfolio': {
                    'essential': perspective_data['holding']['composite_info'].get(
                        'essential', []
                    ),
                    'complete': perspective_data['holding']['composite_info'].get(
                        'complete', []
                    ),
                },
                'benchmark': {
                    'selected': {
                        'essential': perspective_data['selected_reference'][
                            'composite_info'
                        ].get('essential', []), # pylint: disable=line-too-long
                        'complete': perspective_data['selected_reference'][
                            'composite_info'
                        ].get('complete', []), # pylint: disable=line-too-long
                    },
                    'contractual': {
                        'essential': perspective_data['contractual_reference'][
                            'composite_info'
                        ].get('essential', []), # pylint: disable=line-too-long
                        'complete': perspective_data['contractual_reference'][
                            'composite_info'
                        ].get('complete', []), # pylint: disable=line-too-long
                    },
                },
            },
            'scale_factors': scale_factors,
            'scale_holdings': scale_holdings,
        }
