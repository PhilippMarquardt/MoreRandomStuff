"""
Data Ingestion - Handles data loading and preparation from JSON input.
"""

from typing import Dict, List, Tuple, Optional

import polars as pl
import polars.selectors as cs

from ..utils.constants import INT_NULL, FLOAT_NULL
from ..database.loaders.database_loader import DatabaseLoader


class DataIngestion:
    """Handles data loading and preparation from JSON input."""

    @staticmethod
    def build_dataframes(input_json: Dict,
                         required_tables: Dict[str, List[str]],
                         weight_labels: List[str],
                         db_loader: Optional[DatabaseLoader] = None) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Build position and lookthrough dataframes from input JSON.

        Args:
            input_json: Raw input data
            required_tables: Tables required for joins {table_name: [columns]}
            weight_labels: List of weight column names
            db_loader: DatabaseLoader instance for loading reference data

        Returns:
            Tuple of (positions_df, lookthroughs_df) as LazyFrames
        """
        # Extract position and lookthrough data
        positions_data, lookthroughs_data = DataIngestion._extract_data(input_json)

        if not positions_data:
            return pl.LazyFrame(), pl.LazyFrame()

        # Create LazyFrames
        positions_lf = pl.LazyFrame(positions_data, infer_schema_length=None)
        lookthroughs_lf = DataIngestion._create_lookthrough_frame(lookthroughs_data)

        # Normalize column names to lowercase
        positions_lf = positions_lf.rename({c: c.lower() for c in positions_lf.collect_schema().names()})
        if lookthroughs_lf is not None:
            lookthroughs_lf = lookthroughs_lf.rename({c: c.lower() for c in lookthroughs_lf.collect_schema().names()})

        # Standardize columns
        positions_lf = DataIngestion._standardize_columns(positions_lf)
        if lookthroughs_lf is not None:
            lookthroughs_lf = DataIngestion._standardize_columns(lookthroughs_lf)

        # Fill nulls with sentinel values
        positions_lf = DataIngestion._fill_null_values(positions_lf, weight_labels)
        if lookthroughs_lf is not None:
            lookthroughs_lf = DataIngestion._fill_null_values(lookthroughs_lf, weight_labels)

        # Join reference data if needed
        if required_tables and db_loader:
            effective_date = input_json.get('ed')
            if not effective_date:
                raise ValueError("Missing required field 'ed' (effective date)")
            system_version_timestamp = input_json.get('system_version_timestamp')
            positions_lf, lookthroughs_lf = DataIngestion._join_reference_data(
                positions_lf, lookthroughs_lf, required_tables,
                db_loader, system_version_timestamp, effective_date
            )

        return positions_lf, lookthroughs_lf

    @staticmethod
    def get_weight_labels(input_json: Dict) -> Tuple[Dict[str, Tuple[List[str], List[str]]], List[str], List[str]]:
        """
        Extract weight labels per container plus union of all weights.

        Each container can have its own weight labels. Falls back to top-level defaults.
        Matches original behavior in request_position_data.py where each container
        gets its own PositionAndLookthroughData with its own weight labels.

        Args:
            input_json: Raw input data

        Returns:
            Tuple of:
                - per_container: {container_name: (pos_weight_labels, lt_weight_labels)}
                - all_pos_weights: union of all position weights (for null-filling/rescaling)
                - all_lt_weights: union of all lookthrough weights
        """
        per_container = {}
        all_pos = set()
        all_lt = set()

        # Top-level defaults 
        default_pos = [w.lower() for w in (input_json.get('position_weight_labels', []) or [])]
        default_lt = [w.lower() for w in (input_json.get('lookthrough_weight_labels', []) or [])]

        # Check each container for its weight labels 
        for key, container in input_json.items():
            if isinstance(container, dict) and 'positions' in container:
                pos = [w.lower() for w in (container.get('position_weight_labels') or default_pos)]
                lt = [w.lower() for w in (container.get('lookthrough_weight_labels') or default_lt)]
                per_container[key] = (pos, lt)
                all_pos.update(pos)
                all_lt.update(lt)

        # If no containers found, use defaults
        if not per_container:
            all_pos.update(default_pos)
            all_lt.update(default_lt)

        return per_container, list(all_pos), list(all_lt)

    @staticmethod
    def _extract_data(input_json: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Extract position and lookthrough records from input JSON."""
        positions_data = []
        lookthroughs_data = []

        for container_name, container_data in input_json.items():
            if not isinstance(container_data, dict) or "position_type" not in container_data:
                continue

            container_info = {
                "container": container_name,
                "position_type": container_data["position_type"]
            }

            # Extract positions
            if "positions" in container_data:
                for position_id, position_attrs in container_data["positions"].items():
                    positions_data.append({
                        **position_attrs,
                        **container_info,
                        "identifier": position_id,
                        "record_type": "position"
                    })

            # Extract lookthroughs
            for key, lookthrough_data in container_data.items():
                if "lookthrough" in key and isinstance(lookthrough_data, dict):
                    for lookthrough_id, lookthrough_attrs in lookthrough_data.items():
                        lookthroughs_data.append({
                            **lookthrough_attrs,
                            **container_info,
                            "identifier": lookthrough_id,
                            "record_type": key
                        })

        return positions_data, lookthroughs_data

    @staticmethod
    def _create_lookthrough_frame(lookthrough_data: List[Dict]) -> Optional[pl.LazyFrame]:
        """Create a LazyFrame for lookthrough data, or None if no data."""
        if lookthrough_data:
            return pl.LazyFrame(lookthrough_data, infer_schema_length=None)
        return None

    @staticmethod
    def _standardize_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Standardize column names. Only applies safe transformations."""
        columns = lf.collect_schema().names()

        standardizations = []

        # Rename instrument_identifier to instrument_id if exists
        if "instrument_identifier" in columns:
            standardizations.append(
                pl.col("instrument_identifier").alias("instrument_id")
            )

        # sub_portfolio_id is required in input data
        if "sub_portfolio_id" not in columns:
            raise ValueError("sub_portfolio_id column is required in position data")

        if standardizations:
            return lf.with_columns(standardizations)
        return lf

    @staticmethod
    def _fill_null_values(lf: pl.LazyFrame, exclude_columns: List[str]) -> pl.LazyFrame:
        """Fill null values with sentinel values."""
        schema = lf.collect_schema()
        if not schema.names():
            return lf

        # Fill integer nulls
        lf = lf.with_columns(
            cs.numeric().exclude(exclude_columns).fill_null(INT_NULL)
        )

        # Fill float nulls
        float_columns = [
            pl.col(col).fill_null(FLOAT_NULL)
            for col, dtype in schema.items()
            if col not in exclude_columns and dtype in [pl.Float32, pl.Float64]
        ]

        if float_columns:
            lf = lf.with_columns(float_columns)

        return lf

    @staticmethod
    def _join_reference_data(positions_lf: pl.LazyFrame,
                             lookthroughs_lf: pl.LazyFrame,
                             required_tables: Dict[str, List[str]],
                             db_loader: DatabaseLoader,
                             system_version_timestamp: Optional[str],
                             effective_date: str) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Join reference data from database."""
        pos_columns = positions_lf.collect_schema().names()
        lt_columns = lookthroughs_lf.collect_schema().names() if lookthroughs_lf is not None else []

        # Get unique instrument IDs
        pos_ids = positions_lf.select('instrument_id')
        lt_ids = (lookthroughs_lf.select('instrument_id')
                  if lt_columns
                  else pl.LazyFrame(schema={'instrument_id': pl.Int64}))

        unique_ids = pl.concat([pos_ids, lt_ids]).unique().collect().to_series().to_list()

        # Get unique parent_instrument_ids for PARENT_INSTRUMENT lookup (only if column exists)
        if 'parent_instrument_id' in pos_columns:
            parent_ids = positions_lf.select('parent_instrument_id').unique().collect().to_series().to_list()
        else:
            parent_ids = []

        # Get unique asset_allocation_ids for ASSET_ALLOCATION_ANALYTICS_CATEGORY_V lookup
        if 'asset_allocation_id' in pos_columns:
            asset_allocation_ids = positions_lf.select('asset_allocation_id').unique().collect().to_series().to_list()
        else:
            asset_allocation_ids = []

        # Filter out position_data - it's not a reference table, it comes from input JSON
        tables_to_load = {k: v for k, v in required_tables.items()
                          if k.lower() not in ('position_data', 'instrumentinput')}

        # Skip if no tables to load
        if not tables_to_load:
            return positions_lf, lookthroughs_lf

        # Load reference data from database
        ref_data = db_loader.load_reference_data(
            instrument_ids=unique_ids,
            parent_instrument_ids=parent_ids,
            asset_allocation_ids=asset_allocation_ids,
            tables_needed=tables_to_load,
            system_version_timestamp=system_version_timestamp,
            ed=effective_date
        )

        # Join each table 
        for table_name, ref_df in ref_data.items():
            if ref_df.is_empty():
                continue

            ref_lf = ref_df.lazy()
            table_lower = table_name.lower()

            if table_lower == 'parent_instrument':
                # Join on parent_instrument_id 
                if 'parent_instrument_id' in pos_columns:
                    positions_lf = positions_lf.join(
                        ref_lf,
                        left_on='parent_instrument_id',
                        right_on='parent_instrument_id',
                        how='left'
                    )
                    if 'parent_instrument_id' in lt_columns:
                        lookthroughs_lf = lookthroughs_lf.join(
                            ref_lf,
                            left_on='parent_instrument_id',
                            right_on='parent_instrument_id',
                            how='left'
                        )
            elif table_lower == 'asset_allocation_analytics_category_v':
                # Special join: asset_allocation_id <-> analytics_category_id
                if 'asset_allocation_id' in pos_columns:
                    positions_lf = positions_lf.join(
                        ref_lf,
                        left_on='asset_allocation_id',
                        right_on='analytics_category_id',
                        how='left'
                    )
                    if 'asset_allocation_id' in lt_columns:
                        lookthroughs_lf = lookthroughs_lf.join(
                            ref_lf,
                            left_on='asset_allocation_id',
                            right_on='analytics_category_id',
                            how='left'
                        )
            else:
                # Join on instrument_id (default for instrument, instrument_categorization, etc.)
                positions_lf = positions_lf.join(ref_lf, on='instrument_id', how='left')
                if lt_columns:
                    lookthroughs_lf = lookthroughs_lf.join(ref_lf, on='instrument_id', how='left')

        return positions_lf, lookthroughs_lf
