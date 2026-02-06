"""
Database Loader - Single class for all database operations.
Uses Polars read_database with arrow-odbc for efficient Arrow-native queries.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import polars as pl
import pyodbc
from opentelemetry import trace, context as otel_context

tracer = trace.get_tracer(__name__)

class DatabaseLoadError(Exception):
    """Raised when database loading fails."""
    pass


class DatabaseLoader:
    """
    Single class for all database operations.
    Uses pl.read_database() with ODBC connection string (arrow-odbc under the hood).
    """

    def __init__(self, connection_string: str):
        self._connection_string = connection_string

    def _execute_query(self, query: str, use_pyodbc: bool = False) -> pl.DataFrame:
        """Execute query and return Polars DataFrame with normalized column names.

        Args:
            query: SQL query string
            use_pyodbc: If True, use pyodbc connection. If False (default), use arrow-odbc.
        """
        with tracer.start_as_current_span("db.query") as span:
            span.set_attribute("db.system", "mssql")
            span.set_attribute("db.statement", query[:500])
            if use_pyodbc:
                # We can use polars interface as it takes pyodbc if provided an already open connection.
                # CAUTION: This WILL lock GIL when querying much data
                with pyodbc.connect(self._connection_string) as conn:
                    df = pl.read_database(query, conn)
            else:
                # arrow-odbc
                df = pl.read_database(
                    query,
                    self._connection_string,
                    execute_options={"max_text_size": 999999}
                )
            # Normalize column names to lowercase
            df = df.rename({c: c.lower() for c in df.columns})
            span.set_attribute("db.result_rows", len(df))
            return df

    # ==================== PERSPECTIVES ====================

    def load_perspectives(self, system_version_timestamp: Optional[str] = None) -> Dict[int, Dict]:
        """Load perspectives from FN_GET_SUBSETTING_SERVICE_PERSPECTIVES."""
        with tracer.start_as_current_span("db.load_perspectives") as span:
            try:
                sql_timestamp = 'null' if system_version_timestamp is None else repr(system_version_timestamp)
                query = f"SELECT [dbo].[FN_GET_SUBSETTING_SERVICE_PERSPECTIVES]({sql_timestamp})"
                df = self._execute_query(query, use_pyodbc=True)

                if df.is_empty():
                    raise DatabaseLoadError("No perspectives found in database")

                json_str = df.item(0, 0)
                json_data = json.loads(json_str)
                raw_perspectives = json_data.get('perspectives', [])

                # Group by perspective ID
                grouped = {}
                for p in raw_perspectives:
                    pid = p.get('id')
                    if pid not in grouped:
                        grouped[pid] = {
                            'id': pid,
                            'name': p.get('name'),
                            'is_active': p.get('is_active', True),
                            'is_supported': p.get('is_compatible_with_sub_setting_service', True),
                            'rules': []
                        }
                    grouped[pid]['is_active'] &= p.get('is_active', True)
                    grouped[pid]['is_supported'] &= bool(p.get('is_compatible_with_sub_setting_service', True))
                    grouped[pid]['rules'].extend(p.get('rules', []))

                span.set_attribute("num_perspectives", len(grouped))
                return grouped

            except json.JSONDecodeError as e:
                raise DatabaseLoadError(f"Failed to parse perspective JSON: {e}")
            except Exception as e:
                raise DatabaseLoadError(f"Database error loading perspectives: {e}")

    # ==================== REFERENCE DATA ====================

    def load_reference_data(self,
                            instrument_ids: List[int],
                            parent_instrument_ids: List[int],
                            asset_allocation_ids: List[int],
                            tables_needed: Dict[str, List[str]],
                            system_version_timestamp: Optional[str],
                            ed: Optional[str]) -> Dict[str, pl.DataFrame]:
        """Load reference data for specified tables in PARALLEL."""
        with tracer.start_as_current_span("db.load_reference_data") as span:
            span.set_attribute("num_instrument_ids", len(instrument_ids))

            # Build list of (table_name, query) tasks
            tasks = []
            for table_name, columns in tables_needed.items():
                # We now call it position_data but it was once called instrumentinput. Since criteria were never updated we have to have it here
                if table_name.lower() == 'position_data' or table_name.lower() == "instrumentinput":
                    continue

                query = self._build_reference_query(
                    table_name, columns, instrument_ids,
                    parent_instrument_ids, asset_allocation_ids,
                    system_version_timestamp, ed
                )
                if query:
                    tasks.append((table_name, query))

            if not tasks:
                return {}

            span.set_attribute("num_tables", len(tasks))

            # Capture parent context for thread propagation
            parent_ctx = otel_context.get_current()

            def _run_query(table_name: str, query: str) -> pl.DataFrame:
                token = otel_context.attach(parent_ctx)
                try:
                    with tracer.start_as_current_span(f"db.query.{table_name}") as q_span:
                        q_span.set_attribute("db.table", table_name)
                        return self._execute_query(query)
                finally:
                    otel_context.detach(token)

            # Execute ALL queries in parallel (As long as arrow-odbc connection...)
            results = {}
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {
                    executor.submit(_run_query, table_name, query): table_name
                    for table_name, query in tasks
                }
                for future in as_completed(futures):
                    table_name = futures[future]
                    try:
                        # Normalize key to lowercase for consistent lookups
                        results[table_name.lower()] = future.result()
                    except Exception as e:
                        raise DatabaseLoadError(f"Failed to load {table_name}: {e}")

            # Post-process parent_instrument
            # This is needed as the instrument_id of loaded data is actually the data for positions parent_instrument_id
            if 'parent_instrument' in results and not results['parent_instrument'].is_empty():
                df = results['parent_instrument']
                rename_map = {'instrument_id': 'parent_instrument_id'}
                for c in df.columns:
                    if c != 'instrument_id':
                        rename_map[c] = f'parent_{c}'
                results['parent_instrument'] = df.rename(rename_map)

            return results

    def _build_stored_proc_call(self, table_name: str, instrument_ids: List[int],
                                columns: List[str], system_version_timestamp: Optional[str],
                                ed: Optional[str]) -> Optional[str]:
        """Build EXEC call for GET_DATA_FOR_PERSPECTIVE_SERVICE stored procedure.

        Note: The stored proc always returns instrument_id as the first column,
        so we must NOT include it in @REQUIRED_COLUMNS.
        """
        ids_str = ",".join(map(str, instrument_ids))

        # Filter out instrument_id - stored proc always returns it
        cols_to_select = [c for c in columns if c.lower() != 'instrument_id']
        if not cols_to_select:
            # The stored procs always add instrument_id so skip of we just 
            # TODO: Check if this is really correct
            return None
        cols_str = ",".join(cols_to_select)

        query = f"EXEC [dbo].[GET_DATA_FOR_PERSPECTIVE_SERVICE] "
        query += f"@INSTRUMENT_IDS = '{ids_str}', "
        query += f"@TABLE_NAME = '{table_name}', "
        query += f"@REQUIRED_COLUMNS = '{cols_str}'"

        if system_version_timestamp:
            query += f", @SYSTEM_VERSION_TIMESTAMP = '{system_version_timestamp}'"
        if ed:
            query += f", @ED = '{ed}'"

        return query

    def _build_reference_query(self, table_name: str, columns: List[str],
                               instrument_ids: List[int],
                               parent_instrument_ids: List[int],
                               asset_allocation_ids: List[int],
                               system_version_timestamp: Optional[str],
                               ed: Optional[str]) -> Optional[str]:
        """Build query string for a reference table using stored procedure where applicable."""
        table_lower = table_name.lower()

        if table_lower == 'instrument':
            if not instrument_ids:
                return None
            return self._build_stored_proc_call('INSTRUMENT', instrument_ids, columns, None, None)

        elif table_lower == 'parent_instrument':
            # Use stored proc with parent_instrument_ids, columns renamed after in load_reference_data()
            valid_ids = [i for i in parent_instrument_ids if i is not None and i != -2147483648]
            if not valid_ids:
                return None
            return self._build_stored_proc_call('INSTRUMENT', valid_ids, columns, None, None)

        elif table_lower == 'instrument_categorization':
            if not instrument_ids:
                return None
            return self._build_stored_proc_call('INSTRUMENT_CATEGORIZATION', instrument_ids, columns,
                                                system_version_timestamp, ed)

        elif table_lower == 'asset_allocation_analytics_category_v':
            # Cannot use stored proc - uses analytics_category_id
            return self._asset_allocation_query(asset_allocation_ids, columns)

        else:
            # Generic tables - use stored proc
            if not instrument_ids:
                return None
            return self._build_stored_proc_call(table_name.upper(), instrument_ids, columns, None, None)

    def _asset_allocation_query(self, ids: List[int], columns: List[str]) -> Optional[str]:
        """Build query for ASSET_ALLOCATION_ANALYTICS_CATEGORY_V view."""
        valid_ids = [i for i in ids if i is not None and i != -2147483648]
        if not valid_ids:
            return None
        # Ensure analytics_category_id is included for joining
        columns_lower = [c.lower() for c in columns]
        if 'analytics_category_id' not in columns_lower:
            columns = ['analytics_category_id'] + list(columns)
        columns_str = ", ".join(columns)
        ids_str = ",".join(str(x) for x in valid_ids)
        return (
            f"SELECT {columns_str} FROM ASSET_ALLOCATION_ANALYTICS_CATEGORY_V WITH (NOLOCK) "
            f"WHERE analytics_category_id IN ({ids_str})"
        )
