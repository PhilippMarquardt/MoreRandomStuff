"""
PSP outputs (when return_raw_dataframes=True):
- positions_lf: positions with factor/weight columns
- lookthroughs_lf: lookthrough underlyings with record_type, parent_instrument_id
- scale_factors_lf: scale factors per (config, pid, container)

Weight logic by granularity:
- "no": PI = initial_weight, PR = resulting_weight
- lookthrough (has underlying): PI = initial_exposure_weight × weight
- lookthrough (leaf, no underlying): PI = initial_exposure_weight alone
"""

from __future__ import annotations

from typing import Dict, List, Optional

import polars as pl
import polars.selectors as cs
import pytest

from perspective_service.core.engine import PerspectiveEngine


# =============================================================================
# Core Function: calculate_single_lookthrough
# Matches old PositionEngine.calculate_single_lookthrough() exactly
# =============================================================================


def calculate_single_lookthrough(
    positions_lf: pl.LazyFrame,
    lookthroughs_lf: Optional[pl.LazyFrame],
    scale_factors_lf: Optional[pl.LazyFrame],
    config: str,
    pid: int,
    granularity: str,  # "no", "essential_lookthroughs", "complete_lookthroughs"
    value_metadata: Optional[pl.LazyFrame] = None,
    scale_holdings: bool = False,
    additional_attributes: List[str] = None,
) -> pl.LazyFrame:
    """
    Fully replicate old PositionEngine.calculate_single_lookthrough().

    Key differences by granularity:
    - "no": PI = initial_weight, PR = resulting_weight
    - lookthrough (has underlying): PI = initial_exposure_weight × weight
    - lookthrough (leaf, no underlying): PI = initial_exposure_weight alone
    """
    suffix = f"{config}_{pid}"
    fcol = f"f_{suffix}"
    is_lookthrough_level_no = (granularity == "no")

    if additional_attributes is None:
        additional_attributes = []

    # === Step 1: Filter positions by perspective ===
    position_df = positions_lf.filter(pl.col(fcol).is_not_null())

    # === Step 2: Get composites (lookthroughs) for this granularity ===
    if granularity == "no" or lookthroughs_lf is None:
        # No composites for "no" granularity - use empty schema match
        composite_df = None
    else:
        composite_df = (
            lookthroughs_lf
            .filter(pl.col(fcol).is_not_null())
            .filter(pl.col("record_type").cast(pl.Utf8) == granularity)
        )

    # === Step 3: Join positions with composites ===
    if composite_df is not None:
        # Left join: positions keep their rows, gain composite underlyings
        # Match on: instrument_id (position) = parent_instrument_id (lookthrough)
        joined_df = position_df.join(
            composite_df.select([
                "instrument_id",
                "parent_instrument_id",
                "sub_portfolio_id",
                "container",
                "weight",  # lookthrough weight is raw "weight" not renamed
            ]),
            left_on=["instrument_id", "container", "sub_portfolio_id"],
            right_on=["parent_instrument_id", "container", "sub_portfolio_id"],
            how="left",
            suffix="_right",
        )
    else:
        # For "no" granularity, just use positions with null right columns
        joined_df = position_df.with_columns([
            pl.lit(None, dtype=pl.Int64).alias("instrument_id_right"),
            pl.lit(None, dtype=pl.Float64).alias("weight_right"),
        ])

    # === Step 4: Calculate PI, PR, BS, BC ===
    joined_df = joined_df.with_columns([
        # PI calculation (matches old code exactly)
        pl.when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_not_null())  # has underlying
        ).then(pl.col(f"initial_exposure_weight_{suffix}") * pl.col("weight_right"))
        .when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_null())
            & (not is_lookthrough_level_no)  # lookthrough level, leaf position
        ).then(pl.col(f"initial_exposure_weight_{suffix}"))
        .when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_null())
            & is_lookthrough_level_no  # "no" level
        ).then(pl.col(f"initial_weight_{suffix}"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("PI"),

        # PR calculation (same logic with resulting_weight)
        pl.when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_not_null())
        ).then(pl.col(f"resulting_exposure_weight_{suffix}") * pl.col("weight_right"))
        .when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_null())
            & (not is_lookthrough_level_no)
        ).then(pl.col(f"resulting_exposure_weight_{suffix}"))
        .when(
            (pl.col("container").cast(pl.Utf8) == "holding")
            & (pl.col("instrument_id_right").is_null())
            & is_lookthrough_level_no
        ).then(pl.col(f"resulting_weight_{suffix}"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("PR"),

        # BC (contractual benchmark)
        pl.when(pl.col("container").cast(pl.Utf8) == "contractual_reference")
        .then(pl.col(f"weight_{suffix}"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("BC"),

        # BS (selected benchmark)
        pl.when(pl.col("container").cast(pl.Utf8) == "selected_reference")
        .then(pl.col(f"weight_{suffix}"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("BS"),
    ])

    # === Step 5: Set parent_instrument_id ===
    joined_df = joined_df.with_columns([
        pl.when(pl.col("instrument_id_right").is_not_null())
        .then(pl.col("instrument_id"))  # parent is original instrument
        .otherwise(pl.col("instrument_id"))  # leaf: parent = self
        .alias("parent_instrument_id_new"),
    ])

    # === Step 6: Update instrument_id (use underlying if exists) ===
    joined_df = joined_df.with_columns([
        pl.when(pl.col("instrument_id_right").is_not_null())
        .then(pl.col("instrument_id_right"))
        .otherwise(pl.col("instrument_id"))
        .alias("instrument_id"),
    ])

    # Rename parent_instrument_id
    joined_df = joined_df.with_columns([
        pl.col("parent_instrument_id_new").alias("parent_instrument_id"),
    ])

    # === Step 7: Group by columns ===
    group_by_cols = additional_attributes + [
        "instrument_id",
        "parent_instrument_id",
        "sub_portfolio_id",
    ]

    # Build aggregation list
    agg_exprs = [
        pl.col("PI").sum(),
        pl.col("PR").sum(),
        pl.col("BS").sum(),
        pl.col("BC").sum(),
    ]

    # Add shares if present
    schema_names = joined_df.collect_schema().names()
    if "initial_shares" in schema_names:
        agg_exprs.append(pl.col("initial_shares").sum())
    if "resulting_shares" in schema_names:
        agg_exprs.append(pl.col("resulting_shares").sum())
    if "blocked_shares" in schema_names:
        agg_exprs.append(pl.col("blocked_shares").sum())

    joined_df = joined_df.group_by(group_by_cols).agg(agg_exprs)

    # === Step 8: Join with value_metadata (TNAs) if provided ===
    if value_metadata is not None:
        joined_df = joined_df.join(value_metadata, how="cross")
    else:
        # Use constant TNAs as fallback
        joined_df = joined_df.with_columns([
            pl.lit(1_000_000.0).alias("total_net_assets_initial_portfolio"),
            pl.lit(1_000_000.0).alias("total_net_assets_resulting_portfolio"),
            pl.lit(1_000_000.0).alias("total_net_assets_selected_benchmark"),
            pl.lit(1_000_000.0).alias("total_net_assets_contractual_benchmark"),
        ])

    # === Step 9: Apply scale factors and calculate value columns ===
    # Scale factors are per (config, pid, container) from scale_factors_lf
    # When scale_holdings=True, multiply values by scale factor
    if scale_holdings and scale_factors_lf is not None:
        # Join with scale_factors to get scale_factors_holding_initial_weight, etc.
        sf = scale_factors_lf.filter(
            (pl.col("config") == config) & (pl.col("perspective_id") == pid)
        )
        joined_df = joined_df.join(sf, how="cross")

        # PIV = PI × TNA × scale_factor (when scale_holdings)
        joined_df = joined_df.with_columns([
            (pl.col("PI") * pl.col("total_net_assets_initial_portfolio") * pl.col("scale_factors_holding_initial_weight")).alias("PIV"),
            (pl.col("PR") * pl.col("total_net_assets_resulting_portfolio") * pl.col("scale_factors_holding_resulting_weight")).alias("PRV"),
            (pl.col("BS") * pl.col("total_net_assets_selected_benchmark") * pl.col("scale_factors_selected_reference_weight")).alias("BSV"),
            (pl.col("BC") * pl.col("total_net_assets_contractual_benchmark") * pl.col("scale_factors_contractual_reference_weight")).alias("BCV"),
        ])
    else:
        # No scale factors - just multiply weight × TNA
        joined_df = joined_df.with_columns([
            (pl.col("PI") * pl.col("total_net_assets_initial_portfolio")).alias("PIV"),
            (pl.col("PR") * pl.col("total_net_assets_resulting_portfolio")).alias("PRV"),
            (pl.col("BS") * pl.col("total_net_assets_selected_benchmark")).alias("BSV"),
            (pl.col("BC") * pl.col("total_net_assets_contractual_benchmark")).alias("BCV"),
        ])

    # === Step 11: Rescale weights (if scale_holdings) ===
    if scale_holdings:
        rescale_cols = ["PI", "PR", "BS", "BC"]
        for m in rescale_cols:
            total_m = pl.col(m).sum()
            factor = pl.when(total_m.ne(0)).then(1.0 / total_m).otherwise(1.0)
            joined_df = joined_df.with_columns([
                (pl.col(m) * factor).alias(m),
                (pl.col(f"{m}V") * factor).alias(f"{m}V"),
            ])

    # === Step 12: Filter small positions ===
    joined_df = joined_df.with_columns([
        pl.when(pl.col(c).abs().le(1e-15)).then(0.0).otherwise(pl.col(c)).alias(c)
        for c in ["PI", "PR", "BS", "BC"]
    ])

    # === Step 13: Fill nulls in value columns ===
    value_cols = ["PI", "PR", "BS", "BC", "PIV", "PRV", "BSV", "BCV"]
    joined_df = joined_df.with_columns([
        pl.col(c).fill_null(0.0).fill_nan(0.0).alias(c)
        for c in value_cols
    ])

    # === Step 14: Calculate active weights ===
    joined_df = joined_df.with_columns([
        (pl.col("PI") - pl.col("BC")).alias("AIC"),
        (pl.col("PR") - pl.col("BC")).alias("ARC"),
        (pl.col("PI") - pl.col("BS")).alias("AIS"),
        (pl.col("PR") - pl.col("BS")).alias("ARS"),
    ])

    # === Step 15: Calculate active values ===
    joined_df = joined_df.with_columns([
        (pl.col("total_net_assets_initial_portfolio") * pl.col("AIC")).alias("AICV"),
        (pl.col("total_net_assets_resulting_portfolio") * pl.col("ARC")).alias("ARCV"),
        (pl.col("total_net_assets_initial_portfolio") * pl.col("AIS")).alias("AISV"),
        (pl.col("total_net_assets_resulting_portfolio") * pl.col("ARS")).alias("ARSV"),
    ])

    return joined_df


def build_perspective_dict(
    positions_lf: pl.LazyFrame,
    lookthroughs_lf: Optional[pl.LazyFrame],
    scale_factors_lf: Optional[pl.LazyFrame],
    perspective_configs: Dict[str, Dict[str, List[str]]],
    value_metadata: Optional[pl.LazyFrame] = None,
    scale_holdings: bool = False,
    record_types: List[str] = None,
) -> Dict[str, Dict[int, Dict[str, pl.LazyFrame]]]:
    """
    Build dict structure: {config: {pid: {granularity: LazyFrame}}}

    All returned LazyFrames are lazy - call .collect() when you need results.
    """
    if record_types is None:
        record_types = ["essential_lookthroughs", "complete_lookthroughs"]

    result: Dict[str, Dict[int, Dict[str, pl.LazyFrame]]] = {}

    for config, pmap in perspective_configs.items():
        result[config] = {}
        for pid_str in pmap:
            pid = int(pid_str)
            result[config][pid] = {}

            # "no" granularity
            result[config][pid]["no"] = calculate_single_lookthrough(
                positions_lf=positions_lf,
                lookthroughs_lf=lookthroughs_lf,
                scale_factors_lf=scale_factors_lf,
                config=config,
                pid=pid,
                granularity="no",
                value_metadata=value_metadata,
                scale_holdings=scale_holdings,
            )

            # Lookthrough granularities
            for rt in record_types:
                result[config][pid][rt] = calculate_single_lookthrough(
                    positions_lf=positions_lf,
                    lookthroughs_lf=lookthroughs_lf,
                    scale_factors_lf=scale_factors_lf,
                    config=config,
                    pid=pid,
                    granularity=rt,
                    value_metadata=value_metadata,
                    scale_holdings=scale_holdings,
                )

    return result


# =============================================================================
# Tests
# =============================================================================


def test_no_granularity_uses_raw_weights():
    """
    "no" granularity uses initial_weight/resulting_weight directly.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.6,
            "resulting_weight": 0.5,
            "initial_exposure_weight": 0.8,  # NOT used for "no"
            "resulting_exposure_weight": 0.7,
            "weight": None,
        },
        {
            "identifier": "s1",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "selected_reference",
            "record_type": "position",
            "initial_weight": None,
            "resulting_weight": None,
            "initial_exposure_weight": None,
            "resulting_exposure_weight": None,
            "weight": 0.3,
        },
        {
            "identifier": "c1",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "contractual_reference",
            "record_type": "position",
            "initial_weight": None,
            "resulting_weight": None,
            "initial_exposure_weight": None,
            "resulting_exposure_weight": None,
            "weight": 0.25,
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight"], []),
        "selected_reference": (["weight"], []),
        "contractual_reference": (["weight"], []),
    }

    perspective_configs = {"test_cfg": {"-1": []}}
    custom_perspective_rules = {"-1": {"rules": [{"criteria": {}, "apply_to": "both"}]}}

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=None,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    result = calculate_single_lookthrough(
        positions_lf=raw["positions"],
        lookthroughs_lf=None,
        scale_factors_lf=None,
        config="test_cfg",
        pid=-1,
        granularity="no",
    )

    df = result.collect()

    # Should use RAW weights (0.6), NOT exposure (0.8)
    assert df.filter(pl.col("PI").is_not_null())["PI"].to_list()[0] == pytest.approx(0.6)
    assert df.filter(pl.col("PR").is_not_null())["PR"].to_list()[0] == pytest.approx(0.5)

    # Verify active weights
    pi_row = df.filter(pl.col("PI") > 0)
    assert pi_row["AIC"].to_list()[0] == pytest.approx(0.6 - 0.25)
    assert pi_row["AIS"].to_list()[0] == pytest.approx(0.6 - 0.3)


def test_lookthrough_with_underlying_uses_exposure_times_weight():
    """
    Lookthrough with underlying uses initial_exposure_weight × weight.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1",
            "instrument_id": 100,  # Parent (fund)
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.5,  # NOT used
            "resulting_weight": 0.4,
            "initial_exposure_weight": 0.6,  # USED
            "resulting_exposure_weight": 0.5,
            "weight": None,
        },
    ])

    lookthroughs_lf = pl.LazyFrame([
        {
            "identifier": "lt1",
            "instrument_id": 1,  # Underlying
            "parent_instrument_id": 100,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "essential_lookthroughs",
            "initial_weight": None,
            "resulting_weight": None,
            "initial_exposure_weight": None,
            "resulting_exposure_weight": None,
            "weight": 0.3,  # Child weight
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight", "weight"], []),
    }

    perspective_configs = {"test_cfg": {"-1": []}}
    custom_perspective_rules = {"-1": {"rules": [{"criteria": {}, "apply_to": "both"}]}}

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=lookthroughs_lf,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    result = calculate_single_lookthrough(
        positions_lf=raw["positions"],
        lookthroughs_lf=raw["lookthroughs"],
        scale_factors_lf=None,
        config="test_cfg",
        pid=-1,
        granularity="essential_lookthroughs",
    )

    df = result.collect()

    # PI = initial_exposure_weight (0.6) × weight (0.3) = 0.18
    pi_row = df.filter(pl.col("PI") > 0)
    assert len(pi_row) == 1
    assert pi_row["PI"].to_list()[0] == pytest.approx(0.18)
    assert pi_row["PR"].to_list()[0] == pytest.approx(0.5 * 0.3)  # 0.15

    # instrument_id should be the underlying (1), not parent (100)
    assert pi_row["instrument_id"].to_list()[0] == 1
    # parent_instrument_id should be the parent (100)
    assert pi_row["parent_instrument_id"].to_list()[0] == 100


def test_lookthrough_leaf_uses_exposure_alone():
    """
    Lookthrough level but no underlying (leaf) uses initial_exposure_weight alone.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1",
            "instrument_id": 1,  # This position has no lookthrough children
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.5,  # NOT used for lookthrough
            "resulting_weight": 0.4,
            "initial_exposure_weight": 0.6,  # USED (alone, no multiplication)
            "resulting_exposure_weight": 0.5,
            "weight": None,
        },
    ])

    # Empty lookthroughs - no underlyings for this position
    lookthroughs_lf = pl.LazyFrame([
        {
            "identifier": "lt_other",
            "instrument_id": 999,
            "parent_instrument_id": 999,  # Different parent
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "essential_lookthroughs",
            "initial_weight": None,
            "resulting_weight": None,
            "initial_exposure_weight": None,
            "resulting_exposure_weight": None,
            "weight": 0.3,
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight", "weight"], []),
    }

    perspective_configs = {"test_cfg": {"-1": []}}
    custom_perspective_rules = {"-1": {"rules": [{"criteria": {}, "apply_to": "both"}]}}

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=lookthroughs_lf,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    result = calculate_single_lookthrough(
        positions_lf=raw["positions"],
        lookthroughs_lf=raw["lookthroughs"],
        scale_factors_lf=None,
        config="test_cfg",
        pid=-1,
        granularity="essential_lookthroughs",  # Lookthrough level
    )

    df = result.collect()

    # Leaf position at lookthrough level: PI = initial_exposure_weight alone (0.6)
    pi_row = df.filter(pl.col("PI") > 0)
    assert len(pi_row) == 1
    assert pi_row["PI"].to_list()[0] == pytest.approx(0.6)
    assert pi_row["PR"].to_list()[0] == pytest.approx(0.5)

    # instrument_id and parent_instrument_id should be same for leaf
    assert pi_row["instrument_id"].to_list()[0] == 1
    assert pi_row["parent_instrument_id"].to_list()[0] == 1


def test_grouping_sums_duplicate_instruments():
    """
    When same instrument appears multiple times, weights are summed.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1a",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.3,
            "resulting_weight": 0.25,
            "initial_exposure_weight": 0.3,
            "resulting_exposure_weight": 0.25,
            "weight": None,
        },
        {
            "identifier": "h1b",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.2,
            "resulting_weight": 0.15,
            "initial_exposure_weight": 0.2,
            "resulting_exposure_weight": 0.15,
            "weight": None,
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight", "weight"], []),
    }

    perspective_configs = {"test_cfg": {"-1": []}}
    custom_perspective_rules = {"-1": {"rules": [{"criteria": {}, "apply_to": "both"}]}}

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=None,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    result = calculate_single_lookthrough(
        positions_lf=raw["positions"],
        lookthroughs_lf=None,
        scale_factors_lf=None,
        config="test_cfg",
        pid=-1,
        granularity="no",
    )

    df = result.collect()

    # Should be grouped: PI = 0.3 + 0.2 = 0.5
    pi_row = df.filter(pl.col("PI") > 0)
    assert len(pi_row) == 1
    assert pi_row["PI"].to_list()[0] == pytest.approx(0.5)
    assert pi_row["PR"].to_list()[0] == pytest.approx(0.4)


def test_small_positions_filtered():
    """
    Weights <= 1e-15 are set to 0.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1",
            "instrument_id": 1,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 1e-16,  # Too small
            "resulting_weight": 0.5,
            "initial_exposure_weight": 1e-16,
            "resulting_exposure_weight": 0.5,
            "weight": None,
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight", "weight"], []),
    }

    perspective_configs = {"test_cfg": {"-1": []}}
    custom_perspective_rules = {"-1": {"rules": [{"criteria": {}, "apply_to": "both"}]}}

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=None,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    result = calculate_single_lookthrough(
        positions_lf=raw["positions"],
        lookthroughs_lf=None,
        scale_factors_lf=None,
        config="test_cfg",
        pid=-1,
        granularity="no",
    )

    df = result.collect()

    # Small weight should be filtered to 0
    pi_row = df.filter(pl.col("instrument_id") == 1)
    assert pi_row["PI"].to_list()[0] == pytest.approx(0.0)
    assert pi_row["PR"].to_list()[0] == pytest.approx(0.5)


def test_build_perspective_dict_structure():
    """
    Build complete dict structure for all perspectives and granularities.
    """
    positions_lf = pl.LazyFrame([
        {
            "identifier": "h1",
            "instrument_id": 100,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "position",
            "initial_weight": 0.6,
            "resulting_weight": 0.5,
            "initial_exposure_weight": 0.7,
            "resulting_exposure_weight": 0.6,
            "weight": None,
        },
    ])

    lookthroughs_lf = pl.LazyFrame([
        {
            "identifier": "lt1",
            "instrument_id": 1,
            "parent_instrument_id": 100,
            "sub_portfolio_id": 10,
            "container": "holding",
            "record_type": "essential_lookthroughs",
            "initial_weight": None,
            "resulting_weight": None,
            "initial_exposure_weight": None,
            "resulting_exposure_weight": None,
            "weight": 0.5,
        },
    ])

    weight_labels_map = {
        "holding": (["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight", "weight"], []),
    }

    perspective_configs = {"cfg_a": {"-1": [], "-2": []}}
    custom_perspective_rules = {
        "-1": {"rules": [{"criteria": {}, "apply_to": "both"}]},
        "-2": {"rules": [{"criteria": {}, "apply_to": "both"}]},
    }

    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []
    raw = engine.process_dataframes(
        positions_lf=positions_lf,
        lookthroughs_lf=lookthroughs_lf,
        weight_labels_map=weight_labels_map,
        perspective_configs=perspective_configs,
        custom_perspective_rules=custom_perspective_rules,
        return_raw_dataframes=True,
    )

    frames = build_perspective_dict(
        positions_lf=raw["positions"],
        lookthroughs_lf=raw["lookthroughs"],
        scale_factors_lf=raw.get("scale_factors"),
        perspective_configs=perspective_configs,
        record_types=["essential_lookthroughs"],
    )

    # Verify structure
    assert "cfg_a" in frames
    assert -1 in frames["cfg_a"]
    assert -2 in frames["cfg_a"]
    assert "no" in frames["cfg_a"][-1]
    assert "essential_lookthroughs" in frames["cfg_a"][-1]

    # "no" uses raw weight (0.6)
    df_no = frames["cfg_a"][-1]["no"].collect()
    pi_no = df_no.filter(pl.col("PI") > 0)["PI"].to_list()[0]
    assert pi_no == pytest.approx(0.6)

    # lookthrough uses exposure × weight (0.7 × 0.5 = 0.35)
    df_lt = frames["cfg_a"][-1]["essential_lookthroughs"].collect()
    pi_lt = df_lt.filter(pl.col("PI") > 0)["PI"].to_list()[0]
    assert pi_lt == pytest.approx(0.35)
