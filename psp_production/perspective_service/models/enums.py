"""Enum definitions for perspective service."""

from enum import StrEnum

import polars as pl


class Container(StrEnum):
    """Container types - the key names from input JSON."""
    HOLDING = "holding"
    SELECTED_REFERENCE = "selected_reference"
    CONTRACTUAL_REFERENCE = "contractual_reference"
    REFERENCE = "reference"


class RecordType(StrEnum):
    """Record types - position vs lookthrough types."""
    POSITIONS = "positions"
    ESSENTIAL_LOOKTHROUGHS = "essential_lookthroughs"
    COMPLETE_LOOKTHROUGHS = "complete_lookthroughs"


class ApplyTo(StrEnum):
    """Apply-to scoping for rules and modifiers."""
    BOTH = "both"
    HOLDING = "holding"
    REFERENCE = "reference"


class WeightLabel(StrEnum):
    """Weight label types - the possible weight column names."""
    WEIGHT = "weight"
    INITIAL_WEIGHT = "initial_weight"
    RESULTING_WEIGHT = "resulting_weight"
    INITIAL_EXPOSURE_WEIGHT = "initial_exposure_weight"
    RESULTING_EXPOSURE_WEIGHT = "resulting_exposure_weight"


# Polars Enum dtypes derived from StrEnums
ContainerEnum = pl.Enum([str(c) for c in Container])
RecordTypeEnum = pl.Enum([str(r) for r in RecordType])
WeightLabelEnum = pl.Enum([str(w) for w in WeightLabel])
