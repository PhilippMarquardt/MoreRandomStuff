# Perspective Service

A Polars-based service for processing positions and lookthroughs through configurable perspective rules and modifiers.

## Usage

### Basic Usage (JSON Input)

```python
from perspective_service import PerspectiveService

service = PerspectiveService(connection_string="Driver={ODBC Driver 17 for SQL Server};Server=...;...")

result = service.process(request_json)
```

### DataFrame Input

```python
result = service.process_dataframes(
    positions=positions_lf,                    # pl.LazyFrame
    weight_labels_map={
        "holding": (["initial_weight", "resulting_weight"], ["weight"]),
        "selected_reference": (["initial_weight"], [])
    },
    perspective_configs={
        "my_config": {
            "1": ["exclude_class_positions", "scale_holdings_to_100_percent"],
            "2": ["exclude_other_net_assets"]
        }
    },
    lookthroughs=lookthroughs_lf,              # Optional
    effective_date="2024-01-31",               # For DB reference joins
    custom_perspective_rules=custom_rules,     # Optional, negative IDs
    system_version_timestamp="2024-01-31T10:00:00"  # Optional, temporal queries
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `return_raw_dataframes=True` | Return raw DataFrames instead of formatted output |

### DataFrame Interface with Raw Output

This example shows how to use `process_dataframes` when you already have separate DataFrames for each container and record type.

#### Input DataFrames

Assume you have the following DataFrames loaded with all positional data:

```python
# Positions per container
holding_positions: pl.DataFrame        # Holding positions
selected_reference: pl.DataFrame       # Selected reference positions
contractual_reference: pl.DataFrame    # Contractual reference positions

# Lookthroughs (Assuming no reference lts..)
essential_lookthroughs: pl.DataFrame   # Essential lookthroughs
complete_lookthroughs: pl.DataFrame    # Complete lookthroughs
```

#### Step 1: Add Required Columns

Add `container` and `record_type` columns cast to Polars Enum types:

```python
import polars as pl
from perspective_service import PerspectiveEngine
from perspective_service.models.enums import ContainerEnum, RecordTypeEnum

# Add container and record_type to positions
holding_positions = holding_positions.with_columns([
    pl.lit("holding").cast(ContainerEnum).alias("container"),
    pl.lit("positions").cast(RecordTypeEnum).alias("record_type"),
])

selected_reference = selected_reference.with_columns([
    pl.lit("selected_reference").cast(ContainerEnum).alias("container"),
    pl.lit("positions").cast(RecordTypeEnum).alias("record_type"),
])

contractual_reference = contractual_reference.with_columns([
    pl.lit("contractual_reference").cast(ContainerEnum).alias("container"),
    pl.lit("positions").cast(RecordTypeEnum).alias("record_type"),
])

# Add container and record_type to lookthroughs
essential_lookthroughs = essential_lookthroughs.with_columns([
    pl.lit("holding").cast(ContainerEnum).alias("container"),
    pl.lit("essential_lookthroughs").cast(RecordTypeEnum).alias("record_type"),
])

complete_lookthroughs = complete_lookthroughs.with_columns([
    pl.lit("holding").cast(ContainerEnum).alias("container"),
    pl.lit("complete_lookthroughs").cast(RecordTypeEnum).alias("record_type"),
])
```

#### Step 2: Concatenate into Single DataFrames

```python
# Combine all positions
positions_df = pl.concat([
    holding_positions,
    selected_reference,
    contractual_reference,
], how="diagonal")

# Combine all lookthroughs
lookthroughs_df = pl.concat([
    essential_lookthroughs,
    complete_lookthroughs,
], how="diagonal")
```

#### Step 3: Define Weight Labels Map

```python
weight_labels_map = {
    "holding": (
        ["initial_weight", "resulting_weight", "initial_exposure_weight", "resulting_exposure_weight"],  # position weights
        ["weight"]  # lookthrough weights
    ),
    "selected_reference": (
        ["weight"],  # position weights only
        []           # no lookthroughs
    ),
    "contractual_reference": (
        ["weight"],  # position weights only
        []           # no lookthroughs
    ),
}
```

#### Step 4: Process with Raw DataFrames Output

```python
engine = PerspectiveEngine(connection_string=None)

perspective_configs = {
    "config": {
        "-1": ["scale_holdings_to_100_percent"]
    }
}

custom_rules = {
    "-1": {
        "rules": [{
            "apply_to": "both",
            "criteria": {"column": "initial_weight", "operator_type": ">", "value": 0.01}
        }]
    }
}

result = engine.process_dataframes(
    positions_lf=positions_df.lazy(),
    lookthroughs_lf=lookthroughs_df.lazy(),
    weight_labels_map=weight_labels_map,
    perspective_configs=perspective_configs,
    custom_perspective_rules=custom_rules,
    return_raw_dataframes=True,
)

# Result contains LazyFrames
positions_lf = result["positions"]
lookthroughs_lf = result["lookthroughs"]
scale_factors_lf = result["scale_factors"]
```

#### Step 5: Use the Output

The output LazyFrames contain pre-computed weight columns that are **already scaled**:

| Column | Description |
|--------|-------------|
| `f_config_-1` | Factor column (NULL = filtered out, value = kept) |
| `initial_weight_config_-1` | Already scaled: `initial_weight × factor` |
| `resulting_weight_config_-1` | Already scaled: `resulting_weight × factor` |
| `initial_exposure_weight_config_-1` | Already scaled: `initial_exposure_weight × factor` |
| `resulting_exposure_weight_config_-1` | Already scaled: `resulting_exposure_weight × factor` |
| `weight_config_-1` | Already scaled: `weight × factor` (for lookthroughs) |

```python
# Collect results
positions_df = positions_lf.collect()
lookthroughs_df = lookthroughs_lf.collect()
scale_factors_df = scale_factors_lf.collect()

# Filter to kept positions (factor is not null)
kept = positions_df.filter(pl.col("f_config_-1").is_not_null())

# Select the scaled weight columns directly (already multiplied)
holding_weights = kept.filter(
    pl.col("container") == "holding"
).select([
    "instrument_id",
    "initial_weight_config_-1",
    "resulting_weight_config_-1",
    "initial_exposure_weight_config_-1",
    "resulting_exposure_weight_config_-1",
])

# Scale factors show percentage of original weight kept
print(scale_factors_df)
# ┌─────────────┬────────────────┬───────────┬──────────────────────────┬──────────────┐
# │ config_name │ perspective_id │ container │ weight_label             │ scale_factor │
# │ str         │ i64            │ str       │ str                      │ f64          │
# ├─────────────┼────────────────┼───────────┼──────────────────────────┼──────────────┤
# │ config      │ -1             │ holding   │ initial_weight           │ 0.85         │
# │ config      │ -1             │ holding   │ resulting_weight         │ 0.82         │
# │ config      │ -1             │ holding   │ initial_exposure_weight  │ 0.84         │
# │ config      │ -1             │ holding   │ resulting_exposure_weight│ 0.81         │
# └─────────────┴────────────────┴───────────┴──────────────────────────┴──────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT                                │
│  JSON Request  OR  DataFrames + weight_labels_map          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                          │
│     • Extract positions & lookthroughs per container        │
│     • Normalize column names (lowercase)                    │
│     • Cast to enum types (container, record_type)           │
│     • Fill nulls with sentinel values                       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. LOAD REFERENCE DATA (if connection_string provided)     │
│     • INSTRUMENT (join on instrument_id)                    │
│     • PARENT_INSTRUMENT (join on parent_instrument_id)      │
│     • INSTRUMENT_CATEGORIZATION                             │
│     • ASSET_ALLOCATION_ANALYTICS_CATEGORY_V                 │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. PRECOMPUTE NESTED CRITERIA                              │
│     • Cache results of nested In/NotIn evaluations          │
│     • Enables rule_result references in PostProcessing      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. BUILD PERSPECTIVE PLAN                                  │
│     For each (config, perspective_id):                      │
│     • Build keep expression (PreProc → Rules → PostProc)    │
│     • Build scale expression (scaling rules)                │
│     • Synchronize lookthroughs with parent positions        │
│     • Apply rescaling if enabled                            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  5. COLLECT                                                 │
│     • Materialize LazyFrames: positions, lookthroughs,      │
│       scale_factors                                         │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  6. FORMAT OUTPUT                                           │
│     • Structure into perspective_configurations hierarchy   │
│     • Apply weight columns × factors                        │
│     • Include scale_factors per container                   │
└─────────────────────────┴───────────────────────────────────┘
```

## Calculation Logic

### Rules

Rules define filtering criteria for each perspective. They are chained with AND/OR logic.

```python
rule = {
    "criteria": {"column": "initial_weight", "operator_type": ">", "value": 0.05},
    "apply_to": "both",           # "both", "holding", or "reference"
    "condition_for_next_rule": "and"  # "and" or "or"
}
```

**Supported operators:** `==`, `!=`, `<`, `>`, `<=`, `>=`, `In`, `NotIn`, `IsNull`, `IsNotNull`

### Modifiers

Modifiers are reusable filtering/scaling components applied to perspectives.

#### PreProcessing Modifiers

Exclude rows **before** rule evaluation. Matching rows are removed and cannot be seen by subsequent logic.

```
exclude_other_net_assets      → liquidity_type_id = 2
exclude_class_positions       → is_class_position = True
exclude_future_trades         → trade_status_id = 1
exclude_pending_trades        → trade_status_id = 2
exclude_blocked_positions     → is_blocked = True
exclude_simulated_trades      → is_simulated_trade = True
exclude_simulated_cash        → position_source_type_id = 10
```

#### PostProcessing Modifiers

Include "savior" rows **after** rules have filtered. Can reference `rule_result` to check if other rows passed.

```
include_all_trade_cash              → Keep position_source_type_id=10, liquidity_type_id=6
include_trade_cash_within_perspective → Keep if simulated_trade_id in rule_result
exclude_trade_cash                  → Inverse logic
```

**rule_result_operator:**
- `"or"`: Keep row if savior criteria match (even if rule failed)
- `"and"`: Keep row only if rule passed AND savior criteria match

#### Scaling Modifiers

Enable weight rescaling:

```
scale_holdings_to_100_percent      → Rescale position weights per container
scale_lookthroughs_to_100_percent  → Rescale lookthrough weights per parent group
```

### Rescaling

When `scale_holdings_to_100_percent` is enabled, weights are normalized to 100% within the kept set.

**Formula:**
```
factor_column = factor / denominator
where denominator = sum(weight × factor) for all kept rows in container
```

**Example:**
```
Position A: weight=0.40, factor=1.0  → kept
Position B: weight=0.30, factor=0.5  → kept (partial)
Position C: weight=0.30, factor=NULL → removed

denominator = (0.40 × 1.0) + (0.30 × 0.5) = 0.55

Final weights:
  A: 0.40 × (1.0 / 0.55) = 0.727
  B: 0.30 × (0.5 / 0.55) = 0.273
  Total: 1.0 (100%)
```

### Scale Factors

Scale factors measure what percentage of original weights remain after filtering:

```
scale_factor = sum(weight × factor for kept rows) / sum(weight for all rows)
```

Returned per (config, perspective_id, container, weight_label).

## Input Format

```json
{
  "position_weight_labels": ["initial_weight", "resulting_weight"],
  "lookthrough_weight_labels": ["weight"],
  "ed": "2024-01-31",
  "system_version_timestamp": "2024-01-31T10:00:00",

  "perspective_configurations": {
    "config_name": {
      "1": ["exclude_class_positions", "scale_holdings_to_100_percent"],
      "2": ["exclude_other_net_assets"]
    }
  },

  "holding": {
    "positions": {
      "pos_001": {
        "instrument_id": 12345,
        "sub_portfolio_id": 100,
        "initial_weight": 0.50,
        "resulting_weight": 0.45,
        "liquidity_type_id": 1,
        "is_class_position": false
      }
    },
    "essential_lookthroughs": {
      "lt_001": {
        "parent_instrument_id": 12345,
        "instrument_id": 67890,
        "sub_portfolio_id": 100,
        "weight": 0.25
      }
    }
  },

  "selected_reference": {
    "position_weight_labels": ["initial_weight"],
    "positions": { ... }
  },

  "custom_perspective_rules": {
    "-1": {
      "rules": [
        {
          "criteria": {"column": "initial_weight", "operator_type": ">", "value": 0.10},
          "apply_to": "both",
          "condition_for_next_rule": "and"
        }
      ]
    }
  }
}
```

### Containers

| Container | Description |
|-----------|-------------|
| `holding` | Primary holdings |
| `selected_reference` | Selected reference positions |
| `contractual_reference` | Contractual reference positions |
| `reference` | General reference positions |

### Record Types

| Record Type | Description |
|-------------|-------------|
| `positions` | Direct positions |
| `essential_lookthroughs` | Essential lookthrough positions (included in rescaling) |
| `complete_lookthroughs` | Complete lookthrough positions |

## Output Format

```json
{
  "perspective_configurations": {
    "config_name": {
      "1": {
        "holding": {
          "positions": {
            "pos_001": {
              "initial_weight": 0.727,
              "resulting_weight": 0.654
            }
          },
          "essential_lookthroughs": {
            "lt_001": {
              "weight": 0.25
            }
          },
          "scale_factors": {
            "initial_weight": 0.55,
            "resulting_weight": 0.58
          }
        }
      }
    }
  }
}
```

### Raw DataFrames (`return_raw_dataframes=True`)

```python
{
    "positions": pl.DataFrame,      # All position data with factor columns
    "lookthroughs": pl.DataFrame,   # All lookthrough data with factor columns
    "scale_factors": pl.DataFrame,  # Scale factors per perspective/container/weight
    "metadata_map": Dict            # {config: {pid: factor_column_name}}
}
```

## Apply-To Scoping

Rules and modifiers specify which containers they apply to:

| apply_to | Applies to |
|----------|------------|
| `"both"` | All containers |
| `"holding"` | Only holding container |
| `"reference"` | All non-holding containers |

## Lookthrough Synchronization

When a parent position is filtered out (factor=NULL), all its child lookthroughs are also removed. This maintains data consistency between positions and their lookthroughs.

## Custom Perspectives

Custom perspective rules can be defined in the request JSON. Requirements:
- Must use **negative IDs** (e.g., `-1`, `-2`)
- Must include `rules` array with valid criteria

```json
{
  "custom_perspective_rules": {
    "-1": {
      "rules": [
        {
          "criteria": {"column": "weight", "operator_type": ">", "value": 0.01},
          "apply_to": "both"
        }
      ]
    }
  },
  "perspective_configurations": {
    "my_config": {
      "-1": ["scale_holdings_to_100_percent"]
    }
  }
}
```
