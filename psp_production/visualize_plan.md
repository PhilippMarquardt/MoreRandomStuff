# PSP Production - Pipeline Visualization Tool (POC)

## Goal
Create an interactive Dash web app to visualize perspective service request processing for:
- Developer debugging (understand why positions were removed)
- Stakeholder verification (share processing flow with clients)

## Key Constraint: Decoupled POC
**This must be completely separate from the main service:**
- No modifications to existing `perspective_service/` code
- Visualizer imports and uses the service as-is
- Can be deleted without breaking anything
- Lives in its own folder: `visualizer/` (at root level, not inside perspective_service)

---

## Tech Stack

- **Dash** - Main framework (React-based, production-ready)
- **Dash AG Grid** - Interactive DataFrame tables with filtering/sorting
- **Plotly** - Charts and visualizations
- **Dash Cytoscape** - Flow diagram for pipeline visualization
- **dash-bootstrap-components** - Professional styling

---

## UI Layout (Pipeline-Focused)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔍 PSP Pipeline Visualizer                          [📁 Upload JSON]          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────── PIPELINE FLOW ──────────────────────────────┐    │
│  │                                                                         │    │
│  │  ┌─────────────┐                                                        │    │
│  │  │   INPUT     │  holding: 150 pos, 80 lt                              │    │
│  │  │   245 pos   │  reference: 95 pos, 40 lt                             │    │
│  │  └──────┬──────┘                                                        │    │
│  │         │                                                               │    │
│  │         ▼                                                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │  exclude_class_positions                              -3 pos    │   │    │
│  │  │  ───────────────────────────────────────────────────────────    │   │    │
│  │  │  Removed from holding:                                          │   │    │
│  │  │    • 12345 (weight: 0.05) - is_class_position = true           │   │    │
│  │  │    • 12346 (weight: 0.03) - is_class_position = true           │   │    │
│  │  │    • 12399 (weight: 0.02) - is_class_position = true           │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                               │    │
│  │         ▼                                                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │  exclude_other_net_assets                             -8 pos    │   │    │
│  │  │  ───────────────────────────────────────────────────────────    │   │    │
│  │  │  Removed from holding:                                          │   │    │
│  │  │    • 12400 (weight: 0.01) - asset_type_id = 99                 │   │    │
│  │  │    • 12401 (weight: 0.01) - asset_type_id = 99                 │   │    │
│  │  │    ... +6 more [expand]                                         │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                               │    │
│  │         ▼                                                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │  include_all_trade_cash                               +0 pos    │   │    │
│  │  │  ───────────────────────────────────────────────────────────    │   │    │
│  │  │  (No positions saved - none matched criteria)                   │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                               │    │
│  │         ▼                                                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Perspective 545 - Rule 1                             -5 pos    │   │    │
│  │  │  Criteria: liquidity_type_id = 5                                │   │    │
│  │  │  ───────────────────────────────────────────────────────────    │   │    │
│  │  │  Removed from holding:                                          │   │    │
│  │  │    • 12500 (weight: 0.02) - liquidity_type_id = 5              │   │    │
│  │  │    • 12501 (weight: 0.01) - liquidity_type_id = 5              │   │    │
│  │  │  Removed from reference:                                        │   │    │
│  │  │    • 79680 (weight: 1.00) - liquidity_type_id = 5              │   │    │
│  │  │    ... +2 more [expand]                                         │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                               │    │
│  │         ▼                                                               │    │
│  │  ┌─────────────┐                                                        │    │
│  │  │   OUTPUT    │  holding: 140 pos (scale: 5.2)                        │    │
│  │  │   225 pos   │  reference: 85 pos (scale: 1.0)                       │    │
│  │  └─────────────┘                                                        │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  [Collapse All]  [Expand All]  [Show only steps with changes]                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- No separate data table - removed positions shown inline per step
- Each step shows ONLY what it changed (removed IDs + reason)
- Collapsible steps - expand for full list, collapse for overview
- Per-container breakdown within each step

---

## Features

### 1. Request Upload
- Upload JSON file or paste JSON directly

### 2. Pipeline Flow (Main View)
- Vertical flow: Input → Modifiers → Rules → Output
- Each step shows ONLY what it changed
- Inline display of removed positions with:
  - Identifier
  - Weight
  - Reason (which criteria matched)
  - Container breakdown

### 3. Collapsible Steps
- Collapsed: shows step name + count of changes
- Expanded: shows all affected positions
- "Expand All" / "Collapse All" buttons
- "Show only steps with changes" toggle

### 4. Input/Output Summary
- Input: position/lookthrough counts per container
- Output: final counts + scale factors per container

---

## File Structure (Decoupled)

```
psp_production/
├── perspective_service/     # UNCHANGED - existing service
│   └── ...
├── visualizer/              # NEW - completely separate
│   ├── __init__.py
│   ├── app.py               # Main Dash app entry point
│   ├── layout.py            # UI layout components
│   ├── callbacks.py         # Interactivity logic
│   ├── analyzer.py          # Wraps engine, extracts step-by-step data
│   └── assets/
│       └── style.css        # Custom styling
└── run_visualizer.py        # Simple launcher script
```

**To remove POC:** Just delete `visualizer/` folder and `run_visualizer.py`

---

## Implementation Steps

### Step 1: Create Analyzer Wrapper (No engine changes!)
The analyzer wraps the existing engine and extracts data by analyzing the result:
```python
# visualizer/analyzer.py
class PipelineAnalyzer:
    """Analyzes perspective processing without modifying the engine."""

    def __init__(self, db_connection):
        self.engine = PerspectiveEngine(db_connection)

    def analyze_request(self, request_json: dict) -> dict:
        """Process request and extract step-by-step breakdown."""

        # 1. Capture input state
        input_summary = self._summarize_input(request_json)

        # 2. Get modifiers and rules that will be applied
        modifiers, rules = self._get_applied_rules(request_json)

        # 3. Run the actual processing
        result = self.engine.process(request_json, ...)

        # 4. Analyze the result to determine what was removed
        analysis = self._analyze_result(request_json, result, modifiers, rules)

        return {
            "input": input_summary,
            "steps": analysis["steps"],
            "output": result,
            "summary": analysis["summary"]
        }

    def _analyze_result(self, input_json, result, modifiers, rules):
        """Compare input vs output to determine what each step removed."""
        # Extract removed_positions_weight_summary from result
        # Map removals back to rules/modifiers based on criteria
        ...
```

### Step 2: Build Dash App
```python
# visualizer/app.py
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from visualizer.layout import create_layout
from visualizer.callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
```

### Step 3: Create Layout Components
- Upload area for JSON
- Pipeline flow (vertical cards for each step)
- Collapsible sections with removed positions
- Summary cards

### Step 4: Add Callbacks for Interactivity
- On JSON upload → process and display
- On step expand/collapse → show/hide details
- Expand All / Collapse All buttons

---

## Dependencies

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
```

---

## Verification

1. Run visualizer: `python run_visualizer.py`
2. Open browser at http://localhost:8050
3. Upload test request JSON
4. Verify pipeline flow shows correct steps
5. Expand steps, verify removed positions are shown
6. Check removed positions are correctly attributed to rules/modifiers
7. Verify can delete `visualizer/` folder without affecting main service

---

## Future Enhancements (Not in POC)

- Persist request history to SQLite
- Authentication for stakeholder access
- Export visualization as standalone HTML
- Compare two requests side-by-side
