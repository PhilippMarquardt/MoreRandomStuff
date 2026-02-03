"""
Comprehensive Rescaling Combinations Test (World A semantics)

World A semantics implemented (matches your CURRENT PSP):
- Scaling rules affect economic mass (effective weight = w * fcol).
- Essential lookthroughs (ELT) do NOT inherit parent's scaling factor.
  They only inherit REMOVAL via synchronization if parent is removed.
- Holdings rescaling (scale_holdings_to_100_percent) rescales ONLY positions'
  position_weight_labels, but the DENOMINATOR includes:
    kept_positions_mass + kept_essential_lt_mass
  where mass is computed as sum(w * fcol) for that weight label.
- Lookthrough rescaling (scale_lookthroughs_to_100_percent) rescales ONLY
  lookthrough_weight_label(s) (e.g. "weight") within its own universe:
    per (container, parent, record_type, sub_portfolio_id if present)
  again using sum(weight * fcol) for the denominator.

Downstream checks are done per "view"/mode:
A) Positions view:
   PIV_pos = pos_output_weight * TNA * scale_factor
   Sum over positions must equal:
     TNA * kept_positions_effective_mass / total_universe_raw_mass

B) Lookthrough view (per record_type separately):
   PIV_lt_row = (parent_output_pos_weight * child_output_lt_weight) * TNA * scale_factor
   For each parent (within that record_type):
     sum_child(PIV) == parent_PIV * sum_child(child_output_lt_weight)
   and if scale_lookthroughs=True then sum_child(child_output_lt_weight) == 1.0.
"""

import sys
import os
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import pytest

# Note: sys.stdout encoding fix removed as it causes pytest issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspective_service.core.engine import PerspectiveEngine
from perspective_service.models.rule import Rule
from perspective_service.models.modifier import Modifier
from perspective_service.models.enums import ApplyTo, ModifierType


# =============================================================================
# Scenario
# =============================================================================

@dataclass
class TestScenario:
    name: str
    has_filtering: bool
    has_scaling: bool
    scale_holdings: bool
    scale_lookthroughs: bool

    def __str__(self) -> str:
        parts = []
        if self.has_filtering:
            parts.append("filter")
        if self.has_scaling:
            parts.append("scale_rule")
        if self.scale_holdings:
            parts.append("rescale_hold")
        if self.scale_lookthroughs:
            parts.append("rescale_lt")
        return "_".join(parts) if parts else "no_changes"


# =============================================================================
# Test Data (single container, single weight label)
# =============================================================================

CONTAINER = "holding"
PERSPECTIVE_ID = 100
TNA = 1_000_000.0

# Position raw weights (position_weight_label: initial_exposure_weight)
POS_W = {"pos_1": 0.40, "pos_2": 0.20, "pos_3": 0.20}

# Essential lookthrough raw exposure weights (same label) and child "weight" for downstream
ELT_EXPO_W = {"elt_1": 0.10, "elt_2": 0.05, "elt_3": 0.05}
ELT_CHILD_W = {"elt_1": 0.50, "elt_2": 0.25, "elt_3": 0.25}
ELT_PARENT = {"elt_1": "pos_2", "elt_2": "pos_2", "elt_3": "pos_3"}

# Complete lookthrough (not part of holdings universe mass in your semantics, but should behave in LT rescaling)
CLT_EXPO_W = {"clt_1": 0.02}
CLT_CHILD_W = {"clt_1": 0.10}
CLT_PARENT = {"clt_1": "pos_2"}

# Holdings universe for SF + holdings-rescale denom (World A): positions + essential LT only
TOTAL_UNIVERSE_RAW = sum(POS_W.values()) + sum(ELT_EXPO_W.values())  # 1.0


def create_test_input_json(container_name: str = CONTAINER) -> Dict:
    return {
        "position_weight_labels": ["initial_exposure_weight"],
        "lookthrough_weight_labels": ["weight"],
        container_name: {
            "positions": {
                "pos_1": {
                    "instrument_id": 1,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": POS_W["pos_1"],
                    "filter_out": True,      # filtered when has_filtering=True
                    "apply_scale": False,
                },
                "pos_2": {
                    "instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": POS_W["pos_2"],
                    "filter_out": False,
                    "apply_scale": True,     # scaled when has_scaling=True
                },
                "pos_3": {
                    "instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": POS_W["pos_3"],
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
            "essential_lookthroughs": {
                "elt_1": {
                    "instrument_id": 101,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": ELT_EXPO_W["elt_1"],
                    "weight": ELT_CHILD_W["elt_1"],
                    "filter_out": False,
                    "apply_scale": False,    # ELT does NOT match scaling in this test data
                },
                "elt_2": {
                    "instrument_id": 102,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": ELT_EXPO_W["elt_2"],
                    "weight": ELT_CHILD_W["elt_2"],
                    "filter_out": False,
                    "apply_scale": False,
                },
                "elt_3": {
                    "instrument_id": 103,
                    "parent_instrument_id": 3,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": ELT_EXPO_W["elt_3"],
                    "weight": ELT_CHILD_W["elt_3"],
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
            "complete_lookthroughs": {
                "clt_1": {
                    "instrument_id": 201,
                    "parent_instrument_id": 2,
                    "sub_portfolio_id": 100,
                    "initial_exposure_weight": CLT_EXPO_W["clt_1"],
                    "weight": CLT_CHILD_W["clt_1"],
                    "filter_out": False,
                    "apply_scale": False,
                },
            },
        }
    }


# =============================================================================
# Engine Setup
# =============================================================================

def setup_engine(scenario: TestScenario) -> PerspectiveEngine:
    engine = PerspectiveEngine(connection_string=None)
    engine.config.default_modifiers = []

    rules: List[Rule] = []

    # Filtering rule (optional)
    if scenario.has_filtering:
        rules.append(Rule(
            name="filter_rule",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "filter_out", "operator_type": "!=", "value": True},
            condition_for_next_rule="and" if scenario.has_scaling else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))
    else:
        rules.append(Rule(
            name="pass_all",
            apply_to=ApplyTo.BOTH,
            criteria=None,
            condition_for_next_rule="and" if scenario.has_scaling else None,
            is_scaling_rule=False,
            scale_factor=1.0,
        ))

    # Scaling rule (optional)
    if scenario.has_scaling:
        rules.append(Rule(
            name="scaling_rule",
            apply_to=ApplyTo.BOTH,
            criteria={"column": "apply_scale", "operator_type": "==", "value": True},
            condition_for_next_rule=None,
            is_scaling_rule=True,
            scale_factor=0.5,
        ))

    engine.config.perspectives[PERSPECTIVE_ID] = rules

    # Modifiers (these are just presence flags read by PSP)
    if scenario.scale_holdings:
        engine.config.modifiers["scale_holdings_to_100_percent"] = Modifier(
            name="scale_holdings_to_100_percent",
            modifier_type=ModifierType.SCALING,
            apply_to=ApplyTo.BOTH,
            criteria=None,
            rule_result_operator=None,
            override_modifiers=[],
        )

    if scenario.scale_lookthroughs:
        engine.config.modifiers["scale_lookthroughs_to_100_percent"] = Modifier(
            name="scale_lookthroughs_to_100_percent",
            modifier_type=ModifierType.SCALING,
            apply_to=ApplyTo.BOTH,
            criteria=None,  # in this test, rescale applies to all parents
            rule_result_operator=None,
            override_modifiers=[],
        )

    return engine


def get_modifiers_list(scenario: TestScenario) -> List[str]:
    mods = []
    if scenario.scale_holdings:
        mods.append("scale_holdings_to_100_percent")
    if scenario.scale_lookthroughs:
        mods.append("scale_lookthroughs_to_100_percent")
    return mods


# =============================================================================
# Expected Values (matches your PSP implementation)
# =============================================================================

def _position_factor_world_a(scenario: TestScenario, pos_id: str) -> Optional[float]:
    """Return fcol for position row (None means removed)."""
    if scenario.has_filtering and pos_id == "pos_1":
        return None
    if scenario.has_scaling and pos_id == "pos_2":
        return 0.5
    return 1.0


def _elt_factor_world_a(scenario: TestScenario, elt_id: str, parent_pos_id: str) -> Optional[float]:
    """
    Return fcol for ELT row (None means removed).

    World A / your PSP:
    - ELT is removed if parent removed (sync).
    - Otherwise ELT's factor is based on its OWN scaling-rule match.
      In test data apply_scale=False for all ELTs => factor 1.0 even when parent factor is 0.5.
    """
    if _position_factor_world_a(scenario, parent_pos_id) is None:
        return None
    # In this test data, ELTs never match apply_scale=True
    return 1.0


def _clt_factor_world_a(scenario: TestScenario, clt_id: str, parent_pos_id: str) -> Optional[float]:
    """Same semantics as ELT for removal; scaling not matched in this test data."""
    if _position_factor_world_a(scenario, parent_pos_id) is None:
        return None
    return 1.0


def calculate_expected_values(scenario: TestScenario) -> Dict:
    # --- kept sets + factors ---
    kept_positions: List[str] = []
    pos_f: Dict[str, float] = {}
    for pid in POS_W:
        f = _position_factor_world_a(scenario, pid)
        if f is None:
            continue
        kept_positions.append(pid)
        pos_f[pid] = f

    kept_elts: List[str] = []
    elt_f: Dict[str, float] = {}
    for eid, parent in ELT_PARENT.items():
        f = _elt_factor_world_a(scenario, eid, parent)
        if f is None:
            continue
        kept_elts.append(eid)
        elt_f[eid] = f

    kept_clts: List[str] = []
    clt_f: Dict[str, float] = {}
    for cid, parent in CLT_PARENT.items():
        f = _clt_factor_world_a(scenario, cid, parent)
        if f is None:
            continue
        kept_clts.append(cid)
        clt_f[cid] = f

    # --- effective masses for SF / holdings-denom (positions + ELT only) ---
    kept_pos_mass = sum(POS_W[pid] * pos_f[pid] for pid in kept_positions)
    kept_elt_mass = sum(ELT_EXPO_W[eid] * elt_f[eid] for eid in kept_elts)
    kept_total_mass = kept_pos_mass + kept_elt_mass

    # SF = 1.0 when scale_holdings is disabled (per user requirement)
    # SF = kept_total_mass / total_universe when scale_holdings is enabled
    expected_sf = (kept_total_mass / TOTAL_UNIVERSE_RAW) if scenario.scale_holdings else 1.0

    # --- position output weights (what PSP emits for positions) ---
    # if scale_holdings: w_out = (w * f) / kept_total_mass
    # else:             w_out = (w * f)
    pos_out: Dict[str, float] = {}
    for pid in kept_positions:
        eff = POS_W[pid] * pos_f[pid]
        pos_out[pid] = (eff / kept_total_mass) if scenario.scale_holdings else eff

    # --- lookthrough output weights (what PSP emits for lookthrough "weight") ---
    # If scale_lookthroughs: per (parent, record_type) normalize using sum(weight * fcol).
    # Else: weight_out = weight * fcol.
    lt_out_by_record_type: Dict[str, Dict[str, float]] = {
        "essential_lookthroughs": {},
        "complete_lookthroughs": {},
    }

    def _compute_lt_out(
        ids: List[str],
        parent_map: Dict[str, str],
        child_w: Dict[str, float],
        factor_map: Dict[str, float],
        record_type: str
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not ids:
            return out

        if not scenario.scale_lookthroughs:
            for lid in ids:
                out[lid] = child_w[lid] * factor_map[lid]
            return out

        # normalize per parent
        denom_by_parent: DefaultDict[str, float] = defaultdict(float)
        for lid in ids:
            denom_by_parent[parent_map[lid]] += child_w[lid] * factor_map[lid]

        for lid in ids:
            p = parent_map[lid]
            denom = denom_by_parent[p]
            out[lid] = (child_w[lid] * factor_map[lid] / denom) if denom != 0 else 0.0
        return out

    # ELT "weight"
    lt_out_by_record_type["essential_lookthroughs"] = _compute_lt_out(
        kept_elts, ELT_PARENT, ELT_CHILD_W, elt_f, "essential_lookthroughs"
    )
    # CLT "weight"
    lt_out_by_record_type["complete_lookthroughs"] = _compute_lt_out(
        kept_clts, CLT_PARENT, CLT_CHILD_W, clt_f, "complete_lookthroughs"
    )

    return {
        "kept_positions": kept_positions,
        "pos_factor": pos_f,
        "kept_elts": kept_elts,
        "elt_factor": elt_f,
        "kept_clts": kept_clts,
        "clt_factor": clt_f,
        "kept_pos_mass": kept_pos_mass,
        "kept_elt_mass": kept_elt_mass,
        "kept_total_mass": kept_total_mass,
        "total_universe_raw": TOTAL_UNIVERSE_RAW,
        "scale_factor": expected_sf,
        "pos_out": pos_out,
        "lt_out": lt_out_by_record_type,
    }


# =============================================================================
# Helpers to read PSP output
# =============================================================================

def _get_container(psp_output: Dict, config_name: str, pid: int, container: str) -> Dict:
    return (
        psp_output.get("perspective_configurations", {})
        .get(config_name, {})
        .get(pid, {})
        .get(container, {})
    )


def _safe_get_dict(d: Dict, key: str) -> Dict:
    v = d.get(key, {})
    return v if isinstance(v, dict) else {}


def _float_close(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


# =============================================================================
# Verification (matches downstream usage)
# =============================================================================

def verify_world_a(
    scenario: TestScenario,
    psp_output: Dict,
    expected: Dict,
    tna: float = TNA,
    config_name: str = "test_config",
    container_name: str = CONTAINER,
) -> bool:
    container_data = _get_container(psp_output, config_name, PERSPECTIVE_ID, container_name)
    if not container_data:
        print("  [FAIL] Missing container output")
        return False

    positions = _safe_get_dict(container_data, "positions")
    elts = _safe_get_dict(container_data, "essential_lookthroughs")
    clts = _safe_get_dict(container_data, "complete_lookthroughs")
    scale_factors = _safe_get_dict(container_data, "scale_factors")

    # -------------------------------------------------------------------------
    # 1) Scale factor check (World A): SF = kept_total_mass / total_universe_raw
    # -------------------------------------------------------------------------
    actual_sf = scale_factors.get("initial_exposure_weight", None)
    expected_sf = expected["scale_factor"]
    if actual_sf is None:
        print("  [FAIL] Missing scale_factors.initial_exposure_weight")
        return False
    if not _float_close(actual_sf, expected_sf, eps=1e-6):
        print(f"  [FAIL] SF mismatch: expected {expected_sf:.10f}, got {actual_sf:.10f}")
        return False
    print(f"  [OK] SF = {actual_sf:.10f}")

    # -------------------------------------------------------------------------
    # 2) Positions emitted weights check
    # -------------------------------------------------------------------------
    # Kept positions present, removed absent
    for pid in expected["kept_positions"]:
        if pid not in positions:
            print(f"  [FAIL] Missing kept position {pid}")
            return False
    for pid in POS_W.keys():
        if pid not in expected["kept_positions"] and pid in positions:
            print(f"  [FAIL] Position {pid} should be removed but is present")
            return False

    # Weights match expected
    for pid, exp_w in expected["pos_out"].items():
        act_w = positions[pid].get("initial_exposure_weight", None)
        if act_w is None:
            print(f"  [FAIL] Position {pid} missing initial_exposure_weight")
            return False
        if not _float_close(act_w, exp_w, eps=1e-6):
            print(f"  [FAIL] Position {pid} weight mismatch: expected {exp_w:.10f}, got {act_w:.10f}")
            return False
    print("  [OK] Position weights match expected")

    # -------------------------------------------------------------------------
    # 3) Lookthrough emitted weights check (only "weight" label)
    # -------------------------------------------------------------------------
    # ELT presence: only if parent kept
    for eid in expected["kept_elts"]:
        if eid not in elts:
            print(f"  [FAIL] Missing kept ELT {eid}")
            return False
    for eid in ELT_PARENT.keys():
        if eid not in expected["kept_elts"] and eid in elts:
            print(f"  [FAIL] ELT {eid} should be removed but is present")
            return False

    # CLT presence: only if parent kept
    for cid in expected["kept_clts"]:
        if cid not in clts:
            print(f"  [FAIL] Missing kept CLT {cid}")
            return False
    for cid in CLT_PARENT.keys():
        if cid not in expected["kept_clts"] and cid in clts:
            print(f"  [FAIL] CLT {cid} should be removed but is present")
            return False

    # ELT weight values
    for eid, exp_w in expected["lt_out"]["essential_lookthroughs"].items():
        act_w = elts[eid].get("weight", None)
        if act_w is None:
            print(f"  [FAIL] ELT {eid} missing weight")
            return False
        if not _float_close(act_w, exp_w, eps=1e-6):
            print(f"  [FAIL] ELT {eid} weight mismatch: expected {exp_w:.10f}, got {act_w:.10f}")
            return False

    # CLT weight values
    for cid, exp_w in expected["lt_out"]["complete_lookthroughs"].items():
        act_w = clts[cid].get("weight", None)
        if act_w is None:
            print(f"  [FAIL] CLT {cid} missing weight")
            return False
        if not _float_close(act_w, exp_w, eps=1e-6):
            print(f"  [FAIL] CLT {cid} weight mismatch: expected {exp_w:.10f}, got {act_w:.10f}")
            return False

    print("  [OK] Lookthrough weights match expected")

    # Additional invariant when scale_lookthroughs=True:
    if scenario.scale_lookthroughs:
        # per parent per record_type sums to 1
        sums_elt: DefaultDict[str, float] = defaultdict(float)
        for eid, row in elts.items():
            sums_elt[ELT_PARENT[eid]] += row["weight"]
        for parent, s in sums_elt.items():
            if not _float_close(s, 1.0, eps=1e-6):
                print(f"  [FAIL] ELT weights for parent {parent} do not sum to 1: {s:.10f}")
                return False

        sums_clt: DefaultDict[str, float] = defaultdict(float)
        for cid, row in clts.items():
            sums_clt[CLT_PARENT[cid]] += row["weight"]
        for parent, s in sums_clt.items():
            if not _float_close(s, 1.0, eps=1e-6):
                print(f"  [FAIL] CLT weights for parent {parent} do not sum to 1: {s:.10f}")
                return False

        print("  [OK] Lookthrough per-parent sums are 1.0 (when rescaled)")

    # -------------------------------------------------------------------------
    # 4) Downstream simulation checks
    # -------------------------------------------------------------------------
    # (A) Positions view total value
    total_pos_value = 0.0
    for pid, row in positions.items():
        total_pos_value += row["initial_exposure_weight"] * tna * actual_sf

    expected_pos_value = expected["kept_pos_mass"] * tna / expected["total_universe_raw"]
    if abs(total_pos_value - expected_pos_value) > 1.0:
        print(f"  [FAIL] Positions view TNA mismatch: expected {expected_pos_value:.2f}, got {total_pos_value:.2f}")
        return False
    print(f"  [OK] Positions view value = {total_pos_value:.2f} (expected {expected_pos_value:.2f})")

    # (B) Lookthrough view (per record_type separately)
    # Downstream: value(child) = parent_pos_weight * child_weight * TNA * SF
    # We check per parent: sum(children) equals parent_value * sum(child_weight).
    def _check_record_type(record_type: str, lt_rows: Dict, parent_map: Dict[str, str]) -> bool:
        if not lt_rows:
            return True

        sum_child_w_by_parent: DefaultDict[str, float] = defaultdict(float)
        sum_child_value_by_parent: DefaultDict[str, float] = defaultdict(float)

        for lid, row in lt_rows.items():
            p = parent_map[lid]
            child_w = row["weight"]
            parent_w = positions[p]["initial_exposure_weight"]  # PSP output for parent
            val = parent_w * child_w * tna * actual_sf
            sum_child_w_by_parent[p] += child_w
            sum_child_value_by_parent[p] += val

        for p, sum_val in sum_child_value_by_parent.items():
            parent_val = positions[p]["initial_exposure_weight"] * tna * actual_sf
            expected_val = parent_val * sum_child_w_by_parent[p]
            if abs(sum_val - expected_val) > 1.0:
                print(
                    f"  [FAIL] {record_type} parent {p} value mismatch: "
                    f"expected {expected_val:.2f}, got {sum_val:.2f}"
                )
                return False

        return True

    if not _check_record_type("essential_lookthroughs", elts, ELT_PARENT):
        return False
    if not _check_record_type("complete_lookthroughs", clts, CLT_PARENT):
        return False
    print("  [OK] Lookthrough downstream identity holds per record_type")

    return True


# =============================================================================
# Runner
# =============================================================================

def run_scenario(scenario: TestScenario) -> bool:
    """Run a single test scenario (called by run_all_tests, not pytest directly)."""
    print("\n" + "=" * 90)
    print(f"TEST: {scenario}")
    print("=" * 90)

    input_json = create_test_input_json()
    engine = setup_engine(scenario)
    modifiers = get_modifiers_list(scenario)
    input_json["perspective_configurations"] = {"test_config": {str(PERSPECTIVE_ID): modifiers}}

    try:
        output = engine.process(input_json)
    except Exception as e:
        print(f"  [ERROR] PSP failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    expected = calculate_expected_values(scenario)

    ok = verify_world_a(scenario, output, expected, tna=TNA)
    print(f"\n  {'[PASS]' if ok else '[FAIL]'} {scenario}")
    return ok


# Module-level scenarios list for parametrized testing
RESCALING_SCENARIOS = [
    # No changes
    TestScenario("no_changes", False, False, False, False),

    # Single feature
    TestScenario("filter_only", True, False, False, False),
    TestScenario("scale_rule_only", False, True, False, False),
    TestScenario("rescale_holdings_only", False, False, True, False),
    TestScenario("rescale_lt_only", False, False, False, True),

    # Two features
    TestScenario("filter+scale", True, True, False, False),
    TestScenario("filter+rescale_hold", True, False, True, False),
    TestScenario("filter+rescale_lt", True, False, False, True),
    TestScenario("scale+rescale_hold", False, True, True, False),
    TestScenario("scale+rescale_lt", False, True, False, True),
    TestScenario("rescale_hold+lt", False, False, True, True),

    # Three features
    TestScenario("filter+scale+rescale_hold", True, True, True, False),
    TestScenario("filter+scale+rescale_lt", True, True, False, True),
    TestScenario("filter+rescale_hold+lt", True, False, True, True),
    TestScenario("scale+rescale_hold+lt", False, True, True, True),

    # All features
    TestScenario("all_features", True, True, True, True),
]


def run_all_tests() -> bool:
    """Run all test scenarios (for __main__ execution)."""
    results: List[Tuple[TestScenario, bool]] = []
    for s in RESCALING_SCENARIOS:
        try:
            results.append((s, run_scenario(s)))
        except Exception as e:
            print(f"  [ERROR] Exception in scenario {s}: {e}")
            import traceback
            traceback.print_exc()
            results.append((s, False))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    passed = 0
    for s, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {s}")
        passed += int(ok)
    print(f"\n  Total: {passed}/{len(results)} passed")

    return passed == len(results)


@pytest.mark.parametrize("scenario", RESCALING_SCENARIOS, ids=lambda s: s.name)
def test_rescaling_scenario(scenario: TestScenario):
    """Pytest parametrized test for each rescaling scenario."""
    assert run_scenario(scenario), f"Scenario {scenario.name} failed"


if __name__ == "__main__":
    print("=" * 90)
    print("COMPREHENSIVE RESCALING COMBINATIONS TEST (World A semantics)")
    print("=" * 90)

    success = run_all_tests()
    if success:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        raise SystemExit(1)
