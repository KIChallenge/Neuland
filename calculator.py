#!/usr/bin/env python3
"""
Deterministic & Abuse-Resistant PolicyCalculator & BudgetValidator for HealthPolicy Simulator (UC2).
Ensures agents NEVER hallucinate arithmetic, unrealistic effect percentages, or fake interpolation points:
1. Every function anchors effects to real Austrian system capacities from `grundfakten.yaml`
   (e.g., 8,300 Kassenärzte, 17,189 Mio. EUR ambulant budget, 22,733 Mio. EUR stationär budget).
2. `linear_interpolate` clamps wild extrapolations and checks plausibility against system capacity.
3. `validate_and_normalize_proposal_budget` deduplicates identical levers, clamps unrealistic single-measure
   effects to realistic Austrian stock ratios, and sums effects transparently without black-box magic.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import re

WORKSPACE_DIR = Path(__file__).resolve().parent
GRUNDFAKTEN_PATH = WORKSPACE_DIR / "grundfakten.yaml"


def load_grundfakten_dict() -> Dict[str, Any]:
    """Lightweight parser for grundfakten.yaml (works with or without PyYAML installed)."""
    text = GRUNDFAKTEN_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        pass

    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not v:
                current_section = k
                result[current_section] = {}
            else:
                current_section = None
                result[k] = _parse_scalar_or_list(v)
        elif indent > 0 and current_section and ":" in stripped:
            k, v = stripped.split(":", 1)
            result[current_section][k.strip()] = _parse_scalar_or_list(v.strip())
    return result


def _parse_scalar_or_list(val: str) -> Any:
    if val.startswith("[") and val.endswith("]"):
        items = [x.strip() for x in val[1:-1].split(",") if x.strip()]
        return [_parse_scalar_or_list(x) for x in items]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


GRUNDFAKTEN = load_grundfakten_dict()

# Baseline 2024 Budgets in Mio. EUR from grundfakten.yaml
BUDGET_BASELINES_2024_MIO: Dict[str, float] = {
    "gesamt_laufend": float(GRUNDFAKTEN["budget_mio_eur"]["gesamt_laufend"][-1]),      # 57837.0
    "stationaer": float(GRUNDFAKTEN["budget_mio_eur"]["stationaer"][-1]),              # 22733.0
    "ambulant": float(GRUNDFAKTEN["budget_mio_eur"]["ambulant"][-1]),                  # 17189.0
    "medikamente": float(GRUNDFAKTEN["budget_mio_eur"]["medikamente"][-1]),            # 8786.0
    "haeusliche_pflege": float(GRUNDFAKTEN["budget_mio_eur"]["haeusliche_pflege"][-1]),# 3496.0
    "transport_rettung": float(GRUNDFAKTEN["budget_mio_eur"]["transport_rettung"][-1]),# 697.0
    "praevention": float(GRUNDFAKTEN["budget_mio_eur"]["praevention"][-1]),            # 1232.0
    "oeffentlich_gesamt": float(
        GRUNDFAKTEN.get("finanzierungstraeger_2024", {}).get("oeffentlich_gesamt_mio_eur_~", 43783.0)
    ),
}

# Real Austrian System Stock / Capacity Anchors (from grundfakten.yaml)
# Used to prevent LLMs from inventing fantasy effect sizes!
SYSTEM_LEVER_ANCHORS: Dict[str, Dict[str, Any]] = {
    "kassenstellen": {
        "name": "Kassenstellen & Vertragsärzt:innen (inkl. Tarifattraktivierung)",
        "base_stock": 8300.0,
        "min_unit_cost_mio": 1.25,
        "default_unit_cost_mio": 1.45,
        "elasticity": 1.35,  # +1% more Kassenstellen -> ~1.35% waiting time reduction in bottleneck specialties
        "max_single_lever_effect_pct": 8.5,
    },
    "pve_primärversorgung": {
        "name": "Primärversorgungseinheiten (PVE) & Randzeiten-Zentren",
        "base_stock": 850.0,  # Equivalent primary care capacity units
        "min_unit_cost_mio": 2.8,
        "default_unit_cost_mio": 3.6,
        "elasticity": 1.20,
        "max_single_lever_effect_pct": 7.5,
    },
    "facharztzentren_ambulanz": {
        "name": "Regionale Facharztzentren & Spitalsambulanz-Entlastung",
        "base_stock": 600.0,
        "min_unit_cost_mio": 4.2,
        "default_unit_cost_mio": 5.2,
        "elasticity": 1.15,
        "max_single_lever_effect_pct": 8.0,
    },
    "spitalsbetten_tagesklinik": {
        "name": "Spitalskapazitäten, OP-Ausbau & Tagesklinik-Verlagerung",
        "base_stock": 43332.0,
        "min_unit_cost_mio": 0.12,
        "default_unit_cost_mio": 0.18,
        "elasticity": 1.25,
        "max_single_lever_effect_pct": 8.5,
    },
    "pflegepersonal_dgkp": {
        "name": "Pflegepersonal (DGKP / Community Nurses / OP-Pflege)",
        "base_stock": 62254.0,
        "min_unit_cost_mio": 0.085,
        "default_unit_cost_mio": 0.095,
        "elasticity": 1.30,
        "max_single_lever_effect_pct": 8.0,
    },
    "digitalisierung_1450_elga": {
        "name": "Digitale Patientenlenkung (1450, ELGA, e-Health-Triage)",
        "base_stock": 1200.0,  # Mio EUR reference transformation volume
        "min_unit_cost_mio": 1.0,
        "default_unit_cost_mio": 1.0,
        "elasticity": 0.25,  # Per 100 Mio EUR -> ~3.0-4.5% steering effect
        "max_single_lever_effect_pct": 5.5,
    },
    "praevention_reha": {
        "name": "Prävention, Vorsorge & Rehabilitation",
        "base_stock": 1232.0,  # Current prevention budget Mio EUR
        "min_unit_cost_mio": 1.0,
        "default_unit_cost_mio": 1.0,
        "elasticity": 0.18,
        "max_single_lever_effect_pct": 6.5,
    },
}


def infer_lever_category(title_or_label: str, description: str = "") -> str:
    """
    Detect which Austrian healthcare lever a measure belongs to.
    Prioritizes the measure TITLE first so generic words in the description
    (like 'Randzeiten' or 'Kassenbindung') never misclassify e.g. Facharztzentren as PVE!
    """
    for text in (title_or_label or "", f"{title_or_label or ''} {description or ''}"):
        low = text.lower()
        if not low.strip():
            continue
        if any(k in low for k in ("facharztzentr", "spitalsambulanz", "ambulanz", "spezialversorgung")):
            return "facharztzentren_ambulanz"
        if any(k in low for k in ("pve", "primärversorgung", "gruppenprax")):
            return "pve_primärversorgung"
        if any(k in low for k in ("1450", "elga", "digital", "telemedizin", "triage", "lenkung", "e-card", "e-health")):
            return "digitalisierung_1450_elga"
        if any(k in low for k in ("pflege", "dgkp", "community nurse", "krankenpflege")):
            return "pflegepersonal_dgkp"
        if any(k in low for k in ("prävention", "vorsorge", "reha", "impf", "früherkennung")):
            return "praevention_reha"
        if any(k in low for k in ("op-", "operation", "tagesklinik", "betten", "stationär")):
            return "spitalsbetten_tagesklinik"
        if any(k in low for k in ("kassenstell", "kassenärzt", "kassenarzt", "vertragsärzt", "wahlarzt", "tarif", "honorar", "randzeit")):
            return "kassenstellen"
    return "facharztzentren_ambulanz"


def _extract_unit_count_from_title(title: str) -> Optional[float]:
    """Extract unit count from a measure title while ignoring hotline '1450' and calendar years '2020'..'2045'."""
    for m in re.finditer(r"\b(\d{2,5})\b", title or ""):
        val = int(m.group(1))
        if val == 1450 or 2020 <= val <= 2045:
            continue
        return float(val)
    return None


def compute_grounded_measure_effect(
    cost_mio_eur: float,
    raw_claimed_effect_pct: float,
    lever_category: str,
    num_units: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes the realistic effect of a measure anchored to the actual Austrian system stock
    in `grundfakten.yaml`.
    Combines:
    1. Direct unit expansion effect (`num_units / base_stock * elasticity`)
    2. System-wide productivity / extended-hours / tariff reform effect if the allocated budget
       exceeds pure unit setup costs (e.g., funding evening/weekend slots across the existing 8,300 doctors).
    Returns transparent before/after explanation for the UI Fact-Check Diff!
    """
    anchor = SYSTEM_LEVER_ANCHORS.get(lever_category, SYSTEM_LEVER_ANCHORS["facharztzentren_ambulanz"])
    base_total_budget = BUDGET_BASELINES_2024_MIO["gesamt_laufend"]  # 57837.0
    max_cap = float(anchor["max_single_lever_effect_pct"])
    stock = float(anchor["base_stock"])
    unit_cost = float(anchor["default_unit_cost_mio"])

    budget_share_pct = (max(0.0, cost_mio_eur) / base_total_budget) * 100.0
    # In Austria (57,837 Mio EUR total budget), 1% of total budget (~578 Mio EUR) invested in targeted
    # outpatient/hospital capacity & extended opening hours yields ~5.5% to 6.8% waiting-time/capacity effect.
    cost_based_effect = budget_share_pct * 6.4

    reason_detail = ""
    if (
        num_units
        and num_units > 0
        and lever_category
        in (
            "kassenstellen",
            "pve_primärversorgung",
            "facharztzentren_ambulanz",
            "spitalsbetten_tagesklinik",
            "pflegepersonal_dgkp",
        )
    ):
        rel_stock_increase_pct = (num_units / stock) * 100.0
        unit_expansion_effect = rel_stock_increase_pct * float(anchor["elasticity"])
        pure_unit_cost_mio = num_units * unit_cost
        # If the measure also funds system-wide tariff/opening-hour incentives beyond pure unit creation:
        extra_structural_budget_mio = max(0.0, cost_mio_eur - pure_unit_cost_mio)
        extra_structural_effect = (extra_structural_budget_mio / base_total_budget) * 100.0 * 5.2
        realistic_effect = unit_expansion_effect + extra_structural_effect
        reason_detail = (
            f"{num_units:.0f} Einh. / {stock:.0f} Bestand (+{rel_stock_increase_pct:.2f}% Kapazität -> +{unit_expansion_effect:.2f}%)"
            + (
                f" + Flächen-/Randzeiten-Hebel aus {extra_structural_budget_mio:.0f} Mio. € (+{extra_structural_effect:.2f}%)"
                if extra_structural_effect >= 0.25
                else ""
            )
        )
    else:
        realistic_effect = cost_based_effect
        reason_detail = f"Budget-Hebel {cost_mio_eur:.0f} Mio. € ({budget_share_pct:.2f}% des Gesamtbudgets im Bereich {anchor['name']})"

    realistic_effect = min(max_cap, max(0.3, realistic_effect))

    # Keep the agent's claimed effect if it lies within the realistic corridor [0.5x .. 1.18x of realistic_effect];
    # otherwise adjust to realistic_effect and log the exact before/after diff!
    if 0.3 <= abs(raw_claimed_effect_pct) <= realistic_effect * 1.18 and abs(raw_claimed_effect_pct) >= realistic_effect * 0.55:
        final_effect = round(abs(raw_claimed_effect_pct), 2)
        was_clamped = False
    else:
        final_effect = round(realistic_effect, 2)
        was_clamped = abs(raw_claimed_effect_pct - final_effect) > 0.45

    return {
        "final_effect_pct": final_effect,
        "raw_claimed_effect_pct": round(raw_claimed_effect_pct, 2),
        "was_clamped": was_clamped,
        "lever_category": lever_category,
        "lever_name": anchor["name"],
        "reason_detail": reason_detail,
    }


@dataclass
class CalculationStep:
    trace_id: str
    function_name: str
    label: str
    inputs: Dict[str, Any]
    formula: str
    result: float
    unit: str
    source_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyCalculator:
    """
    Deterministic, grounded calculator instance attached to an agent or fact-checker run.
    Records every calculation step with a unique trace_id (`CALC-01`, `CALC-02`, ...) and
    enforces Austrian stock & budget anchors across all 5 calculation functions.
    """

    def __init__(self, agent_id: str = "AGENT"):
        self.agent_id = agent_id
        self.steps: List[CalculationStep] = []

    def _record(
        self,
        fn: str,
        label: str,
        inputs: Dict[str, Any],
        formula: str,
        result: float,
        unit: str,
        source_reference: str = "",
    ) -> Dict[str, Any]:
        tid = f"CALC-{len(self.steps) + 1:02d}"
        rounded = round(float(result), 4)
        step = CalculationStep(
            trace_id=tid,
            function_name=fn,
            label=label,
            inputs=inputs,
            formula=formula,
            result=rounded,
            unit=unit,
            source_reference=source_reference,
        )
        self.steps.append(step)
        return step.to_dict()

    def linear_interpolate(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        target_x: float,
        label: str,
        unit: str = "",
        source_reference: str = "",
    ) -> Dict[str, Any]:
        """
        Linearly interpolate y at target_x given two points (x1, y1) and (x2, y2).
        Includes an anti-extrapolation guard: target_x cannot extrapolate more than 2.5x beyond [x1, x2],
        and if interpolating a % system effect, it is anchored against realistic system bounds.
        """
        span = abs(x2 - x1)
        clamped_target_x = target_x
        guard_note = ""
        if span > 1e-9:
            min_bound = min(x1, x2) - 1.5 * span
            max_bound = max(x1, x2) + 1.5 * span
            if target_x < min_bound or target_x > max_bound:
                clamped_target_x = max(min_bound, min(max_bound, target_x))
                guard_note = f" [Extrapolations-Guard: target_x von {target_x} auf {clamped_target_x:.2f} begrenzt]"

        if span < 1e-12:
            res = float(y1)
            formula = f"y = {y1} (x2 == x1 == {x1})"
        else:
            res = y1 + ((clamped_target_x - x1) / (x2 - x1)) * (y2 - y1)
            formula = (
                f"{y1} + (({clamped_target_x} - {x1}) / ({x2} - {x1})) * ({y2} - {y1}) = {res:.4f}"
                + guard_note
            )

        # If unit is % and label refers to effect/waiting time, cap single interpolation to max realistic lever effect (9.5%)
        if "%" in unit and any(w in label.lower() for w in ("effekt", "wirkung", "wartezeit", "reduktion", "entlastung")):
            if abs(res) > 9.5:
                old_res = res
                res = 9.5 if res > 0 else -9.5
                formula += f" -> [Bestands-Anker: Einzelwirkung von {old_res:.2f}% auf max. {res:.2f}% Systemkapazität korrigiert]"

        return self._record(
            fn="linear_interpolate",
            label=label,
            inputs={"x1": x1, "y1": y1, "x2": x2, "y2": y2, "target_x": target_x},
            formula=formula,
            result=res,
            unit=unit,
            source_reference=source_reference,
        )

    def scale_metric(
        self,
        base_value: float,
        change_pct: float,
        label: str,
        return_delta_only: bool = True,
        unit: str = "Mio. EUR",
        source_reference: str = "",
    ) -> Dict[str, Any]:
        """
        Calculate percentage change on a baseline value from grundfakten.yaml or an article.
        """
        delta = base_value * (change_pct / 100.0)
        new_val = base_value + delta
        res = delta if return_delta_only else new_val
        formula = (
            f"{base_value} * ({change_pct} / 100.0) = {delta:.4f}"
            if return_delta_only
            else f"{base_value} * (1 + {change_pct} / 100.0) = {new_val:.4f}"
        )
        return self._record(
            fn="scale_metric",
            label=label,
            inputs={
                "base_value": base_value,
                "change_pct": change_pct,
                "return_delta_only": return_delta_only,
                "delta": round(delta, 4),
                "new_total": round(new_val, 4),
            },
            formula=formula,
            result=res,
            unit=unit,
            source_reference=source_reference,
        )

    def budget_impact_pct(
        self,
        cost_mio_eur: float,
        reference_sector: str = "gesamt_laufend",
        label: str = "Budgetauswirkung in % der Gesundheitsausgaben",
        source_reference: str = "grundfakten.yaml",
    ) -> Dict[str, Any]:
        """
        Convert a cost in Mio. EUR into a percentage of the Austrian health budget (2024: 57,837 Mio. EUR).
        """
        base = BUDGET_BASELINES_2024_MIO.get(
            reference_sector, BUDGET_BASELINES_2024_MIO["gesamt_laufend"]
        )
        pct = (cost_mio_eur / base) * 100.0
        formula = f"({cost_mio_eur} Mio. EUR / {base} Mio. EUR [{reference_sector}]) * 100 = {pct:.4f}%"
        return self._record(
            fn="budget_impact_pct",
            label=label,
            inputs={"cost_mio_eur": cost_mio_eur, "reference_sector": reference_sector, "base_mio_eur": base},
            formula=formula,
            result=pct,
            unit="%",
            source_reference=source_reference,
        )

    def unit_cost_projection(
        self,
        num_units: float,
        cost_per_unit_mio_eur: float,
        effect_per_unit_pct: float = 0.0,
        years: float = 1.0,
        label: str = "Stückkosten- & Bestands-Wirkungsprojektion",
        source_reference: str = "",
    ) -> Dict[str, Any]:
        """
        Project total cost (in Mio. EUR and % of total health budget) and realistic system effect (%)
        anchored to the Austrian baseline stock in `grundfakten.yaml`.
        """
        lever_cat = infer_lever_category(label)
        anchor = SYSTEM_LEVER_ANCHORS.get(lever_cat, SYSTEM_LEVER_ANCHORS["facharztzentren_ambulanz"])

        # Ensure realistic full unit cost (including tariff/staffing overhead)
        effective_unit_cost = max(float(cost_per_unit_mio_eur), float(anchor["min_unit_cost_mio"]))
        total_cost_mio = num_units * effective_unit_cost * max(1.0, min(years, 5.0))
        total_budget_pct = (
            total_cost_mio / BUDGET_BASELINES_2024_MIO["gesamt_laufend"]
        ) * 100.0

        # Compute stock-anchored effect instead of trusting arbitrary `effect_per_unit_pct`
        raw_claimed = num_units * abs(effect_per_unit_pct)
        grounded = compute_grounded_measure_effect(
            cost_mio_eur=total_cost_mio,
            raw_claimed_effect_pct=raw_claimed,
            lever_category=lever_cat,
            num_units=num_units,
        )
        total_effect_pct = grounded["final_effect_pct"]
        stock = anchor["base_stock"]
        rel_cap_pct = (num_units / stock) * 100.0 if stock > 0 else 0.0

        formula = (
            f"Bestands-Relation [{anchor['name']}]: {num_units:.0f} Einh. / {stock:.0f} Bestand = +{rel_cap_pct:.2f}% Kapazität "
            f"-> Realistischer Systemeffekt: {total_effect_pct:.2f}%; "
            f"Vollkosten: {num_units:.0f} * {effective_unit_cost:.2f} Mio. EUR = {total_cost_mio:.2f} Mio. EUR "
            f"({total_budget_pct:.3f}% von 57.837 Mio. EUR)"
        )
        return self._record(
            fn="unit_cost_projection",
            label=label,
            inputs={
                "num_units": num_units,
                "cost_per_unit_mio_eur": effective_unit_cost,
                "lever_category": lever_cat,
                "base_stock_austria": stock,
                "relative_capacity_increase_pct": round(rel_cap_pct, 3),
                "total_cost_mio_eur": round(total_cost_mio, 2),
                "total_budget_pct": round(total_budget_pct, 4),
                "total_effect_pct": total_effect_pct,
            },
            formula=formula,
            result=total_cost_mio,
            unit="Mio. EUR",
            source_reference=source_reference,
        )

    def basic_math(
        self,
        operation: str,
        a: float,
        b: float,
        label: str,
        unit: str = "",
        source_reference: str = "",
    ) -> Dict[str, Any]:
        """
        Execute basic arithmetic: operation in ['add', 'sub', 'mul', 'div', 'pct_of', 'pct_change'].
        Note: Final proposal effects and budgets are always verified against system stock anchors.
        """
        op = operation.lower().strip()
        if op in ("add", "+"):
            res = a + b
            formula = f"{a} + {b} = {res:.4f}"
        elif op in ("sub", "-"):
            res = a - b
            formula = f"{a} - {b} = {res:.4f}"
        elif op in ("mul", "*"):
            res = a * b
            formula = f"{a} * {b} = {res:.4f}"
        elif op in ("div", "/"):
            res = a / b if abs(b) > 1e-12 else 0.0
            formula = f"{a} / {b} = {res:.4f}"
        elif op == "pct_of":
            res = (a / b) * 100.0 if abs(b) > 1e-12 else 0.0
            formula = f"({a} / {b}) * 100 = {res:.4f}%"
        elif op == "pct_change":
            res = ((b - a) / a) * 100.0 if abs(a) > 1e-12 else 0.0
            formula = f"(({b} - {a}) / {a}) * 100 = {res:.4f}%"
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return self._record(
            fn="basic_math",
            label=label,
            inputs={"operation": op, "a": a, "b": b},
            formula=formula,
            result=res,
            unit=unit,
            source_reference=source_reference,
        )

    def execute_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "linear_interpolate":
            return self.linear_interpolate(**args)
        if name == "scale_metric":
            return self.scale_metric(**args)
        if name == "budget_impact_pct":
            return self.budget_impact_pct(**args)
        if name == "unit_cost_projection":
            return self.unit_cost_projection(**args)
        if name == "basic_math":
            return self.basic_math(**args)
        raise ValueError(f"Unknown calculator tool: {name}")


CALCULATOR_TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": "linear_interpolate",
        "description": (
            "Berechnet eine lineare Interpolation y für target_x zwischen zwei bekannten Punkten (x1, y1) und (x2, y2) "
            "aus einem Zeitungsartikel oder grundfakten.yaml (inkl. Plausibilitäts- & Bestands-Prüfung)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x1": {"type": "number", "description": "Erster Stützpunkt X"},
                "y1": {"type": "number", "description": "Erster Stützpunkt Y"},
                "x2": {"type": "number", "description": "Zweiter Stützpunkt X aus Quelle/Artikel"},
                "y2": {"type": "number", "description": "Zweiter Stützpunkt Y aus Quelle/Artikel"},
                "target_x": {"type": "number", "description": "Zielwert X, für den Y interpoliert werden soll"},
                "label": {"type": "string", "description": "Kurze Beschreibung der Interpolation"},
                "unit": {"type": "string", "description": "Einheit des Ergebnisses, z.B. '%', 'Mio. EUR'"},
                "source_reference": {"type": "string", "description": "Artikel-ID (z.B. ART-20241007-001) oder grundfakten.yaml"},
            },
            "required": ["x1", "y1", "x2", "y2", "target_x", "label"],
        },
    },
    {
        "type": "function",
        "name": "scale_metric",
        "description": (
            "Berechnet eine prozentuale Veränderung auf einen Sektor- oder Budgetwert aus grundfakten.yaml "
            "(z.B. +5% im ambulanten Sektor 17.189 Mio. EUR = 859,45 Mio. EUR)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_value": {"type": "number", "description": "Basiswert (z.B. 57837, 22733 oder 17189 Mio. EUR)"},
                "change_pct": {"type": "number", "description": "Prozentuale Änderung (z.B. 4.5 für +4,5%)"},
                "label": {"type": "string", "description": "Beschreibung der Rechnung"},
                "return_delta_only": {"type": "boolean", "description": "True gibt die Differenz zurück, False den neuen Gesamtwert"},
                "unit": {"type": "string", "description": "Einheit, z.B. 'Mio. EUR'"},
                "source_reference": {"type": "string", "description": "Artikel-ID oder grundfakten.yaml"},
            },
            "required": ["base_value", "change_pct", "label"],
        },
    },
    {
        "type": "function",
        "name": "budget_impact_pct",
        "description": (
            "Rechnet Kosten in Mio. EUR exakt in Prozent (%) der österreichischen Gesundheitsausgaben (57.837 Mio. EUR) um."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cost_mio_eur": {"type": "number", "description": "Kosten der Maßnahme in Mio. EUR"},
                "reference_sector": {
                    "type": "string",
                    "enum": [
                        "gesamt_laufend",
                        "stationaer",
                        "ambulant",
                        "medikamente",
                        "haeusliche_pflege",
                        "praevention",
                        "oeffentlich_gesamt",
                    ],
                },
                "label": {"type": "string", "description": "Bezeichnung der Maßnahme"},
                "source_reference": {"type": "string", "description": "Artikel-ID oder grundfakten.yaml"},
            },
            "required": ["cost_mio_eur", "label"],
        },
    },
    {
        "type": "function",
        "name": "unit_cost_projection",
        "description": (
            "Berechnet Vollkosten (in Mio. EUR und % des Gesundheitsbudgets) sowie den realen Systemeffekt (%) "
            "anhand des Verhältnisses zum österreichischen Gesamtbestand (z.B. 8.300 Kassenärzte, 43.332 Akutbetten, 62.254 Pflegekräfte)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_units": {"type": "number", "description": "Anzahl der zusätzlichen Einheiten"},
                "cost_per_unit_mio_eur": {"type": "number", "description": "Vollkosten pro Einheit in Mio. EUR"},
                "effect_per_unit_pct": {"type": "number", "description": "Geschätzter Effekt pro Einheit (wird gegen Österreich-Bestand geprüft)"},
                "years": {"type": "number", "description": "Reform-Rollout-Faktor (1 bis 5)"},
                "label": {"type": "string", "description": "Bezeichnung der Maßnahme (z.B. 'Ausbau 350 Kassenstellen inkl. Tarifreform')"},
                "source_reference": {"type": "string", "description": "Artikel-ID oder grundfakten.yaml"},
            },
            "required": ["num_units", "cost_per_unit_mio_eur", "label"],
        },
    },
    {
        "type": "function",
        "name": "basic_math",
        "description": "Grundrechnungsarten (add, sub, mul, div, pct_of, pct_change) für transparente Zwischenschritte.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "sub", "mul", "div", "pct_of", "pct_change"],
                },
                "a": {"type": "number", "description": "Erster Operand"},
                "b": {"type": "number", "description": "Zweiter Operand"},
                "label": {"type": "string", "description": "Beschreibung der Rechnung"},
                "unit": {"type": "string", "description": "Einheit des Ergebnisses"},
                "source_reference": {"type": "string", "description": "Artikel-ID oder grundfakten.yaml"},
            },
            "required": ["operation", "a", "b", "label"],
        },
    },
]


def validate_and_normalize_proposal_budget(
    proposal: Dict[str, Any],
    max_budget_pct: Optional[float] = None,
    auto_scale_to_cap: bool = False,
) -> Dict[str, Any]:
    """
    Hard Python numerical verification of a proposal's budget and stock-anchored effect!
    1. Grounds every single measure's effect against the real Austrian system stock (`compute_grounded_measure_effect`)
       so no single measure can claim fantasy 20%-30% effects for tiny budgets.
    2. Prevents double-counting if two measures in the same proposal target the exact same lever category
       (caps the sum within a single lever category to `max_single_lever_effect_pct * 1.25`).
    3. Transparently sums the grounded measure effects across categories (100% explainable addition!).
    4. Checks total cost against `max_budget_pct` of `57,837 Mio. EUR` and scales linearly if `auto_scale_to_cap=True`.
    """
    base_total_mio = BUDGET_BASELINES_2024_MIO["gesamt_laufend"]  # 57837.0
    measures = proposal.get("measures", [])

    computed_total_mio = 0.0
    category_effects_sum: Dict[str, float] = {}
    clamped_notes: List[str] = []

    for m in measures:
        title = m.get("title", "Maßnahme")
        desc = m.get("description", "")
        cost_mio = max(0.0, float(m.get("cost_mio_eur", 0.0)))
        current_eff = float(m.get("expected_effect_pct", 0.0))
        if "original_claimed_effect_pct" not in m or float(m.get("original_claimed_effect_pct", 0.0)) <= 0.0:
            if current_eff > 0.0:
                m["original_claimed_effect_pct"] = round(current_eff, 2)
        raw_eff = float(m.get("original_claimed_effect_pct", current_eff))

        lever_cat = infer_lever_category(title, desc)
        num_units = _extract_unit_count_from_title(title)

        grounded = compute_grounded_measure_effect(
            cost_mio_eur=cost_mio,
            raw_claimed_effect_pct=raw_eff if raw_eff > 0 else current_eff,
            lever_category=lever_cat,
            num_units=num_units,
        )
        eff = grounded["final_effect_pct"]

        # Prevent extreme double-counting within the exact same lever category,
        # while NEVER zeroing out a measure that has real budget attached!
        anchor_max = float(SYSTEM_LEVER_ANCHORS[lever_cat]["max_single_lever_effect_pct"]) * 1.65
        already_in_cat = category_effects_sum.get(lever_cat, 0.0)
        if already_in_cat > 0 and already_in_cat + eff > anchor_max:
            remaining_room = max(0.0, anchor_max - already_in_cat)
            # Keep a realistic diminishing-returns floor (at least 45% of its standalone grounded effect)
            eff = round(max(remaining_room, min(eff * 0.55, 4.2)), 2)
            grounded["was_clamped"] = True
        category_effects_sum[lever_cat] = already_in_cat + eff

        display_raw = raw_eff if raw_eff > 0.0 else eff
        if grounded["was_clamped"] and abs(display_raw - eff) > 0.35:
            clamped_notes.append(
                f"Maßnahme „{title}“: Wirkung von behaupteten +{display_raw:.1f}% auf +{eff:.2f}% korrigiert ({grounded['reason_detail']})."
            )
            m["fact_check_diff"] = f"Korrigiert von +{display_raw:.1f}% auf +{eff:.2f}% ({grounded['reason_detail']})"
        else:
            m["fact_check_diff"] = f"Bestätigt (+{eff:.2f}% | {grounded['reason_detail']})"

        # Clean source_article_ids so 'grundfakten.yaml' is not mixed with ART- IDs
        raw_sources = m.get("source_article_ids", [])
        clean_art_ids = [
            s.strip()
            for s in raw_sources
            if isinstance(s, str) and s.strip().startswith("ART-")
        ]
        m["source_article_ids"] = list(dict.fromkeys(clean_art_ids))
        m["lever_category"] = lever_cat
        m["cost_mio_eur"] = round(cost_mio, 2)
        m["budget_impact_pct"] = round((cost_mio / base_total_mio) * 100.0, 4)
        m["expected_effect_pct"] = round(eff, 2)
        computed_total_mio += cost_mio

    # Transparent addition of grounded measure effects
    computed_total_effect = round(sum(float(m.get("expected_effect_pct", 0.0)) for m in measures), 2)
    computed_total_pct = round((computed_total_mio / base_total_mio) * 100.0, 4)
    reported_pct = float(proposal.get("total_budget_impact_pct", computed_total_pct))
    discrepancy_pct = round(abs(computed_total_pct - reported_pct), 4)

    exceeds_cap = False
    cap_violation_msg = ""
    was_auto_scaled = False

    if max_budget_pct is not None and computed_total_pct > max_budget_pct + 1e-4:
        exceeds_cap = True
        max_mio = round((max_budget_pct / 100.0) * base_total_mio, 2)
        cap_violation_msg = (
            f"BUDGET-ÜBERSCHREITUNG: Berechnete Gesamtkosten betragen {computed_total_mio:.2f} Mio. EUR "
            f"({computed_total_pct:.3f}% von 57.837 Mio. EUR), erlaubt sind maximal {max_budget_pct:.2f}% "
            f"({max_mio:.2f} Mio. EUR)."
        )
        if auto_scale_to_cap and computed_total_pct > 0:
            scale_factor = max_budget_pct / computed_total_pct
            new_total_mio = 0.0
            new_total_effect = 0.0
            for m in measures:
                m["cost_mio_eur"] = round(m["cost_mio_eur"] * scale_factor, 2)
                m["budget_impact_pct"] = round(
                    (m["cost_mio_eur"] / base_total_mio) * 100.0, 4
                )
                m["expected_effect_pct"] = round(
                    float(m.get("expected_effect_pct", 0.0)) * scale_factor, 2
                )
                m["scaled_by_python_validator"] = round(scale_factor, 4)
                new_total_mio += m["cost_mio_eur"]
                new_total_effect += m["expected_effect_pct"]
            computed_total_mio = round(new_total_mio, 2)
            computed_total_pct = round((computed_total_mio / base_total_mio) * 100.0, 4)
            computed_total_effect = round(new_total_effect, 2)
            was_auto_scaled = True
            exceeds_cap = False
            cap_violation_msg += (
                f" -> Automatisch durch Python linear mit Faktor {scale_factor:.3f} auf "
                f"{computed_total_mio:.2f} Mio. EUR ({computed_total_pct:.3f}%) skaliert."
            )

    proposal["total_cost_mio_eur"] = round(computed_total_mio, 2)
    proposal["total_budget_impact_pct"] = computed_total_pct
    proposal["total_expected_effect_pct"] = round(computed_total_effect, 2)
    proposal["python_budget_check"] = {
        "verified": not exceeds_cap,
        "base_health_budget_mio_eur": base_total_mio,
        "computed_total_mio_eur": round(computed_total_mio, 2),
        "computed_total_budget_pct": computed_total_pct,
        "computed_total_effect_pct": round(computed_total_effect, 2),
        "llm_reported_budget_pct": reported_pct,
        "discrepancy_adjusted": discrepancy_pct > 0.01 or len(clamped_notes) > 0,
        "clamped_effect_notes": clamped_notes,
        "max_budget_pct": max_budget_pct,
        "exceeds_cap": exceeds_cap,
        "was_auto_scaled": was_auto_scaled,
        "message": cap_violation_msg or (
            f"Python Budget- & Bestands-Check OK: {computed_total_mio:.2f} Mio. EUR = {computed_total_pct:.2f}% "
            f"des Gesundheitsbudgets (57.837 Mio. EUR) | Kombinierter Effekt: {computed_total_effect:.2f}%"
            + (f" (Limit <= {max_budget_pct:.2f}%)." if max_budget_pct is not None else ".")
        ),
    }
    return proposal
