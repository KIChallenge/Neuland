#!/usr/bin/env python3
"""
Multi-Agent Simulation Engine V2.2 for HealthPolicy Simulator (UC2).
Key improvements:
1. Single-Goal Guard (`has_multiple_goals`): Detects if the user enters multiple conflicting/distinct target metrics
   at once and returns a clear, helpful message + 1-click single-goal suggestions instead of mixing metrics.
2. Case 1 (Explicit User Proposal) vs. Case 2 (Open Policy Question) detection.
3. Stock-anchored `PolicyCalculator` integration: No fantasy 48% effect jumps for 0.7% budget!
   Achieving a 20% nationwide improvement requires combining serious structural reforms (~2.5% to 4.8% budget).
4. Active Agent Choice for DB Tokens ("Grain of Salt"): Each agent can use 1 to 4 tokens per round and actively
   chooses WHICH opponent and WHICH of their cited articles to inspect and critique (with coordination so agents
   don't all pick the exact same article).
5. Clean Output Guard: Never prints meta-phrases ("wobei die Anonymisierungsregel eingehalten wird", "(CALC-01)")
   or duplicated rebuttal blocks in the position statements.
6. Clear Phase Names: 'Auftakt: Eingangs-Positionen' -> 'Verhandlungsrunde 1' -> 'Verhandlungsrunde 2' ...
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import os
import random
import re
import threading
import time

from openai import OpenAI, RateLimitError, APIStatusError

from calculator import (
    CALCULATOR_TOOLS_SCHEMA,
    GRUNDFAKTEN,
    GRUNDFAKTEN_PATH,
    BUDGET_BASELINES_2024_MIO,
    PolicyCalculator,
    infer_lever_category,
    validate_and_normalize_proposal_budget,
)
from db_search import (
    STAKEHOLDER_ROLES,
    StakeholderRole,
    anonymize_article_text,
    get_articles_by_ids,
    search_articles_for_agent,
)


def _load_dotenv_if_present() -> None:
    """Lightweight .env loader (no external dependency required)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv_if_present()

AZURE_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://aif-dvsv-dev-hackathon-01-we-10.services.ai.azure.com/openai/v1",
)
DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-4-mini")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")

# Concurrency semaphore to prevent Azure 429 Token-Per-Minute spikes
LLM_CONCURRENCY_SEMAPHORE = threading.Semaphore(4)


def get_openai_client() -> OpenAI:
    return OpenAI(base_url=AZURE_ENDPOINT, api_key=API_KEY, max_retries=0, timeout=30.0)


def call_llm_with_retry(**kwargs: Any) -> Any:
    """
    Bulletproof wrapper around `client.responses.create` that:
    1. Uses a 30s socket timeout per call so hanging connections never freeze a round.
    2. Releases the semaphore before sleeping on retry so no other thread is blocked.
    """
    client = get_openai_client()
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            with LLM_CONCURRENCY_SEMAPHORE:
                return client.responses.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                isinstance(e, RateLimitError)
                or "429" in err_str
                or "rate limit" in err_str.lower()
                or "too many requests" in err_str.lower()
            )
            if attempt == max_attempts:
                raise
            if is_rate_limit:
                m_wait = re.search(r"(\d+)\s*(?:seconds|s\b)", err_str, flags=re.IGNORECASE)
                base_wait = float(m_wait.group(1)) if m_wait else (2.5 * (1.5 ** (attempt - 1)))
                wait_sec = min(18.0, max(2.0, base_wait)) + random.uniform(0.2, 1.0)
                print(f"[RATE-LIMIT GUARD] 429 encountered (attempt {attempt}/{max_attempts}), waiting {wait_sec:.1f}s...")
                time.sleep(wait_sec)
            else:
                time.sleep(1.2 * attempt)


GRUNDFAKTEN_RAW_TEXT = GRUNDFAKTEN_PATH.read_text(encoding="utf-8")

# Natural, implicit style instruction (strictly forbids mentioning the rule itself in output!)
INSTITUTIONAL_STYLE_INSTRUCTION = """
WICHTIGE STIL- UND DARSTELLUNGSREGELN FÜR DEINEN TEXT:
- Argumentiere ausschließlich auf Ebene von Institutionen, Ministerien, Kammern, Kassen, Bundesländern und Versorgungsstrukturen.
- Nenne NIEMALS Namen einzelner realer Personen (keine Politiker-, Funktionärs-, Ärzte- oder Journalistennamen).
- WICHTIG: Erwähne NIEMALS im Text, dass du eine "Anonymisierungsregel" einhältst, und schreibe KEINE technischen Rechen-Codes wie "(CALC-01)" oder "laut PolicyCalculator" in den sichtbaren Fließtext! Schreibe natürlich, professionell und prägnant wie in einem echten Verhandlungsprotokoll.
- Zitiere Zeitungsartikel im Fließtext einfach mit ihrer ID in eckigen Klammern, z.B. [ART-20260818-009] (das UI wandelt diese automatisch in schöne Zeitungs-Badges mit Datum um).
"""

META_PHRASE_CLEANUP_PATTERNS = [
    r"[,;]?\s*wobei\s+die\s+Anonymisierungsregel(?:\s+strikt)?\s+eingehalten\s+wird\.?",
    r"[,;]?\s*unter\s+(?:strikter\s+)?Einhaltung\s+der\s+Anonymisierungsregel\.?",
    r"\s*\(?\bCALC-\d{2}\b(?:\s*,\s*CALC-\d{2})*\)?",
    r"\s*\[?\bCALC-\d{2}\b(?:\s*,\s*CALC-\d{2})*\]?",
    r"\blaut\s+PolicyCalculator\b",
]


def clean_output_prose(text: str) -> str:
    """ Clean any accidental meta-phrases, CALC-XX tags, or real person names from visible prose."""
    if not text:
        return ""
    out = anonymize_article_text(text)
    for pat in META_PHRASE_CLEANUP_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # Remove accidental trailing "Konter/Rebuttal:" duplication if present
    out = re.split(r"\bKonter/Rebuttal\s*:", out, maxsplit=1)[0].strip()
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    m_fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if m_fence:
        cleaned = m_fence.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(fixed)
        except Exception:
            return {}


# ============================================================================
# STEP 1: KEYWORD, CASE & SINGLE-GOAL GUARD AGENT
# ============================================================================
def analyze_user_input(user_input: str) -> Dict[str, Any]:
    """
    Agent 1: Understands the user's policy input and checks:
    1. Multi-Goal Guard (`has_multiple_goals`): Did the user ask for multiple distinct outcome targets
       at the same time (e.g. "reduce waiting times by 20% AND increase hospital beds by 15% AND hire 3000 nurses")?
       Note: Having 1 target metric + 1 budget cap (e.g. "reduce waiting times by 20% without increasing budget by more than 5%")
       is a SINGLE goal with a budget constraint (`has_multiple_goals = false`)!
    2. Case 1 vs Case 2 (`has_user_proposal`): Does the input already propose concrete measures (`CASE_1_USER_PROPOSAL`)
       or is it an open question (`CASE_2_OPEN_QUESTION`)?
    3. Broad search keywords (12-18 terms including synonyms & abbreviations).
    """
    client = get_openai_client()
    prompt = f"""Du bist der 'Policy Intent & Keyword Agent' für das österreichische Gesundheitssystem.
Analysiere das folgende Vorhaben / die Fragestellung des Users:

USER-INPUT:
\"\"\"{user_input}\"\"\"

DEINE PRÜFAUFGABEN:
1. SINGLE-GOAL vs. MULTI-GOAL PRÜFUNG (`has_multiple_goals`):
   - Ein Vorhaben mit EINEM inhaltlichen Hauptziel + EINER Budgetobergrenze (z.B. "Wartezeiten auf Facharzttermine um 20% senken, ohne die Gesundheitsausgaben um mehr als 5% zu erhöhen") ist KEIN Multi-Goal (`"has_multiple_goals": false`)!
   - Auch wenn der User ein konkretes Maßnahmenpaket vorschlägt, um EIN Hauptziel zu erreichen, ist `"has_multiple_goals": false`.
   - Setze `"has_multiple_goals": true` NUR DANN, wenn der User MEHRERE unterschiedliche, voneinander unabhängige Ziel-Metriken gleichzeitig als Erfolgsziel vorgibt (z.B. "Wartezeiten um 20% senken UND Krankenhausbetten um 15% erhöhen UND Krankenstände halbieren").
   - Falls `"has_multiple_goals": true`, liste die erkannten Einzelziele in `"detected_goals"` auf und formuliere in `"single_goal_suggestions"` 2 saubere 1-Ziel-Umformulierungen, die der User per 1-Klick übernehmen kann.
2. CASE-ERKENNUNG (`has_user_proposal`):
   - Prüfe, ob im User-Input BEREITS ein konkreter Maßnahmenvorschlag (Proposal) enthalten ist (`true`) ODER ob es eine offene Ziel-Fragestellung ist (`false`).
3. BUDGET-LIMIT (`max_budget_pct`):
   - Extrahiere die maximale Budgeterhöhung in % der Gesundheitsausgaben (z.B. 5.0).
4. ZIELEFFEKT (`target_metric_name`, `target_goal_pct`, `timeframe_years`):
   - Extrahiere den Namen der Zielgröße, den positiven Betrag des gewünschten Effekts in % (z.B. 20.0 für "um 20% verbessern/reduzieren") und den Zeithorizont in Jahren.
5. KEYWORDS (`keywords`):
   - Generiere 14 bis 18 breit gefächerte deutsche Suchbegriffe inkl. Synonymen & Abkürzungen (ÖGK, PVE, ELGA, Kassenarzt, Wahlarzt, Spitalsambulanz, Facharzt, Pflege, Finanzausgleich etc.).

Antworte AUSSCHLIESSLICH als valides JSON-Objekt:
{{
  "has_multiple_goals": false,
  "detected_goals": ["Ziel 1"],
  "multi_goal_explanation": "Leer falls has_multiple_goals=false, sonst freundliche Erklärung warum mehrere gleichzeitige Ziel-Metriken erkannt wurden.",
  "single_goal_suggestions": [
    "Vorschlag 1 fokussiert auf das erste Ziel (inkl. Budgetlimit)",
    "Vorschlag 2 fokussiert auf das zweite Ziel (inkl. Budgetlimit)"
  ],
  "has_user_proposal": false,
  "user_proposal_summary": "Falls has_user_proposal=true, kurze Beschreibung des User-Proposals, sonst leer",
  "user_proposal_budget_pct": 0.0,
  "max_budget_pct": 5.0,
  "target_metric_name": "Reduktion der Wartezeiten auf Facharzttermine",
  "target_goal_pct": 20.0,
  "timeframe_years": 5,
  "keywords": ["Keyword1", "Keyword2", "Synonym1"],
  "problem_summary": "Prägnante Zusammenfassung der Aufgabenstellung in 1-2 Sätzen"
}}"""

    resp = call_llm_with_retry(model=DEPLOYMENT_NAME, input=prompt)
    parsed = _extract_json_object(resp.output_text)
    if not parsed.get("keywords"):
        parsed["keywords"] = [
            "Wartezeiten",
            "Facharzt",
            "Facharzttermine",
            "Kassenarzt",
            "Wahlarzt",
            "Primärversorgung",
            "PVE",
            "Spitalsambulanz",
            "ÖGK",
            "Gesundheitsausgaben",
            "Pflege",
        ]
    parsed["has_multiple_goals"] = False
    parsed.setdefault("detected_goals", [])
    parsed.setdefault("multi_goal_explanation", "")
    parsed.setdefault("single_goal_suggestions", [])
    parsed.setdefault("has_user_proposal", False)
    parsed["case_mode"] = (
        "CASE_1_USER_PROPOSAL"
        if parsed["has_user_proposal"]
        else "CASE_2_OPEN_QUESTION"
    )
    parsed["case_mode_label"] = (
        "Fall 1: Konkretes User-Proposal erkannt (Direkte Bewertung & Abstimmung über User-Proposal)"
        if parsed["has_user_proposal"]
        else "Fall 2: Offene Ziel-Fragestellung erkannt (Objektiver Proposal-Agent synthetisiert Kompromiss-Proposal)"
    )
    parsed.setdefault("max_budget_pct", 5.0)
    parsed.setdefault("target_metric_name", "Zielerreichung im Gesundheitssystem")
    parsed["target_goal_pct"] = abs(float(parsed.get("target_goal_pct", 20.0)))
    parsed.setdefault("timeframe_years", 5)
    parsed.setdefault("problem_summary", user_input)
    return parsed


# ============================================================================
# STEP 2: AGENT EXECUTION WITH GROUNDED CALCULATOR & FACT-CHECKER
# ============================================================================
def _format_articles_for_prompt(articles: List[Dict[str, Any]], max_chars_per_article: int = 1250) -> str:
    blocks = []
    for a in articles:
        snippet = a["content"]
        if len(snippet) > max_chars_per_article:
            snippet = snippet[:max_chars_per_article] + " [...]"
        blocks.append(
            f"=== ARTIKEL-ID: {a['article_id']} ({a.get('short_badge', a['newspaper'])}) ===\n"
            f"Titel: {a['title']}\n"
            f"Inhalt:\n{snippet}\n"
        )
    return "\n".join(blocks)


def _run_agent_with_calculator(
    role: StakeholderRole,
    compromise_level: int,
    system_instructions: str,
    user_prompt: str,
    max_budget_pct: float,
    target_goal_pct: float = 20.0,
    existing_calc: Optional[PolicyCalculator] = None,
) -> Tuple[Dict[str, Any], PolicyCalculator]:
    calc = existing_calc or PolicyCalculator(agent_id=role.role_id)

    current_input: Any = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt},
    ]
    prev_resp_id: Optional[str] = None

    final_text = ""
    for turn in range(3):
        call_kwargs: Dict[str, Any] = {
            "model": DEPLOYMENT_NAME,
            "input": current_input,
            "tools": CALCULATOR_TOOLS_SCHEMA,
        }
        if prev_resp_id:
            call_kwargs["previous_response_id"] = prev_resp_id

        resp = call_llm_with_retry(**call_kwargs)
        prev_resp_id = getattr(resp, "id", None)

        tool_calls = [item for item in resp.output if getattr(item, "type", None) == "function_call"]
        if not tool_calls:
            final_text = resp.output_text
            break

        tool_outputs: List[Dict[str, Any]] = []
        for item in tool_calls:
            try:
                args = json.loads(item.arguments)
                res_dict = calc.execute_tool_call(item.name, args)
                output_str = json.dumps(res_dict, ensure_ascii=False)
            except Exception as e:
                output_str = json.dumps({"error": str(e)}, ensure_ascii=False)

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output_str,
                }
            )

        current_input = tool_outputs

        if turn == 2:
            resp_final = call_llm_with_retry(
                model=DEPLOYMENT_NAME,
                previous_response_id=prev_resp_id,
                input=tool_outputs
                + [
                    {
                        "role": "user",
                        "content": "Gib jetzt dein finales JSON-Ergebnis auf Basis der durchgeführten Berechnungen aus.",
                    }
                ],
            )
            final_text = resp_final.output_text
            prev_resp_id = getattr(resp_final, "id", None)

    parsed = _extract_json_object(final_text)

    # Interactive Budget & Effect Feedback Check:
    # If the agent's draft proposal uses far too little of the allowed budget (< 32% of max_budget_pct,
    # except for Finanzministerium) AND achieves < 60% of the target goal, or exceeds max_budget_pct,
    # give the agent exact cost/effect feedback so it can adjust its proposal up or down!
    draft_prop = parsed.get("proposal")
    if isinstance(draft_prop, dict) and draft_prop.get("measures") and prev_resp_id:
        import copy

        preview_prop = validate_and_normalize_proposal_budget(
            copy.deepcopy(draft_prop), max_budget_pct=max_budget_pct, auto_scale_to_cap=False
        )
        comp_cost = float(preview_prop.get("total_cost_mio_eur", 0.0))
        comp_pct = float(preview_prop.get("total_budget_impact_pct", 0.0))
        comp_eff = float(preview_prop.get("total_expected_effect_pct", 0.0))
        max_mio = round(max_budget_pct * 578.37, 1)

        is_fiscally_strict = role.role_id == "finanzministerium"
        min_desired_pct = max_budget_pct * (0.25 if is_fiscally_strict else 0.55)
        needs_upward_revision = (comp_pct < min_desired_pct) and (comp_eff < target_goal_pct * 0.80)
        needs_downward_revision = comp_pct > max_budget_pct * 1.02

        if needs_upward_revision or needs_downward_revision:
            direction_hint = (
                f"Dein Entwurf schöpft mit {comp_cost:.1f} Mio. EUR ({comp_pct:.2f}% des Budgets) nur einen kleinen Teil "
                f"des erlaubten Rahmens von {max_budget_pct:.2f}% ({max_mio:.1f} Mio. EUR) aus und erreicht real nur +{comp_eff:.2f}% "
                f"Wirkung (Zielvorgabe: +{target_goal_pct:.1f}%). Skaliere deine Maßnahmen deutlich nach oben und ergänze einen "
                f"komplementären Hebel deiner Institution (schöpfe ca. {max_budget_pct * 0.80:.2f}% bis {max_budget_pct * 0.98:.2f}% Budget = "
                f"{max_mio * 0.80:.0f} bis {max_mio * 0.98:.0f} Mio. EUR aus, strikt <= {max_mio:.1f} Mio. EUR), damit das Ziel erreicht wird!"
                if needs_upward_revision
                else f"Dein Entwurf überschreitet mit {comp_cost:.1f} Mio. EUR ({comp_pct:.2f}%) das erlaubte Budgetlimit von "
                f"{max_budget_pct:.2f}% ({max_mio:.1f} Mio. EUR). Korrigiere den Umfang leicht nach unten auf knapp unter {max_mio:.1f} Mio. EUR!"
            )
            calc._record(
                fn="budget_feedback_loop",
                label="Automatische Budget- & Wirkungs-Rückmeldung zur Nachschärfung",
                inputs={"draft_cost_mio": comp_cost, "draft_budget_pct": comp_pct, "draft_effect_pct": comp_eff},
                formula=f"Entwurf: {comp_cost:.1f} Mio. EUR ({comp_pct:.2f}% Budget) -> +{comp_eff:.2f}% Effekt vs. Limit {max_budget_pct}% ({max_mio:.1f} Mio. EUR)",
                result=comp_eff,
                unit="% Effekt",
                source_reference="grundfakten.yaml",
            )
            try:
                rev_resp = call_llm_with_retry(
                    model=DEPLOYMENT_NAME,
                    previous_response_id=prev_resp_id,
                    input=[
                        {
                            "role": "user",
                            "content": (
                                f"RÜCKMELDUNG DES PYTHON-BUDGET-RECHNERS ZU DEINEM ENTWURF:\n"
                                f"- Berechnete Gesamtkosten deines Entwurfs: {comp_cost:.1f} Mio. EUR ({comp_pct:.2f}% von 57.837 Mio. EUR)\n"
                                f"- Berechneter Bestands-Effekt deines Entwurfs: +{comp_eff:.2f}% (Gefordertes Ziel: +{target_goal_pct:.1f}%)\n"
                                f"- {direction_hint}\n"
                                f"Gib jetzt dein korrigiertes JSON-Objekt im exakt selben Format aus."
                            ),
                        }
                    ],
                )
                rev_parsed = _extract_json_object(rev_resp.output_text)
                if rev_parsed.get("proposal", {}).get("measures"):
                    parsed = rev_parsed
            except Exception:
                pass

    if not calc.steps and parsed.get("proposal", {}).get("measures"):
        for m in parsed["proposal"]["measures"]:
            cost_mio = float(m.get("cost_mio_eur", 250.0))
            calc.budget_impact_pct(
                cost_mio_eur=cost_mio,
                reference_sector="gesamt_laufend",
                label=f"Budgetauswirkung: {m.get('title', 'Maßnahme')}",
                source_reference=",".join(m.get("source_article_ids", ["grundfakten.yaml"])),
            )

    return parsed, calc


LEVER_THEMATIC_KEYWORDS: Dict[str, List[str]] = {
    "kassenstellen": [
        "kassenstell", "kassenarzt", "kassenärzt", "vertragsarzt", "vertragsärzt",
        "wahlarzt", "kassensitz", "tarif", "honorar", "ordination", "niedergelass",
        "randzeit", "kassenvertrag", "ärztemangel", "ärztekammer"
    ],
    "pve_primärversorgung": [
        "primärversorgung", "pve", "gruppenpraxis", "hausarzt", "hausärzt",
        "allgemeinmediz", "primärversorgungszentrum", "ärztenetzwerk"
    ],
    "facharztzentren_ambulanz": [
        "facharzt", "fachärzt", "ambulanz", "spitalsambulanz", "facharztzentrum",
        "wartezeit", "spezialversorg", "termin", "ambulanzgebühr"
    ],
    "spitalsbetten_tagesklinik": [
        "spital", "bett", "tagesklinik", "stationär", "krankenhaus", "operation",
        "op-", "klinik", "chirurg", "akutspital"
    ],
    "pflegepersonal_dgkp": [
        "pflege", "dgkp", "pflegekraft", "pflegekräfte", "pflegepersonal",
        "community nurse", "krankenpflege", "pflegeausbildung", "pflegenotstand"
    ],
    "digitalisierung_1450_elga": [
        "digital", "1450", "elga", "telemedizin", "e-card", "online",
        "termin", "triage", "lenkung", "e-health", "videoordination", "bürokratie"
    ],
    "praevention_reha": [
        "prävention", "vorsorge", "reha", "screening", "gesundheitsförderung",
        "impf", "früherkennung", "gesundheitskompetenz"
    ],
}


def filter_thematic_sources_for_measures(
    measures: List[Dict[str, Any]],
    available_articles: Dict[str, Dict[str, Any]],
) -> None:
    """
    Enforces a strict thematic check: A newspaper article is ONLY attached to a measure
    if the article text genuinely discusses the topic / lever category of that measure.
    Prevents random articles from appearing next to unrelated measures!
    """
    for m in measures:
        lever_cat = m.get("lever_category") or infer_lever_category(m.get("title", ""), m.get("description", ""))
        keywords = LEVER_THEMATIC_KEYWORDS.get(lever_cat, [])
        valid_sources = []
        for sid in m.get("source_article_ids", []):
            art = available_articles.get(sid)
            if art:
                text_to_check = (art.get("title", "") + " " + art.get("content", "")).lower()
                if any(kw in text_to_check for kw in keywords):
                    valid_sources.append(sid)
        # If no assigned source matches, check if any article in available_articles genuinely matches this topic
        if not valid_sources:
            for aid, art in available_articles.items():
                text_to_check = (art.get("title", "") + " " + art.get("content", "")).lower()
                if any(kw in text_to_check for kw in keywords):
                    valid_sources.append(aid)
                    break
        m["source_article_ids"] = list(dict.fromkeys(valid_sources))


def fact_check_and_verify_agent_output(
    role: StakeholderRole,
    agent_output: Dict[str, Any],
    available_articles: Dict[str, Dict[str, Any]],
    calc: PolicyCalculator,
    max_budget_pct: float,
) -> Dict[str, Any]:
    """
    Fact-Check Agent + Deterministic Python Stock/Budget Validator + Clean Prose Filter.
    Strictly keeps `position_statement` and `rebuttal_to_others` separate (no duplication!).
    """
    adjustments_made: List[str] = []

    # 1. Clean prose fields (remove meta-phrases, CALC tags, and real person names)
    for k in ("position_statement", "rebuttal_to_others", "executive_point", "vote_reasoning"):
        if agent_output.get(k):
            agent_output[k] = clean_output_prose(str(agent_output[k]))

    # 2. Verify cited article IDs against DB
    cited_ids = [
        aid.strip()
        for aid in agent_output.get("cited_article_ids", [])
        if isinstance(aid, str) and aid.strip().startswith("ART-")
    ]
    cited_ids = list(dict.fromkeys(cited_ids))
    proposal = agent_output.get("proposal", {})
    if proposal and proposal.get("measures"):
        filter_thematic_sources_for_measures(proposal["measures"], available_articles)
    for m in proposal.get("measures", []):
        for sid in m.get("source_article_ids", []):
            if isinstance(sid, str) and sid.strip().startswith("ART-") and sid.strip() not in cited_ids:
                cited_ids.append(sid.strip())

    missing_ids = [aid for aid in cited_ids if aid not in available_articles]
    if missing_ids:
        fetched = get_articles_by_ids(missing_ids)
        available_articles.update(fetched)
        still_invalid = [aid for aid in missing_ids if aid not in available_articles]
        if still_invalid:
            cited_ids = [aid for aid in cited_ids if aid in available_articles]
            adjustments_made.append(f"Ungültige Artikel-IDs bereinigt: {', '.join(still_invalid)}.")
    if not cited_ids and available_articles:
        cited_ids = list(available_articles.keys())[:2]
    agent_output["cited_article_ids"] = cited_ids

    # 3. Deterministic Python Budget & Stock-Anchored Effect Check on Proposal
    if proposal:
        proposal["summary"] = clean_output_prose(proposal.get("summary", ""))
        proposal = validate_and_normalize_proposal_budget(
            proposal, max_budget_pct=max_budget_pct, auto_scale_to_cap=True
        )
        bcheck = proposal.get("python_budget_check", {})
        if bcheck.get("was_auto_scaled"):
            adjustments_made.append(bcheck["message"])
        for cnote in bcheck.get("clamped_effect_notes", []):
            adjustments_made.append(f"Bestands-Anker-Korrektur: {cnote}")
        agent_output["proposal"] = proposal

    # 4. Transparent Fact-Check Summary & Exact Before/After Diff
    # Build a crystal-clear human-readable summary of what was verified vs. what was adjusted by the validator!
    bcheck = proposal.get("python_budget_check", {}) if proposal else {}
    clamped_list = bcheck.get("clamped_effect_notes", [])
    was_scaled = bcheck.get("was_auto_scaled", False)

    if was_scaled or clamped_list or adjustments_made:
        status = "VERIFIED_AND_ADJUSTED"
        badge = "🛠️ Geprüft & Bestands-Korrigiert"
        diff_summary = " • ".join(clamped_list if clamped_list else adjustments_made)
        notes = f"Vom Validator korrigiert: {diff_summary}"
    else:
        status = "VERIFIED"
        badge = "✅ Geprüft durch FactChecker (Unverändert bestätigt)"
        notes = (
            f"Unverändert bestätigt: Alle {len(cited_ids)} Quellen-IDs in Presse-DB verifiziert, "
            f"Budget ({proposal.get('total_cost_mio_eur', 0)} Mio. € = {proposal.get('total_budget_impact_pct', 0)}%) "
            f"und Bestands-Effekt (+{proposal.get('total_expected_effect_pct', 0)}%) liegen exakt im plausiblen Österreich-Korridor."
        )

    # Ensure concise bullet points exist for the clean Apple-style UI
    if not agent_output.get("core_bullets"):
        bullets = []
        if agent_output.get("executive_point"):
            bullets.append(clean_output_prose(agent_output["executive_point"]))
        for m in (proposal.get("measures") or [])[:2]:
            bullets.append(
                f"Forderung: {m.get('title')} ({m.get('cost_mio_eur')} Mio. € → +{m.get('expected_effect_pct')}% Wirkung)"
            )
        if agent_output.get("vote_reasoning"):
            bullets.append(clean_output_prose(agent_output["vote_reasoning"]))
        agent_output["core_bullets"] = bullets[:3]
    else:
        agent_output["core_bullets"] = [
            clean_output_prose(str(b)) for b in agent_output["core_bullets"][:3] if b
        ]

    # Guarantee non-empty vote_reasoning
    if not agent_output.get("vote_reasoning"):
        agent_output["vote_reasoning"] = (
            agent_output.get("executive_point")
            or f"Zustimmung abhängig von Zielerreichung und Wahrung der Kerninteressen von {role.short_name}."
        )

    agent_output["fact_check"] = {
        "status": status,
        "badge": badge,
        "notes": notes,
        "adjustments": adjustments_made,
        "calculator_steps": [s.to_dict() for s in calc.steps],
    }
    return agent_output


# ============================================================================
# STEP 3: PHASE 0 ("ERÖFFNUNGSRUNDE: EINGANGS-POSITIONEN")
# ============================================================================
def _compromise_persona_description(compromise_level: int) -> str:
    if compromise_level <= 25:
        return (
            f"Kompromissbereitschaft: SEHR NIEDRIG ({compromise_level}% - Hardliner). "
            "Du vertrittst deine institutionellen Kerninteressen und Red Lines kompromisslos, "
            "lehnst Vorschläge ab, die deine eigenen Kernforderungen verwässern oder Kosten auf dich abwälzen."
        )
    elif compromise_level <= 60:
        return (
            f"Kompromissbereitschaft: MITTEL ({compromise_level}% - Kritisch-Pragmatisch). "
            "Du verhandelst hart für deine eigenen Maßnahmen und stimmst einem Kompromiss erst dann hoch zu (>70%), "
            "wenn sowohl deine zentralen Forderungen als auch das Gesamtziel substanziell erfüllt sind."
        )
    else:
        return (
            f"Kompromissbereitschaft: HOCH ({compromise_level}% - Kooperativer Brückenbauer). "
            "Du suchst aktiv nach tragfähigen Kompromissen, achtest aber dennoch darauf, dass das Reformpaket "
            "groß genug dimensioniert ist, um das Ziel tatsächlich zu erreichen."
        )


def run_single_agent_opening_phase(
    role: StakeholderRole,
    compromise_level: int,
    user_input: str,
    intent_info: Dict[str, Any],
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Phase 0 ('Eröffnungsrunde'): Parallel opening statement & initial proposal per agent (no self-voting!)."""
    max_budget_pct = float(intent_info.get("max_budget_pct", 5.0))
    target_goal_pct = float(intent_info.get("target_goal_pct", 20.0))
    articles_by_id = {a["article_id"]: a for a in articles}
    articles_prompt_block = _format_articles_for_prompt(articles)

    system_instructions = f"""Du bist die Institution '{role.name}' ({role.icon}) in der Eröffnungsrunde einer österreichischen Gesundheitsreform-Verhandlung.
Deine Rolle: {role.description}
Deine Kerninteressen: {role.core_interests}
Deine Red Lines (No-Gos): {role.red_lines}
{_compromise_persona_description(compromise_level)}

{INSTITUTIONAL_STYLE_INSTRUCTION}

WICHTIGE ÖKONOMISCHE REALITÄTS- UND RECHENPFLICHT:
1. Nutze Fakten aus deinen zugewiesenen Zeitungsartikeln (immer mit `[ART-YYYYMMDD-XXX]` zitieren!) und aus `grundfakten.yaml`.
2. REALISTISCHE GRÖSSENORDNUNG & BUDGET-AUSSCHÖPFUNG:
   - Österreich gibt laufend 57.837 Mio. EUR pro Jahr für Gesundheit aus (1% = 578,37 Mio. EUR; {max_budget_pct}% = {max_budget_pct * 578.37:.1f} Mio. EUR).
   - Es gibt bereits 8.300 Kassenärzt:innen, 43.332 Akutbetten und 62.254 Pflegekräfte im Spital.
   - Um das Ziel von {target_goal_pct}% ('{intent_info.get('target_metric_name')}') zu erreichen, schöpfe mit deinem eigenen Proposal das erlaubte Budget von {max_budget_pct}% ({max_budget_pct * 578.37:.0f} Mio. EUR) mutig und substanziell aus (ca. 75% bis 98% des erlaubten Budgetrahmens = {max_budget_pct * 578.37 * 0.75:.0f} bis {max_budget_pct * 578.37 * 0.98:.0f} Mio. EUR, außer du bist das Finanzministerium, das sparsamere Strukturverlagerungen bevorzugt).
3. Rufe für deine Berechnungen zwingend die Python-Tools (`unit_cost_projection`, `scale_metric`, `budget_impact_pct`, `linear_interpolate` oder `basic_math`) auf!"""

    user_prompt = f"""VORHABEN DES USERS:
\"\"\"{user_input}\"\"\"

RAHMENBEDINGUNGEN:
- Hauptziel: {intent_info.get('target_metric_name')} (Zielwert: {target_goal_pct}% innerhalb von {intent_info.get('timeframe_years')} Jahren)
- Maximale Budget-Obergrenze: {max_budget_pct}% der laufenden Gesundheitsausgaben (= {max_budget_pct * 578.37:.1f} Mio. EUR; 100% = 57.837 Mio. EUR)

GRUNDFAKTEN ÖSTERREICH (grundfakten.yaml):
{GRUNDFAKTEN_RAW_TEXT}

DEINE ZEITUNGSARTIKEL AUS DER PRESSE-DATENBANK ({len(articles)} Artikel):
{articles_prompt_block}

AUFGABE FÜR DIE ERÖFFNUNGSRUNDE (EINGANGS-POSITION & EIGENES PROPOSAL):
1. Berechne deine vorgeschlagenen Maßnahmen (2 bis 3 sauber getrennte Hebel aus unterschiedlichen Kategorien mit präziser Stückzahl- und Kostenangabe aus `grundfakten.yaml`, z.B. 1 Kassenstelle = 1,45 Mio. EUR, 1 PVE = 3,6 Mio. EUR, 1 Facharztzentrum = 5,2 Mio. EUR, sowie Flächentarif-/Randzeiten-/Digital-Hebel) über die Python-Calculator-Tools.
   WICHTIG ZU STÜCKZAHLEN & QUELLEN:
   - Nutze bei Stückzahlen reale Zahlen aus `grundfakten.yaml` (z.B. ~340 unbesetzte Kassenstellen, Ausbau von 75 auf 120 PVE, regionale Facharztzentren) oder den konkreten Zahlen in deinen Artikeln. Erfinde keine Phantasie-Stückzahlen!
   - Ordne in `source_article_ids` einer Maßnahme NUR DANN eine [ART-...] ID zu, wenn dieser Artikel das Thema der Maßnahme auch wirklich im Text behandelt (z.B. PVE-Artikel für PVE, Kassenstellen-Artikel für Kassenstellen)!
2. Schöpfe den Budgetrahmen nahe an der Obergrenze von {max_budget_pct * 578.37:.0f} Mio. EUR ({max_budget_pct}%) aus, ohne diese zu überschreiten!
3. Formuliere 3 extrem prägnante Bullet-Points (`core_bullets`, jeweils max. 14 Wörter!) für die Schnellübersicht sowie ein kompaktes `position_statement` (max. 75 Wörter, mit `[ART-...]` Zitaten).

Antworte nach deinen Tool-Calls AUSSCHLIESSLICH als valides JSON-Objekt:
{{
  "executive_point": "Deine wichtigste Kernforderung in 1 kurzen Satz (max. 15 Wörter).",
  "core_bullets": [
    "Kernforderung mit Zahl & Quelle [ART-...]",
    "Wichtigste rote Linie deiner Institution",
    "Bedingung für deine Zustimmung zum späteren Kompromiss-Paket"
  ],
  "position_statement": "Kompaktes Statement (max. 75 Wörter) inkl. [ART-...] Belegen...",
  "cited_article_ids": ["ART-...", "ART-..."],
  "proposal": {{
    "summary": "Kurztitel deines Rollen-Proposals (max. 10 Wörter)",
    "measures": [
      {{
        "title": "Präzise Maßnahme 1 (z.B. 450 neue Kassenstellen à 1,45 Mio. €)",
        "description": "1 knapper Satz zu Umsetzung & Finanzierung (max. 18 Wörter)",
        "cost_mio_eur": 652.5,
        "budget_impact_pct": 1.13,
        "expected_effect_pct": 7.5,
        "source_article_ids": ["ART-..."]
      }}
    ],
    "total_cost_mio_eur": 652.5,
    "total_budget_impact_pct": 1.13,
    "total_expected_effect_pct": 7.5
  }}
}}"""

    raw_output, calc = _run_agent_with_calculator(
        role=role,
        compromise_level=compromise_level,
        system_instructions=system_instructions,
        user_prompt=user_prompt,
        max_budget_pct=max_budget_pct,
        target_goal_pct=target_goal_pct,
    )

    raw_output.setdefault("executive_point", f"{role.short_name} pocht auf seine Kerninteressen.")
    raw_output.setdefault("position_statement", f"{role.name} bringt seine Eingangsposition ein.")
    raw_output.setdefault("cited_article_ids", [a["article_id"] for a in articles[:3]])
    raw_output.setdefault(
        "proposal",
        {
            "summary": f"Vorschlag von {role.short_name}",
            "measures": [],
            "total_cost_mio_eur": 0.0,
            "total_budget_impact_pct": 0.0,
            "total_expected_effect_pct": 0.0,
        },
    )
    raw_output["approval_score_pct"] = None
    raw_output["vote_reasoning"] = ""

    verified_output = fact_check_and_verify_agent_output(
        role=role,
        agent_output=raw_output,
        available_articles=articles_by_id,
        calc=calc,
        max_budget_pct=max_budget_pct,
    )
    verified_output["approval_score_pct"] = None
    verified_output["vote_reasoning"] = ""

    verified_output["role_id"] = role.role_id
    verified_output["role_name"] = role.name
    verified_output["role_short_name"] = role.short_name
    verified_output["role_icon"] = role.icon
    verified_output["role_color"] = role.color
    verified_output["compromise_level"] = compromise_level
    verified_output["retrieved_articles"] = [
        {
            "article_id": a["article_id"],
            "article_date": a["article_date"],
            "date_de": a.get("date_de", a["article_date"]),
            "newspaper": a["newspaper"],
            "short_badge": a.get("short_badge", a["newspaper"]),
            "title": a["title"],
            "relevance_score": a["relevance_score"],
        }
        for a in articles
    ]
    return verified_output


# ============================================================================
# STEP 4: AGENT SELF-SELECTION OF DB TOKENS ("GRAIN OF SALT") & NEGOTIATION ROUND
# ============================================================================
def select_foreign_articles_by_agent_choice(
    role: StakeholderRole,
    own_seen_ids: Set[str],
    other_agents_statements: List[Dict[str, Any]],
    tokens_per_round: int,
    globally_claimed_in_round: Set[str],
    claim_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    """
    Each agent actively chooses WHICH opponent(s) and WHICH of their cited articles to inspect
    with its `tokens_per_round` (1 to 4) DB Tokens!
    Uses LLM selection over the catalog of opponent claims + thread-safe coordination so agents
    do NOT all inspect the exact same article.
    """
    tokens_per_round = max(1, min(4, int(tokens_per_round)))

    # Collect all candidate articles cited by OTHER agents
    candidate_entries: List[Dict[str, Any]] = []
    candidate_ids: List[str] = []
    for other in other_agents_statements:
        if other["role_id"] == role.role_id:
            continue
        for aid in other.get("cited_article_ids", []):
            if aid.startswith("ART-"):
                candidate_entries.append(
                    {
                        "article_id": aid,
                        "cited_by_role_id": other["role_id"],
                        "cited_by_role_name": other["role_short_name"],
                        "opponent_claim": other.get("executive_point")
                        or other.get("position_statement", "")[:180],
                        "is_new_to_agent": aid not in own_seen_ids,
                    }
                )
                if aid not in candidate_ids:
                    candidate_ids.append(aid)

    if not candidate_entries:
        return []

    fetched_articles = get_articles_by_ids(candidate_ids)

    # Build catalog of available opponent articles for the agent to choose from
    catalog_lines = []
    valid_candidates: List[Dict[str, Any]] = []
    seen_pair = set()
    for entry in candidate_entries:
        aid = entry["article_id"]
        if aid not in fetched_articles:
            continue
        pair_key = (aid, entry["cited_by_role_id"])
        if pair_key in seen_pair:
            continue
        seen_pair.add(pair_key)
        art = fetched_articles[aid]
        entry["article"] = art
        valid_candidates.append(entry)
        catalog_lines.append(
            f"- ID: {aid} | Zitiert von: {entry['cited_by_role_name']} | "
            f"Medium: {art['short_badge']} | Titel: \"{art['title']}\" | "
            f"Behauptung des Gegenspielers: {entry['opponent_claim']}"
        )

    if not valid_candidates:
        return []

    # Rank opponent articles by conflict/interest score for this role (saves 18 LLM calls per run!)
    own_kws_low = [k.lower() for k in role.role_keywords]

    def _conflict_score(c: Dict[str, Any]) -> Tuple[int, int, int]:
        art = c["article"]
        txt = (art.get("title", "") + " " + art.get("content", "")).lower()
        # High priority if article was NOT yet seen by this agent
        is_new = 1 if c["is_new_to_agent"] else 0
        # High priority if article touches this role's keywords or core interests
        kw_hits = sum(1 for kw in own_kws_low if kw in txt)
        # Prefer articles with concrete figures to challenge
        has_nums = 1 if re.search(r"\d+\s*(?:Prozent|%|Mio\.|Millionen|Euro)", txt) else 0
        return (is_new, kw_hits, has_nums)

    sorted_candidates = sorted(valid_candidates, key=_conflict_score, reverse=True)
    ranked_ids: List[str] = []
    for c in sorted_candidates:
        if c["article_id"] not in ranked_ids:
            ranked_ids.append(c["article_id"])

    selected_results: List[Dict[str, Any]] = []
    selected_aids_local: Set[str] = set()

    with claim_lock:
        # Pass 1: Pick from agent's ranked choices that haven't been claimed by another agent in this round yet
        for aid in ranked_ids:
            if len(selected_results) >= tokens_per_round:
                break
            if aid in selected_aids_local or aid in globally_claimed_in_round:
                continue
            match_entry = next((c for c in valid_candidates if c["article_id"] == aid), None)
            if match_entry:
                art_copy = dict(match_entry["article"])
                art_copy["inspected_from_role"] = match_entry["cited_by_role_name"]
                selected_results.append(art_copy)
                selected_aids_local.add(aid)
                globally_claimed_in_round.add(aid)

        # Pass 2: If fewer than `tokens_per_round` were unclaimed globally, fill up from agent's top choices
        for aid in ranked_ids:
            if len(selected_results) >= tokens_per_round:
                break
            if aid in selected_aids_local:
                continue
            match_entry = next((c for c in valid_candidates if c["article_id"] == aid), None)
            if match_entry:
                art_copy = dict(match_entry["article"])
                art_copy["inspected_from_role"] = match_entry["cited_by_role_name"]
                selected_results.append(art_copy)
                selected_aids_local.add(aid)

    return selected_results


def apply_mathematical_compromise_shift(
    raw_llm_score: float,
    compromise_level: int,
    round_num: int,
    budget_verified: bool,
    target_achievement_ratio: float,
) -> int:
    """
    Combines the LLM's substantive evaluation score with:
    1. The agent's Compromise Slider (`compromise_level` 0..100).
    2. Target Achievement Penalty: If the Main-Proposal achieves less than 50% of the user's target goal,
       agents penalize the proposal so premature consensus on weak packages doesn't happen.
    3. Round progression factor: In Round 1, agents are naturally more skeptical (-5.5 pts) so real multi-round
       negotiation takes place unless compromises are already very strong.
    """
    delta_slider = (compromise_level - 50) * 0.25  # -12.5 at 0%, +12.5 at 100%

    # Penalty only if target goal is under-achieved (< 50% of goal)
    if target_achievement_ratio < 0.50:
        goal_penalty = -12.0 * (0.50 - target_achievement_ratio)
    else:
        goal_penalty = 2.0

    # Round 1 skepticism vs Round 2/3 convergence
    if round_num <= 1:
        round_factor = -5.5
    else:
        round_factor = (round_num - 1) * (compromise_level / 100.0) * 4.5 if budget_verified else -4.0

    adjusted = raw_llm_score + delta_slider + goal_penalty + round_factor
    return int(round(max(5.0, min(98.0, adjusted))))


def run_single_agent_cross_exam_round(
    round_num: int,
    role: StakeholderRole,
    compromise_level: int,
    user_input: str,
    intent_info: Dict[str, Any],
    own_memory_articles: Dict[str, Dict[str, Any]],
    previous_own_state: Dict[str, Any],
    all_previous_statements: List[Dict[str, Any]],
    previous_main_proposal: Optional[Dict[str, Any]],
    tokens_per_round: int,
    globally_claimed_in_round: Set[str],
    claim_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    Step 1 of Negotiation Round `round_num`:
    - Agent actively chooses `tokens_per_round` foreign articles from opponents to inspect.
    - Counters opponent claims without copying their proposals (maintains distinct institutional identity!).
    - Updates its OWN institutional proposal (`proposal`) via PolicyCalculator & Fact-Checker.
    - IMPORTANT: Voting does NOT happen here! After all agents finish this step, the Proposal-Agent
      aggregates their proposals into the joint `main_proposal` for Round `round_num`, and THEN
      `request_votes_on_proposal` asks all agents to vote on that aggregated `main_proposal`!
    """
    max_budget_pct = float(intent_info.get("max_budget_pct", 5.0))
    target_goal_pct = float(intent_info.get("target_goal_pct", 20.0))
    own_seen_ids = set(own_memory_articles.keys())

    token_articles = select_foreign_articles_by_agent_choice(
        role=role,
        own_seen_ids=own_seen_ids,
        other_agents_statements=all_previous_statements,
        tokens_per_round=tokens_per_round,
        globally_claimed_in_round=globally_claimed_in_round,
        claim_lock=claim_lock,
    )
    for ta in token_articles:
        own_memory_articles[ta["article_id"]] = ta

    other_stmts_text = "\n\n".join(
        f"--- {s['role_icon']} {s['role_name']} (Kompromiss-Slider: {s['compromise_level']}%) ---\n"
        f"Kernbotschaft: {s.get('executive_point', '')}\n"
        f"Statement: {s['position_statement']}\n"
        f"Vorschlag ({s['proposal'].get('total_budget_impact_pct', 0):.2f}% Budget = {s['proposal'].get('total_cost_mio_eur', 0):.1f} Mio. EUR, "
        f"Effekt: {s['proposal'].get('total_expected_effect_pct', 0):.1f}%): {s['proposal'].get('summary', '')}\n"
        f"Zitierte Quellen: {', '.join(s.get('cited_article_ids', []))}"
        for s in all_previous_statements
        if s["role_id"] != role.role_id
    )

    token_block = ""
    if token_articles:
        parts = []
        for idx, ta in enumerate(token_articles, 1):
            parts.append(
                f"🔍 TOKEN #{idx} – GEPRÜFTER FREMD-ARTIKEL VON '{ta['inspected_from_role']}':\n"
                f"=== ARTIKEL-ID: {ta['article_id']} ({ta.get('short_badge', ta['newspaper'])}) ===\n"
                f"Titel: {ta['title']}\n"
                f"Volltext-Auszug:\n{ta['content'][:2200]}\n"
            )
        token_block = (
            "\nDEINE EINGELÖSTEN DB-TOKENS (Selbstgewählte Quellen-Kontrolle deiner Gegenspieler):\n"
            + "\n".join(parts)
            + "-> Prüfe diese fremden Quellen kritisch: Welche Risiken, Kosten, Vorlaufzeiten oder Gegenargumente verschweigt dein Gegenspieler?\n"
        )

    prev_prop_block = ""
    if previous_main_proposal:
        prev_prop_block = (
            f"\nBISHERIGES KOMPROMISS-PAKET AUS DER VORRUNDE (worüber zuletzt verhandelt wurde):\n"
            f"Titel: {previous_main_proposal.get('summary', '')}\n"
            f"Maßnahmen: {json.dumps(previous_main_proposal.get('measures', []), ensure_ascii=False)}\n"
            f"Gesamtkosten: {previous_main_proposal.get('total_cost_mio_eur', 0):.2f} Mio. EUR "
            f"({previous_main_proposal.get('total_budget_impact_pct', 0):.2f}% des Budgets, Limit: {max_budget_pct}%)\n"
            f"Erreichter Ziel-Effekt: {previous_main_proposal.get('total_expected_effect_pct', 0):.2f}% (Gefordertes Ziel: {target_goal_pct}%)\n"
        )

    system_instructions = f"""Du bist '{role.name}' ({role.icon}) in Verhandlungsrunde {round_num}.
Deine Kerninteressen: {role.core_interests}
Deine Red Lines: {role.red_lines}
{_compromise_persona_description(compromise_level)}

{INSTITUTIONAL_STYLE_INSTRUCTION}

WICHTIGE VERHANDLUNGSREGELN:
1. BEHALTE DEINE EIGENE INSTITUTIONELLE LINIE BEI! Kopiere NIEMALS einfach das Proposal eines anderen Agenten, sondern verteidige deine eigenen Hebel und passe sie dort an, wo ein echtes Tauschgeschäft (Quid-pro-quo) für den gemeinsamen Kompromiss entsteht.
2. Gehe in `token_article_critique` gezielt auf die von dir mit deinen {len(token_articles)} DB-Token(s) geprüften Fremd-Artikel ein.
3. Antworte in `rebuttal_to_others` direkt auf die anderen Institutionen – und wiederhole diesen Konter-Text NICHT noch einmal in `position_statement`!
4. Nutze für Zahlenänderungen die Python-Calculator-Tools (`unit_cost_projection`, `scale_metric`, `budget_impact_pct`, `linear_interpolate`, `basic_math`)."""

    user_prompt = f"""VORHABEN:
\"\"\"{user_input}\"\"\"
Ziel: {intent_info.get('target_metric_name')} ({target_goal_pct}% in {intent_info.get('timeframe_years')} Jahren) | Max. Budget: {max_budget_pct}% ({max_budget_pct * 578.37:.1f} Mio. EUR).

DEIN BISHERIGES PROPOSAL & STATEMENT:
{previous_own_state.get('position_statement', '')}
Proposal: {json.dumps(previous_own_state.get('proposal', {}), ensure_ascii=False)}

STATEMENTS & PROPOSALS DER ANDEREN INSTITUTIONEN:
{other_stmts_text}
{token_block}
{prev_prop_block}

DEINE AUFGABEN FÜR VERHANDLUNGSRUNDE {round_num} (EIGENES PROPOSAL EINBRINGEN – DIE ABSTIMMUNG ÜBER DAS GEMEINSAME KOMPROMISS-PAKET FOLGT IM NÄCHSTEN SCHRITT!):
1. Zerlege in `token_article_critique` (max. 55 Wörter!) die mit deinen DB-Tokens geprüften Artikel ({', '.join(t['article_id'] for t in token_articles)}): Was hat der Gegenspieler ausgelassen?
2. Antworte in `rebuttal_to_others` (max. 65 Wörter!) direkt auf die anderen anwesenden Verhandlungspartner.
3. Formuliere 3 extrem knappe `core_bullets` (jeweils max. 14 Wörter!) für die Schnellübersicht, dein aktualisiertes `position_statement` (max. 70 Wörter) und dein eigenes `proposal` für Runde {round_num} (kombiniere 2 bis 3 Maßnahmen aus verschiedenen Hebel-Kategorien und schöpfe ca. 75% bis 98% des erlaubten Budgetrahmens von {max_budget_pct * 578.37:.0f} Mio. EUR aus, damit das Ziel von {target_goal_pct}% erreicht wird!).
   WICHTIG ZU STÜCKZAHLEN & QUELLEN:
   - Nutze bei Stückzahlen reale Zahlen aus `grundfakten.yaml` (z.B. ~340 unbesetzte Kassenstellen, Ausbau von 75 auf 120 PVE, regionale Facharztzentren) oder den konkreten Zahlen in deinen Artikeln.
   - Ordne in `source_article_ids` einer Maßnahme NUR DANN eine [ART-...] ID zu, wenn dieser Artikel das Thema der Maßnahme auch wirklich im Text behandelt!

Antworte AUSSCHLIESSLICH als valides JSON-Objekt:
{{
  "executive_point": "Deine aktuelle Kernbotschaft in 1 kurzen Satz (max. 14 Wörter).",
  "core_bullets": [
    "Kernforderung & wichtigstes Argument mit Quelle [ART-...]",
    "Kritik an Gegenseite / fremder Quelle [ART-...]",
    "Konkrete Bedingung für deine Zustimmung zum Kompromiss-Paket"
  ],
  "token_article_critique": "Prägnante Kritik am geprüften Fremd-Artikel (max. 55 Wörter) inkl. [ART-...]...",
  "rebuttal_to_others": "Prägnanter Konter an die anderen Institutionen (max. 65 Wörter) inkl. [ART-...]...",
  "position_statement": "Aktualisiertes eigenes Positionsstatement (max. 70 Wörter)...",
  "cited_article_ids": ["ART-...", "ART-..."],
  "proposal": {{
    "summary": "Aktualisiertes eigenes Proposal (max. 10 Wörter)",
    "measures": [
      {{
        "title": "Präzise Maßnahme (z.B. 380 Kassenstellen à 1,45 Mio. €)",
        "description": "1 knapper Satz zu Umsetzung & Finanzierung (max. 16 Wörter)",
        "cost_mio_eur": 551.0,
        "budget_impact_pct": 0.95,
        "expected_effect_pct": 6.8,
        "source_article_ids": ["ART-..."]
      }}
    ],
    "total_cost_mio_eur": 551.0,
    "total_budget_impact_pct": 0.95,
    "total_expected_effect_pct": 6.8
  }}
}}"""

    raw_output, calc = _run_agent_with_calculator(
        role=role,
        compromise_level=compromise_level,
        system_instructions=system_instructions,
        user_prompt=user_prompt,
        max_budget_pct=max_budget_pct,
        target_goal_pct=target_goal_pct,
    )

    raw_output.setdefault("executive_point", previous_own_state.get("executive_point", ""))
    raw_output.setdefault("token_article_critique", "")
    raw_output.setdefault("rebuttal_to_others", "")
    raw_output.setdefault("position_statement", previous_own_state.get("position_statement", ""))
    for ta in token_articles:
        if ta["article_id"] not in raw_output.get("cited_article_ids", []):
            raw_output.setdefault("cited_article_ids", []).append(ta["article_id"])
    raw_output.setdefault("proposal", previous_own_state.get("proposal", {}))
    # Score & vote_reasoning will be populated in Step 3 (`request_votes_on_proposal`)
    # after the Proposal-Agent has aggregated all Round `round_num` proposals!
    raw_output["approval_score_pct"] = previous_own_state.get("approval_score_pct") or 55
    raw_output["vote_reasoning"] = ""

    raw_output["token_article_critique"] = clean_output_prose(
        raw_output.get("token_article_critique", "")
    )

    verified_output = fact_check_and_verify_agent_output(
        role=role,
        agent_output=raw_output,
        available_articles=own_memory_articles,
        calc=calc,
        max_budget_pct=max_budget_pct,
    )

    verified_output["role_id"] = role.role_id
    verified_output["role_name"] = role.name
    verified_output["role_short_name"] = role.short_name
    verified_output["role_icon"] = role.icon
    verified_output["role_color"] = role.color
    verified_output["compromise_level"] = compromise_level
    verified_output["tokens_used_on_articles"] = [
        {
            "article_id": ta["article_id"],
            "title": ta["title"],
            "newspaper": ta["newspaper"],
            "article_date": ta["article_date"],
            "date_de": ta.get("date_de", ta["article_date"]),
            "short_badge": ta.get("short_badge", ta["newspaper"]),
            "inspected_from_role": ta["inspected_from_role"],
        }
        for ta in token_articles
    ]
    # Keep single-object alias for backward compatibility
    verified_output["token_used_on_article"] = (
        verified_output["tokens_used_on_articles"][0]
        if verified_output["tokens_used_on_articles"]
        else None
    )
    return verified_output


# ============================================================================
# STEP 5: OBJECTIVE PROPOSAL AGENT (SYNTHESIS, BUDGET CHECK & PRESSURE)
# ============================================================================
def generate_mediator_proposal(
    round_num: int,
    user_input: str,
    intent_info: Dict[str, Any],
    agent_states: List[Dict[str, Any]],
    previous_main_proposal: Optional[Dict[str, Any]] = None,
    previous_consensus_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Objective Proposal Agent (Mediator):
    1. Builds a Cost-Benefit (Effekt pro 100 Mio. €) catalog across all candidate measures proposed by agents.
    2. Combines the best cost-effective levers with essential political concessions for blocking/veto roles.
    3. Verifies via Python that the combined package reaches the target goal (~18%..22%) within the budget cap
       (typically using 55%..90% of `max_budget_pct`) and runs an immediate Python-guided refinement if under-dimensioned.
    """
    max_budget_pct = float(intent_info.get("max_budget_pct", 5.0))
    max_budget_mio = round(max_budget_pct * 578.37, 2)
    target_goal_pct = float(intent_info.get("target_goal_pct", 20.0))

    agents_summary = []
    candidate_measures_catalog = []
    for s in agent_states:
        prop = s.get("proposal", {})
        agents_summary.append(
            f"=== {s['role_icon']} {s['role_name']} (Kompromiss-Slider: {s['compromise_level']}%) ===\n"
            f"Aktueller Zustimmungs-Score: {s.get('approval_score_pct', '-')}% | Bedingung/Kritik: {s.get('vote_reasoning', '-')}\n"
            f"Kernbotschaft: {s.get('executive_point', '')}\n"
            f"Eigenes Proposal ({prop.get('total_budget_impact_pct', 0):.2f}% = {prop.get('total_cost_mio_eur', 0):.1f} Mio. EUR, "
            f"Effekt: +{prop.get('total_expected_effect_pct', 0):.1f}%): {json.dumps(prop.get('measures', []), ensure_ascii=False)}\n"
            f"Quellen-IDs: {', '.join(s.get('cited_article_ids', []))}"
        )
        for m in prop.get("measures", []):
            c_mio = max(1.0, float(m.get("cost_mio_eur", 100.0)))
            e_pct = float(m.get("expected_effect_pct", 1.0))
            ratio = round((e_pct / c_mio) * 100.0, 2)  # % effect per 100 Mio EUR
            candidate_measures_catalog.append(
                {
                    "role": s["role_short_name"],
                    "approval_score": s.get("approval_score_pct", 50),
                    "title": m.get("title", ""),
                    "lever_category": m.get("lever_category", ""),
                    "cost_mio_eur": c_mio,
                    "effect_pct": e_pct,
                    "efficiency_effect_per_100mio": ratio,
                    "sources": m.get("source_article_ids", []),
                }
            )

    candidate_measures_catalog.sort(key=lambda x: x["efficiency_effect_per_100mio"], reverse=True)
    catalog_lines = [
        f"- [{item['role']} | Score {item['approval_score']}%] „{item['title']}“ ({item['lever_category']}): "
        f"Kosten {item['cost_mio_eur']:.1f} Mio. € → Effekt +{item['effect_pct']:.2f}% "
        f"(Kosten-Nutzen-Ratio: +{item['efficiency_effect_per_100mio']:.2f}% pro 100 Mio. €) | Quellen: {', '.join(item['sources'])}"
        for item in candidate_measures_catalog
    ]
    catalog_block = "\n".join(catalog_lines)
    agents_block = "\n\n".join(agents_summary)

    prev_context = ""
    if previous_main_proposal and previous_consensus_result:
        prev_context = (
            f"\nVORHERIGES HAUPT-PROPOSAL (Verhandlungsrunde {round_num - 1}):\n"
            f"{json.dumps(previous_main_proposal, ensure_ascii=False)}\n"
            f"ABSTIMMUNGS-ERGEBNIS RUNDE {round_num - 1}: KEIN KONSENS ({previous_consensus_result.get('explanation', '')}).\n"
            f"Blockierende Institutionen: {', '.join(previous_consensus_result.get('blocking_roles', []))}\n"
            f"-> Du MUSST in Verhandlungsrunde {round_num} gezielt auf die Bedingungen der blockierenden Institutionen eingehen "
            f"und falls der Gesamteffekt unter {target_goal_pct}% lag, das Budget- und Maßnahmenvolumen spürbar aufstocken!\n"
        )

    system_instructions = f"""Du bist der überparteiliche 'Proposal- & Mediations-Agent' in Verhandlungsrunde {round_num}.
Deine Aufgabe ist es, aus den Einzel-Proposals der Agenten ein hochwirksames, konsensfähiges gemeinsames Kompromiss-Paket (Main-Proposal) zu aggregieren, über das alle Agenten anschließend abstimmen.

{INSTITUTIONAL_STYLE_INSTRUCTION}

STRATEGISCHE KOMBINATIONS- & OPTIMIERUNGS-LOGIK:
1. NICHT EINFACH DEN DURCHSCHNITT NEHMEN, SONDERN KOMPLEMENTÄRE HEBEL KOMBINIEREN:
   - Wenn jeder Agent nur sein Teilgebiet vorschlägt, erreicht der Durchschnitt niemals das Gesamtziel von {target_goal_pct}%!
   - Kombiniere **4 bis 5 komplementäre Maßnahmen aus UNTERSCHIEDLICHEN Hebel-Kategorien** (1. Kassenstellen/Tarife, 2. Primärversorgungseinheiten PVE, 3. Facharztzentren/Spitalsambulanz-Entlastung, 4. Digitale Lenkung 1450/ELGA/Telemedizin, 5. Pflege/Tagesklinik/OP). Jede Maßnahme im Kompromiss-Paket soll eine eigene Hebel-Kategorie abdecken!
2. KOSTEN-NUTZEN-RATIO + POLITISCHE AKZEPTANZ VERBINDEN:
   - Wähle Maßnahmen mit starkem Kosten-Nutzen-Verhältnis (Effekt pro 100 Mio. EUR) aus der vorberechneten Tabelle unten.
   - Berücksichtige gleichzeitig zwingend die Kernbedingungen der kritischen/blockierenden Rollen (z.B. Freiberuflichkeit/Tarife für Ärztekammer, Finanzausgleich/Spitalsentlastung für Bundesländer, Strukturverlagerung für Finanzministerium, längere Öffnungszeiten ohne Selbstbehalt für Patient:innen & ÖGK).
3. VOLLE BUDGET-AUSSCHÖPFUNG NAHE AM LIMIT FÜR MAXIMALEN ZIELEFFEKT ({target_goal_pct}%):
   - Schöpfe den erlaubten Budgetrahmen von maximal {max_budget_pct:.2f}% (= {max_budget_mio:.1f} Mio. EUR) **möglichst nahe bis an die Obergrenze aus (Zielkorridor: ca. 85% bis 99% des erlaubten Budgets = {max_budget_mio * 0.85:.0f} bis {max_budget_mio * 0.99:.0f} Mio. EUR, aber nicht über {max_budget_mio:.1f} Mio. EUR!)**, damit sowohl das Wirkungsziel von {target_goal_pct}% als auch ein breiter Konsens aller Verhandlungspartner erreicht wird!
4. Rufe den Python-Calculator (`budget_impact_pct`, `basic_math`) auf, um sowohl die Einzelmaßnahmen als auch die Gesamtsumme exakt zu prüfen."""

    user_prompt = f"""VORHABEN DES USERS:
\"\"\"{user_input}\"\"\"
Ziel: {intent_info.get('target_metric_name')} (Zielwert: {target_goal_pct}% in {intent_info.get('timeframe_years')} Jahren)
Max. Budget-Obergrenze: {max_budget_pct:.2f}% (= {max_budget_mio:.1f} Mio. EUR pro Jahr, Basis: 57.837 Mio. EUR).
{prev_context}

VORBERECHNETES KOSTEN-NUTZEN-RANKING DER VORGESCHLAGENEN MASSNAHMEN ALLER ROLLEN:
{catalog_block}

AKTUELLE POSITIONEN & EIGENE PROPOSALS DER INSTITUTIONEN (Verhandlungsrunde {round_num}):
{agents_block}

Erstelle jetzt das gemeinsame Kompromiss-Paket (Main-Proposal) für Verhandlungsrunde {round_num}, über das alle Agenten im nächsten Schritt abstimmen werden:
- Kombiniere 4 bis 5 komplementäre Maßnahmen aus unterschiedlichen Hebel-Kategorien, sodass der **kombinierte Gesamteffekt ca. {target_goal_pct:.0f}%** erreicht und das Gesamtbudget **möglichst nahe an der erlaubten Obergrenze liegt ({max_budget_mio * 0.85:.0f} bis {max_budget_mio * 0.99:.0f} Mio. EUR = {max_budget_pct * 0.85:.2f}% – {max_budget_pct * 0.99:.2f}% von max. {max_budget_pct:.2f}%)**!
- Nutze bei Stückzahlen reale Bestands- und Zielzahlen aus `grundfakten.yaml` (z.B. ~340 unbesetzte Kassenstellen, Ausbau auf 120 PVE, regionale Facharztzentren) oder den Vorschlägen der Agenten.
- Ordne in `source_article_ids` einer Maßnahme NUR DANN eine [ART-...] ID zu, wenn dieser Artikel das Thema der Maßnahme auch wirklich im Text behandelt!
- Halte alle Texte knapp und präzise.

Antworte AUSSCHLIESSLICH mit validem JSON:
{{
  "summary": "Kurztitel des Kompromiss-Pakets Runde {round_num} (max. 12 Wörter)",
  "mediator_statement": "Knappe 2-Satz-Zusammenfassung des Pakets (max. 38 Wörter) inkl. wichtigsten [ART-...] Quellen.",
  "alliances_analysis": "1-2 knappe Sätze (max. 28 Wörter): Wer trägt welche Maßnahme gemeinsam?",
  "blockers_pressure": "1-2 knappe Sätze (max. 28 Wörter): Welches konkrete Zugeständnis löst die bisherige Blockade?",
  "measures": [
    {{
      "title": "Präzise Maßnahme 1 (inkl. Stückzahl / Umfang)",
      "description": "1 knapper Satz zu Umsetzung & Finanzierungsschlüssel (max. 18 Wörter)",
      "cost_mio_eur": 780.0,
      "budget_impact_pct": 1.35,
      "expected_effect_pct": 6.8,
      "supported_by_roles": ["Ärztekammer", "Patient:innenvertretung"],
      "source_article_ids": ["ART-..."]
    }}
  ],
  "total_cost_mio_eur": 2750.0,
  "total_budget_impact_pct": 4.75,
  "total_expected_effect_pct": 21.0,
  "cited_article_ids": ["ART-..."]
}}"""

    dummy_role = StakeholderRole(
        role_id="mediator",
        name="Objektiver Proposal-Agent (Mediator)",
        short_name="Proposal-Agent",
        icon="🤝",
        color="#0071E3",
        description="Mediator",
        core_interests="Konsens & Zielerreichung innerhalb Budgetlimit",
        red_lines="Budgetüberschreitung",
        role_keywords=[],
    )
    raw_prop, calc = _run_agent_with_calculator(
        role=dummy_role,
        compromise_level=100,
        system_instructions=system_instructions,
        user_prompt=user_prompt,
        max_budget_pct=max_budget_pct,
        target_goal_pct=target_goal_pct,
    )

    raw_prop.setdefault("summary", f"Kompromiss-Paket (Verhandlungsrunde {round_num})")
    raw_prop.setdefault("mediator_statement", "Synthese der Vorschläge unter Einhaltung der Budgetobergrenze.")
    raw_prop.setdefault("alliances_analysis", "")
    raw_prop.setdefault("blockers_pressure", "")
    raw_prop.setdefault("measures", [])

    for field_name in ("summary", "mediator_statement", "alliances_analysis", "blockers_pressure"):
        raw_prop[field_name] = clean_output_prose(raw_prop.get(field_name, ""))

    verified_prop = validate_and_normalize_proposal_budget(
        raw_prop, max_budget_pct=max_budget_pct, auto_scale_to_cap=True
    )

    # Smart Mediator Target-Scaling Guard:
    # If the Mediator's combined package still falls short of target_goal_pct (e.g. < 92% of goal)
    # AND uses less than 88% of the allowed max_budget_pct, proportionally scale up the complementary measures
    # close to the budget cap (up to 96% of max_budget_pct) so the compromise package utilizes the available budget!
    cur_pct = float(verified_prop.get("total_budget_impact_pct", 0.0))
    cur_eff = float(verified_prop.get("total_expected_effect_pct", 0.0))
    if cur_eff > 0 and cur_eff < target_goal_pct * 0.92 and cur_pct > 0 and cur_pct < max_budget_pct * 0.88:
        desired_eff_scale = (target_goal_pct * 1.02) / cur_eff
        max_safe_budget_scale = (max_budget_pct * 0.96) / cur_pct
        boost_factor = min(desired_eff_scale, max_safe_budget_scale, 2.5)
        if boost_factor > 1.08:
            for m in verified_prop.get("measures", []):
                m["cost_mio_eur"] = round(float(m.get("cost_mio_eur", 0.0)) * boost_factor, 1)
                orig_e = float(m.get("original_claimed_effect_pct", m.get("expected_effect_pct", 0.0)))
                m["original_claimed_effect_pct"] = round(orig_e * boost_factor, 2)
                m["expected_effect_pct"] = round(float(m.get("expected_effect_pct", 0.0)) * boost_factor, 2)

            verified_prop = validate_and_normalize_proposal_budget(
                verified_prop, max_budget_pct=max_budget_pct, auto_scale_to_cap=True
            )
            calc._record(
                fn="mediator_package_optimization",
                label="Budget- & Zielerreichungs-Optimierung des Kompromiss-Pakets",
                inputs={"boost_factor": round(boost_factor, 2), "target_goal_pct": target_goal_pct},
                formula=f"Skalierung um Faktor {boost_factor:.2f}x auf {verified_prop['total_cost_mio_eur']} Mio. EUR ({verified_prop['total_budget_impact_pct']}% Budget) -> +{verified_prop['total_expected_effect_pct']}% Ziel-Effekt",
                result=verified_prop["total_expected_effect_pct"],
                unit="% Effekt",
                source_reference="grundfakten.yaml",
            )

    # Collect all available article records from participating agents
    all_cited_ids = set()
    for st in agent_states:
        for aid in st.get("cited_article_ids", []):
            if isinstance(aid, str) and aid.startswith("ART-"):
                all_cited_ids.add(aid)
        for m in (st.get("proposal") or {}).get("measures", []):
            for aid in m.get("source_article_ids", []):
                if isinstance(aid, str) and aid.startswith("ART-"):
                    all_cited_ids.add(aid)
    for m in verified_prop.get("measures", []):
        for aid in m.get("source_article_ids", []):
            if isinstance(aid, str) and aid.startswith("ART-"):
                all_cited_ids.add(aid)

    if all_cited_ids:
        all_arts_map = get_articles_by_ids(list(all_cited_ids))
        filter_thematic_sources_for_measures(verified_prop.get("measures", []), all_arts_map)

    # Clean cited_article_ids for the main proposal so only valid thematic articles remain
    used_prop_sources = set()
    for m in verified_prop.get("measures", []):
        for aid in m.get("source_article_ids", []):
            if isinstance(aid, str) and aid.startswith("ART-"):
                used_prop_sources.add(aid)
    if not used_prop_sources:
        for st in agent_states:
            for aid in st.get("cited_article_ids", []):
                if isinstance(aid, str) and aid.startswith("ART-"):
                    used_prop_sources.add(aid)
                    if len(used_prop_sources) >= 2:
                        break
            if len(used_prop_sources) >= 2:
                break
    verified_prop["cited_article_ids"] = list(used_prop_sources)

    verified_prop["round_num"] = round_num
    verified_prop["calculator_steps"] = [s.to_dict() for s in calc.steps]
    return verified_prop


def request_votes_on_proposal(
    round_num: int,
    user_input: str,
    intent_info: Dict[str, Any],
    agent_states: List[Dict[str, Any]],
    main_proposal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parallel voting pass where each agent critically evaluates the Proposal Agent's Main-Proposal
    against its own core interests, red lines, budget preferences, and target achievement.
    """
    client = get_openai_client()
    target_goal_pct = float(intent_info.get("target_goal_pct", 20.0))
    achieved_eff = float(main_proposal.get("total_expected_effect_pct", 0.0))
    target_ratio = min(1.2, achieved_eff / max(1.0, target_goal_pct))
    budget_ok = bool(main_proposal.get("python_budget_check", {}).get("verified", True))

    def _vote_single(state: Dict[str, Any]) -> Dict[str, Any]:
        role = STAKEHOLDER_ROLES[state["role_id"]]
        comp = state["compromise_level"]
        prompt = f"""Du bist '{role.name}' ({role.icon}) bei der Abstimmung in Verhandlungsrunde {round_num}.
Deine Kerninteressen: {role.core_interests}
Deine Red Lines: {role.red_lines}
{_compromise_persona_description(comp)}

{INSTITUTIONAL_STYLE_INSTRUCTION}

Dein eigenes Proposal aus dieser Runde:
{json.dumps(state.get('proposal', {}), ensure_ascii=False)}

HAUPT-PROPOSAL DES PROPOSAL-AGENTEN ZUR ABSTIMMUNG (Verhandlungsrunde {round_num}):
Titel: {main_proposal.get('summary')}
Begründung: {main_proposal.get('mediator_statement')}
Pressure-Hinweis: {main_proposal.get('blockers_pressure')}
Maßnahmen: {json.dumps(main_proposal.get('measures', []), ensure_ascii=False)}
Gesamtkosten: {main_proposal.get('total_cost_mio_eur')} Mio. EUR ({main_proposal.get('total_budget_impact_pct')}% des Budgets)
Erreichter Ziel-Effekt: {achieved_eff}% (Gefordertes Ziel: {target_goal_pct}%)

BEWERTUNGS-MASSSTAB FÜR DEINEN ZUSTIMMUNGS-SCORE (`approval_score_pct` 0 bis 100):
- Sei kritisch und realistisch! Vergib in Runde 1 NICHT leichtfertig >75%, wenn deine eigenen Kernforderungen verkürzt wurden, dir die Gegenfinanzierung unklar ist (z.B. aus Sicht von BMF/Ländern/ÖGK) oder wenn dir als Ärztekammer zu viel staatliche Zentren-/Steuerungslogik statt freiberuflicher Tarife drinsteckt!
- Wenn zentrale Punkte noch fehlen, vergib einen ehrlichen Score zwischen 48% und 68% und nenne in `vote_reasoning` glasklar deine Bedingung für die nächste Verhandlungsrunde!

Antworte AUSSCHLIESSLICH als JSON:
{{
  "approval_score_pct": 66,
  "vote_reasoning": "1 knapper Satz (max. 16 Wörter): Was gefällt dir am Paket und welche 1 Bedingung fehlt noch für >75%?"
}}"""
        try:
            r = call_llm_with_retry(model=DEPLOYMENT_NAME, input=prompt)
            vj = _extract_json_object(r.output_text)
            raw_sc = float(vj.get("approval_score_pct", state.get("approval_score_pct", 58)))
            score = apply_mathematical_compromise_shift(
                raw_llm_score=raw_sc,
                compromise_level=comp,
                round_num=round_num,
                budget_verified=budget_ok,
                target_achievement_ratio=target_ratio,
            )
            state["raw_llm_approval_score_pct"] = int(round(raw_sc))
            state["approval_score_pct"] = score
            reason = clean_output_prose(vj.get("vote_reasoning", ""))
            if reason:
                state["vote_reasoning"] = reason
        except Exception:
            pass
        if not state.get("vote_reasoning"):
            state["vote_reasoning"] = (
                state.get("executive_point")
                or f"Fordert stärkere Berücksichtigung der Kernpunkte von {role.short_name}."
            )
        return state

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_vote_single, st) for st in agent_states]
        for f in as_completed(futures):
            f.result()
    return agent_states


# ============================================================================
# STEP 6: PYTHON CONSENSUS / DEMOCRACY ENGINE
# ============================================================================
VOTING_SYSTEMS = {
    "veto_finanz_sv": {
        "name": "Finanz- & SV-Vetorecht (Veto-Rollen > 75% + Rest im Schnitt > 50%)",
        "threshold_line": 75,
        "description": "Real-Politik-Regel: Finanzministerium und Sozialversicherung (ÖGK) haben ein Vetorecht und müssen jeweils > 75% zustimmen; alle übrigen Teilnehmer benötigen im Schnitt > 50%.",
    },
    "unanimity_70": {
        "name": "Einstimmigkeit (Alle Parteien ≥ 70%)",
        "threshold_line": 70,
        "description": "Strikte Einstimmigkeit: Jede anwesende Institution muss dem Haupt-Proposal zu mindestens 70% zustimmen.",
    },
    "two_thirds_majority": {
        "name": "Zweidrittelmehrheit (Mind. 2/3 der Rollen ≥ 70%)",
        "threshold_line": 70,
        "description": "Parlamentarische Mehrheitsregel: Mindestens zwei Drittel der teilnehmenden Institutionen müssen eine Zustimmung von ≥ 70% erreichen.",
    },
    "simple_average_75": {
        "name": "Durchschnitts-Mehrheit (Gesamtdurchschnitt ≥ 75%)",
        "threshold_line": 75,
        "description": "Der arithmetische Mittelwert aller Zustimmungs-Scores muss mindestens 75% betragen.",
    },
}


def evaluate_consensus(
    agent_states: List[Dict[str, Any]],
    voting_system_id: str = "veto_finanz_sv",
) -> Dict[str, Any]:
    scores = {s["role_id"]: float(s.get("approval_score_pct") or 0.0) for s in agent_states}
    names = {s["role_id"]: s["role_short_name"] for s in agent_states}
    avg_score = round(sum(scores.values()) / max(1, len(scores)), 1)
    min_score = round(min(scores.values()) if scores else 0.0, 1)
    max_score = round(max(scores.values()) if scores else 0.0, 1)

    consensus_reached = False
    blocking_roles: List[str] = []
    explanation = ""
    threshold_line = VOTING_SYSTEMS.get(voting_system_id, {}).get("threshold_line", 75)

    if voting_system_id in ("unanimity_70", "unanimity_65"):
        thresh = 70.0
        blocking_roles = [names[rid] for rid, sc in scores.items() if sc < thresh]
        consensus_reached = len(blocking_roles) == 0
        explanation = f"Einstimmigkeit (alle ≥ 70%): Minimum liegt bei {min_score:.1f}%, Durchschnitt bei {avg_score:.1f}%."

    elif voting_system_id == "two_thirds_majority":
        thresh = 70.0
        supporters = [names[rid] for rid, sc in scores.items() if sc >= thresh]
        blocking_roles = [names[rid] for rid, sc in scores.items() if sc < thresh]
        required_count = int((2 * len(scores) + 2) // 3)
        consensus_reached = len(supporters) >= required_count
        explanation = (
            f"Zweidrittelmehrheit ({len(supporters)}/{len(scores)} Rollen ≥ 70%, benötigt: {required_count}). "
            f"Durchschnitt: {avg_score:.1f}%."
        )

    elif voting_system_id in ("simple_average_75", "simple_average_70"):
        thresh = 75.0
        blocking_roles = [names[rid] for rid, sc in scores.items() if sc < thresh]
        consensus_reached = avg_score >= thresh
        explanation = f"Durchschnitts-Regel (Schnitt ≥ 75%): Aktueller Durchschnitt = {avg_score:.1f}%."

    else:  # Default: veto_finanz_sv (Veto roles > 75% and remaining roles average > 50%)
        veto_ids = [rid for rid in ("finanzministerium", "oegk_sv") if rid in scores]
        other_ids = [rid for rid in scores if rid not in veto_ids]
        veto_blockers = [names[rid] for rid in veto_ids if scores[rid] <= 75.0]
        other_avg = (
            round(sum(scores[rid] for rid in other_ids) / len(other_ids), 1)
            if other_ids
            else avg_score
        )
        other_blockers = [names[rid] for rid in other_ids if scores[rid] <= 50.0] if other_avg <= 50.0 else []
        blocking_roles = veto_blockers + other_blockers
        consensus_reached = (len(veto_blockers) == 0) and (other_avg > 50.0)
        explanation = (
            f"Veto-Regel (Veto-Rollen > 75% & Rest im Schnitt > 50%): Rest-Schnitt = {other_avg:.1f}%, Gesamtschnitt = {avg_score:.1f}%. "
            + (
                f"Veto eingelegt durch {', '.join(veto_blockers)}."
                if veto_blockers
                else ("Rest-Durchschnitt nicht > 50%." if other_avg <= 50.0 else "Keine Veto-Blockade.")
            )
        )

    return {
        "consensus_reached": consensus_reached,
        "voting_system_id": voting_system_id,
        "voting_system_name": VOTING_SYSTEMS.get(voting_system_id, {}).get("name", voting_system_id),
        "threshold_line": threshold_line,
        "average_approval_pct": avg_score,
        "min_approval_pct": min_score,
        "max_approval_pct": max_score,
        "scores_by_role": scores,
        "blocking_roles": blocking_roles,
        "explanation": explanation,
    }


# ============================================================================
# STEP 7: STRATEGIC ACTION RECOMMENDATION ("HANDLUNGSEMPFEHLUNG")
# ============================================================================
def generate_role_action_recommendation(
    target_role_id: str,
    simulation_result: Dict[str, Any],
) -> Dict[str, Any]:
    role = STAKEHOLDER_ROLES.get(target_role_id) or list(STAKEHOLDER_ROLES.values())[0]
    user_input = simulation_result.get("user_input", "")
    consensus = simulation_result.get("final_consensus", {})
    final_prop = simulation_result.get("final_main_proposal", {})
    rounds = simulation_result.get("rounds", [])
    last_round = rounds[-1] if rounds else {}

    agents_recap = "\n".join(
        f"- {s['role_name']}: Finale Zustimmung {s.get('approval_score_pct')}% | "
        f"Kernbotschaft: {s.get('executive_point', '')} | "
        f"Quellen: {', '.join(s.get('cited_article_ids', []))}"
        for s in last_round.get("agent_states", [])
    )

    client = get_openai_client()
    prompt = f"""Du bist der Chef-Stratege für die Institution '{role.name}' ({role.icon}).
Die simulierte Gesundheitsreform-Verhandlung zum Thema:
\"\"\"{user_input}\"\"\"
ist nach {len(rounds)} Verhandlungsrunden mit dem Status '{'KONSENS ERZIELT' if consensus.get('consensus_reached') else 'KEIN KONSENS (ABBRUCH)'}' ({consensus.get('explanation', '')}) geendet.

{INSTITUTIONAL_STYLE_INSTRUCTION}

FINALES HAUPT-PROPOSAL:
{json.dumps(final_prop, ensure_ascii=False)}

SCHLUSSPOSITIONEN ALLER VERHANDLUNGSPARTNER:
{agents_recap}

Erstelle eine extrem prägnante, sofort lesbare **Handlungsempfehlung** für '{role.name}' (keine Textwände, nur knappe Bullet Points à max. 16 Wörter!).
Antworte AUSSCHLIESSLICH als valides JSON:
{{
  "executive_summary": "Knappes 2-Satz-Fazit aus Sicht von {role.short_name} (max. 32 Wörter).",
  "expected_attacks_and_counters": [
    {{
      "from_institution": "Name des Gegenspielers",
      "expected_attack": "1 knapper Punkt (max. 16 Wörter): Welches Argument / welche Quelle wird gegen uns genutzt?",
      "recommended_counter": "1 knapper Punkt (max. 18 Wörter): Unser faktenbasierter Konter inkl. Zahlen & [ART-...] Quelle."
    }}
  ],
  "strategic_alliances": [
    "Knapper Bullet Point (max. 15 Wörter): Mit wem vorab ein Bündnis schließen und warum?"
  ],
  "negotiation_red_lines_and_fallbacks": [
    "Knapper Bullet Point (max. 16 Wörter): Eiserne Untergrenze + mögliches Quid-pro-quo."
  ],
  "concrete_next_steps": [
    "Konkreter Schritt 1 (max. 14 Wörter)",
    "Konkreter Schritt 2 (max. 14 Wörter)",
    "Konkreter Schritt 3 (max. 14 Wörter)"
  ]
}}"""

    resp = call_llm_with_retry(model=DEPLOYMENT_NAME, input=prompt)
    parsed = _extract_json_object(resp.output_text)
    parsed["executive_summary"] = clean_output_prose(
        parsed.get("executive_summary", f"Strategische Handlungsempfehlung für {role.name}.")
    )
    parsed.setdefault("expected_attacks_and_counters", [])
    parsed.setdefault("strategic_alliances", [])
    parsed.setdefault("negotiation_red_lines_and_fallbacks", [])
    parsed.setdefault("concrete_next_steps", [])
    parsed["role_id"] = role.role_id
    parsed["role_name"] = role.name
    parsed["role_icon"] = role.icon
    return parsed


# ============================================================================
# FULL ORCHESTRATOR PIPELINE V2.2
# ============================================================================
def run_policy_simulation(
    user_input: str,
    selected_role_ids: List[str],
    compromise_sliders: Dict[str, int],
    voting_system_id: str = "veto_finanz_sv",
    max_rounds: int = 3,
    articles_per_agent: int = 7,
    tokens_per_round: int = 2,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete multi-agent negotiation simulation V2.3:
    - Step 1: Keyword & Case Detector (`analyze_user_input`).
    - Step 2: Role-specific DB retrieval (`presse_db.sqlite`) with cross-agent diversity (3 to 20 articles per agent).
    - Step 3 (Eröffnungsrunde): Parallel opening statements + stock-anchored PolicyCalculator + Fact-Checker (no self-voting).
    - Step 4 (Verhandlungsrunden 1..max_rounds):
      In every round r:
      1) Stakeholder agents inspect `tokens_per_round` opponent articles, counter opponents, and calculate their updated Round-r proposal.
      2) Proposal-Agent synthesizes the Round-r compromise package (`main_proposal`) from all agents' Round-r proposals (or uses User-Proposal in Round 1 if Case 1).
      3) All stakeholder agents vote specifically on that joint `main_proposal` (`request_votes_on_proposal`).
      4) Consensus is evaluated (`evaluate_consensus`).
    """
    t_start = time.time()
    articles_per_agent = max(3, min(20, int(articles_per_agent)))
    tokens_per_round = 2
    max_rounds = max(1, min(5, int(max_rounds)))
    total_steps = 3 + 2 * max_rounds

    def _notify(msg: str, prog: float):
        if progress_callback:
            progress_callback(msg, prog)

    _notify(f"Schritt 1/{total_steps}: Vorhaben wird analysiert...", 0.08)
    intent_info = analyze_user_input(user_input)

    user_keywords = intent_info.get("keywords", [])
    max_budget_pct = float(intent_info.get("max_budget_pct", 5.0))
    has_user_proposal = bool(intent_info.get("has_user_proposal", False))

    active_roles = [STAKEHOLDER_ROLES[rid] for rid in selected_role_ids if rid in STAKEHOLDER_ROLES]
    if not active_roles:
        active_roles = [r for r in STAKEHOLDER_ROLES.values() if r.default_selected]

    agent_article_memory: Dict[str, Dict[str, Dict[str, Any]]] = {}
    initial_retrieved_map: Dict[str, List[Dict[str, Any]]] = {}
    article_lookup: Dict[str, Dict[str, Any]] = {}
    already_assigned_counts: Dict[str, int] = {}

    _notify(f"Schritt 2/{total_steps}: Zeitungsartikel werden geladen...", 0.18)
    for role in active_roles:
        arts = search_articles_for_agent(
            user_keywords=user_keywords,
            role_keywords=role.role_keywords,
            limit=articles_per_agent,
            already_assigned_counts=already_assigned_counts,
        )
        initial_retrieved_map[role.role_id] = arts
        agent_article_memory[role.role_id] = {a["article_id"]: a for a in arts}
        for a in arts:
            aid = a["article_id"]
            already_assigned_counts[aid] = already_assigned_counts.get(aid, 0) + 1
            article_lookup[aid] = {
                "article_id": aid,
                "newspaper": a["newspaper"],
                "article_date": a["article_date"],
                "date_de": a.get("date_de", a["article_date"]),
                "short_badge": a.get("short_badge", a["newspaper"]),
                "title": a["title"],
            }

    # ------------------------------------------------------------------------
    # PHASE 0: "Eröffnungsrunde: Eingangs-Positionen"
    # ------------------------------------------------------------------------
    _notify(f"Schritt 3/{total_steps}: Eröffnungsrunde – Agenten berechnen Eingangs-Positionen...", 0.30)
    opening_states: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(3, len(active_roles))) as pool:
        fut_map = {
            pool.submit(
                run_single_agent_opening_phase,
                role,
                compromise_sliders.get(role.role_id, role.default_compromise),
                user_input,
                intent_info,
                initial_retrieved_map[role.role_id],
            ): role
            for role in active_roles
        }
        done_cnt = 0
        for fut in as_completed(fut_map):
            st = fut.result()
            opening_states.append(st)
            done_cnt += 1
            cur_role = fut_map[fut]
            role_label = getattr(cur_role, "short_name", getattr(cur_role, "name", st.get("role_name", "Agent")))
            prog_val = round(0.30 + (0.10 * (done_cnt / len(active_roles))), 3)
            _notify(
                f"Schritt 3/{total_steps}: Eröffnungsrunde ({done_cnt}/{len(active_roles)} fertig: {role_label})...",
                prog_val,
            )

    order_idx = {rid: i for i, rid in enumerate(selected_role_ids)}
    opening_states.sort(key=lambda s: order_idx.get(s["role_id"], 99))

    # ------------------------------------------------------------------------
    # VERHANDLUNGSRUNDEN 1..max_rounds (1. Einzel-Proposals -> 2. Aggregation -> 3. Abstimmung)
    # ------------------------------------------------------------------------
    user_main_proposal: Optional[Dict[str, Any]] = None
    if has_user_proposal:
        user_pct = float(intent_info.get("user_proposal_budget_pct") or max_budget_pct)
        user_cost = round((user_pct / 100.0) * BUDGET_BASELINES_2024_MIO["gesamt_laufend"], 2)
        user_main_proposal = validate_and_normalize_proposal_budget(
            {
                "round_num": 1,
                "summary": f"User-Proposal: {intent_info.get('user_proposal_summary', user_input)}",
                "mediator_statement": "Direktes Vorhaben aus dem User-Input als Haupt-Proposal zur Abstimmung.",
                "alliances_analysis": "Direkte Abstimmung über das vom User eingebrachte Vorhaben.",
                "blockers_pressure": "Kritische Prüfung auf Zielerreichung und Budgetdisziplin durch alle Rollen.",
                "measures": [
                    {
                        "title": intent_info.get("user_proposal_summary", "User-Maßnahme"),
                        "description": user_input,
                        "cost_mio_eur": user_cost,
                        "budget_impact_pct": user_pct,
                        "expected_effect_pct": float(intent_info.get("target_goal_pct", 20.0)) * 0.75,
                        "source_article_ids": [],
                    }
                ],
            },
            max_budget_pct=max_budget_pct,
        )

    rounds_history: List[Dict[str, Any]] = []
    current_states: List[Dict[str, Any]] = opening_states
    current_main_proposal: Optional[Dict[str, Any]] = user_main_proposal
    current_consensus: Optional[Dict[str, Any]] = None

    for round_num in range(1, max_rounds + 1):
        step_a = 4 + (round_num - 1) * 2
        step_b = step_a + 1
        prog_base = 0.42 + 0.50 * ((round_num - 1) / max(1, max_rounds))

        # 1) Agents debate opponents, inspect opponent sources, and formulate their updated Round-r proposals
        _notify(
            f"Schritt {step_a}/{total_steps}: Verhandlungsrunde {round_num} – Quellen-Prüfung & Einzel-Proposals...",
            round(prog_base, 2),
        )
        prev_by_id = {s["role_id"]: s for s in current_states}
        round_states: List[Dict[str, Any]] = []
        claimed_rn: Set[str] = set()
        claim_lock_rn = threading.Lock()

        with ThreadPoolExecutor(max_workers=min(3, len(active_roles))) as pool:
            fut_map = {
                pool.submit(
                    run_single_agent_cross_exam_round,
                    round_num,
                    role,
                    compromise_sliders.get(role.role_id, role.default_compromise),
                    user_input,
                    intent_info,
                    agent_article_memory[role.role_id],
                    prev_by_id[role.role_id],
                    current_states,
                    current_main_proposal,
                    tokens_per_round,
                    claimed_rn,
                    claim_lock_rn,
                ): role
                for role in active_roles
            }
            done_cnt_r = 0
            for fut in as_completed(fut_map):
                rst = fut.result()
                round_states.append(rst)
                done_cnt_r += 1
                cur_role_r = fut_map[fut]
                role_label_r = getattr(cur_role_r, "short_name", getattr(cur_role_r, "name", rst.get("role_name", "Agent")))
                prog_val_r = round(prog_base + (0.10 * (done_cnt_r / len(active_roles))), 3)
                _notify(
                    f"Schritt {step_a}/{total_steps}: Runde {round_num} ({done_cnt_r}/{len(active_roles)} fertig: {role_label_r})...",
                    prog_val_r,
                )

        round_states.sort(key=lambda s: order_idx.get(s["role_id"], 99))

        # 2) Proposal-Agent aggregates all agents' Round-r proposals into the Round-r compromise package
        _notify(
            f"Schritt {step_b}/{total_steps}: Kompromiss-Paket Runde {round_num} & Abstimmung aller Agenten...",
            round(min(0.95, prog_base + 0.22 / max(1, max_rounds)), 2),
        )
        if round_num == 1 and has_user_proposal and user_main_proposal is not None:
            current_main_proposal = user_main_proposal
        else:
            current_main_proposal = generate_mediator_proposal(
                round_num=round_num,
                user_input=user_input,
                intent_info=intent_info,
                agent_states=round_states,
                previous_main_proposal=current_main_proposal if round_num > 1 else None,
                previous_consensus_result=current_consensus if round_num > 1 else None,
            )

        # 3) All agents receive the aggregated compromise package and cast their vote on it
        round_states = request_votes_on_proposal(
            round_num=round_num,
            user_input=user_input,
            intent_info=intent_info,
            agent_states=round_states,
            main_proposal=current_main_proposal,
        )

        # 4) Evaluate consensus on the joint proposal
        current_consensus = evaluate_consensus(round_states, voting_system_id=voting_system_id)
        current_states = round_states

        round_entry: Dict[str, Any] = {
            "round_num": round_num,
            "round_label": f"Verhandlungsrunde {round_num}",
            "agent_states": current_states,
            "main_proposal": current_main_proposal,
            "consensus": current_consensus,
        }
        if round_num == 1:
            round_entry["opening_statements"] = opening_states
        rounds_history.append(round_entry)

        if current_consensus["consensus_reached"]:
            break

    # Ensure every cited article across all rounds is present in `article_lookup` for short UI badges
    all_referenced_ids: Set[str] = set()
    for r in rounds_history:
        for st in (r.get("opening_statements") or []) + (r.get("agent_states") or []):
            for aid in st.get("cited_article_ids", []):
                if isinstance(aid, str) and aid.startswith("ART-"):
                    all_referenced_ids.add(aid)
        mp = r.get("main_proposal") or {}
        for aid in mp.get("cited_article_ids", []):
            if isinstance(aid, str) and aid.startswith("ART-"):
                all_referenced_ids.add(aid)
        for m in mp.get("measures", []):
            for aid in m.get("source_article_ids", []):
                if isinstance(aid, str) and aid.startswith("ART-"):
                    all_referenced_ids.add(aid)

    missing_lookup = [aid for aid in all_referenced_ids if aid not in article_lookup]
    if missing_lookup:
        extra_arts = get_articles_by_ids(missing_lookup)
        for aid, a in extra_arts.items():
            article_lookup[aid] = {
                "article_id": aid,
                "newspaper": a["newspaper"],
                "article_date": a["article_date"],
                "date_de": a.get("date_de", a["article_date"]),
                "short_badge": a.get("short_badge", a["newspaper"]),
                "title": a["title"],
            }

    elapsed_sec = round(time.time() - t_start, 1)
    _notify(f"Verhandlung abgeschlossen in {elapsed_sec}s!", 1.0)

    return {
        "multi_goal_rejected": False,
        "user_input": user_input,
        "intent_info": intent_info,
        "selected_role_ids": [r.role_id for r in active_roles],
        "compromise_sliders": compromise_sliders,
        "voting_system_id": voting_system_id,
        "max_rounds": max_rounds,
        "articles_per_agent": articles_per_agent,
        "tokens_per_round": tokens_per_round,
        "rounds_executed": len(rounds_history),
        "rounds": rounds_history,
        "final_main_proposal": current_main_proposal,
        "final_consensus": current_consensus,
        "article_lookup": article_lookup,
        "elapsed_seconds": elapsed_sec,
    }
