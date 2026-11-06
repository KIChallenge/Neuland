#!/usr/bin/env python3
"""
Database Retrieval, Article-Level Anonymization & Institutional Role Profiles for HealthPolicy Simulator (UC2).
Provides:
1. Full-text search over the 11,412 articles in `presse_db.sqlite` with diversity weighting
   (so each agent gets distinct articles, configurable from 3 to 20 articles per agent).
2. Direct article lookup by `article_id` with automatic real-person anonymization (`Instructions.docx` compliance)
   and short readable badge formatting: `📰 Zeitung (TT.MM.JJJJ)`.
3. Definitions of all 10 institutional stakeholder roles.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import gzip
import re
import shutil
import sqlite3

WORKSPACE_DIR = Path(__file__).resolve().parent
DB_PATH = WORKSPACE_DIR / "presse_db.sqlite"
DB_GZ_PATH = WORKSPACE_DIR / "presse_db.sqlite.gz"


def ensure_sqlite_db_present() -> None:
    """
    If the repository was cloned from GitHub without the >100 MB raw `presse_db.sqlite`,
    automatically unpack `presse_db.sqlite.gz` on first run.
    """
    if not DB_PATH.exists() and DB_GZ_PATH.exists():
        print(f"[DB] Entpacke {DB_GZ_PATH.name} -> {DB_PATH.name} ...")
        with gzip.open(DB_GZ_PATH, "rb") as f_in, open(DB_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("[DB] Datenbank erfolgreich entpackt.")


ensure_sqlite_db_present()


@dataclass
class StakeholderRole:
    role_id: str
    name: str
    short_name: str
    icon: str
    color: str
    description: str
    core_interests: str
    red_lines: str
    role_keywords: List[str]
    default_selected: bool = True
    default_compromise: int = 50


STAKEHOLDER_ROLES: Dict[str, StakeholderRole] = {
    "oegk_sv": StakeholderRole(
        role_id="oegk_sv",
        name="Sozialversicherung (ÖGK & Dachverband)",
        short_name="Sozialversicherung (ÖGK)",
        icon="🏛️",
        color="#3b82f6",
        description="Gesetzliche Krankenversicherung; finanziert Kassenärzte, PVE und Medikamente.",
        core_interests="Ausbau von Primärversorgungseinheiten (PVE), eigenen Ambulatorien und Kassenstellen mit längeren Öffnungszeiten; Dämpfung der Arzneimittel- und Spitalskosten; Sachleistungsprinzip (e-card statt Wahlarzt-Privatrechnung); Telemedizin.",
        red_lines="Unfinanzierte Leistungsausweitungen ohne Bundes-Ausgleich; pauschale Honorarerhöhungen für Einzelordinationen ohne längere Öffnungszeiten; Subventionierung des Wahlarztsystems.",
        role_keywords=[
            "ÖGK",
            "Gesundheitskasse",
            "Sozialversicherung",
            "Dachverband",
            "Kassenvertrag",
            "Kassenstellen",
            "Primärversorgung",
            "PVE",
            "Ambulatorium",
            "e-card",
            "Wahlarzt",
            "Kostenerstattung",
        ],
        default_selected=True,
        default_compromise=50,
    ),
    "aerztekammer": StakeholderRole(
        role_id="aerztekammer",
        name="Österreichische Ärztekammer (ÖÄK)",
        short_name="Ärztekammer",
        icon="🩺",
        color="#10b981",
        description="Interessenvertretung der Kassen-, Wahl- und Spitalsärzt:innen.",
        core_interests="Deutliche Attraktivierung der Kassenverträge durch höhere Tarife/Honorare und Bürokratieabbau; Schutz der Freiberuflichkeit, Einzelordinationen und des Wahlarztsystems; Ablehnung von Zwangsverpflichtungen oder kassen-eigenen Ambulatorien.",
        red_lines="Abschaffung oder Schlechterstellung der Wahlärzt:innen, Zwangsmaßnahmen gegen Mediziner:innen, Ersetzung freiberuflicher Praxen durch staatliche/Kassen-Ambulatorien, Honorardeckelungen.",
        role_keywords=[
            "Ärztekammer",
            "Ärztemangel",
            "Kassenarzt",
            "Wahlarzt",
            "Honorar",
            "Tarife",
            "Spitalsärzte",
            "Ordination",
            "Einzelordination",
            "Bürokratie",
            "Wartezeiten",
        ],
        default_selected=True,
        default_compromise=35,
    ),
    "bundeslaender": StakeholderRole(
        role_id="bundeslaender",
        name="Bundesländer (Landesgesundheitsreferent:innen)",
        short_name="Bundesländer",
        icon="🏰",
        color="#f59e0b",
        description="Träger der Landeskrankenanstalten und der regionalen Pflegeplanung.",
        core_interests="Entlastung der Spitalsambulanzen durch vorgelagerte Versorgung; mehr Mittel aus dem Finanzausgleich vom Bund und höhere Zuschüsse der Sozialversicherung; Erhalt regionaler Spitals- und Klinikstandorte.",
        red_lines="Kostenabwälzung auf die Länderbudgets ohne Gegenfinanzierung durch Bund/SV; Schließung regionaler Versorgungsstrukturen ohne Ersatzangebot.",
        role_keywords=[
            "Bundesländer",
            "Landesklinikum",
            "Spitäler",
            "Spitalsambulanzen",
            "Finanzausgleich",
            "Gastpatienten",
            "Tagesklinik",
            "Krankenhaus",
            "Abgangsdeckung",
            "Landesgesundheitsfonds",
        ],
        default_selected=True,
        default_compromise=45,
    ),
    "finanzministerium": StakeholderRole(
        role_id="finanzministerium",
        name="Finanzministerium (BMF)",
        short_name="Finanzministerium",
        icon="💶",
        color="#ef4444",
        description="Zuständig für Bundesbudget, Finanzausgleich und Ausgabenkontrolle.",
        core_interests="Minimierung der Budgetbelastung (deutlich unter der Maximalgrenze bleiben!); Effizienzsteigerung und Strukturreform statt frischem Steuergeld (Verlagerung von teuer-stationär 9.726 EUR/Fall zu tagesklinisch/ambulant); Kopplung jedes Euros an verbindliche Ziel-Meilensteine.",
        red_lines="Ausschöpfung des vollen Budgetrahmens ohne nachgewiesene Gegenfinanzierung/Einsparung im Spitalsbereich; reine Gießkannen-Honorarerhöhungen.",
        role_keywords=[
            "Finanzministerium",
            "Budget",
            "Defizit",
            "Einsparung",
            "Finanzausgleich",
            "Rechnungshof",
            "Effizienz",
            "Kostenexplosion",
            "Steuergeld",
            "Gesundheitsausgaben",
        ],
        default_selected=True,
        default_compromise=30,
    ),
    "gesundheitsministerium": StakeholderRole(
        role_id="gesundheitsministerium",
        name="Gesundheits- & Sozialministerium (BMSGPK)",
        short_name="Gesundheitsministerium",
        icon="⚖️",
        color="#8b5cf6",
        description="Zuständig für Bundessteuerung, Prävention, ELGA und Digitalisierung.",
        core_interests="Verbindliche Patientenlenkung über Gesundheitshotline 1450 und ELGA, Ausbau von PVE und Facharztzentren, Stärkung von Prävention/Vorsorge, sozial gerechter Zugang ohne Zwei-Klassen-Medizin.",
        red_lines="Verschlechterung des Zugangs für sozial Schwache; Kürzungen bei Prävention oder Patientensicherheit; ungesteuerter Ausbau ohne digitale Anbindung.",
        role_keywords=[
            "Gesundheitsministerium",
            "Gesundheitsreform",
            "Prävention",
            "Vorsorge",
            "ELGA",
            "1450",
            "Zwei-Klassen-Medizin",
            "Pflege",
            "Patientenlenkung",
            "Digitalisierung",
        ],
        default_selected=True,
        default_compromise=60,
    ),
    "patientenvertretung": StakeholderRole(
        role_id="patientenvertretung",
        name="Patient:innenvertretung & Patientenanwaltschaft",
        short_name="Patient:innenvertretung",
        icon="🧑‍🦽",
        color="#06b6d4",
        description="Vertretung der Patient:innen für rasche, wohnortnahe und kostenfreie Versorgung.",
        core_interests="Drastische Verkürzung der Wartezeiten auf Facharzttermine und OPs; rasche Wahlarzt-Kostenerstattung; Ausbau von Abend- und Wochenend-Öffnungszeiten; verständliche Patientenlenkung ohne Selbstbehalte.",
        red_lines="Einführung von Selbstbehalten, Ambulanzgebühren oder verpflichtenden Hürden für Patient:innen; Leistungskürzungen; zu kleine Reformpakete, die das 20%-Ziel verfehlen.",
        role_keywords=[
            "Patienten",
            "Patientenanwalt",
            "Wartezeit",
            "Wartezeiten",
            "Facharzttermin",
            "Selbstbehalt",
            "Kostenerstattung",
            "Zwei-Klassen-Medizin",
            "OP-Wartezeit",
            "Versorgung",
        ],
        default_selected=True,
        default_compromise=55,
    ),
    "krankenanstalten": StakeholderRole(
        role_id="krankenanstalten",
        name="Krankenanstalten & Spitalsträger",
        short_name="Krankenanstalten",
        icon="🏥",
        color="#ec4899",
        description="Betrieb der öffentlichen und gemeinnützigen Akutspitäler und Ambulanzen.",
        core_interests="Entlastung der Spitalsambulanzen von Bagatellfällen; Bekämpfung des Pflege- und Anästhesiemangels für OP-Kapazitäten; Ausbau von Tageskliniken; leistungsgerechte LKF-Finanzierung.",
        red_lines="Zusätzliche Aufgaben für Spitalsambulanzen ohne Personal; Abzug von Spitalsärzt:innen in den niedergelassenen Bereich ohne Kompensation.",
        role_keywords=[
            "Krankenanstalten",
            "Spital",
            "Klinik",
            "Ambulanz",
            "OP",
            "Operation",
            "Tagesklinik",
            "Pflegenotstand",
            "Betten",
            "Auslastung",
        ],
        default_selected=False,
        default_compromise=50,
    ),
    "pflegevertretung": StakeholderRole(
        role_id="pflegevertretung",
        name="Pflegevertretung & Gesundheitsberufe",
        short_name="Pflegevertretung",
        icon="👩‍⚕️",
        color="#14b8a6",
        description="Berufsvertretung der Krankenpflege (DGKP), Pflegeassistenz und MTD-Berufe.",
        core_interests="Bessere Arbeitsbedingungen, höhere Gehälter, Ausbildungsprämien, Einsatz von Community Nurses und Advanced Practice Nurses in PVE zur Entlastung von Ärzt:innen und Spitälern.",
        red_lines="Mehrbelastung des bestehenden Pflegepersonals ohne Personalaufbau; reine Ärzte-Honorarreform ohne Berücksichtigung der Pflege.",
        role_keywords=[
            "Pflege",
            "Pflegekräfte",
            "Pflegemangel",
            "DGKP",
            "Community Nurse",
            "Pflegebedürftige",
            "Spital",
            "Arbeitsbedingungen",
            "Ausbildung",
        ],
        default_selected=False,
        default_compromise=55,
    ),
    "public_health": StakeholderRole(
        role_id="public_health",
        name="Public-Health-Expert:innen & Wissenschaft",
        short_name="Public Health",
        icon="🔬",
        color="#84cc16",
        description="Wissenschaftliche Versorgungsforschung, Prävention und Gesundheitsökonomie.",
        core_interests="Evidenzbasierte Steuerung: verbindliche Patientenpfade (1450 -> Primärversorgung -> Facharzt -> Spital), Ausbau von Prävention und Gesundheitskompetenz, Abbau der Überhospitalisierung in Österreich.",
        red_lines="Evidenzfreie Gießkannen-Subventionen, Beibehaltung der fragmentierten Finanzierung zwischen Ländern und Kassen.",
        role_keywords=[
            "Public Health",
            "Studie",
            "Prävention",
            "GÖG",
            "Gesundheitsökonomie",
            "Primärversorgung",
            "Telemedizin",
            "Vorsorge",
            "Versorgungsforschung",
        ],
        default_selected=False,
        default_compromise=70,
    ),
    "pharma_industrie": StakeholderRole(
        role_id="pharma_industrie",
        name="Gesundheits- & Pharmaindustrie (Pharmig / MedTech)",
        short_name="Industrie & Pharma",
        icon="💊",
        color="#f97316",
        description="Hersteller von Arzneimitteln, Diagnostika und Medizintechnik.",
        core_interests="Rascher Marktzugang und faire Erstattungspreise für innovative Therapien und Generika; Versorgungssicherheit bei Medikamenten; Vermeidung neuer Sonderabgaben.",
        red_lines="Preisdiktate unter den Produktionskosten, einseitige Kostenabwälzung auf Arzneimittelhersteller.",
        role_keywords=[
            "Pharma",
            "Medikamente",
            "Arzneimittel",
            "Pharmig",
            "Impfung",
            "Generika",
            "Krebsmedizin",
            "Therapie",
            "Lieferengpässe",
        ],
        default_selected=False,
        default_compromise=50,
    ),
}


# ============================================================================
# DB-LEVEL & OUTPUT-LEVEL PERSON ANONYMIZATION (`Instructions.docx`)
# ============================================================================
KNOWN_PERSON_REPLACEMENTS = [
    (r"\b(?:Stadtrat\s+)?(?:Peter\s+)?Hacker\b", "[Wiener Gesundheitsressort]"),
    (r"\b(?:Minister\s+|Gesundheitsminister\s+)?(?:Johannes\s+)?Rauch\b", "[Gesundheitsministerium]"),
    (r"\b(?:SPÖ-Gesundheitsministerin\s+|Ministerin\s+)?(?:Korinna\s+)?Schumann\b", "[Gesundheitsministerium]"),
    (r"\b(?:ÖGK-Obmann\s+|Obmann\s+)?(?:Andreas\s+)?Huss\b", "[ÖGK-Führung]"),
    (r"\b(?:ÖGK-Co-Obmann\s+|Obmann\s+)?(?:Peter\s+)?McDonald\b", "[ÖGK-Verwaltungsrat]"),
    (r"\b(?:Ärztekammer-Präsident\s+|Präsident\s+)?(?:Johannes\s+)?Steinhart\b", "[Ärztekammer-Spitze]"),
    (r"\b(?:Vizepräsident\s+)?(?:Harald\s+)?Mayer\b", "[Kurie Angestellte Ärzte]"),
    (r"\b(?:Vizepräsident\s+)?(?:Edgar\s+)?Wutscher\b", "[Kurie Niedergelassene Ärzte]"),
    (r"\b(?:Finanzminister\s+)?(?:Magnus\s+)?Brunner\b", "[Finanzministerium]"),
    (r"\b(?:Finanzminister\s+)?(?:Markus\s+)?Marterbauer\b", "[Finanzministerium]"),
    (r"\b(?:Bundeskanzler\s+|Kanzler\s+)?(?:Karl\s+)?Nehammer\b", "[Bundeskanzleramt]"),
    (r"\b(?:Bundeskanzler\s+|Kanzler\s+)?(?:Christian\s+)?Stocker\b", "[Bundeskanzleramt]"),
    (r"\b(?:Vizekanzler\s+)?(?:Andreas\s+)?Babler\b", "[Vizekanzleramt]"),
    (r"\b(?:Landesrat\s+)?(?:Karlheinz\s+)?Kornhäusl\b", "[Landesgesundheitsressort Steiermark]"),
    (r"\b(?:Landesrätin\s+)?(?:Beate\s+)?Prettner\b", "[Landesgesundheitsressort Kärnten]"),
    (r"\b(?:LH-Stellvertreterin\s+|Landesrätin\s+)?(?:Christine\s+)?Haberlander\b", "[Landesgesundheitsressort OÖ]"),
    (r"\b(?:Landesrätin\s+)?(?:Cornelia\s+)?Hagele\b", "[Landesgesundheitsressort Tirol]"),
    (r"\b(?:Landesrätin\s+)?(?:Martina\s+)?Rüscher\b", "[Landesgesundheitsressort Vorarlberg]"),
    (r"\b(?:Landesrat\s+)?(?:Ludwig\s+)?Schleritzko\b", "[Klinikressort NÖ]"),
    (r"\b(?:Landesrat\s+)?(?:Anton\s+)?Kasser\b", "[Gesundheitsressort NÖ]"),
    (r"\b(?:Pharmig-Generalsekretär\s+|Generalsekretär\s+)?(?:Alexander\s+)?Herzog\b", "[Pharmig-Vertretung]"),
    (r"\b(?:Ex-Neos-Abgeordnete[rn]?\s+|Abgeordnete[rn]?\s+)?(?:Gerald\s+)?Loacker\b", "[Parlamentarische Kontrolle]"),
]

# Generic pattern for "Von <Vorname> <Nachname>" author bylines and "<Funktion> <Vorname> <Nachname>"
BYLINE_RE = re.compile(r"(?m)^Von\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß-]+){1,2}\s*$")
ROLE_PERSON_RE = re.compile(
    r"\b(Gesundheitsminister(?:in)?|Finanzminister(?:in)?|Sozialminister(?:in)?|Landesr(?:at|ätin)|"
    r"Stadtr(?:at|ätin)|Klinikvorstand|Abteilungsvorstand|Primar(?:a|ä)rzt(?:in)?|"
    r"Kassenobmann|Kammerpräsident(?:in)?|Vizepräsident(?:in)?|Generalsekretär(?:in)?|"
    r"Sektionsleiter(?:in)?|Patientenanw(?:alt|ältin)|Sprecher(?:in)?)\s+"
    r"([A-ZÄÖÜ][a-zäöüß-]+\s+[A-ZÄÖÜ][a-zäöüß-]+)"
)


def anonymize_article_text(text: str) -> str:
    """
    Anonymizes real person names inside newspaper articles and statements so that
    only institutional roles are represented (`Instructions.docx` rule).
    """
    if not text:
        return ""
    out = BYLINE_RE.sub("Redaktioneller Bericht", text)
    for pattern, repl in KNOWN_PERSON_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    out = ROLE_PERSON_RE.sub(r"[\1-Vertretung]", out)
    return out


def format_date_de(iso_date: str) -> str:
    """Convert '2026-08-18' to '18.08.2026'."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", (iso_date or "").strip())
    if m:
        y, mo, d = m.groups()
        return f"{d}.{mo}.{y}"
    return iso_date or ""


def format_short_badge(newspaper: str, iso_date: str) -> str:
    """Return short readable badge label: e.g. 'Die Presse (18.08.2026)'."""
    paper = (newspaper or "Presse").strip()
    if len(paper) > 26:
        paper = paper[:24] + "…"
    return f"{paper} ({format_date_de(iso_date)})"


def enrich_article_dict(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach anonymized content & short badge label (`Zeitung (Datum)`) to an article dict."""
    anon_title = anonymize_article_text(raw_row.get("title", ""))
    anon_content = anonymize_article_text(raw_row.get("content", ""))
    date_de = format_date_de(raw_row.get("article_date", ""))
    short_badge = format_short_badge(raw_row.get("newspaper", ""), raw_row.get("article_date", ""))
    out = dict(raw_row)
    out["title"] = anon_title
    out["content"] = anon_content
    out["date_de"] = date_de
    out["short_badge"] = short_badge
    return out


def _sanitize_fts_term(term: str) -> str:
    cleaned = re.sub(r"[^\w\säöüÄÖÜß-]", " ", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def search_articles_for_agent(
    user_keywords: List[str],
    role_keywords: List[str],
    limit: int = 7,
    exclude_ids: Optional[Set[str]] = None,
    already_assigned_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top `limit` (configurable 3..20) most relevant articles from `presse_db.sqlite`
    for an agent, combining user keywords, role keywords, recency boost, and cross-agent diversity
    so different agents get genuinely different articles.
    """
    limit = max(1, min(25, int(limit)))
    exclude_ids = exclude_ids or set()
    already_assigned_counts = already_assigned_counts or {}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    clean_user_kws = [
        _sanitize_fts_term(k) for k in user_keywords if len(_sanitize_fts_term(k)) >= 2
    ]
    clean_role_kws = [
        _sanitize_fts_term(k) for k in role_keywords if len(_sanitize_fts_term(k)) >= 2
    ]

    all_terms = []
    seen_lower = set()
    for kw in clean_user_kws + clean_role_kws:
        low = kw.lower()
        if low not in seen_lower:
            seen_lower.add(low)
            all_terms.append(kw)

    if not all_terms:
        all_terms = ["Gesundheit", "Spital", "ÖGK", "Wartezeiten"]

    fts_parts = []
    for t in all_terms[:35]:
        if " " in t:
            fts_parts.append(f'"{t}"')
        else:
            fts_parts.append(f'"{t}"*')
    match_expr = " OR ".join(fts_parts)

    sql = """
        SELECT
            a.article_id,
            a.article_date,
            a.newspaper,
            a.title,
            a.author,
            a.media_type,
            a.content,
            a.char_count,
            bm25(articles_fts, 0.0, 3.0, 0.5, 1.0) AS raw_bm25
        FROM articles_fts
        JOIN articles a ON a.article_id = articles_fts.article_id
        WHERE articles_fts MATCH ?
          AND a.char_count > 220
        ORDER BY raw_bm25 ASC
        LIMIT 250
    """
    try:
        rows = cur.execute(sql, (match_expr,)).fetchall()
    except sqlite3.OperationalError:
        simple_expr = " OR ".join(f'"{t.split()[0]}"' for t in all_terms[:15] if t.split())
        rows = cur.execute(sql, (simple_expr,)).fetchall()

    conn.close()

    ref_date = datetime(2026, 9, 30)
    scored_candidates = []
    user_kws_lower = [k.lower() for k in clean_user_kws]
    role_kws_lower = [k.lower() for k in clean_role_kws]

    for r in rows:
        aid = r["article_id"]
        if aid in exclude_ids:
            continue

        title_low = (r["title"] or "").lower()
        text_low = (r["content"] or "").lower()
        combined_low = title_low + "\n" + text_low

        matched_user_kws = [kw for kw in user_kws_lower if kw in combined_low]
        matched_role_kws = [kw for kw in role_kws_lower if kw in combined_low]
        title_hits = sum(1 for kw in (user_kws_lower + role_kws_lower) if kw in title_low)

        user_kw_score = len(matched_user_kws) * 3.0 if matched_user_kws else 0.0
        role_kw_score = len(matched_role_kws) * 4.8
        if matched_user_kws and matched_role_kws:
            role_kw_score += 4.5
        bm25_score = max(0.0, -float(r["raw_bm25"]))

        try:
            dt = datetime.strptime(r["article_date"], "%Y-%m-%d")
            days_old = max(0, (ref_date - dt).days)
        except Exception:
            days_old = 500
        # Smooth recency weighting: newer articles are favored, but older articles still have >0 probability
        recency_factor = max(0.30, 1.0 - (days_old / 1250.0))
        recency_bonus = recency_factor * 5.0

        has_numbers_bonus = (
            2.5
            if re.search(
                r"\b\d+(?:[.,]\d+)?\s*(?:Prozent|%|Mio\.|Millionen|Milliarden|Mrd\.|Euro|Tage|Monate)\b",
                r["content"],
            )
            else 0.0
        )

        # Diversity penalty if this article was already assigned to other agents in Round 1
        assigned_times = already_assigned_counts.get(aid, 0)
        diversity_penalty = assigned_times * 9.5

        total_score = (
            user_kw_score
            + role_kw_score
            + (title_hits * 2.0)
            + (bm25_score * 0.8)
            + recency_bonus
            + has_numbers_bonus
            - diversity_penalty
        )

        enriched = enrich_article_dict(
            {
                "article_id": aid,
                "article_date": r["article_date"],
                "newspaper": r["newspaper"],
                "title": r["title"],
                "media_type": r["media_type"],
                "content": r["content"],
                "char_count": r["char_count"],
                "relevance_score": round(total_score, 2),
                "recency_factor": round(recency_factor, 2),
                "matched_user_keywords": matched_user_kws,
                "matched_role_keywords": matched_role_kws,
                "total_keyword_hits": len(matched_user_kws) + len(matched_role_kws),
            }
        )
        scored_candidates.append(enriched)

    scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    selected = []
    seen_titles = set()
    for cand in scored_candidates:
        norm_title = re.sub(r"\W+", "", cand["title"].lower())[:40]
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        selected.append(cand)
        if len(selected) >= limit:
            break

    return selected


def get_articles_by_ids(article_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch full anonymized article records (including `short_badge`) by exact `article_id` list."""
    clean_ids = [aid for aid in article_ids if isinstance(aid, str) and aid.startswith("ART-")]
    if not clean_ids:
        return {}
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in clean_ids)
    rows = cur.execute(
        f"""
        SELECT article_id, article_date, newspaper, title, media_type, content, char_count
        FROM articles
        WHERE article_id IN ({placeholders})
        """,
        clean_ids,
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        result[r["article_id"]] = enrich_article_dict(
            {
                "article_id": r["article_id"],
                "article_date": r["article_date"],
                "newspaper": r["newspaper"],
                "title": r["title"],
                "media_type": r["media_type"],
                "content": r["content"],
                "char_count": r["char_count"],
            }
        )
    return result


def get_db_statistics() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return {"total_articles": 0}
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    min_d, max_d = cur.execute(
        "SELECT MIN(article_date), MAX(article_date) FROM articles"
    ).fetchone()
    top_papers = cur.execute(
        "SELECT newspaper, COUNT(*) as c FROM articles GROUP BY newspaper ORDER BY c DESC LIMIT 8"
    ).fetchall()
    conn.close()
    return {
        "total_articles": total,
        "min_date": min_d,
        "max_date": max_d,
        "top_newspapers": [{"newspaper": r[0], "count": r[1]} for r in top_papers],
    }
