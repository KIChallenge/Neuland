#!/usr/bin/env python3
"""
Interactive Web Server & API for HealthPolicy Simulator (UC2).
Provides:
- Real-time Server-Sent Events (SSE) / Background Job streaming so the user sees every step live
  (Keyword Agent -> DB Retrieval -> Round 1a Initial Statements + PolicyCalculator -> Round 1b 1-Token "Grain of Salt"
  -> Proposal Agent & Pressure -> Consensus Check -> Round 2..N Loop).
- Endpoints for inspecting full articles from `presse_db.sqlite` (11,412 articles).
- Endpoint for generating strategic Action Recommendations ("Handlungsempfehlung") for any chosen role.
- Serves the rich interactive Single-Page Application (`index.html`).
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Dict, List
import json
import threading
import time
import urllib.parse
import uuid

from agents_engine import (
    STAKEHOLDER_ROLES,
    VOTING_SYSTEMS,
    generate_role_action_recommendation,
    run_policy_simulation,
)
from calculator import GRUNDFAKTEN, BUDGET_BASELINES_2024_MIO
from db_search import get_articles_by_ids, get_db_statistics, search_articles_for_agent

WORKSPACE_DIR = Path(__file__).resolve().parent
STATIC_HTML_PATH = WORKSPACE_DIR / "index.html"
LOGO_PATH = WORKSPACE_DIR / "logo.png"

# In-memory store for simulation jobs & results
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _start_simulation_job(payload: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())[:8]
    user_input = payload.get(
        "user_input",
        "Wie können die Wartezeiten auf Facharzttermine innerhalb von fünf Jahren um 20 % reduziert werden, ohne die Gesundheitsausgaben um mehr als 5% zu erhöhen?",
    )
    selected_role_ids = payload.get(
        "selected_role_ids",
        [r.role_id for r in STAKEHOLDER_ROLES.values() if r.default_selected],
    )
    compromise_sliders = {
        k: int(v) for k, v in payload.get("compromise_sliders", {}).items()
    }
    voting_system_id = payload.get("voting_system_id", "veto_finanz_sv")
    max_rounds = int(payload.get("max_rounds", 3))
    articles_per_agent = int(payload.get("articles_per_agent", 7))
    tokens_per_round = int(payload.get("tokens_per_round", 2))

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.02,
            "message": "Initialisiere Multi-Agenten-Simulation...",
            "logs": [
                {
                    "time": time.strftime("%H:%M:%S"),
                    "msg": "Simulation gestartet.",
                    "progress": 0.02,
                }
            ],
            "result": None,
            "error": None,
            "created_at": time.time(),
        }

    def _progress_cb(msg: str, prog: float):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["message"] = msg
                JOBS[job_id]["progress"] = round(float(prog), 3)
                JOBS[job_id]["logs"].append(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "msg": msg,
                        "progress": round(float(prog), 3),
                    }
                )

    def _worker():
        try:
            res = run_policy_simulation(
                user_input=user_input,
                selected_role_ids=selected_role_ids,
                compromise_sliders=compromise_sliders,
                voting_system_id=voting_system_id,
                max_rounds=max_rounds,
                articles_per_agent=articles_per_agent,
                tokens_per_round=tokens_per_round,
                progress_callback=_progress_cb,
            )
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["progress"] = 1.0
                JOBS[job_id]["result"] = res
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"[ERROR] Simulation job {job_id} failed:\n{tb}")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(e)
                JOBS[job_id]["message"] = f"Fehler: {e}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id


class SimulatorRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Keep console logs clean
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            if STATIC_HTML_PATH.exists():
                raw = STATIC_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json({"error": "index.html not found"}, 404)
            return

        if path == "/logo.png":
            if LOGO_PATH.exists():
                raw = LOGO_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json({"error": "logo.png not found"}, 404)
            return

        if path == "/api/config":
            roles_list = [
                {
                    "role_id": r.role_id,
                    "name": r.name,
                    "short_name": r.short_name,
                    "icon": r.icon,
                    "color": r.color,
                    "description": r.description,
                    "core_interests": r.core_interests,
                    "red_lines": r.red_lines,
                    "role_keywords": r.role_keywords,
                    "default_selected": r.default_selected,
                    "default_compromise": r.default_compromise,
                }
                for r in STAKEHOLDER_ROLES.values()
            ]
            self._send_json(
                {
                    "roles": roles_list,
                    "voting_systems": VOTING_SYSTEMS,
                    "db_stats": get_db_statistics(),
                    "grundfakten": GRUNDFAKTEN,
                    "budget_baselines_2024_mio": BUDGET_BASELINES_2024_MIO,
                    "example_prompts": [
                        {
                            "label": "Wie können die Wartezeiten auf Facharzttermine innerhalb von fünf Jahren um 20 % reduziert werden, ohne die Gesundheitsausgaben um mehr als 5% zu erhöhen?",
                            "text": "Wie können die Wartezeiten auf Facharzttermine innerhalb von fünf Jahren um 20 % reduziert werden, ohne die Gesundheitsausgaben um mehr als 5% zu erhöhen?",
                        },
                        {
                            "label": "Wie können die Gesundheitsausgaben bis 2035 um 10 % gesenkt werden, ohne die Versorgungsqualität wesentlich zu verschlechtern?",
                            "text": "Wie können die Gesundheitsausgaben bis 2035 um 10 % gesenkt werden, ohne die Versorgungsqualität wesentlich zu verschlechtern?",
                        },
                    ],
                }
            )
            return

        if path == "/api/job":
            job_id = qs.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self._send_json({"error": "Job not found"}, 404)
                    return
                self._send_json(job)
            return

        if path == "/api/article":
            aid = qs.get("id", [""])[0]
            arts = get_articles_by_ids([aid])
            if aid in arts:
                self._send_json(arts[aid])
            else:
                self._send_json({"error": f"Article {aid} not found"}, 404)
            return

        if path == "/api/search_db":
            q = qs.get("q", [""])[0]
            kws = [k.strip() for k in q.split(",") if k.strip()] or [q]
            results = search_articles_for_agent(kws, [], limit=15)
            self._send_json({"results": results})
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body_raw)
        except Exception:
            payload = {}

        if path == "/api/simulate":
            job_id = _start_simulation_job(payload)
            self._send_json({"job_id": job_id, "status": "running"})
            return

        if path == "/api/recommendation":
            job_id = payload.get("job_id", "")
            role_id = payload.get("role_id", "oegk_sv")
            sim_result = payload.get("simulation_result")
            if not sim_result and job_id:
                with JOBS_LOCK:
                    sim_result = JOBS.get(job_id, {}).get("result")
            if not sim_result:
                self._send_json({"error": "Keine Simulationsergebnisse gefunden."}, 400)
                return
            rec = generate_role_action_recommendation(role_id, sim_result)
            self._send_json(rec)
            return

        self._send_json({"error": "Unknown endpoint"}, 404)


def run_server(port: int = 8501) -> None:
    server = ThreadedHTTPServer(("127.0.0.1", port), SimulatorRequestHandler)
    print(f"HealthPolicy Simulator server running on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port)
