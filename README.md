# HealthPolicy Simulator Österreich

Ein interaktiver, faktenbasierter **Multi-Agenten-Verhandlungssimulator für österreichische Gesundheitspolitik**, verankert in den offiziellen Strukturdaten von Statistik Austria (2022-2024; `grundfakten.yaml`) und einer volltextindizierten Pressedatenbank (`presse_db.sqlite` mit **11.412 Zeitungsartikeln** aus den Jahren 2024–2026).

---

## 1. Beschreibung der Lösung und des zugrunde liegenden Ansatzes

Gesundheitspolitische Reformen in Österreich scheitern selten an fehlenden Ideen, sondern an den komplexen Zielkonflikten zwischen **Bund (Gesundheits- & Finanzministerium)**, **Sozialversicherung (ÖGK)**, **Bundesländern (Spitalsträgern)**, **Ärztekammer** und **Patientenanwaltschaft**.

Der **HealthPolicy Simulator Österreich** bildet diesen realpolitischen Aushandlungsprozess als mehrstufige, mathematisch und empirisch kontrollierte Multi-Agenten-Simulation ab:

### Kernarchitektur & Ablauf
1. **Intent- & Ziel-Analyse (`analyze_user_input`)**:
   - Erkennt automatisch, ob der User eine **offene Zielfrage** (z. B. *„Wartezeiten um 20 % senken bei max. 5 % Budgeterhöhung“*) oder ein **konkretes Maßnahmen-Proposal** eingibt, und extrahiert Zielgröße, Zeithorizont, Budget-Obergrenze (`max_budget_pct`) sowie Suchbegriffe.
2. **Diversifizierte Presse-Retrieval-Engine (`db_search.py`)**:
   - Durchsucht **11.412 österreichische Presseartikel (2024–2026)** mittels SQLite FTS5 Volltextindex.
   - Kombiniert thematische Relevanz (BM25), rollenspezifische Schlüsselwörter, einen **Cross-Agent-Diversitäts-Malus** (damit nicht alle Agenten dieselben Artikel erhalten) sowie eine geglättete Aktualitätsgewichtung, durch die neuere Artikel bevorzugt werden, ältere Artikel aber weiterhin eine faire Chance haben, wenn sie viele Schlüsselwörter enthalten.
3. **Eröffnungsrunde (Phase 0 – Eingangs-Positionen)**:
   - Alle ausgewählten Institutionen formulieren parallel ihre Eingangs-Position auf Basis ihrer Presse-Quellen und berechnen über das Tool `PolicyCalculator` ihr eigenes Start-Maßnahmenpaket. In der Eröffnungsrunde findet noch keine Selbst-Abstimmung statt.
4. **Dreistufige Verhandlungsrunden (Runde 1 bis $N$)**:
   - **Schritt A – Quellen-Kritik & Einzel-Proposals**: Jeder Agent wählt gezielt 2 von Gegenspielern zitierte Zeitungsartikel aus, liest deren Original-Volltext aus der Datenbank, prüft sie kritisch auf Übertreibungen oder selektive Zitate und berechnet sein aktualisiertes Einzel-Proposal (mit aktivem Kosten-/Effekt-Feedback im Tool-Loop, um das erlaubte Budget von z. B. 5 % bestmöglich auszuschöpfen).
   - **Schritt B – Intelligente Aggregation durch den Proposal-Agenten**: Erst nachdem alle Einzel-Proposals der Runde vorliegen, erstellt der **Proposal-Agent (Kompromiss-Architekt)** auf Basis einer Kosten-Wirksamkeits-Matrix (Cost-Effectiveness-Ratio € pro +1 % Effekt) und politischer Koalitionslogik das gemeinsame **Kompromiss-Paket (`main_proposal`)** der Runde.
   - **Schritt C – Separates Abstimmungs-Votum**: Erst nach Vorliegen des aggregierten Kompromiss-Pakets erhalten alle Agenten diesen Entwurf und stimmen verbindlich darüber ab.
5. **Doppelte Anti-Halluzinations-Schranke (`PolicyCalculator` + `Fact-Checker`)**:
   - **Kein Kopfrechnen durch Sprachmodelle**: Agenten dürfen Zahlen nicht frei erfinden, sondern müssen eines der 6 deterministischen Rechentools in `calculator.py` aufrufen.
   - **Bestands-Verankerung (Stock-Flow-Ratio)**: Jede neue Kapazität (z. B. +120 Primärversorgungseinheiten oder +60 Facharztzentren) wird gegen den realen österreichischen Gesamtbestand (`grundfakten.yaml`) ins Verhältnis gesetzt und mit einer Sättigungskurve (abnehmender Grenznutzen) sowie Doppelzählungs-Schutz bewertet.
   - **Deterministischer Fact-Checker**: Überprüft nach jedem LLM-Aufruf Kosten, Budget-Prozentsatz ($1\,\% = 578{,}37\text{ Mio. €}$) und System-Effekt und korrigiert überhöhte Effekt-Versprechen automatisch auf den physikalisch-ökonomisch belegbaren Ankerwert.
6. **Strategische Handlungsempfehlung (Tab 3)**:
   - Generiert auf Knopfdruck für jede wählbare Institution ein konkretes Verhandlungs-Briefing mit erwarteten Angriffen der Gegenseite, faktenbasierten Kontern (inkl. `[ART-...]` Quelle), empfohlenen Koalitionen sowie *Red Lines & Quid-pro-quo*-Tauschgeschäften.

---

## 2. Anleitung zur Ausführung des Prototyps

### Option A: Ein-Klick-Start (macOS)
Doppelklick auf die Datei **`Start_Simulator.command`** im Projektordner. Das Skript wechselt automatisch in das Projektverzeichnis, startet den lokalen Webserver auf Port `8501` und öffnet `http://127.0.0.1:8501` im Standard-Browser.

### Option B: Start über das Terminal (macOS / Linux / Windows)
```bash
# 1. In das Projektverzeichnis wechseln
cd Pfad/zum/Projektordner

# 2. Webserver starten (entpackt beim ersten Start automatisch presse_db.sqlite.gz)
python3 server.py 8501
```
Anschließend im Browser folgende Adresse öffnen:
👉 **`http://127.0.0.1:8501`**

*(Hinweis: Beim allerersten Start nach dem Klonen von GitHub entpackt `db_search.py` automatisch das komprimierte Datenbank-Archiv `presse_db.sqlite.gz` in die Datei `presse_db.sqlite`.)*

---

## 3. Notwendige Abhängigkeiten und Voraussetzungen

* **Python-Version**: Python `3.10` oder höher.
* **Python-Pakete** (installierbar über `pip install -r requirements.txt`):
  * `openai>=1.40.0` (für die Kommunikation mit dem Azure OpenAI Responses/Chat-Endpunkt)
  * `pyyaml>=6.0.0` (zum Laden von `grundfakten.yaml`)
  * `pypdf>=4.0.0` (nur erforderlich, falls die Pressedatenbank über `build_db.py` neu aus Roh-PDFs aufgebaut werden soll)
* **Datenbank**: SQLite3 mit aktiviertem `FTS5`-Modul (in der Python-Standardbibliothek auf macOS, Linux und Windows standardmäßig enthalten).
* **Internetverbindung**: Für API-Aufrufe an Azure OpenAI sowie zum Laden von `Chart.js` im Browser.

Installation der Abhängigkeiten:
```bash
pip install -r requirements.txt
```

---

## 4. Gegebenenfalls erforderliche Konfigurationsschritte

Damit keine geheimen API-Schlüssel im Quellcode oder auf GitHub gespeichert werden, liest die Engine ihre Zugangsdaten aus Umgebungsvariablen bzw. einer lokalen Datei **`.env`** im Projektverzeichnis.

1. Vorlage kopieren:
   ```bash
   cp .env.example .env
   ```
2. In der Datei `.env` den eigenen API-Schlüssel eintragen:
   ```env
   AZURE_OPENAI_API_KEY=ihr_api_schluessel_hier
   AZURE_OPENAI_BASE_URL=https://my-openai-4115.openai.azure.com/openai/v1/
   AZURE_OPENAI_DEPLOYMENT=gpt-5.2-chat
   ```
*(Die Datei `.env` ist in `.gitignore` eingetragen und wird nicht zu GitHub hochgeladen.)*

---

## 5. Bekannte Einschränkungen und noch offene Punkte

1. **Laufzeit & API-Ratenlimits (Latenz bei mehreren Verhandlungsrunden)**:
   - Da in jeder Verhandlungsrunde zunächst alle teilnehmenden Institutionen parallel ihre Quellen-Kritik und ihr Einzel-Proposal inkl. `PolicyCalculator`-Feedback-Schleife berechnen, anschließend der Proposal-Agent das Kompromiss-Paket aggregiert und danach alle Agenten darüber abstimmen, dauern Simulationen mit 6 Agenten und 3–5 Runden je nach Azure-OpenAI-Antwortzeit ca. **35 bis 90 Sekunden**.
2. **Strikte Fakten-Verankerung auf den Zeitraum 2024–2026**:
   - Die mitgelieferte Pressedatenbank (`presse_db.sqlite`) umfasst **11.412 kuratierte Artikel** aus österreichischen Leitmedien der Jahre **2024 bis März 2026**. Ältere Pressespiegel-Jahrgänge (z. B. 2020–2023) sind bewusst nicht enthalten, um veraltete COVID-19-Sonderregelungen und historische Budgetzahlen nicht mit der aktuellen Finanzausgleichs-Periode (2024–2028) zu vermischen.
3. **Makro-Modellierung im `PolicyCalculator`**:
   - Die 7 kalibrierten Hebel-Kategorien (`pve_ausbau`, `kassenstellen_tarif`, `facharztzentren_ambulanzen`, `digital_1450_elga`, `pflege_personal_betten`, `spitalsambulanzen_entlastung`, `praevention_screening`) bilden die österreichischen Makro-Strukturdaten aus `grundfakten.yaml` auf Bundesebene ab. Regionale Detailunterschiede zwischen einzelnen Bundesländern (z. B. unterschiedliche Landesgesundheitsfonds-Abrechnungsmodelle) werden aggregiert betrachtet.
4. **In-Memory Job-Verwaltung**:
   - Der lokale HTTP-Server (`server.py`) hält laufende Simulations-Jobs im Arbeitsspeicher (`JOBS`). Für einen produktiven Multi-User-Cloud-Betrieb sollte der Job-Store auf Redis/PostgreSQL ausgelagert werden.
  
### Zum Modell-Design & Weiterentwicklung

5. **Lineare Aggregation der Einzeleffekte**:
   - Der Gesamteffekt eines Kompromiss-Pakets ergibt sich derzeit aus der **linearen Summe** der Einzeleffekte aller Maßnahmen (nur begrenzt durch die Hebel-Deckel und den Double-Counting Guard). In der Realität überlagern sich Maßnahmen: Sind z. B. bereits mehr Kassenstellen mit kaum Wartezeit vorhanden, bringt eine digitale Steuerung über 1450 nicht so viel zusätzlichen Nutzen.
   - **Weiterentwicklung:** Wechselwirkungen und abnehmender Grenznutzen zwischen Maßnahmen sollen in die Berechnung des Gesamteffekts einfließen.
6. **Wirkungsannahmen statt kalibriertem Wirkungsmodell**:
   - Die Effekte der Maßnahmen werden aus Presseartikeln und den Grundzahlen der Statistik-Austria-Reports abgeleitet. Elastizitäten, Flächenhebel-Faktor und Maximaleffekte pro Hebel sind feste Modellannahmen in `grundfakten.yaml`, keine empirisch geschätzten Werte.
   - **Weiterentwicklung:** Ein echtes, kalibriertes Rechenmodell auf Basis von Versorgungs-, Kapazitäts- und Wartezeitdaten würde die Aussagekraft deutlich erhöhen.

> **Fazit:** Die Qualität der Simulation steht und fällt mit der Datengrundlage – KI-Systeme brauchen reliable Daten!
