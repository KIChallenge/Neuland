#!/usr/bin/env python3
"""
Fast & Exact Parser for APA-Comm / APA-DeFacto Pressespiegel PDFs -> SQLite FTS5 Database.
Extracts only clean digital plain-text article pages (skipping duplicate newspaper scan pages)
and builds `presse_db.sqlite` with full metadata and BM25 + recency search support.
"""

import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
PRESSE_DIR = WORKSPACE_DIR / "presse_data"
DB_PATH = WORKSPACE_DIR / "presse_db.sqlite"
PDF_DUMP_BIN = Path("/tmp/pdf_dump")

# Compile the Objective-C PDFKit helper if not present
OBJC_SOURCE = r"""
#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) return 1;
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        PDFDocument *doc = [[PDFDocument alloc] initWithURL:[NSURL fileURLWithPath:path]];
        if (!doc) return 2;
        NSInteger count = [doc pageCount];
        for (NSInteger i = 0; i < count; i++) {
            PDFPage *page = [doc pageAtIndex:i];
            NSString *str = [page string];
            if (str) {
                printf("<<<PAGE_%ld>>>\n%s\n", (long)(i+1), [str UTF8String]);
            } else {
                printf("<<<PAGE_%ld>>>\n\n", (long)(i+1));
            }
        }
    }
    return 0;
}
"""


def ensure_pdf_dumper():
    if not PDF_DUMP_BIN.exists():
        subprocess.run(
            [
                "clang",
                "-O2",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "PDFKit",
                "-x",
                "objective-c",
                "-",
                "-o",
                str(PDF_DUMP_BIN),
            ],
            input=OBJC_SOURCE.encode("utf-8"),
            check=True,
        )


TOC_LINE_RE = re.compile(r"^S\.\s*(\d+)\s+(.+)$")
DATE_SOURCE_RE = re.compile(r"^(\d{2}\s*\.\s*\d{2}\s*\.\s*\d{4})\s*,\s*(.+)$")
FOOTER_RE = re.compile(r"^Seite\s+\d+\s+von\s+\d+\s+©\s+APA-(?:Comm|DeFacto)\s+GmbH.*$")
META_LINE_RE = re.compile(
    r"^(Quelle|Sender|Mutation|Typ|Land|Autor:in|Sprache)\s+(.+)$"
)


def normalize_date(raw_date: str) -> str:
    """Convert '07 .10.2024' or '07.10.2024' to ISO '2024-10-07'."""
    clean = re.sub(r"\s+", "", raw_date)
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", clean)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{mo}-{d}"
    return clean


def extract_issue_date_from_filename(pdf_path: Path) -> str:
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", pdf_path.name)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{mo}-{d}"
    return "2024-01-01"


def parse_single_pdf(pdf_path: Path) -> list[dict]:
    """Extract all articles from a single APA-Comm Pressespiegel PDF."""
    try:
        proc = subprocess.run(
            [str(PDF_DUMP_BIN), str(pdf_path)],
            capture_output=True,
            timeout=60,
        )
        raw_out = proc.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[WARN] Failed reading {pdf_path}: {e}", file=sys.stderr)
        return []

    # Split into pages (1-indexed dict)
    pages: dict[int, str] = {}
    parts = re.split(r"<<<PAGE_(\d+)>>>\n", raw_out)
    for idx in range(1, len(parts) - 1, 2):
        pnum = int(parts[idx])
        pages[pnum] = parts[idx + 1]

    if not pages:
        return []

    max_page = max(pages.keys())
    issue_date_iso = extract_issue_date_from_filename(pdf_path)
    issue_compact = issue_date_iso.replace("-", "")

    # 1. Find Inhaltsverzeichnis pages (usually page 2, sometimes continues on page 3)
    toc_entries = []
    for pnum in range(1, min(6, max_page + 1)):
        ptxt = pages.get(pnum, "")
        if "Inhaltsverzeichnis" in ptxt or (toc_entries and pnum == 3 and "S. " in ptxt):
            lines = [ln.strip() for ln in ptxt.splitlines() if ln.strip()]
            i = 0
            while i < len(lines):
                m_toc = TOC_LINE_RE.match(lines[i])
                if m_toc:
                    start_page = int(m_toc.group(1))
                    title = m_toc.group(2).strip()
                    art_date_iso = issue_date_iso
                    source_info = ""
                    if i + 1 < len(lines):
                        m_ds = DATE_SOURCE_RE.match(lines[i + 1])
                        if m_ds:
                            art_date_iso = normalize_date(m_ds.group(1))
                            source_info = m_ds.group(2).strip()
                            i += 1
                    toc_entries.append(
                        {
                            "start_page": start_page,
                            "toc_title": title,
                            "art_date_iso": art_date_iso,
                            "toc_source_info": source_info,
                        }
                    )
                i += 1

    # Fallback if TOC was not matched
    if not toc_entries:
        return []

    articles = []
    for idx, entry in enumerate(toc_entries):
        start_p = entry["start_page"]
        end_p = (
            toc_entries[idx + 1]["start_page"] - 1
            if idx + 1 < len(toc_entries)
            else max_page
        )

        # Collect only digital plain-text pages of this article!
        # Digital plain-text pages ALWAYS start with "Pressespiegel" on the first non-empty line
        # and end right when the metadata block ("Quelle ...", "Sprache Deutsch" / "Land AT") appears.
        # Scan pages that follow do NOT start with "Pressespiegel".
        text_lines = []
        meta = {
            "Quelle": "",
            "Sender": "",
            "Mutation": "",
            "Typ": "",
            "Land": "AT",
            "Autor:in": "",
            "Sprache": "Deutsch",
        }
        reached_meta_end = False

        for p in range(start_p, end_p + 1):
            ptxt = pages.get(p, "")
            if not ptxt:
                continue
            plines = [ln.strip() for ln in ptxt.splitlines() if ln.strip()]
            if not plines:
                continue

            # Check if this is an APA digital text page (starts with "Pressespiegel")
            if plines[0] != "Pressespiegel" and p > start_p:
                # This page is the scanned newspaper clipping -> skip!
                break

            # Remove top "Pressespiegel" header and bottom "© APA-..." footer
            body_lines = []
            for ln in plines:
                if ln == "Pressespiegel":
                    continue
                if FOOTER_RE.match(ln):
                    continue
                if ln.startswith("CoreGraphics PDF has logged"):
                    continue
                body_lines.append(ln)

            # Parse metadata at the bottom of the digital text page if present
            for ln in body_lines:
                m_meta = META_LINE_RE.match(ln)
                if m_meta:
                    k, v = m_meta.group(1), m_meta.group(2).strip()
                    meta[k] = v
                    if k in ("Sprache", "Land"):
                        reached_meta_end = True
                else:
                    text_lines.append(ln)

            if reached_meta_end:
                # Digital text ends right at the metadata block; subsequent pages in range are scans
                break

        # Clean header lines if first line repeats "<Quelle> vom <Datum>" or title
        if text_lines and re.search(r"\bvom\s+\d{2}\.\d{2}\.\d{4}$", text_lines[0]):
            header_source_line = text_lines.pop(0)
            if not meta["Quelle"]:
                meta["Quelle"] = re.sub(
                    r"\s+vom\s+\d{2}\.\d{2}\.\d{4}$", "", header_source_line
                ).strip()

        # Determine newspaper/medium name
        newspaper = meta["Quelle"] or meta["Sender"]
        if not newspaper and entry["toc_source_info"]:
            newspaper = entry["toc_source_info"].split(",")[0].split(" Seite")[0].strip()

        full_text = "\n".join(text_lines).strip()
        if not full_text:
            continue

        article_id = f"ART-{issue_compact}-{idx+1:03d}"
        # Handle rare case where multiple PDFs share the same date
        rel_pdf = str(pdf_path.relative_to(PRESSE_DIR))

        articles.append(
            {
                "article_id": article_id,
                "issue_date": issue_date_iso,
                "article_date": entry["art_date_iso"],
                "newspaper": newspaper or "Unbekannt",
                "title": entry["toc_title"],
                "author": meta["Autor:in"],
                "media_type": meta["Typ"],
                "mutation": meta["Mutation"],
                "country": meta["Land"],
                "pdf_path": rel_pdf,
                "start_page": start_p,
                "end_page": end_p,
                "content": full_text,
                "char_count": len(full_text),
            }
        )

    return articles


def build_database():
    t0 = time.time()
    ensure_pdf_dumper()

    pdf_files = sorted(
        p for p in PRESSE_DIR.rglob("*.pdf") if "2020-2023" not in str(p)
    )
    print(f"Found {len(pdf_files)} PDF files (2024-2026) in {PRESSE_DIR}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY,
            issue_date TEXT NOT NULL,
            article_date TEXT NOT NULL,
            newspaper TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            media_type TEXT,
            mutation TEXT,
            country TEXT,
            pdf_path TEXT NOT NULL,
            start_page INTEGER,
            end_page INTEGER,
            content TEXT NOT NULL,
            char_count INTEGER
        );

        CREATE INDEX idx_articles_date ON articles(article_date DESC);
        CREATE INDEX idx_articles_newspaper ON articles(newspaper);

        CREATE VIRTUAL TABLE articles_fts USING fts5(
            article_id UNINDEXED,
            title,
            newspaper,
            content,
            tokenize = 'unicode61 remove_diacritics 0'
        );
        """
    )

    all_articles = []
    seen_ids = set()

    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_pdf = {pool.submit(parse_single_pdf, p): p for p in pdf_files}
        done_count = 0
        for fut in as_completed(future_to_pdf):
            done_count += 1
            arts = fut.result()
            all_articles.extend(arts)
            if done_count % 100 == 0 or done_count == len(pdf_files):
                print(
                    f"  Parsed {done_count}/{len(pdf_files)} PDFs ({len(all_articles)} articles so far)..."
                )

    # Sort deterministically by issue_date, pdf_path, start_page
    all_articles.sort(key=lambda a: (a["issue_date"], a["pdf_path"], a["start_page"]))

    rows_main = []
    rows_fts = []
    for a in all_articles:
        aid = a["article_id"]
        if aid in seen_ids:
            suffix = 2
            while f"{aid}-{suffix}" in seen_ids:
                suffix += 1
            aid = f"{aid}-{suffix}"
            a["article_id"] = aid
        seen_ids.add(aid)

        rows_main.append(
            (
                a["article_id"],
                a["issue_date"],
                a["article_date"],
                a["newspaper"],
                a["title"],
                a["author"],
                a["media_type"],
                a["mutation"],
                a["country"],
                a["pdf_path"],
                a["start_page"],
                a["end_page"],
                a["content"],
                a["char_count"],
            )
        )
        rows_fts.append((a["article_id"], a["title"], a["newspaper"], a["content"]))

    cur.executemany(
        """
        INSERT INTO articles (
            article_id, issue_date, article_date, newspaper, title,
            author, media_type, mutation, country, pdf_path,
            start_page, end_page, content, char_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_main,
    )
    cur.executemany(
        """
        INSERT INTO articles_fts (article_id, title, newspaper, content)
        VALUES (?, ?, ?, ?)
        """,
        rows_fts,
    )
    conn.commit()

    elapsed = time.time() - t0
    print(
        f"\nSUCCESS: Built {DB_PATH} with {len(rows_main)} articles from {len(pdf_files)} PDFs in {elapsed:.1f}s!"
    )
    conn.close()


if __name__ == "__main__":
    build_database()
