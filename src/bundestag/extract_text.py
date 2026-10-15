"""Schritt 2: PDF -> Text in korrekter Lesereihenfolge.

Die Protokolle sind zweispaltig gesetzt. Eine naive Extraktion (z. B.
``page.get_text()`` oder ``pypdf``) liefert die Textblöcke in der Reihenfolge,
in der sie im PDF-Inhaltsstrom stehen. Das ist bei diesen Dateien meist, aber
nicht immer spaltenweise korrekt; Zwischenrufe und Kolumnentitel geraten
dabei auch an falsche Stellen. Deshalb wird hier jede Textzeile mit ihren
Koordinaten und Schriftmerkmalen aus PyMuPDF gelesen und explizit nach
(Seite, Spalte, y, x) sortiert.

Ausgabe je Sitzung:
* ``data/lines/20NNN.jsonl`` – eine Zeile pro Textzeile inkl. Layout-Merkmalen
  (Schrift und Einrückung trennen hier Rednerköpfe und Zwischenrufe ab; der Parser
  liest daraus Datum und Inhaltsverzeichnis, validate.py die PDF-Seite).
* ``data/txt/20NNN.txt`` – bereinigter Fließtext mit Absätzen und ohne
  Silbentrennung (Eingabe für Parser, Metriken und Sprachmodell).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from functools import partial

import pymupdf

from . import config

BEGIN_RE = re.compile(r"^Beginn: ?(\d{1,2})[.:](\d{2}) ?Uhr")
END_RE = re.compile(r"^\(Schluss: ?(\d{1,2})[.:](\d{2}) ?Uhr\)")
ANLAGE_RE = re.compile(r"^Anlage \d+$")
# Ergänzungsstrich am Zeilenende vor Konjunktion: "Außen-" + "und Sicherheitspolitik".
# Nicht aufgenommen, weil in data/lines überwiegend Silbenfragmente: "so-|wie" (111),
# "an-|statt" (19), "den-|noch" (13), "Canna-|bis" (12).
CONJ_RE = re.compile(r"^(?:und|oder|bzw\.|sowie|beziehungsweise|respektive|als auch)(?=[\s,;:/)]|$)")
# Versal-Silbentrennung im Fraktionsnamen ("GRÜ-|NEN" 4681, "BÜND-|NIS" 1778,
# "BÜNDNIS-|SES" 1416 in data/lines). Andere Versalwörter (EU-SILC, EU-ETS vs.
# UNIFIL, OSZE) sind nicht eindeutig und behalten den Strich.
CAPS_SPLIT = ("GRÜNEN", "BÜNDNISSES", "BÜNDNIS")
# Die längste der 1947 Kopf-Fortsetzungen in Times ("NEN):", "Amt:") hat in data/lines 58 Zeichen.
KOPF_FORTSETZUNG_MAX = 60
# Sitzungen sind voneinander unabhängig und rein CPU-gebunden (PyMuPDF), daher ein Prozess je Kern.
# Messung 15.10.2026 auf M1 Pro mit 8 Kernen: sequenziell 170,6 s, 8 Prozesse 28,7 s, alle 201 Dateien in data/txt
# und data/lines bytegleich (SHA-256).
WORKER = os.cpu_count() or 1
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

log = logging.getLogger(__name__)


@dataclass
class Line:
    page: int
    col: int          # 0 = linke, 1 = rechte Spalte
    x0: float
    y0: float
    x1: float
    text: str
    speaker_font: bool  # beginnt mit Arial fett (Rednerkopf)
    bold_chars: int     # Anzahl fett gesetzter Zeichen (Times fett: Titel, Drucksachen)
    size: float
    section: str = ""   # "toc" | "body" | "annex"


def _page_lines(page: pymupdf.Page, pno: int) -> list[Line]:
    """Alle Spans einer Seite einsammeln und zu Zeilen je Spalte gruppieren.

    Gruppiert wird über die Grundlinie (y1) statt über PyMuPDFs eigene
    Zeilenobjekte, weil Rednername (Arial fett) und Partei (Times) teils als
    getrennte Zeilenobjekte mit 1 pt Versatz geliefert werden.
    """
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                if y0 < config.HEADER_Y or y0 > config.FOOTER_Y:
                    continue  # Kopfzeile mit Kolumnentitel (y < 72) und Druckvermerk (y > 790) sind kein Redetext
                if x0 < config.TEXT_X_MIN or x0 > config.TEXT_X_MAX:
                    continue  # Marginalien (A)–(D) stehen außerhalb x 60–538 und gehören nicht zum Text
                col = 0 if x0 < config.COLUMN_SPLIT_X else 1
                spans.append((col, y1, x0, s))
    spans.sort(key=lambda t: (t[0], t[1], t[2]))

    lines: list[Line] = []
    cur: list = []

    def flush():
        if not cur:
            return
        cur.sort(key=lambda t: t[2])
        text = "".join(t[3]["text"] for t in cur)
        first = cur[0][3]
        lines.append(Line(
            page=pno,
            col=cur[0][0],
            x0=round(min(t[3]["bbox"][0] for t in cur), 1),
            y0=round(min(t[3]["bbox"][1] for t in cur), 1),
            x1=round(max(t[3]["bbox"][2] for t in cur), 1),
            text=re.sub(r"\s+", " ", text).strip(),
            speaker_font=first["font"].startswith(config.SPEAKER_FONT),
            bold_chars=sum(len(t[3]["text"].strip()) for t in cur
                           if config.BOLD_FONT_MARKER in t[3]["font"] and "Arial" not in t[3]["font"]),
            size=round(first["size"], 1),
        ))
        cur.clear()

    for t in spans:
        # gleiche Spalte und Grundlinie innerhalb 2,5 pt -> dieselbe Zeile (Versatz Name/Partei 1 pt, Zeilenabstand ≈ 11 pt)
        if cur and (t[0] != cur[-1][0] or abs(t[1] - cur[-1][1]) > 2.5):
            flush()
        cur.append(t)
    flush()
    return lines


def extract_lines(pdf_path) -> tuple[list[Line], dict]:
    doc = pymupdf.open(pdf_path)
    lines: list[Line] = []
    for pno, page in enumerate(doc):
        lines.extend(_page_lines(page, pno))

    meta = assign_sections(lines, source=str(pdf_path))
    meta["seiten"] = doc.page_count
    return lines, meta


def assign_sections(lines: list[Line], source: str = "") -> dict:
    """Abschnitte bestimmen: Inhaltsverzeichnis | Sitzungsverlauf | Anlagen.

    "Beginn:" und "(Schluss: …)" stehen in jedem Protokoll genau einmal als
    eigene Zeile; die Uhrzeiten werden für die Kalibrierung der Redezeit gebraucht.
    """
    meta = {"beginn": None, "schluss": None}
    section = "toc"
    for ln in lines:
        if section == "toc" and (m := BEGIN_RE.match(ln.text)):
            section = "body"
            meta["beginn"] = f"{int(m.group(1)):02d}:{m.group(2)}"
        ln.section = section
        if section == "body" and (m := END_RE.match(ln.text)):
            meta["schluss"] = f"{int(m.group(1)):02d}:{m.group(2)}"
            section = "annex"
    if meta["schluss"] is None:
        # Ohne Schlusszeile liefe der gesamte Anhang (Namenslisten, Anlagen-Reden) in den Body.
        start = next((i for i, ln in enumerate(lines)
                      if ln.section == "body" and ANLAGE_RE.match(ln.text)), None)
        if start is not None:
            for ln in lines[start:]:
                ln.section = "annex"
        meta["warnung"] = ("Schluss-Zeile fehlt, Anlagen-Beginn als Ende"
                           if start is not None else "Schluss-Zeile und Anlagen-Beginn fehlen")
        log.warning("%s: %s", source, meta["warnung"])
    return meta


def _col_left(ln: Line) -> float:
    return config.COL_LEFT_X[ln.col]


def is_interjection_start(ln: Line) -> bool:
    """Zwischenrufe/Beifall: eingerückt und mit '(' beginnend."""
    return ln.text.startswith("(") and ln.x0 - _col_left(ln) >= config.INTERJECTION_INDENT


def join_hyphenated(parts: list[str]) -> str:
    """Zeilen zu Fließtext verbinden und Silbentrennung entfernen.

    'Bun-' + 'destag' -> 'Bundestag'. Steht nach dem Bindestrich ein
    Großbuchstabe ('CDU/CSU-' + 'Fraktion') oder endet die Zeile auf '/',
    bleibt das Zeichen erhalten, da es sich dann um echte Bindestriche bzw.
    Parteinamen handelt. Folgt eine Konjunktion (CONJ_RE), ist der Strich ein
    Ergänzungsstrich: 'Außen-' + 'und Sicherheitspolitik' -> 'Außen- und …'.
    """
    out = ""
    for p in parts:
        if not out:
            out = p
        elif out.endswith("-") and CONJ_RE.match(p):
            out += " " + p
        elif (m := re.search(r"([A-ZÄÖÜ]+)-$", out)) and any(
                w != m.group(1) and w.startswith(m.group(1)) and p.startswith(w[len(m.group(1)):])
                for w in CAPS_SPLIT):
            out = out[:-1] + p
        elif re.search(r"[a-zäöüß]-$", out) and p[:1].islower():
            out = out[:-1] + p
        elif out.endswith("-") or out.endswith("/"):
            out += p
        else:
            out += " " + p
    return out


def to_paragraphs(lines: list[Line]) -> list[str]:
    """Zeilen des Sitzungsverlaufs zu Absätzen zusammenfassen (für die .txt-Datei)."""
    paras: list[str] = []
    buf: list[str] = []
    mode = None  # "speaker" | "inter" | "text" | "block"
    prev: Line | None = None

    def flush():
        if buf:
            paras.append(join_hyphenated(buf))
            buf.clear()

    for ln in lines:
        indent = ln.x0 - _col_left(ln)
        # Vertikaler Abstand > 13 pt (Zeilenabstand ≈ 11 pt) = Leerzeile = neuer Absatz
        gap = prev is not None and prev.page == ln.page and prev.col == ln.col and ln.y0 - prev.y0 > 13
        prev = ln
        # Kopf-Fortsetzung in Times statt Arial fett: "… (BÜNDNIS 90/DIE GRÜ-" | "NEN):"
        # oder "…, Staatsminister im Auswärtigen" | "Amt:" (1947 Fälle, alle echte Köpfe).
        if mode == "speaker" and buf and not buf[-1].endswith(":") \
                and len(ln.text) <= KOPF_FORTSETZUNG_MAX and ln.text.endswith(":"):
            buf.append(ln.text)
            flush()
            mode = None
            continue
        if ln.speaker_font:
            if mode != "speaker":
                flush()
            mode = "speaker"
            buf.append(ln.text)
            if ln.text.endswith(":"):
                flush()
                mode = None
            continue
        if is_interjection_start(ln):
            flush()
            mode = "inter"
            buf.append(ln.text)
            continue
        if mode == "inter":
            buf.append(ln.text)
            # Zwischenruf endet mit schließender Klammer am Zeilenende
            if ln.text.count(")") and " ".join(buf).count("(") <= " ".join(buf).count(")"):
                flush()
                mode = None
            continue
        # Eingerückte Blöcke ohne Klammer: Aufruf eines Tagesordnungspunkts
        # (Gesetzestitel, Drucksachen) – jeweils durch Leerzeilen gegliedert.
        if indent >= config.INTERJECTION_INDENT:
            if mode != "block" or gap:
                flush()
            mode = "block"
            buf.append(ln.text)
            continue
        # Absatzbeginn im Fließtext: Erstzeileneinzug von ca. 10 pt
        if 6 <= indent < config.INTERJECTION_INDENT or mode != "text" or gap:
            flush()
        mode = "text"
        buf.append(ln.text)
    flush()
    return paras


def process(nr: int) -> dict:
    stem = f"{config.WAHLPERIODE}{nr:03d}"
    lines, meta = extract_lines(config.PDF_DIR / f"{stem}.pdf")
    config.LINES_DIR.mkdir(parents=True, exist_ok=True)
    config.TXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LINES_DIR / f"{stem}.jsonl", "w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for ln in lines:
            f.write(json.dumps(asdict(ln), ensure_ascii=False) + "\n")
    body = [ln for ln in lines if ln.section == "body"]
    paras = to_paragraphs(body)
    (config.TXT_DIR / f"{stem}.txt").write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    # Anlagen getrennt: zu Protokoll gegebene Reden (gold 20146: 9 von 177 Reden) fehlten sonst im Parser
    annex = to_paragraphs([ln for ln in lines if ln.section == "annex"])
    (config.TXT_DIR / f"{stem}_anlagen.txt").write_text("\n\n".join(annex) + "\n", encoding="utf-8")
    meta.update(sitzung=nr, zeilen=len(lines), absaetze=len(paras), absaetze_anlagen=len(annex),
                woerter=sum(len(p.split()) for p in paras))
    return meta


def load_lines(nr: int) -> tuple[list[Line], dict]:
    stem = f"{config.WAHLPERIODE}{nr:03d}"
    with open(config.LINES_DIR / f"{stem}.jsonl") as f:
        meta = json.loads(f.readline())["meta"]
        lines = [Line(**json.loads(row)) for row in f]
    return lines, meta


def _process_sicher(nr: int) -> tuple[tuple, str]:
    """Logzeile und Traceback einer Sitzung; die Ausnahme bleibt im Worker, sonst bräche sie die übrigen ab."""
    try:
        m = process(nr)
    except Exception as exc:
        return (nr, "ERROR", 0, 0, f"{type(exc).__name__}: {exc}"), traceback.format_exc()
    status = "WARN" if m.get("warnung") or m["beginn"] is None else "OK"
    msg = m.get("warnung") or f"Beginn {m['beginn']}, Schluss {m['schluss']}"
    return (nr, status, m["zeilen"], m["woerter"], msg), ""


def extract_all(sitzungen=None, log_path=None, worker: int = WORKER) -> dict[str, int]:
    """Alle Sitzungen extrahieren, je Sitzung ein Prozess aus dem Pool; Fehler je Sitzung protokollieren statt
    abzubrechen. worker=1 rechnet im eigenen Prozess (Tests, Fehlersuche mit Debugger)."""
    sitzungen = list(config.SITZUNGEN if sitzungen is None else sitzungen)
    log_path = config.EXTRACT_LOG if log_path is None else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_count = {"OK": 0, "WARN": 0, "ERROR": 0}
    pool = None
    if worker > 1 and len(sitzungen) > 1:
        # Worker starten per spawn (macOS) ohne die Logging-Konfiguration aus __main__; ohne Initializer fehlte
        # der Warnung "Schluss-Zeile fehlt" das Format.
        pool = ProcessPoolExecutor(max_workers=min(worker, len(sitzungen)),
                                   initializer=partial(logging.basicConfig, level=logging.INFO, format=LOG_FORMAT))
        auftraege = [pool.submit(_process_sicher, nr) for nr in sitzungen]
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("sitzung\tstatus\tzeilen\twoerter\tmeldung\n")
            # In Sitzungsreihenfolge abholen: extract_log.txt bleibt sortiert wie beim sequenziellen Lauf.
            for i, nr in enumerate(sitzungen):
                try:
                    row, tb = auftraege[i].result() if pool else _process_sicher(nr)
                except BrokenProcessPool as exc:  # Absturz eines Workers (z. B. in MuPDF) trifft die offenen Sitzungen
                    row, tb = (nr, "ERROR", 0, 0, f"{type(exc).__name__}: {exc}"), ""
                if tb:
                    print(tb, file=sys.stderr, end="", flush=True)
                status_count[row[1]] += 1
                f.write("\t".join(map(str, row)) + "\n")
                f.flush()
                print(f"[extract] {row[0]} {row[1]}: {row[2]} Zeilen, {row[3]} Wörter, {row[4]}", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    print(f"[extract] fertig: {status_count}", flush=True)
    return status_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    t0 = time.time()
    extract_all()
    print(f"[extract] {time.time() - t0:.1f} s mit {min(WORKER, len(config.SITZUNGEN))} Prozessen", flush=True)
