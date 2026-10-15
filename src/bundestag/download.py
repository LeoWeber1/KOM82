"""Schritt 1: Automatisierter Download der Plenarprotokolle.

Die PDF-URLs folgen einem festen Muster (https://dserver.bundestag.de/btp/20/20199.pdf),
daher ist kein Scraping der Übersichtsseite nötig. Bereits vorhandene Dateien
werden übersprungen, sodass der Schritt beliebig oft wiederholt werden kann.
Die XML-Fassung wird zusätzlich geladen, aber ausschließlich als Referenz
(Goldstandard) für die Qualitätsbewertung der PDF-Extraktion verwendet.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

# Stammdaten-Archiv des Bundestags (geprüft 15.10.2026: HTTP 200, 952 696 Bytes wie die lokale Kopie)
STAMMDATEN_URL = "https://www.bundestag.de/resource/blob/472878/MdB-Stammdaten.zip"
# Dateianfang je Endung: eine Wartungs- oder Fehlerseite mit HTTP 200 (HTML) würde sonst als .pdf gespeichert und
# wegen Größe > 0 nie neu geladen. Geprüft an allen 67 PDFs (%PDF-) und 67 XMLs (<?xml) in data/.
MAGIC = {".pdf": b"%PDF-", ".xml": b"<?xml", ".zip": b"PK\x03\x04"}
# requests-Zeitlimit je Verbindungsaufbau und je Lesevorgang, nicht für die ganze Datei: 20211.pdf (1,4 MB) kam am
# 15.10.2026 nach 0,15 s erstem Byte in 0,3 s; 60 s lassen Lastspitzen zu, danach greifen die 5 Versuche aus _session.
TIMEOUT_S = 60
BLOCK = 1 << 20           # Schreiben in 1-MiB-Blöcken: die größte Datei (20147.pdf, 4,7 MB) liegt nie ganz im Speicher


class FalscherInhalt(requests.RequestException):
    """Unterklasse von RequestException, damit download_all den Fall wie einen Netzfehler je Datei markiert."""


def _session() -> requests.Session:
    # Wiederholungen mit exponentiellem Backoff fangen sporadische 5xx/Timeouts ab.
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "KOM82-Bundestagsprotokollanalyse/1.0 (Studienprojekt)"
    return s


def _gueltig(pfad, magic: bytes) -> bool:
    try:
        with open(pfad, "rb") as f:
            return f.read(len(magic)) == magic
    except OSError:
        return False


def _fetch(s: requests.Session, url: str, target) -> tuple[int, int]:
    """Lädt url nach target, sofern dort keine gültige Datei liegt; Rückgabe (HTTP-Status, Bytes).

    Der Körper geht erst in target.part und wird nach der Kopfprüfung per os.replace übernommen; ein Abbruch oder
    Fehlinhalt hinterlässt so nie eine halbe oder falsche Datei unter dem Zielnamen.
    """
    magic = MAGIC.get(target.suffix, b"")  # nach target, nicht nach tmp: dessen Endung ist .part
    if target.exists():
        if target.stat().st_size > 0 and _gueltig(target, magic):
            return 200, target.stat().st_size
        target.unlink()             # leer oder Fehlinhalt eines früheren Laufs: verwerfen und neu laden
        print(f"[download] {target.name}: vorhandene Datei ohne gültigen Dateianfang verworfen, lade neu")
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with s.get(url, timeout=TIMEOUT_S, stream=True) as r:
            if r.status_code != 200:
                return r.status_code, 0
            with open(tmp, "wb") as f:
                for block in r.iter_content(BLOCK):
                    f.write(block)
            groesse = tmp.stat().st_size
            if not _gueltig(tmp, magic):
                raise FalscherInhalt(f"Dateianfang ist nicht {magic!r} "
                                     f"(Content-Type {r.headers.get('Content-Type', '?')}, {groesse} Bytes)")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    time.sleep(0.5)                 # Server schonen: 135 Abrufe in Folge
    return 200, groesse


def stammdaten_laden(s: requests.Session) -> dict:
    """MdB-Stammdaten (Partei je Abgeordnetem und Zeitraum) für stammdaten.py, nur wenn die XML fehlt."""
    from .stammdaten import STAMMDATEN_XML
    zip_pfad = STAMMDATEN_XML.parent / "MdB-Stammdaten.zip"
    zeile = {"sitzung": "", "typ": "stammdaten", "url": STAMMDATEN_URL, "status": 200, "bytes": 0}
    if not STAMMDATEN_XML.exists():
        STAMMDATEN_XML.parent.mkdir(parents=True, exist_ok=True)
        zeile["status"], zeile["bytes"] = _fetch(s, STAMMDATEN_URL, zip_pfad)
        if zeile["status"] == 200:
            with zipfile.ZipFile(zip_pfad) as z:
                for name in ("MDB_STAMMDATEN.XML", "MDB_STAMMDATEN.DTD"):
                    z.extract(name, STAMMDATEN_XML.parent)
    print(f"[download] {STAMMDATEN_XML.name}: HTTP {zeile['status']}")
    return zeile


def download_all(with_xml: bool = True) -> tuple[int, int]:
    """Lädt fehlende Dateien; Fehler je Datei stehen im Manifest, statt den Lauf abzubrechen.

    Rückgabe (fehlgeschlagen, gesamt), damit der Aufrufer nur bei Totalausfall abbricht.
    """
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    config.XML_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    rows = []
    for nr in config.SITZUNGEN:
        kinds = [("pdf", config.PDF_URL, config.PDF_DIR)]
        if with_xml:
            kinds.append(("xml", config.XML_URL, config.XML_DIR))
        for kind, pattern, folder in kinds:
            url = pattern.format(wp=config.WAHLPERIODE, nr=nr)
            target = folder / f"{config.WAHLPERIODE}{nr:03d}.{kind}"
            try:
                status, size = _fetch(s, url, target)
            except requests.RequestException as exc:  # 5 Versuche aus _session erschöpft: markieren, weiter
                status, size = 0, 0
                print(f"[download] {target.name}: FEHLER {type(exc).__name__}: {exc}")
            rows.append({"sitzung": nr, "typ": kind, "url": url, "status": status, "bytes": size})
            print(f"[download] {target.name}: HTTP {status}, {size/1e6:.1f} MB")
    try:
        rows.append(stammdaten_laden(s))
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
        rows.append({"sitzung": "", "typ": "stammdaten", "url": STAMMDATEN_URL, "status": 0, "bytes": 0})
        print(f"[download] Stammdaten: FEHLER {type(exc).__name__}: {exc}")
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    fehler = [r for r in rows if r["status"] != 200]
    print(f"[download] fertig: {len(rows) - len(fehler)} von {len(rows)} Dateien vorhanden, Fehler: "
          + (", ".join(f"{r['sitzung']} {r['typ']}" for r in fehler) or "keine"))
    return len(fehler), len(rows)


if __name__ == "__main__":
    n_fehler, n = download_all()
    sys.exit(1 if n_fehler == n else 0)  # ≠ 0 nur bei Totalausfall, d. h. keine der 135 Dateien (kein Netz)
