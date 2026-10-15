"""Gesamte Pipeline in fester Reihenfolge, von den PDFs bis zu LaTeX/zahlen.tex.

download → extract → stammdaten → gold → parse → validate → applaus → sentiment → sentiment_kappa
→ metrics → stats → stats_sentiment → llm → visualize → zahlen

Aufruf aus der Projektwurzel:
  python run_all.py                                 alle Schritte, Sitzungen 146–212
  python run_all.py --skip-download --skip-llm      vorhandene PDFs, LLM-Ausgaben aus dem letzten Lauf
  python run_all.py --sitzungen 146-150 --nur extract parse
  python run_all.py --skip-download --skip-llm --ab sentiment_kappa   fortsetzen ab einem Schritt

Jeder Schritt läuft als eigener Prozess (python -m bundestag.<modul>). Seine Ausgabe steht in
logs/run_all/<schritt>.log, Start, Ende, Dauer und Status in logs/run_all.log, die Laufzeiten in
data/processed/laufzeiten.csv (schritt, sekunden, sitzungen, status; je Schritt der letzte Lauf).

Teillauf: Ein Bereich außer 146-212 schreibt alle Ergebnisse, Protokolle und LaTeX/zahlen.tex unter
data/teillauf/<von>-<bis>/ (config.setze_bereich); PDFs, XML, txt und lines je Sitzung bleiben in data/.

Status: OK; TEIL = einzelne Sitzungen mit Fehler (stehen im Log, der Schritt läuft weiter);
FEHLER = Schritt insgesamt gescheitert; ÜBERSPRUNGEN = per Option oder weil ein Vorgänger aus ABHAENGIG
FEHLER meldet; FEHLT = Modul nicht vorhanden.
Rückgabecode 1 nur bei Totalausfall, d. h. wenn ein Schritt FEHLER meldet; sonst 0.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import config

ROOT = config.ROOT
SPALTEN = ["schritt", "sekunden", "sitzungen", "status"]

log = logging.getLogger("run_all")


# (Sitzungen ohne Fehler, Sitzungen mit Fehler) je Schritt: so meldet ein Schritt mit einzelnen Fehlsitzungen TEIL statt FEHLER.
def _tsv_status(pfad: Path, bereich: range) -> tuple[int, list[int]]:
    """extract_text und parse schreiben je Sitzung eine Zeile mit status OK/WARN/ERROR."""
    with open(pfad, encoding="utf-8") as f:
        zeilen = [z for z in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                  if z["sitzung"].isdigit() and int(z["sitzung"]) in bereich]
    fehler = sorted(int(z["sitzung"]) for z in zeilen if z["status"] == "ERROR")
    return len(zeilen) - len(fehler), fehler


def _pruefe_download(pfad: Path, bereich: range) -> tuple[int, list[int]]:
    with open(pfad, encoding="utf-8") as f:
        zeilen = [z for z in csv.DictReader(f) if z["sitzung"].isdigit() and int(z["sitzung"]) in bereich]
    fehler = sorted({int(z["sitzung"]) for z in zeilen if z["status"] != "200"})
    return len({z["sitzung"] for z in zeilen}) - len(fehler), fehler


def _pruefe_validierung(pfad: Path, bereich: range) -> tuple[int, list[int]]:
    with open(pfad, encoding="utf-8") as f:
        da = {int(z["sitzung"]) for z in csv.DictReader(f) if int(z["sitzung"]) in bereich}
    return len(da), sorted(set(bereich) - da)


@dataclass(frozen=True)
class Schritt:
    """befehle: Argumentlisten nach "python", nacheinander; protokoll: liefert die Datei mit Status je Sitzung, die
    pruefe() nach dem Lauf auswertet (erst zur Laufzeit, weil der Teillauf sie nach data/teillauf/ verlegt)."""
    name: str
    befehle: tuple[tuple[str, ...], ...]
    protokoll: Callable[[], Path] | None = field(default=None, compare=False)
    pruefe: Callable[[Path, range], tuple[int, list[int]]] | None = field(default=None, compare=False)

    def fehlt(self) -> str | None:
        for b in self.befehle:
            if b[0] == "-m" and importlib.util.find_spec(b[1]) is None:
                return b[1]
            if b[0].endswith(".py") and not Path(b[0]).exists():
                return b[0]
        return None


SCHRITTE = [
    # Zuerst, weil alle Folgeschritte data/pdf und data/xml lesen. Lädt nur Fehlendes (download._fetch prüft
    # Größe > 0 und Dateianfang %PDF-, <?xml, PK): mit allen 135 Dateien vorhanden dauert der Schritt 0,5 s.
    Schritt("download", (("-m", "bundestag.download"),), lambda: config.MANIFEST, _pruefe_download),
    # Nach dem Download: liest data/pdf, schreibt data/lines (Parser-Eingabe) und data/txt (Applaus, LLM).
    Schritt("extract", (("-m", "bundestag.extract_text"),), lambda: config.EXTRACT_LOG, _tsv_status),
    # Direkt vor dem Parser, ihrem ersten Leser: Partei zum Sitzungsdatum braucht fraktionen.parquet
    # (stammdaten.FRAKTIONEN_PQ, parse._lookup). Nach download, der das Stammdaten-ZIP holt.
    Schritt("stammdaten", (("-c", "from bundestag.stammdaten import build; build()"),)),
    # Gold nur als Validierungsreferenz: liest allein data/xml (gold.build), gelesen nur von validate (Daten);
    # vor dem Parser, damit die Referenz nicht vom Parserstand abhängt.
    Schritt("gold", (("-m", "bundestag.gold"),)),
    # Braucht data/lines aus extract und fraktionen.parquet aus stammdaten; schreibt reden/beitraege/gesetze.
    Schritt("parse", (("-m", "bundestag.parse"),), lambda: config.LOGS / "parse.log", _tsv_status),
    # Direkt nach dem Parser und vor jeder Analyse: Reden-F1 und Partei-Quote je Sitzung sollen feststehen,
    # bevor Kennzahlen darauf aufbauen; validierung.csv speist make_zahlen.
    Schritt("validate", (("-m", "bundestag.validate", "--alle"),), lambda: config.PROCESSED / "validierung.csv",
            _pruefe_validierung),
    # Vor stats: schreibt applaus_ereignisse.parquet, das stats für H2 über applaus.ereignisse_je_rede liest
    # (zusammen mit beitraege.parquet aus dem Parser); applaus.main selbst braucht data/txt und die Stammdaten.
    Schritt("applaus", (("-m", "bundestag.applaus"),)),
    # Nach dem Parser (sentiment.lies_reden liest reden.parquet). Immer --voll: --sitzungen ist nur ein
    # Probelauf ohne Ausgabedatei (sentiment.main). Der Fenster-Cache data/cache/sentiment_bert.parquet
    # macht den Neulauf für unveränderte Fenster ≈ 0 s statt 481 s (Erstlauf ohne Cache).
    Schritt("sentiment", (("-m", "bundestag.sentiment", "--voll"),)),
    # Nach sentiment: misst die Güte genau dieser BERT-Labels gegen die Rater des Gold-Pools
    # (sentiment_kappa.lade). --gold-pool läuft nie mit, er würde die annotierte Stichprobe G01–G60 (_v2) neu ziehen.
    Schritt("sentiment_kappa", (("-m", "bundestag.sentiment_kappa"),)),
    # Nach sentiment, weil abgeordnete.parquet das Sentiment je MdB enthält (metrics.lade_sentiment);
    # schreibt top10_gesetze.csv, das llm liest (llm.top10).
    Schritt("metrics", (("-m", "bundestag.metrics"),)),
    # Nach metrics, dessen Funktionen stats importiert (stats.py Kopf), und nach applaus, dessen
    # applaus_ereignisse.parquet H2 liest (applaus.ereignisse_je_rede; 0,02 s statt 15,4 s für die Neuberechnung).
    Schritt("stats", (("-m", "bundestag.stats"),)),
    # Nach stats, dessen Bootstrap und KI es nutzt (stats_sentiment.py Kopf). Ohne sentiment.parquet endet es
    # mit 0 (stats_sentiment.main), daher bricht --skip-sentiment nichts.
    Schritt("stats_sentiment", (("-m", "bundestag.stats_sentiment"),)),
    # Spät, weil es Ollama braucht und am längsten läuft; liest gesetze.parquet und top10_gesetze.csv aus
    # metrics (llm.top10). Ohne erreichbares Ollama mit --nur-cache, d. h. nur aus data/llm/cache.
    # "vergleich" fehlt bewusst: Modellvergleich auf 30 Blöcken als Beleg der Toolwahl (llm --help).
    # --auswahl gesetzentwurf: die Auswertung deckt alle 249 Gesetzentwurf-Aufrufe ab; der breitere Filter
    # (516 Blöcke) bräuchte ≈ 45 min mehr Ollama-Laufzeit für Anträge und Berichte ohne Gesetzesbezug.
    Schritt("llm", (("-m", "bundestag.llm", "--schritt", "gesetze", "--auswahl", "gesetzentwurf"),
                    ("-m", "bundestag.llm", "--schritt", "zusammenfassung"))),
    # Nach allen Analysen einschließlich LLM: die Abbildungen lesen deren Ausgaben (visualize importiert
    # metrics und stats, Kopf von visualize.py, und liest data/processed/metrics/).
    Schritt("visualize", (("-m", "bundestag.visualize"),)),
    # Zuletzt: sammelt kennzahlen.csv aller Skripte und die Laufzeiten dieses Laufs in LaTeX/zahlen.tex.
    Schritt("zahlen", ((str(ROOT / "tools" / "make_zahlen.py"),),)),
]
NAMEN = [s.name for s in SCHRITTE]
# Nachfolger, die nicht starten, wenn einer dieser Vorgänger in diesem Lauf FEHLER meldet: dessen Ausgaben fehlen
# dann oder stammen aus einem früheren Lauf, und zahlen mischte beide Stände in LaTeX/zahlen.tex (ein Teillauf
# 146–147 schrieb so trotz metrics-FEHLER zahlen.tex mit +626/−996 Zeilen neu).
ABHAENGIG = {
    "llm": ("metrics",),                                         # top10_gesetze.csv (llm.top10)
    "visualize": ("metrics", "stats", "stats_sentiment"),        # metrics/*.parquet, top20_redner.csv
    "zahlen": ("validate", "sentiment_kappa", "metrics", "stats", "stats_sentiment"),  # kennzahlen.csv, validierung.csv
}


def bereich(text: str) -> range:
    """'146-212' oder '150' → range; Grenzen einschließlich."""
    try:
        teile = [int(t) for t in text.split("-")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Bereich wie 146-212 erwartet, nicht {text!r}")
    if len(teile) not in (1, 2) or teile[0] > teile[-1] or teile[0] < 1:
        raise argparse.ArgumentTypeError(f"Bereich wie 146-212 erwartet, nicht {text!r}")
    return range(teile[0], teile[-1] + 1)


def ollama_erreichbar() -> bool:
    # Ein laufendes Ollama antwortet auf /api/tags in 2 bis 4 ms (Messung 15.10.2026, 3 Abrufe), ohne Server wird
    # die Verbindung sofort abgewiesen; 2 s begrenzen nur den Fall eines offenen Ports mit hängendem Server.
    try:
        with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=2):
            return True
    except OSError:
        return False


def schreibe_laufzeit(zeile: dict) -> None:
    """Ersetzt die Zeile des Schritts; die übrigen bleiben aus früheren Läufen stehen (z. B. bei --nur)."""
    laufzeiten = config.PROCESSED / "laufzeiten.csv"
    alt = []
    if laufzeiten.exists():
        with open(laufzeiten, encoding="utf-8") as f:
            alt = [z for z in csv.DictReader(f) if z.get("schritt") != zeile["schritt"]]
    rang = {n: i for i, n in enumerate(NAMEN)}
    zeilen = sorted(alt + [zeile], key=lambda z: rang.get(z["schritt"], len(rang)))
    laufzeiten.parent.mkdir(parents=True, exist_ok=True)
    tmp = laufzeiten.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPALTEN, extrasaction="ignore")
        w.writeheader()
        w.writerows(zeilen)
    os.replace(tmp, laufzeiten)


def fuehre_aus(s: Schritt, sitzungen: range, env: dict) -> dict:
    zeile = {"schritt": s.name, "sekunden": "", "sitzungen": "", "status": "OK"}
    fehlt = s.fehlt()
    if fehlt:
        log.info("FEHLT  %-16s %s nicht vorhanden", s.name, fehlt)
        return zeile | {"status": "FEHLT"}
    befehle = s.befehle
    if s.name == "llm" and not ollama_erreichbar():
        log.info("       llm: Ollama unter %s nicht erreichbar, Ausgaben nur aus dem Cache", config.OLLAMA_URL)
        befehle = tuple((*b, "--nur-cache") for b in befehle)
    (config.LOGS / "run_all").mkdir(parents=True, exist_ok=True)
    pfad = config.LOGS / "run_all" / f"{s.name}.log"
    log.info("START  %-16s → %s", s.name, pfad.relative_to(ROOT))
    t0 = time.time()
    rc = 0
    with open(pfad, "w", encoding="utf-8") as out:
        for b in befehle:
            out.write(f"$ python {' '.join(b)}\n")
            out.flush()
            rc = subprocess.run([sys.executable, *b], cwd=ROOT, env=env, stdout=out,
                                stderr=subprocess.STDOUT).returncode
            if rc:
                break
    sek = time.time() - t0
    zeile["sekunden"] = f"{sek:.1f}"
    if rc:
        zeile["status"] = "FEHLER"
        rest = pfad.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        log.info("       %s: Rückgabecode %d, letzte Zeilen:\n         %s", s.name, rc, "\n         ".join(rest))
    elif s.pruefe and s.protokoll and (prot := s.protokoll()).exists() and prot.stat().st_mtime >= t0 - 1:
        # nur ein in diesem Lauf geschriebenes Protokoll zählt (mtime ab Start − 1 s Auflösung), sonst stammten
        # die gemeldeten Fehler aus einem früheren Lauf
        try:
            ok, fehler = s.pruefe(prot, sitzungen)
        except (OSError, KeyError, ValueError) as exc:   # Spalten sitzung/status fehlen (Format wie extract_all)
            log.info("       %s: Protokoll %s nicht lesbar (%s)", s.name, prot.name, exc)
            ok, fehler = "", []
        zeile["sitzungen"] = ok
        if fehler:
            zeile["status"] = "FEHLER" if ok == 0 else "TEIL"
            log.info("       %s: %d Sitzungen mit Fehler: %s (Details %s)", s.name, len(fehler),
                     ", ".join(map(str, fehler)), prot.relative_to(ROOT))
    log.info("ENDE   %-16s %7.1f s  %-6s %s", s.name, sek, zeile["status"],
             f"{zeile['sitzungen']} Sitzungen" if zeile["sitzungen"] != "" else "")
    return zeile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_all.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sitzungen", type=bereich, default=config.SITZUNGEN, metavar="VON-BIS",
                    help="Sitzungsbereich, Standard 146-212 (wirkt auf download, extract, parse, validate; "
                         "die Analysen nehmen, was der Parser geschrieben hat); ein anderer Bereich schreibt "
                         "alle Ausgaben nach data/teillauf/VON-BIS/")
    ap.add_argument("--skip-download", action="store_true", help="vorhandene PDFs/XMLs nutzen, nichts laden")
    ap.add_argument("--skip-llm", action="store_true", help="LLM-Schritt auslassen (Ausgaben aus data/llm bleiben)")
    ap.add_argument("--skip-sentiment", action="store_true",
                    help="BERT-Lauf auslassen; sentiment_kappa/stats_sentiment nutzen das vorhandene sentiment.parquet "
                         "(im Teillauf fehlt es, dann entfallen Sentiment-Statistik und Abbildungen 1–3)")
    auswahl = ap.add_mutually_exclusive_group()
    auswahl.add_argument("--nur", nargs="+", choices=NAMEN, metavar="SCHRITT",
                         help="nur diese Schritte (in Pipeline-Reihenfolge), Optionen --skip-* gelten dann nicht: "
                              + ", ".join(NAMEN))
    auswahl.add_argument("--ab", choices=NAMEN, metavar="SCHRITT",
                         help="ab diesem Schritt bis zum Ende, z. B. nach Neuberechnung eines Vorgängers; "
                              "--skip-* gelten weiter, frühere Zeilen in laufzeiten.csv bleiben")
    a = ap.parse_args(argv)

    von, bis = a.sitzungen[0], a.sitzungen[-1]
    config.setze_bereich(von, bis)   # Ausgabepfade dieses Prozesses (Protokolle, Laufzeiten) wie in den Schritten
    config.LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.FileHandler(config.LOGS / "run_all.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log.info("=== Lauf: %s | Sitzungen %d–%d", " ".join(sys.argv[1:] if argv is None else argv) or "(alle Schritte)",
             von, bis)
    if config.AUSGABE != config.DATA:
        log.info("Teillauf: Ausgaben unter %s, die Ergebnisse für %d–%d bleiben unverändert",
                 config.AUSGABE.relative_to(ROOT), config.SITZUNGEN_VOLL[0], config.SITZUNGEN_VOLL[-1])
    env = os.environ | {
        "BUNDESTAG_SITZUNGEN": f"{von}-{bis}",          # liest config.SITZUNGEN in jedem Unterprozess
        "PYTHONUNBUFFERED": "1",                         # bei Abbruch fehlen sonst bis 8 KiB Puffer im Log
        "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH")) if p),
    }
    ausgelassen = {"download": a.skip_download, "llm": a.skip_llm, "sentiment": a.skip_sentiment}
    t0 = time.time()
    zeilen = []
    for s in SCHRITTE:
        if (a.nur and s.name not in a.nur) or (a.ab and NAMEN.index(s.name) < NAMEN.index(a.ab)):
            continue
        gescheitert = [v for v in ABHAENGIG.get(s.name, ()) if v in {z["schritt"] for z in zeilen
                                                                       if z["status"] == "FEHLER"}]
        if not a.nur and ausgelassen.get(s.name):
            log.info("SKIP   %-16s per Option", s.name)
            zeile = {"schritt": s.name, "sekunden": "", "sitzungen": "", "status": "ÜBERSPRUNGEN"}
        elif gescheitert:
            log.info("SKIP   %-16s weil %s FEHLER meldet", s.name, ", ".join(gescheitert))
            zeile = {"schritt": s.name, "sekunden": "", "sitzungen": "", "status": "ÜBERSPRUNGEN"}
        else:
            zeile = fuehre_aus(s, a.sitzungen, env)
        schreibe_laufzeit(zeile)
        zeilen.append(zeile)
    fehler = [z["schritt"] for z in zeilen if z["status"] == "FEHLER"]
    log.info("=== Gesamt %.1f s; %s; Fehler: %s", time.time() - t0,
             ", ".join(f"{z['schritt']} {z['status']}" for z in zeilen), ", ".join(fehler) or "keine")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
