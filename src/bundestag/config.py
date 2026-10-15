"""Zentrale Konfiguration der Pipeline.

Alle Pfade, Sitzungsbereiche und Schwellwerte stehen hier an einer Stelle,
damit die einzelnen Schritte reproduzierbar und ohne Code-Änderung
angepasst werden können.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
# Eingaben: je Sitzung eine Datei, ein Teillauf ergänzt sie nur und bleibt daher im Hauptbaum
PDF_DIR = DATA / "pdf"            # Originalprotokolle (Rohdaten)
XML_DIR = DATA / "xml"            # offizielle XML-Fassung, nur als Goldstandard zur Validierung
TXT_DIR = DATA / "txt"            # lesbarer Text in korrekter Lesereihenfolge (Eingabe für Parser, Applaus, Metriken, LLM)
LINES_DIR = DATA / "lines"        # Zeilen mit Layout-Metadaten (Datum, Inhaltsverzeichnis, PDF-Seite einer Fundstelle)
REFERENCE = DATA / "reference"    # Referenzdaten: Sentiment-Rater, Abstimmungsurteile, gold_gesetze.csv, benchmark.csv
STAMMDATEN_XML = DATA / "raw" / "stammdaten" / "MDB_STAMMDATEN.XML"
# Caches je Inhalt (Fenster-Hash bzw. Modell + Prompt-Hash) gelten für jeden Bereich; ein Teillauf nutzt die 31 065
# BERT-Fenster und 294 LLM-Antworten des Hauptbaums, statt sie neu zu rechnen
BERT_CACHE = DATA / "cache" / "sentiment_bert.parquet"
LLM_CACHE = DATA / "llm" / "cache"

WAHLPERIODE = 20
# 146. bis einschließlich 212. Sitzung laut Aufgabenstellung
SITZUNGEN_VOLL = range(146, 213)


def setze_bereich(von: int, bis: int) -> None:
    """Sitzungsbereich und alle Ausgabepfade. Ein Teillauf schreibt nach data/teillauf/<von>-<bis>/, weil er sonst die
    Ergebnisse für 146–212 überschriebe (ein Teillauf 146–147 änderte vor der Trennung 61 Dateien, zahlen.tex
    +626/−996 Zeilen).
    run_all.py ruft sie nach --sitzungen für den eigenen Prozess auf; die Schritte lesen den Bereich beim Import."""
    global SITZUNGEN, AUSGABE, PROCESSED, FIGURES, LLM_DIR, GOLD_DIR, MANIFEST, EXTRACT_LOG, LOGS, ZAHLEN_TEX, \
        ABB_TITEL_TEX
    SITZUNGEN = range(von, bis + 1)
    voll = SITZUNGEN == SITZUNGEN_VOLL
    AUSGABE = DATA if voll else DATA / "teillauf" / f"{von}-{bis}"
    PROCESSED = AUSGABE / "processed"          # Dataframes (Parquet + CSV)
    FIGURES = AUSGABE / "figures"              # Visualisierungen
    LLM_DIR = AUSGABE / "llm"                  # LLM-Ausgaben (Cache: LLM_CACHE)
    GOLD_DIR = AUSGABE / "reference"           # gold_*.parquet aus data/xml (gold.py)
    MANIFEST = AUSGABE / "download_manifest.csv"
    EXTRACT_LOG = AUSGABE / "extract_log.txt"
    LOGS = ROOT / "logs" if voll else AUSGABE / "logs"
    latex = ROOT / "LaTeX" if voll else AUSGABE / "LaTeX"
    ZAHLEN_TEX = latex / "zahlen.tex"
    ABB_TITEL_TEX = latex / "abbildungen_titel.tex"


# run_all.py --sitzungen setzt den Bereich als Umgebungsvariable, weil download/extract_text/parse/validate ihn ohne
# eigene Option hier lesen
_BEREICH = os.environ.get("BUNDESTAG_SITZUNGEN", f"{SITZUNGEN_VOLL[0]}-{SITZUNGEN_VOLL[-1]}").split("-")
setze_bereich(int(_BEREICH[0]), int(_BEREICH[-1]))
PDF_URL = "https://dserver.bundestag.de/btp/{wp}/{wp}{nr:03d}.pdf"
XML_URL = "https://dserver.bundestag.de/btp/{wp}/{wp}{nr:03d}.xml"

# Layoutwerte in PDF-Punkten (Seite 595 x 842); PyMuPDF misst y ab der Oberkante, die Kopfzeile hat kleine y-Werte.
# Kopfzeile (Seitenzahl, Sitzungsangabe, lebender Kolumnentitel mit dem
# Namen des Präsidiums) liegt oberhalb von y=72 und wird verworfen.
HEADER_Y = 72
# Fußzeile (Druckerei-Vermerk) liegt unterhalb davon.
FOOTER_Y = 790
# Satzspiegel: linke Spalte x≈64–293, rechte Spalte x≈305–534. Die Marginalien
# (A)–(D) stehen außerhalb dieses Bereichs und sind keine Inhalte.
TEXT_X_MIN = 60
TEXT_X_MAX = 538
COLUMN_SPLIT_X = 299              # Mitte des Spaltenzwischenraums
# Bezugskante für Einzüge: häufigster Zeilenanfang x0 im Sitzungsverlauf aller 67 Sitzungen (data/lines),
# 64 pt links (281 291 von 433 931 Zeilen) und 305 pt rechts (270 575 von 433 162 Zeilen).
COL_LEFT_X = (64.0, 305.0)
# Zwischenrufe/Beifall beginnen frühestens 19,5 pt rechts dieser Kante (data/lines: 28 453 der 74 575
# Klammerzeilen bei 19,5 bis 19,9 pt), der Erstzeileneinzug von Absätzen bei 10 pt; zwischen 10 und 18 pt
# beginnt keine Klammerzeile. Ein Wert von 20 verlöre die Zwischenrufe unter 20 pt an den Fließtext.
INTERJECTION_INDENT = 18

# Schriftarten, an denen sich Strukturelemente erkennen lassen
SPEAKER_FONT = "Arial-BoldMT"     # Rednerkopf, z. B. "Friedrich Merz (CDU/CSU):"
BOLD_FONT_MARKER = "Bold"         # Gesetzestitel / Drucksachen im Aufruf eines TOP

# Sentiment: Fenstergröße und Gewichtung begründet sentiment.py; die κ-Schwelle für H3 nennt sentiment_kappa.py.
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"
# Commit des Modells auf dem Hugging-Face-Hub, mit dem alle 31 065 Fenster gerechnet sind. Fest statt "main":
# der Ordnername geht in den Cache-Schlüssel ein (sentiment.baue_fenster), ein neuer Commit auf dem Hub
# machte sonst den Cache ungültig und erzwänge einen Neulauf aller Fenster (Erstlauf 481 s) mit anderen Gewichten.
SENTIMENT_REVISION = "b1177ff59e305c966836ba2825d3dc2efc53f125"
SENTIMENT_MAX_TOKENS = 512        # Kontextlänge von BERT

# LLM lokal über Ollama (Standardport 11434); qwen2.5:7b begründet llm.MODELL und der Modellvergleich in llm.py.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
