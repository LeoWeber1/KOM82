# KOM82 Bundestagsprotokollanalyse

Die Pipeline lädt die 67 Plenarprotokolle der Sitzungen 146 bis 212 (20. Wahlperiode) als PDF, bringt den zweispaltigen Text in Lesereihenfolge und zerlegt ihn in Reden mit Partei zum Sitzungsdatum, Gesetzesaufrufe und Abstimmungen. Darauf rechnet sie Kennzahlen je Abgeordnetem und Fraktion, ein Sentiment je Rede (german-sentiment-bert), sieben Abbildungen (fünf davon in der Dokumentation; H1-Sitzanteile und Beifallsmatrix nur in `data/figures/`) und einen Abgleich der Gesetzesextraktion mit einem lokalen Sprachmodell (qwen2.5:7b über Ollama). Die Dokumentation dazu ist eine Arbeit im Modul KOM82 (AKAD University, 2026); ihr Text liegt nicht in diesem Repository.

## Installation

Die Pipeline braucht Python 3.12 oder neuer (`pyproject.toml`); getestet ist sie mit Python 3.12.9 auf macOS arm64.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pinnt die geprüften Versionen; torch und transformers machen den größten Teil des Downloads aus. `pip install -e .` installiert das Paket `bundestag` aus `src/`; `run_all.py` und die Tests setzen `src/` auch selbst auf den Suchpfad.

Der Schritt sentiment lädt das Modell oliverguhr/german-sentiment-bert (Revision b1177ff) beim ersten Lauf vom Hugging-Face-Hub, danach kommt es aus dem lokalen Cache; auf Apple Silicon rechnet er auf MPS, sonst auf der CPU.

Ollama braucht nur der Schritt llm: `ollama pull qwen2.5:7b`, der Dienst muss unter `http://localhost:11434` antworten. Ist er nicht erreichbar, liest der Schritt die Antworten mit `--nur-cache` aus `data/llm/cache/` (im Repository enthalten); `--skip-llm` lässt ihn ganz aus und behält die Ausgaben in `data/llm/`.

## Daten

Die Daten liegen zusammen mit dem Code in diesem Repository: `data/pdf` (67 Protokolle), `data/txt` (Text je Sitzung in Lesereihenfolge, Hauptteil und Anlagen), `data/processed` (Dataframes als Parquet und CSV), `data/llm` (LLM-Antworten, Divergenzen, Schiedsverfahren, Modellvergleich, Cache), `data/reference` (XML-Referenz, Benchmark, Stichproben-Urteile, Sentiment-Gold) und `data/figures` (alle sieben Abbildungen). Geklont (`git clone https://github.com/LeoWeber1/KOM82`) oder als ZIP geladen, spart es den PDF-Download; die XML-Fassungen und die MdB-Stammdaten liegen nicht im Repository, der Schritt download holt sie nach. Zeilenangaben wie `zeile_start` zählen ab 1 in `data/txt/20NNN.txt` (Anlagen-Reden: `20NNN_anlagen.txt`).

Die wichtigsten Dataframes in `data/processed`:

- Datei `abgeordnete`: je Person `name`, `partei`, `fraktion` (zum Sitzungsdatum), `reden_n`, `woerter_mittel`, `woerter_gesamt`, `minuten_gesamt`, `sentiment_netto_mittel`; dazu Zähler für Kurzinterventionen, Zwischenfragen, Befragungen, Beiträge zu Protokoll.
- Datei `reden`: eine Zeile je Rede (Beiträge derselben Person im selben Tagesordnungspunkt) mit `rede_id`, `sitzung`, `datum`, `name`, `partei`, `fraktion`, `rolle`, `top_id`, `redetyp`, `zeile_start`, `zeile_ende`, `text`, `woerter`; die Hauptstatistik nimmt `redetyp == "plenarrede"`.
- Datei `beitraege`: ein Rednerkopf je Zeile, Spalten wie `reden` plus `fraktion_pdf` aus dem Kopf.
- Datei `gesetze`: ein Beratungsgegenstand je Aufruf mit `drucksache`, `titel`, `kurztitel`, `typ`, `beratungsstufe`, `punkt`, `abgestimmt`, `ergebnis_text`, `beschlussart` (angenommen, abgelehnt, zurückverwiesen, überwiesen, vertagt, offen).
- Datei `gesetze_einheit`: ein Gesetzentwurf je Drucksache über alle Lesungen mit `sitzungen`, `beratungsstufen`, `beschlussart_final`, `n_reden`, `n_redner`, `fraktionen`, `gegenstaende_im_top_max` (größte Zahl gemeinsam beratener Vorlagen).
- Datei `gesetz_beteiligte`: Rede mal Gegenstand mit `rede_id`, `sitzung`, `drucksache`, `name`, `partei`, `fraktion`, `redetyp`, `gegenstaende_im_top` (Vorlagen im selben Tagesordnungspunkt; jede Rede zählt bei jeder).
- Datei `sentiment`: je Rede `p_pos`, `p_neu`, `p_neg`, `netto` und `label` aus Fenstern von 150 Wörtern; `sentiment_fenster` je Fenster.
- Dateien `fraktionen_kennzahlen`, `top10_gesetze`, `top20_redner`: Kennzahlen je Fraktion und Phase sowie die Tabellen und Abbildungen des Berichts.
- Dateien `validierung`, `kennzahlen`, `laufzeiten`: Validierung gegen das XML je Sitzung, alle Kennzahlen des Berichts mit Quelle sowie Status und Dauer je Schritt.

`data/reference` hält die Referenz aus dem XML (`gold_*.parquet`), den Extraktions-Benchmark, die Gold-Stichprobe der Gesetzesaufrufe (`gold_gesetze.csv`), die Urteile der Abstimmungs-Stichproben (v2 vor, v3 nach der Korrektur der Abstimmungserkennung, v4 nach den ergänzten Regeln zur Beschlussart) und das Sentiment-Gold: 60 Fenster, annotiert von zwei LLM-Instanzen (`sentiment_llm_1_v2.csv`, `sentiment_llm_2_v2.csv`) und von qwen2.5:7b (`sentiment_qwen_v2.csv`). Ein menschliches Gold fehlt, Cohens Kappa misst daher nur die Übereinstimmung der Modelle.

## Ausführen

Alle Aufrufe starten in der Projektwurzel mit aktiviertem venv; `python run_all.py --help` erklärt alle Schritte und Optionen.

```
python run_all.py                                                   # alle 15 Schritte, Sitzungen 146-212
python run_all.py --skip-download --skip-llm                        # vorhandene PDFs und XML, LLM-Ausgaben aus data/llm
python run_all.py --sitzungen 146-147 --skip-llm --skip-sentiment   # Teillauf, Ausgaben unter data/teillauf/146-147/
python run_all.py --skip-download --skip-llm --ab sentiment_kappa   # fortsetzen ab einem Schritt
python run_all.py --nur extract parse                               # nur einzelne Schritte, --skip-* gelten dann nicht
```

Die Schritte laufen in fester Reihenfolge als eigene Prozesse: download, extract, stammdaten, gold, parse, validate, applaus, sentiment, sentiment_kappa, metrics, stats, stats_sentiment, llm, visualize, zahlen. Status (OK, TEIL, FEHLER, ÜBERSPRUNGEN, FEHLT) und Dauer je Schritt stehen in `data/processed/laufzeiten.csv`; der Rückgabecode ist 1, sobald ein Schritt FEHLER meldet, und llm, visualize und zahlen starten nicht, wenn ein Vorgänger gescheitert ist. Ein Sitzungsbereich außer 146-212 ist ein Teillauf und schreibt seine Ergebnisse unter `data/teillauf/<von>-<bis>/`; mit `--skip-sentiment` fehlen dort Sentiment-Statistik und Abbildungen 1 bis 3. Der letzte Schritt zahlen schreibt die Kennzahlen des Berichts nach `LaTeX/zahlen.tex`.

Laufzeiten des letzten Laufs mit `--skip-download` über 67 Sitzungen auf einem M1 Pro, zusammen 146 s:

- Extraktion und Parser: 27,9 s extract mit 8 Prozessen (seriell 171 s), 0,80 s stammdaten, 7,80 s gold, 13,4 s parse, 26 s validate.
- Analyse: 15,1 s applaus, 1,30 s sentiment_kappa, 6,20 s metrics, 1,30 s stats, 1,20 s stats_sentiment, 3,10 s visualize.
- Schritt sentiment: 7,90 s aus dem Fenster-Cache; ohne Cache rechnet BERT 59,2 Fenster je Sekunde und brauchte 367 s für gut zwei Drittel der Fenster. Der Cache liegt nicht im Repository, ein erster Lauf rechnet daher alle Fenster neu.
- Schritt llm: 34 s aus dem Cache; der Kaltlauf braucht für die 248 Aufruf-Blöcke mit Gesetzentwurf 2721 s Wandzeit (45,4 min), im Mittel 10,8 s je Block.

## Tests

`pytest` läuft die Suite in `tests/`. Tests mit dem Marker `rohdaten` brauchen `data/xml`, `data/txt` oder `data/lines` und werden ohne diese Dateien übersprungen (`tests/conftest.py`).

## Projektstruktur

```
data/                          Daten, siehe Abschnitt Daten
README.md
requirements.txt               gepinnte Pakete des geprüften Laufs
pyproject.toml                 Paket bundestag (Python ab 3.12), pytest-Marker rohdaten
run_all.py                     Startpunkt der Pipeline
src/bundestag/
  config.py                    Pfade, Sitzungsbereich, Layoutkonstanten, Modellnamen und Revision
  download.py                  PDFs und XML von dserver.bundestag.de, Stammdaten-ZIP, Manifest je Datei
  extract_text.py              PyMuPDF, Spaltentrennung und Lesereihenfolge nach data/lines und data/txt
  stammdaten.py                MdB-Stammdaten, Partei zum Sitzungsdatum
  gold.py                      Referenz aus den XML-Fassungen nach data/reference
  parse.py                     Rednerköpfe, Reden, Gesetzesaufrufe und Abstimmungen nach data/processed
  validate.py                  Parser gegen die XML-Referenz, validierung.csv und fehler.csv
  applaus.py                   Beifall-Klammern je Fraktion
  sentiment.py                 german-sentiment-bert in Fenstern von 150 Wörtern
  sentiment_kappa.py           Cohens Kappa gegen den Gold-Pool
  sentiment_fenstercheck.py    Einmalprüfung 200 gegen 150 Wörter, nicht in run_all
  metrics.py                   Kennzahlen je Abgeordnetem und Fraktion, Top-10-Gesetze
  stats.py                     Bootstrap-Konfidenzintervalle, Kruskal-Wallis, Hypothesen H1 und H2
  stats_sentiment.py           Sentiment-Statistik und Hypothese H3
  llm.py                       Ollama mit qwen2.5:7b, Pydantic-Schema, Cache, Abgleich mit dem Parser
  visualize.py                 Abbildungen 1 bis 7 nach data/figures
  run_all.py                   15 Schritte als Unterprozesse, Status und Laufzeiten
tests/                         pytest-Suite, conftest.py mit Marker rohdaten
tools/
  make_zahlen.py               Kennzahlen und Laufzeiten nach LaTeX/zahlen.tex (Schritt zahlen)
  kennzahlen_text.py           Kennzahlen aus Ergebnisdateien, die kein Pipeline-Schritt schreibt
  latex_bausteine.py           Tabelle 3 und die Sätze zum Zustrombegrenzungsgesetz für den Bericht
  bench_extract.py             Benchmark von fünf Extraktionsvarianten gegen das XML (PyMuPDF, pdfminer.six, pdfplumber)
  sentiment_qwen_rater.py      qwen2.5:7b als dritter Annotator der Sentiment-Gold-Fenster
```

## Nicht enthalten

- Der Text der Arbeit, bis sie bewertet ist.
- `data/xml`, `data/raw`, `data/lines` und `data/cache`: `run_all.py` lädt bzw. erzeugt sie neu.
