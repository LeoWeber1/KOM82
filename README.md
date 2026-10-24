# KOM82 – Bundestagsprotokollanalyse

Pipeline zur Extraktion und Analyse der Plenarprotokolle der 20. Wahlperiode des Deutschen Bundestages (Sitzungen 146 bis 212).

Die Pipeline extrahiert die zweispaltigen PDF-Protokolle in Lesereihenfolge, trennt Reden, Gesetzesberatungen und Abstimmungen, berechnet Kennzahlen je Abgeordnetem und Fraktion und führt eine Sentiment-Analyse mit `german-sentiment-bert` durch.

## Installation

Benötigt Python 3.12+ (empfohlen in einer virtuellen Umgebung):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

*Hinweis:* Für den optionalen LLM-Vergleich (`qwen2.5:7b`) wird eine lokale [Ollama](https://ollama.ai/)-Instanz verwendet. Ist Ollama nicht aktiv, werden vorhandene Cache-Ergebnisse aus `data/llm/` genutzt oder der Schritt mit `--skip-llm` übersprungen.

## Ausführung

Die Pipeline kann vollständig oder schrittweise über `run_all.py` ausgeführt werden:

```bash
# Gesamte Pipeline ausführen
python run_all.py

# Schnelldurchlauf mit vorhandenen Daten
python run_all.py --skip-download --skip-llm

# Einzelne Schritte ausführen (z. B. nur Extraktion und Parsing)
python run_all.py --nur extract parse
```

## Daten und Ergebnisse

- `data/txt/`: Extrahierte Volltexte der Plenarprotokolle.
- `data/processed/`: Aufbereitete Datensätze als CSV und Parquet (Abgeordnete, Reden, Gesetze, Sentiment).
- `data/figures/`: Generierte Diagramme und Visualisierungen.
- `data/reference/`: Referenzdaten zur Validierung gegen offizielle XML-Daten.

## Tests

Test-Suite mit pytest ausführen:

```bash
pytest
```
