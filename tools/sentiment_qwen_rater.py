#!/usr/bin/env python3
"""Dritter Rater der Sentiment-Gold-Stichprobe: qwen2.5:7b lokal über Ollama, Codebuch als Systemprompt.

Aufruf: python tools/sentiment_qwen_rater.py [--gold v1|v2] → data/reference/sentiment_qwen[_v2].csv (id, label)
Erst nach dem BERT-Vollauf starten, beide Modelle teilen sich die GPU.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
# v2 = G01–G60 aus 150-Wort-Fenstern (Produktionseinheit nach dem Fenstercheck, bundestag.sentiment_fenstercheck),
# v1 = F01–F60 aus 200-Wort-Fenstern der Erstfassung.
VERSIONEN = {"v2": "_v2", "v1": ""}
URL = "http://localhost:11434/api/chat"
MODELL = "qwen2.5:7b"     # dritter Rater aus einer anderen Modellfamilie als BERT und die beiden LLM-Annotatoren
LABELS = ["positiv", "neutral", "negativ"]
# Das JSON-Schema zwingt die Antwort auf ein Label; freier Text bräuchte eine fehleranfällige Nachbearbeitung.
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}}, "required": ["label"]}
# temperature 0 und Seed 82 machen den Lauf wiederholbar; 4096 Tokens fassen Codebuch und ein Fenster bis 250 Wörter.
OPTIONEN = {"temperature": 0, "seed": 82, "num_ctx": 4096}
VERSUCHE = 2              # zweiter Versuch, falls die Antwort kein gültiges Label enthält
TIMEOUT = 300             # der erste Aufruf lädt das 4,7-GB-Modell in den Speicher
AUFTRAG = "Ordne das folgende Fenster nach dem Codebuch ein. Antworte als JSON mit dem Feld label.\n\nFenster:\n"
# Codebuch der Gold-Annotation, wörtlich der Systemprompt beider Läufe (v1 und v2); „rund 200 Wörter“ stammt aus der
# Erstfassung und bleibt stehen, damit ein Neulauf dieselbe Eingabe sieht wie die Labels in data/reference.
CODEBUCH = """# Codebuch Sentiment (Fenster aus Bundestagsreden, WP 20)

Jedes Fenster (rund 200 Wörter aus einer Rede, ohne Zwischenrufe und Beifall) erhält genau ein Label für die Haltung und Tonalität des Sprechers in diesem Ausschnitt. Bewertet wird, wie der Sprecher spricht, nicht ob das Thema erfreulich ist und nicht, ob man ihm zustimmt. Partei und Person des Sprechers sowie der Rest der Rede sind unbekannt und spielen keine Rolle.

## Labels
- Positiv: Der Sprecher lobt, dankt, wirbt zustimmend für eine Maßnahme, äußert Zuversicht oder Stolz; Kritik kommt höchstens am Rand vor.
- Negativ: Der Sprecher kritisiert, greift an, warnt, empört sich, wirft Versagen oder Täuschung vor oder schildert einen Zustand anklagend; Lob kommt höchstens am Rand vor.
- Neutral: Sachliche Darstellung ohne erkennbare Wertung, oder Lob und Kritik halten sich die Waage.

## Regeln
1. Maßgeblich ist die überwiegende Wertung im ganzen Fenster, nicht der erste oder letzte Satz.
2. Ironie und Spott zählen nach der gemeinten Aussage: „Großartig, wieder ein Jahr verschenkt“ ist negativ.
3. Höfliche Angriffe bleiben Angriffe: „Bei allem Respekt, Herr Kollege, Ihr Entwurf geht an der Wirklichkeit vorbei“ ist negativ.
4. Zitate und Positionen des Gegners zählen nach der Haltung des Sprechers dazu: Wer eine Aussage zitiert, um sie zurückzuweisen, spricht negativ, auch wenn das Zitat selbst positiv klingt.
5. Sachaufzählungen (Paragrafen, Beträge, Termine, Verfahrensschritte) sind neutral, auch bei ernstem Thema (Krieg, Insolvenz, Hochwasser). Wertende Wörter wie „endlich“, „leider“ oder „skandalös“ verschieben das Label.
6. Anrede- und Dankesformeln („Frau Präsidentin! Liebe Kolleginnen und Kollegen!“, „Vielen Dank“) machen ein Fenster allein nicht positiv.
7. Forderungen („Die Regierung muss jetzt handeln“) sind neutral; verbunden mit einem Vorwurf („… statt weiter zu zögern“) sind sie negativ.
8. Mischfenster mit Lob und Kritik: Das Label folgt der Wertung, die mehr Raum einnimmt; bei Gleichgewicht neutral.
9. Betroffenheit über Opfer oder Katastrophen ist negativ, wenn der Sprecher anklagt oder sich empört; Würdigung und Dank an Helfer sind positiv.
10. Bleibt nach den Regeln 1–9 ein echter Zweifel zwischen neutral und einem wertenden Label, gilt neutral.

## Beispiele (erfunden, nicht aus der Stichprobe)
- Positiv: „Dieses Gesetz entlastet Familien spürbar, und ich danke allen, die in den Ausschüssen daran mitgearbeitet haben.“
- Neutral: „Der Entwurf sieht drei Stufen vor: ab Januar 30 Euro, ab Juli 45 Euro, danach entscheidet eine Kommission über die Anpassung.“
- Negativ: „Sie haben drei Jahre lang Ankündigungen produziert und nichts geliefert; die Leute draußen zahlen die Rechnung.“
- Negativ (Ironie): „Herzlichen Glückwunsch, Herr Minister, so schnell hat noch niemand eine Branche ins Ausland vertrieben.“
- Negativ (Zitat): „Der Kollege sagt, alles laufe nach Plan. Nach welchem Plan denn? Die Zahlen sagen das Gegenteil.“

## Ausgabe
Genau ein Label aus positiv, neutral, negativ.
"""


def bewerte(text, system):
    anfrage = {"model": MODELL, "stream": False, "format": SCHEMA, "options": OPTIONEN,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": AUFTRAG + text}]}
    for _ in range(VERSUCHE):
        antwort = requests.post(URL, json=anfrage, timeout=TIMEOUT)
        antwort.raise_for_status()
        try:
            label = json.loads(antwort.json()["message"]["content"])["label"]
        except (ValueError, KeyError, TypeError):
            continue
        if label in LABELS:
            return label
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", choices=list(VERSIONEN), default="v2")
    suffix = VERSIONEN[ap.parse_args().gold]
    fenster = pd.read_csv(REFERENCE / f"sentiment_gold_fenster{suffix}.csv")
    system = CODEBUCH
    start = time.time()
    labels = []
    for nr, (fid, text) in enumerate(zip(fenster.id, fenster.text), 1):
        labels.append(bewerte(text, system))
        print(f"{nr}/{len(fenster)} {fid} {labels[-1] or 'UNGÜLTIG'}", flush=True)
    pd.DataFrame({"id": fenster.id, "label": labels}).to_csv(REFERENCE / f"sentiment_qwen{suffix}.csv", index=False)
    verteilung = pd.Series(labels).value_counts().to_dict()
    print(f"{len(labels)} Fenster in {time.time() - start:.0f} s, Labels {verteilung}", flush=True)


if __name__ == "__main__":
    main()
