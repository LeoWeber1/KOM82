"""Schritt 3: Rednerköpfe, Reden, TOPs, Gesetze und Abstimmungen aus data/txt.

Fundstellen in den Kommentaren stehen als Sitzung:Zeile, z. B. 20211:3519 = data/txt/20211.txt, Zeile 3519."""
from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import date
from functools import lru_cache

import pandas as pd

from . import config, stammdaten
from .extract_text import BEGIN_RE, END_RE, join_hyphenated
from .stammdaten import KOPF_ABG, KOPF_REG, KOPF_REG_START, FRAKTION_KOPF

log = logging.getLogger(__name__)

# Präsidium nur mit Whitelist-Namen: "Vizepräsident hier im Hohen Haus, wörtlich:" mitten im Satz ist Fließtext
PRAESIDIUM_RE = re.compile(
    r"^(?P<rolle>(?:Vize)?[Pp]räsident(?:in)?) (?P<name>(?:[A-ZÄÖÜ][\w.-]+ )*?"
    r"(?:Bas|Kubicki|Özoğuz|Göring-Eckardt|Pau|Magwas)):$")
# Aufruf vor oder nach der Nummer ("Ich rufe auf den …" / "Ich rufe … auf"), auch "Jetzt rufe ich auf die …"
# (20211:4327) und mitten im Absatz nach einer Vorrede (20160:15, 20211:17)
AUFRUF_RE = re.compile(
    r"(?:\bIch rufe\b|\brufe ich\b)(?: [a-zäöü]+){0,4}?"
    r"\s+(?P<liste>(?:Tagesordnungspunkt|Zusatzpunkt)e?\b[^:.]{0,160}?)"
    r"(?:\s+auf)?\s*(?=[:.]|$)")
# Wiederaufnahme nach Unterbrechung: 20211 ZP 35 Aufruf Z. 1065, „… komme zurück zu Zusatzpunkt 35“ Z. 3435
WIEDER_RE = re.compile(
    r"(?:Ich komme (?:\w+ )?(?:zurück )?zu[mr]?|Wir kommen (?:\w+ )?(?:zurück )?zu[mr]?|Wir gehen weiter und kommen zu[m]?|"
    r"Ich kehre zurück zu[m]?) (?:dem |den |der )?(?P<liste>(?:Tagesordnungspunkt|Zusatzpunkt)e?\b[^.:]{0,60}?)[.:]")
# Buchstabe nur mit Wortgrenze, sonst frisst "9 und Zusatzpunkt 5" das u (20160:3109)
NUMMER = r"(?:\d+|[IVX]+(?:\.\d+)?)"  # Haushaltswoche nummeriert römisch: "Tagesordnungspunkt I.10" (20150:23), "I a und b" (20149:35)
PUNKT_RE = re.compile(r"(Tagesordnungspunkt|Zusatzpunkt)e?\s*((?:" + NUMMER + r"(?:\s*[a-z]{1,2}\b)?)(?:\s*(?:,|und|sowie|bis)\s*(?:" + NUMMER + r"\s*)?(?:[a-z]{1,2}\b)?)*)")
NUM_RE = re.compile(r"(" + NUMMER + r")(?:\s*([a-z]{1,2})\b)?")


def _buchstaben(a: str, b: str) -> list[str]:
    """a…z, dann aa, bb, … (Zählweise der Tagesordnung: 20203 ZP 31 a bis bb = 28 Punkte)."""
    idx = lambda l: ord(l[0]) - 97 + 26 * (len(l) - 1)
    return [chr(97 + i % 26) * (i // 26 + 1) for i in range(idx(a), idx(b) + 1)]
# Ankündigungen sind keine Aufrufe: "werden abgesetzt" (20146:151), "wird als Zusatzpunkt 51 … aufgerufen" (20211:249)
# Aufruf ohne "rufe": "Das ist der Tagesordnungspunkt 11:" (20210:4907), „… setzen die Haushaltsberatungen – Tagesordnungspunkt 1 – fort:“ (20186:9)
TOP_ENDE_RE = re.compile(r"(?P<liste>(?:Tagesordnungspunkt|Zusatzpunkt)e?\s+" + NUMMER + r"[^:–]{0,40}?)\s*(?:–\s*fort)?:$")
# Unnummerierte Aufrufe: Befragung 20159 TOP 3 („… kommen … zur Befragung“), Einzelpläne der Haushaltswoche (20184:24).
# „Wir kommen zur Abstimmung über den Einzelplan 25“ (20149:3867, 22 Zeilen) ist kein Aufruf: die Abstimmung fehlte sonst,
# und der Abstimmungsblock verschluckte den nächsten Aufruf (20150:5477 TOP I.13). „Ich komme zurück zum Einzelplan 04“
# (20150:2839) kündigt nur das Ende der namentlichen Abstimmung an, die Debatte zu Einzelplan 05 läuft weiter
FREI_RE = re.compile(r"^(?:Wir kommen|Ich komme|Wir beginnen|Wir setzen|Ich rufe)(?![^.]{0,60}?(?:Abstimmung|zurück))[^.]{0,120}?"
                     r"(?P<frei>Befragung der Bundesregierung|Fragestunde|Einzelplan \d{2})")
EINZELPLAN_RE = re.compile(r"Einzelplan (\d{2})")
ABGESETZT_RE = re.compile(r"abgesetzt|aufgerufen|aufrufen")

# Prosa-Beginn beendet den Aufrufblock (Titel/Drucksachen stehen eingerückt, Prosa beginnt mit Präsidiumsfloskel)
PROSA_RE = re.compile(
    r"^(?:\(?(?:Beifall|Zuruf|Lachen|Heiterkeit|Widerspruch)|Ich |Wir |Für die Aussprache|Hierzu|Nach einer|"
    r"Interfraktionell|Über den|Über die|Die Reden|Es liegt|Es liegen|Bevor|Es ist|Dazu|Zu |Das Wort|Als |Die Aussprache|"
    r"Dann |Jetzt |Nun |Zunächst|Mir liegt|Sie sind damit|Gibt es|Wird das|Der Ältestenrat|Vielen Dank|Liebe Kolleg|"
    r"Meine Damen|Sehr geehrte|Herr |Frau |Dem |Damit|Erhebt sich|Gestatten|Darf ich|Ich eröffne)")
# Beratungsstufe teilt den Aufrufblock in Items, auch innerhalb eines Absatzes (20211:17 "ZP 26–Zweite und dritte Beratung …")
ITEM_RE = re.compile(
    r"(?:(?P<stufe>Erste|Zweite|Zweite und dritte|Dritte) Beratung des|"
    r"Beratung (?:des|der) (?:Antrags|Antrages|Beschlussempfehlung|Berichts|Unterrichtung|Großen Anfrage|"
    r"Gesetzentwurfs)|Befragung der Bundesregierung|Fragestunde|Aktuelle Stunde|Abgabe einer Regierungserklärung|"
    r"Vereinbarte Debatte|Wahl |hier: Einzelplan)")  # Haushaltswoche: "a) hier: Einzelplan 04 … Drucksachen 20/8604, 20/8661" (20150:23)
# Drucksachen-Formen: "Drucksache 20/N" 1569, "Drucksachen 20/N, 20/N" 434 Fälle; über Umbruch gespaltene Nummern
# kommen nicht vor
DRS_RE = re.compile(r"20/(\d{3,5})")
# Kurzbezeichnung steht als Klammer hinter dem Titel: "(Zustrombegrenzungsgesetz)" 20211:1069; auch "-Gesetz" und
# "Gesetz zur …" ("(Gesundes-Herz-Gesetz – GHG)" 20196, "(Gesetz zur Trennung von Amt und Mandat)" 20200) und lange
# Abkürzungen ("(ERP-Wirtschaftsplangesetz 2025 – ERPWiPlanG 2025)" 20194, "(Gesetz zur Einführung der
# Begründungspflicht)" 20200); 8 von 70 XML-Kurztiteln fehlten sonst
KURZTITEL_RE = re.compile(r"\(([^()]{0,80}[Gg]esetz[^()]{0,60})\)")
STUFE_MAP = {"Erste": "erste", "Zweite": "zweite", "Zweite und dritte": "zweite und dritte", "Dritte": "dritte"}

# Abstimmung im laufenden TOP; "und Schlussabstimmung." steht als Absatzanfang nach Zwischenruf (20211:1037, 4317).
# Bloßes "namentliche Abstimmung" zählt nicht: in 162 Zeilen ohne andere Formel ist es meist ein Hinweis auf eine laufende
# Abstimmung eines früheren Punkts ("dass die namentliche Abstimmung bald endet", 20178:8715). Formel sind nur Verlangen
# und Eröffnung (20152:1445, 20147:4125, 20152:2619). "Wer stimmt gegen" steht vor Überweisung/Zurückverweisung (33 Zeilen),
# "Wer stimmt dafür?" ohne andere Formel vor Sammelübersichten und Einzelplänen (301 Zeilen, 20147:4201, 20149:77)
ABSTIMMUNG_RE = re.compile(
    r"Wir kommen (?:nun |jetzt |damit |somit )?zur Abstimmung|dritten? Beratung und Schlussabstimmung|^und Schlussabstimmung|"
    r"namentliche Abstimmung(?:en)? (?:verlangt|beantragt)|eröffne ich (?:\w+ ){0,2}die (?:\w+ )?namentliche Abstimmung|"
    r"Wer stimmt für|Wer stimmt (?:\w+ )?dafür|Wer stimmt gegen|Ich eröffne die (?:\w+ )?(?:namentliche )?Abstimmung|"
    r"Abstimmung über den")
# Ergebnisformeln ("ist damit angenommen" 99, "Damit ist … angenommen" 65, "Der Gesetzentwurf ist abgelehnt" 20211:3519);
# Sammelübersichten zu Petitionen und Wahlvorschläge (21 und 11 Abstimmungen ohne Ergebnis, "Die Sammelübersicht ist angenommen" 20169:3973);
# "Damit ist auch dieser Antrag mit großer Mehrheit angenommen" (20210:9079)
ERGEBNIS_RE = re.compile(
    r"((?:Der|Die|Das|(?:Damit|Dann|Somit) ist (?:auch )?(?:der|die|das|dieser|diese|dieses)) "
    r"(?:Gesetzentwurf|Entwurf|Antrag|Entschließungsantrag|Änderungsantrag|Beschlussempfehlung|Vorlage|Verordnung|Einzelplan|"
    r"Sammelübersicht|Wahlvorschlag)"
    r"[^.]{0,220}?\b(?:angenommen|abgelehnt)\b)")
# Überweisung in erster Beratung: "Interfraktionell wird Überweisung der Vorlage auf Drucksache 20/10722 … vorgeschlagen. … Dann ist so beschlossen." (20160 nach Z. 1009);
# bei strittiger Federführung "Damit ist der Überweisungsvorschlag angenommen" (20181:3225)
UEBERWEISUNG_RE = re.compile(
    r"Überweisung(?:en|svorschl[aä]ge?)?\b[^.]{0,200}?(?:vorgeschlagen|einverstanden|beschlossen)|"
    r"Dann (?:ist|sind) (?:die Überweisung(?:en)? )?so beschlossen|(?:ist|sind) die Überweisung(?:en)? (?:so )?beschlossen|"
    r"an die [^.]{0,160}?Ausschüsse (?:zu )?überw|Überweisungsvorschlag (?:ist )?(?:damit |somit |also )?angenommen")
# "die Vorlagen an die … Ausschüsse zu überweisen" (Plural) gilt für alle nicht abgestimmten Gegenstände des Sammelaufrufs,
# nicht nur für den zuletzt genannten (20172:5971, 20194:5137, 20207:6261: 11, 13 und 64 Gegenstände blieben "offen")
SAMMEL_UEBERWEISUNG_RE = re.compile(r"\bdie Vorlagen\b")
VERTAGT_RE = re.compile(r"\bvertagt\b|wird abgesetzt|von der Tagesordnung (?:ab)?gesetzt")
# Namentliche Abstimmung: Ergebnis wird später, oft in einem anderen TOP, verkündet (20211 ZP 35: Aufruf Z. 1065, Ergebnis Z. 3519)
# auch "Ergebnis der namentlichen Schlussabstimmung" (20167:1313) und "der zweiten namentlichen Abstimmung"
NAMENTLICH_ERGEBNIS_RE = re.compile(r"Ergebnis der (?:\w+ )?namentlichen (?:Schluss)?[Aa]bstimmung")
ZWEITE_BERATUNG = "in zweiter Beratung"

# Zwischenruf am Absatzanfang vor dem Redetext (12676 Fälle); ohne schließende Klammer gilt der ganze Absatz als
# Zwischenruf (43 Fälle)
KLAMMER_RE = re.compile(r"^\((?:[^()]|\([^()]*\))*\)\s*")
# str.split() zählte Gedankenstriche „–“ und Halbgeviertstriche als Wörter mit
WORT_RE = re.compile(r"\w+(?:[-'’]\w+)*")
# Wortdefinition: Wort = Whitespace-Token mit ≥ 1 Buchstaben/Ziffer; WORT_RE zählte "1,5" und "Herr/Frau" doppelt
BUCHSTABE_RE = re.compile(r"[^\W_]")


def zaehle_woerter(text: str) -> int:
    return sum(1 for tok in text.split() if BUCHSTABE_RE.search(tok))

# Präsidium fragt meist "Gestatten Sie eine Frage oder Bemerkung …?" (20160:5513, 5323, 1791), seltener "Zwischenfrage"
ZWISCHENFRAGE_WORT = re.compile(r"Zwischenfrage|Zwischenbemerkung|Nachfrage|[Gg]estatten Sie eine|[Ee]rlauben Sie eine|Frage oder Bemerkung")
KURZINTERVENTION_WORT = re.compile(r"Kurzintervention")
# Ankündigungsformen; ohne sie liefen Kurzinterventionen als plenarrede (60 → 70 erkannt): "zu einer
# Kurzintervention", "bittet um eine Kurzintervention" (20211:2405), "Kurzintervention von Bettina Hagedorn" (20146:4535), Zwischenruf "Kurzintervention!" (20211:1889)
KURZINTERVENTION_ANKUENDIGUNG = re.compile(r"zu einer Kurzintervention|um eine Kurzintervention|Kurzintervention (?:hat|des|der|erhält|von)|Kurzintervention!")
# Erwiderung: "Frau Baerbock, möchten Sie antworten?" (20211:1923), "Sie möchten antworten?" (20211:2337), "Möglichkeit zur Erwiderung"
ERWIDERUNG_WORT = re.compile(r"möchten Sie antworten|Sie möchten antworten|Möglichkeit zur Erwiderung|zur Erwiderung|können Sie (?:auf beide |darauf )?antworten")
# Aufruf des nächsten Redners löscht offene Marker für Zwischenfrage/Kurzintervention
# "hat nun der Abgeordnete Jan Ralf Nolte das Wort" (20160:5369): Name steht zwischen "hat" und "das Wort"
NAECHSTER_REDNER = re.compile(r"das Wort|Redner|Rednerin")
# download_manifest.csv hat kein Datum; Titelseite "Berlin, Donnerstag, den 21. März 2024" (20160.jsonl, Zeile 5)
DATUM_RE = re.compile(r"den (\d{1,2})\. (\w+) (\d{4})")
MONATE = {m: i + 1 for i, m in enumerate(
    ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"])}

REGIERUNG = {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}  # FDP nur bis 196 (Ampel-Bruch 06.11.2024, Sitzung 197 am 07.11.)


def phase(nr: int) -> str:
    return "koalition" if nr <= 196 else "minderheit" if nr <= 204 else "nach_vertrauensfrage"


def rolle_regierung(nr: int, fraktion: str | None) -> bool:
    return fraktion in REGIERUNG or (fraktion == "FDP" and nr <= 196)


def sitzungsdatum(nr: int) -> tuple[date | None, str]:
    """Datum von der Titelseite (data/lines, section toc); Rückfall XML-Attribut sitzung-datum."""
    p = config.LINES_DIR / f"{config.WAHLPERIODE}{nr:03d}.jsonl"
    if p.exists():
        with open(p) as f:
            for _, row in zip(range(15), f):
                m = DATUM_RE.search(json.loads(row).get("text", ""))
                if m and m.group(2) in MONATE:
                    return date(int(m.group(3)), MONATE[m.group(2)], int(m.group(1))), "titelseite"
    return stammdaten.sitzungsdaten().get(nr), "xml"


# Anlagen: "Anlage 3" / "Zu Protokoll gegebene Rede" / Titelabsätze / "(Tagesordnungspunkt 20)" / "Name (Fraktion):" (20160_anlagen.txt Z. 665–681)
ANLAGE_RE = re.compile(r"^Anlage (\d+)$")
# XML zählt auch "Erklärungen nach § 31 GO" als <rede> (gold 20146: 8 von 9 Anlagen-Reden, 20146_anlagen.txt Z. 921)
ZU_PROTOKOLL_RE = re.compile(r"^(?P<art>Zu Protokoll gegebene Reden?|Erklärung(?:en)? nach § 31 GO)")
TOP_KLAMMER_RE = re.compile(r"\((?P<liste>(?:Tagesordnungspunkt|Zusatzpunkt)e?[^)]{1,60})\)")


def parse_anlagen(nr: int, datum: date | None, top_gesetze: dict) -> list[dict]:
    """Zu Protokoll gegebene Reden aus data/txt/<nr>_anlagen.txt (gold 20146: 9 von 177 Reden stehen dort)."""
    p = config.TXT_DIR / f"{config.WAHLPERIODE}{nr:03d}_anlagen.txt"
    if not p.exists():
        return []
    zeilen = p.read_text(encoding="utf-8").splitlines()
    drs_top = {d: tid for tid, items in top_gesetze.items() for g in items for d in g["drucksachen_alle"]}
    out: list[dict] = []
    anlage, titel, cur, sammle = None, None, None, False
    i = 0
    while i < len(zeilen):
        t = zeilen[i]
        zeile = i + 1
        if not t.strip():
            i += 1
            continue
        if m := ANLAGE_RE.match(t):
            anlage, titel, cur, sammle = int(m.group(1)), None, None, False
            i += 1
            continue
        if m := ZU_PROTOKOLL_RE.match(t):
            titel, cur, sammle = [], None, True
            redetyp = "zu_protokoll" if m["art"].startswith("Zu") else "erklaerung_go31"
            i += 1
            continue
        if titel is None:
            # Anlage 1/2 sind Namenslisten (Entschuldigte, Abstimmungslisten), keine Reden
            i += 1
            continue
        kopf, skip = (t, 0) if t.endswith(":") and erkenne_kopf(t) else join_kopf(zeilen, i)
        info = erkenne_kopf(kopf) if kopf else None
        if info and info["art"] != "praesidium":
            sammle = False
            titeltext = " ".join(titel)
            top_id = None
            if (mt := TOP_KLAMMER_RE.search(titeltext)) and (pk := expand_punkte(mt["liste"])):
                top_id = top_id_von(pk[0])
            elif (d := DRS_RE.search(titeltext)):
                top_id = drs_top.get("20/" + d.group(1))
            top_typ = top_gesetze[top_id][0]["top_typ"] if top_gesetze.get(top_id) else "Debatte"
            cur = {"sitzung": nr, "datum": datum, "zeile_start": zeile, "zeile_ende": zeile, "name": info["name"],
                   "ort": info["ort"], "fraktion_pdf": info["fraktion_pdf"], "rolle": info["rolle"],
                   "top_id": top_id, "top_typ": top_typ, "redetyp": redetyp, "anlage": anlage,
                   "anlage_titel": titeltext[:300], "_text": []}
            out.append(cur)
            i += 1 + skip
            continue
        if sammle:
            titel.append(t)
        elif cur:
            cur["_text"].append(strip_zwischenruf(t))
            cur["zeile_ende"] = zeile
        i += 1
    return out


# Rednerköpfe: data/txt trägt keine Schriftart, daher erkennt der Parser sie am Muster mit Doppelpunkt (KOPF_ABG, KOPF_REG)
def join_kopf(zeilen: list[str], i: int) -> tuple[str, int]:
    """Kopf ab zeilen[i] mit bis zu zwei Folgezeilen verbinden: umbrochene Regierungsfunktion (1577 Fälle, auch mit
    Silbentrennung), umbrochene Fraktionsklammer der Grünen (111 Fälle); ohne Treffer ("", 0)."""
    l = zeilen[i]
    if l.endswith(":") or len(l) > 140:
        return (l, 0) if l.endswith(":") else ("", 0)
    offen = l.count("(") > l.count(")")
    if not (offen or KOPF_REG_START.match(l)):
        return "", 0
    verbraucht = 0
    for j in range(i + 1, min(i + 4, len(zeilen))):
        nxt = zeilen[j]
        if not nxt.strip():
            continue
        verbraucht = j - i
        l = l[:-1] + nxt if l.endswith("-") and nxt[:1].islower() else l + " " + nxt
        if l.endswith(":") or len(l) > 200:
            break
    if l.endswith(":") and (KOPF_ABG.match(l) or KOPF_REG.match(l)):
        return l, verbraucht
    return "", 0


def erkenne_kopf(text: str) -> dict | None:
    if m := PRAESIDIUM_RE.match(text):
        return {"name": m["name"], "rolle": m["rolle"], "fraktion_pdf": None, "ort": None, "art": "praesidium"}
    if m := KOPF_ABG.match(text):
        fr = m["fr"].strip()
        fr_norm = FRAKTION_KOPF.get(fr.lower())
        # Ministerpräsident-Köpfe laufen über KOPF_REG; eine Klammer ohne Fraktionsname ist Land/Ort
        return {"name": m["name"].strip(), "rolle": "MdB", "fraktion_pdf": fr_norm or None,
                "ort": m["ort"] or (fr if fr_norm is None else None), "art": "abgeordnet"}
    if m := KOPF_REG.match(text):
        return {"name": m["name"].strip(), "rolle": m["rolle"].strip(), "fraktion_pdf": None, "ort": None, "art": "regierung"}
    return None


@lru_cache(maxsize=None)
def _lookup(name: str, datum: date, fraktion_pdf: str | None, ort: str | None) -> dict:
    return stammdaten.lookup(name, datum, fraktion_pdf, ort)


# TOPs einzeln auflösen, weil verbundene Punkte alle Gegenstände des Aufrufs erben (Beteiligte je Gesetz)
def expand_punkte(liste: str) -> list[str]:
    """"Tagesordnungspunkte 36 a bis 36 h sowie die Zusatzpunkte 9 a und 9 b" -> ["TOP 36 a", …, "ZP 9 b"]."""
    out: list[str] = []
    for m in PUNKT_RE.finditer(liste):
        art = "TOP" if m.group(1) == "Tagesordnungspunkt" else "ZP"
        nums = m.group(2)
        teile = re.split(r"\s*(?:,|und|sowie)\s*", nums)
        for t in teile:
            if "bis" in t:
                a, b = [NUM_RE.match(x.strip()) for x in t.split("bis")]
                if not (a and b):
                    continue
                if a.group(2) and b.group(2) and a.group(1) == b.group(1):
                    out += [f"{art} {a.group(1)} {l}" for l in _buchstaben(a.group(2), b.group(2))]
                elif not a.group(2) and not b.group(2) and a.group(1).isdigit() and b.group(1).isdigit():
                    out += [f"{art} {n}" for n in range(int(a.group(1)), int(b.group(1)) + 1)]
                else:
                    out += [f"{art} {a.group(1)} {a.group(2) or ''}".strip(), f"{art} {b.group(1)} {b.group(2) or ''}".strip()]
            elif (n := NUM_RE.match(t.strip())):
                out.append(f"{art} {n.group(1)} {n.group(2) or ''}".strip())
            elif t.strip().isalpha() and len(t.strip()) <= 2 and out:
                # "31 a und b": Buchstabe erbt die Nummer
                out.append(f"{out[-1].rsplit(' ', 1)[0]} {t.strip()}")
    return list(dict.fromkeys(out))


def top_id_von(punkt: str) -> str:
    if not punkt.startswith(("TOP", "ZP")):
        return punkt
    art, nr = punkt.split(" ")[:2]
    return ("Tagesordnungspunkt " if art == "TOP" else "Zusatzpunkt ") + nr


def top_typ_von(block: str) -> str:
    if re.search(r"Befragung der Bundesregierung", block):
        return "Befragung"
    if re.search(r"\bFragestunde\b", block):
        return "Fragestunde"
    if re.search(r"Aktuelle Stunde", block):
        return "Aktuelle Stunde"
    if re.search(r"Regierungserklärung", block):
        return "Regierungserklärung"
    if re.search(r"\bWahl(?:en)?\b|Geschäftsordnung(?!\s*\n)|Geschäftsordnungs", block[:400]) and not DRS_RE.search(block):
        return "Wahl/GO"
    return "Debatte"


# Inhaltsverzeichnis: "Zusatzpunkt 31:" / "a)Erste Beratung des von der Bundesregie-" / "… . . . . 26194 C" / "Drucksache 20/13954"
TOC_TOP_RE = re.compile(r"^(Tagesordnungspunkt|Zusatzpunkt) (" + NUMMER + r")\s*:?$")
TOC_SUB_RE = re.compile(r"^([a-z]{1,2})\)\s*(.*)$")
SEITENREF_RE = re.compile(r"(?:\s*\.){2,}\s*\d+\s*[A-D]$|\s+\d{4,5}\s[A-D]$")


@lru_cache(maxsize=None)
def tagesordnung(nr: int) -> dict[str, dict]:
    """Punkt ("TOP 31 a", "ZP 34") -> Titel und Drucksachen aus dem Inhaltsverzeichnis. Nur Nachschlagequelle für
    Sammelaufrufe ohne Drucksachen im Aufrufblock (284 von 552 fehlenden Drucksachen), nie selbst ein Aufruf."""
    p = config.LINES_DIR / f"{config.WAHLPERIODE}{nr:03d}.jsonl"
    if not p.exists():
        return {}
    eintraege: dict[str, dict] = {}
    cur = None
    with open(p) as f:
        next(f)
        for row in f:
            ln = json.loads(row)
            if ln["section"] != "toc":
                break
            t = ln["text"].strip()
            if m := TOC_TOP_RE.match(t):
                art = "TOP" if m.group(1) == "Tagesordnungspunkt" else "ZP"
                cur = f"{art} {m.group(2)}"
                eintraege[cur] = {"teile": [], "drucksachen": []}
                continue
            if cur is None:
                continue
            if m := TOC_SUB_RE.match(t):
                basis = cur.rsplit(" ", 1)[0] if cur.rsplit(" ", 1)[-1].isalpha() else cur
                cur = f"{basis} {m.group(1)}"
                eintraege[cur] = {"teile": [], "drucksachen": []}
                t = m.group(2)
            e = eintraege[cur]
            e["drucksachen"] += ["20/" + d for d in DRS_RE.findall(t) if "20/" + d not in e["drucksachen"]]
            t = SEITENREF_RE.sub("", t).strip()
            if t and not t.startswith("Drucksache"):
                e["teile"].append(t)
    return {k: {"titel": join_hyphenated(v["teile"]), "drucksachen": v["drucksachen"]}
            for k, v in eintraege.items() if v["drucksachen"]}


def aufruf_block(zeilen: list[str], start: int, rest_erster: str) -> tuple[list[tuple[int, str]], int]:
    """Absätze des Aufrufblocks (Zeilennr, Text) ab dem Aufruf bis zum ersten Prosa-Absatz oder Kopf."""
    block = [(start, rest_erster)] if rest_erster.strip() else []
    j = start + 1
    while j < len(zeilen) and j < start + 80:
        t = zeilen[j - 1]
        if t.strip():
            if erkenne_kopf(t) or PROSA_RE.match(t) or AUFRUF_RE.search(t):
                break
            block.append((j, t))
        j += 1
    return block, j


# Ordinaltitel: "Entwurfs eines Zweiten Gesetzes zur Änderung …", "eines 2. Gesetzes", "eines Zweiten Haushaltsfinanzierungsgesetzes 2024",
# "eines … Gesetzes" (LLM-Abgleich: 61 von 334 Entwürfen standen als sonstiges/bericht, 38 Drucksachen fehlten in gesetze_einheit)
ENTWURF_RE = re.compile(r"Entwurfs? (?:eines?|des|der)(?:\s+\S{1,40}){0,4}?\s+\S*[Gg]esetz|Gesetzentwurf")


def _item(seg: str, drs: list[str], punkt: str | None, quelle: str) -> dict:
    m = ITEM_RE.match(seg)
    stufe = STUFE_MAP.get(m["stufe"]) if m and m["stufe"] else "beratung"
    if (ms := re.match(r"(Erste|Zweite|Zweite und dritte|Dritte) Beratung", seg)):
        stufe = STUFE_MAP[ms.group(1)]
    if seg.startswith("hier: Einzelplan"):
        typ, stufe = "gesetzentwurf", "zweite"  # Einzelpläne sind Teile des Haushaltsgesetzes in zweiter Beratung
    elif ENTWURF_RE.search(seg):
        typ = "gesetzentwurf"
    elif re.search(r"Befragung|Fragestunde", seg):
        typ = "befragung"
    elif re.match(r"^(?:Beratung des )?Antrag", seg):
        typ = "antrag"
    elif re.search(r"Beschlussempfehlung|Bericht|Unterrichtung", seg):
        typ = "bericht"
    else:
        typ = "sonstiges"
    # Trennstriche aus eingerückten Titelzeilen ("Bundes- regierung", 20211:17) schließen; Ergänzungsstriche vor einer
    # Konjunktion bleiben ("Strom- und Energiesteuerrecht", 20188:9667; gefunden über den Titelabgleich mit dem XML, 55 Titel betroffen)
    seg = re.sub(r"([a-zäöüß])- (?=[a-zäöüß])(?!(?:und|oder|bzw\.|sowie|beziehungsweise|respektive)\b)", r"\1", seg)
    titel = re.split(r"\s+Drucksachen?\s+20/", seg, maxsplit=1)[0]
    titel = re.sub(r"\s+(?:ZP|TOP)\s*\d+.*$", "", titel).strip(" –-")
    kurz = KURZTITEL_RE.search(titel)
    # Beschlussempfehlung ist Attribut des Entwurfs. Eigenes Item nur bei "Beratung der Beschlussempfehlung …"
    # (zu Anträgen/Berichten: Drucksachen 20/9312 (Antrag), 20/9880 (Beschlussempfehlung), 20160:2229)
    if re.match(r"^(?:Beratung der )?Beschlussempfehlung", seg):
        typ_fein = "beschlussempfehlung"
        be = drs[-1] if len(drs) > 1 else (drs[0] if drs else None)
    else:
        typ_fein = typ if typ in ("gesetzentwurf", "antrag", "bericht") else "sonstiges"
        mb = re.search(r"Beschlussempfehlung.*?(20/\d{3,5})", seg)
        be = mb.group(1) if mb and drs and mb.group(1) != drs[0] else None
    gesetz_titel, kurz2 = gesetz_titel_von(titel)
    return {"drucksache": drs[0] if drs else None, "drucksachen_alle": drs, "titel": titel[:800], "gesetz_titel": gesetz_titel,
            "kurztitel_ende": kurz2,
            "kurztitel": kurz.group(1) if kurz else None, "typ": typ, "beratungsstufe": stufe, "punkt": punkt,
            "typ_fein": typ_fein, "ist_gesetz": typ_fein == "gesetzentwurf", "beschlussempfehlung_drucksache": be,
            "bezug_drucksache": None, "quelle_drucksache": quelle}


BERATUNGSFORMEL_RE = re.compile(r"^(?:(?:Erste|Zweite|Zweite und dritte|Dritte) Beratung des (?:von|vom) .*? eingebrachten |"
                                r"Beratung (?:des|der) (?:Antrags|Antrages|Beschlussempfehlung und des Berichts|Unterrichtung|Großen Anfrage) )")


def gesetz_titel_von(titel: str) -> tuple[str, str | None]:
    """Titel ohne Beratungsformel ("Zweite und dritte Beratung des von der Fraktion der CDU/CSU eingebrachten Entwurfs eines
    Gesetzes …" -> "Entwurf eines Gesetzes …"), die vorher 202 von 221 Einheitstitel einleitete;
    Kurztitel aus der Klammer am Titelende."""
    m = re.search(r"Entwurfs? (?:eines?|des|der) .*$", titel)
    t = titel[m.start():] if m else BERATUNGSFORMEL_RE.sub("", titel)
    t = re.sub(r"^Entwurfs ", "Entwurf ", t).strip(" :–-")
    k = re.search(r"\(([^()]{3,80})\)\s*$", t)
    return t[:400], (k.group(1) if k and not re.search(r"Ausschuss|Drucksache", k.group(1)) else None)


def parse_gesetze(block: list[tuple[int, str]], punkte: list[str], to: dict | None = None) -> list[dict]:
    text = " ".join(t for _, t in block)
    starts = [m.start() for m in ITEM_RE.finditer(text)]
    starts.append(len(text))
    items = []
    labels = iter(punkte)
    for a, b in zip(starts, starts[1:]):
        seg = text[a:b].strip()
        drs = ["20/" + d for d in dict.fromkeys(DRS_RE.findall(seg))]
        items.append(_item(seg, drs, next(labels, punkte[-1] if punkte else None), "aufruf"))
    items = [it for it in items if it["drucksache"]]
    # Sammelaufruf ohne Drucksachen im Block ("Zusatzpunkte 31 a bis 31 bb", 20203:3811): Inhaltsverzeichnis nachschlagen
    bekannt = {d for it in items for d in it["drucksachen_alle"]}
    vorhanden = {it["punkt"] for it in items}
    for pk in punkte:
        e = (to or {}).get(pk)
        if pk in vorhanden or not e or set(e["drucksachen"]) & bekannt:
            continue
        items.append(_item(e["titel"], e["drucksachen"], pk, "tagesordnung"))
    entwuerfe = [it["drucksache"] for it in items if it["ist_gesetz"]]
    for it in items:
        if it["typ_fein"] == "beschlussempfehlung":
            # Bezug: beratene Vorlage (erste Drucksache vor der Beschlussempfehlung), sonst der einzige Gesetzentwurf des Aufrufs
            eigene = it["drucksachen_alle"]
            it["bezug_drucksache"] = (eigene[0] if len(eigene) > 1 else None) or (entwuerfe[0] if len(entwuerfe) == 1 else None)
    # Ohne Drucksache (Aktuelle Stunde, Befragung, Wahlen) gibt es keinen Schlüssel; nur Drucksachen-Gegenstände sind Gesetze/Anträge
    return items


def _abstimmung_im_block(items: list[dict], block: list[tuple[int, str]]) -> None:
    """Beschlussfassung ohne Aussprache: Abstimmungsformel steht im Aufrufblock selbst ("… Drucksachen 20/4874, 20/9892
    Der Ausschuss empfiehlt … Wer stimmt für diese Beschlussempfehlung?", 20147:4257, 20185:4145) und wird dem genannten Gegenstand zugeordnet."""
    aktiv: list[dict] = []
    for zeile, t in block:
        ziel = _ziel_items(items, t, aktiv)
        if ziel is not items or len(items) == 1:
            aktiv = ziel
        if ABSTIMMUNG_RE.search(t):
            for g in ziel:
                if not g["abgestimmt"]:
                    g["abgestimmt"], g["abstimmung_zeile"] = True, zeile
        if UEBERWEISUNG_RE.search(t):
            sammel = [g for g in items if not g["abgestimmt"]] if SAMMEL_UEBERWEISUNG_RE.search(t) else []
            for g in ziel + [g for g in sammel if g not in ziel]:
                g["_ueberwiesen"] = True
        if ERGEBNIS_RE.search(t):
            for g in ziel:
                if g["abgestimmt"] and not g["ergebnis_text"]:
                    g["ergebnis_text"] = ergebnis_fuer(t, g)


def _ziel_items(ziel: list[dict], t: str, aktiv: list[dict]) -> list[dict]:
    """Gegenstände, auf die sich ein Präsidiumsabsatz bezieht: genannte Drucksache > genannter Punkt ("Tagesordnungspunkt 6 a",
    20207 ZP 7 erbte sonst die Abstimmung von TOP 6 a) > laufender Gegenstand > Gattungswort > alle."""
    genannt = {"20/" + d for d in DRS_RE.findall(t)}
    if (h := [g for g in ziel if g["drucksache"] in genannt or set(g["drucksachen_alle"]) & genannt]):
        return h
    # Haushaltswoche: Einzelpläne teilen sich Drucksache 20/8661 und werden einzeln abgestimmt (20149:1447 Einzelplan 08, 1449 Einzelplan 20)
    plaene = {f"Einzelplan {n}" for n in EINZELPLAN_RE.findall(t)}
    if plaene and (h := [g for g in ziel if any(p in (g["titel"] or "") for p in plaene)]):
        return h
    pk = set(expand_punkte(t))
    if pk and (h := [g for g in ziel if g["punkt"] in pk]):
        return h
    if aktiv:
        return aktiv
    for wort, typen in (("Gesetzentwurf", {"gesetzentwurf"}), ("Beschlussempfehlung", {"beschlussempfehlung"}), ("Antrag", {"antrag"})):
        if wort in t and (h := [g for g in ziel if g["typ_fein"] in typen]):
            # Abstimmungen laufen in Aufrufreihenfolge: erster noch offener Gegenstand (20164:1009 traf sonst beide Entwürfe)
            return [g for g in h if not g["abgestimmt"]][:1] or h[-1:]
    return ziel


# "Antrag auf Zurückverweisung" bzw. "Zurücküberweisung" (20203:4049)
ZURUECK_RE = re.compile(r"Zurück(?:verweisung|überweisung)")
# Nummer im Ergebnissatz muss zur Nummer im Titel passen: "Die Sammelübersicht 566 ist angenommen" gehört nicht zu
# Sammelübersicht 563 (20166 TOP 27 h), "Damit ist der Einzelplan 05 angenommen" nicht zu Einzelplan 04 (20150)
NUMMER_IM_ERGEBNIS_RE = re.compile(r"(Sammelübersicht|Einzelplan) (\d+)")


def ergebnis_passt(satz: str, typ: str, titel: str | None = None) -> bool:
    # "Der Antrag auf Zurückverweisung ist angenommen" (20207:7023) beendet die Beratung jeder Vorlage; abgelehnt geht
    # die Abstimmung über die Vorlage weiter, deren Ergebnis danach folgt
    if ZURUECK_RE.search(satz):
        return "angenommen" in satz
    # Änderungs- und Entschließungsanträge sind Nebenabstimmungen vor oder nach der Vorlage (20197 TOP 6: "Der
    # Änderungsantrag ist damit abgelehnt", danach "Damit ist der Antrag angenommen")
    if "Änderungsantrag" in satz or "Entschließungsantrag" in satz:
        return False
    if titel is not None and (mn := NUMMER_IM_ERGEBNIS_RE.search(satz)):
        # nur Sammelübersichten bzw. Einzelpläne mit derselben Nummer (20166 ZP 5 a erbte sonst Sammelübersicht 564)
        mt = re.search(mn.group(1) + r" (\d+)", titel or "")
        if not mt or mt.group(1) != mn.group(2):
            return False
    if "Gesetzentwurf" in satz:
        return typ == "gesetzentwurf"
    if "antrag" in satz.lower():
        return typ == "antrag"
    if "Beschlussempfehlung" in satz:
        return typ == "bericht"
    return True


def ergebnis_fuer(text: str, g: dict) -> str | None:
    """Erster Ergebnissatz im Absatz, der zum Gegenstand passt; ein Absatz nennt oft erst den Änderungsantrag, dann die
    Vorlage."""
    return next((m.group(1).strip() for m in ERGEBNIS_RE.finditer(text) if ergebnis_passt(m.group(1), g["typ"], g["titel"])),
                None)


def text_ohne_zwischenruf(t: str, zustand: dict) -> str:
    """Wie strip_zwischenruf, aber eine unten offene Klammer (43 Fälle) läuft bis zur schließenden Klammer,
    höchstens 3 Absätze weiter (längere Reste sind Extraktionsfehler, nicht Zwischenrufe)."""
    if zustand.get("offen", 0):
        zustand["offen"] -= 1
        if ")" in t:
            zustand["offen"] = 0
            return t.split(")", 1)[1].strip()
        return ""
    if t.startswith("(") and ")" not in t:
        zustand["offen"] = 3
        return ""
    return strip_zwischenruf(t)


def strip_zwischenruf(t: str) -> str:
    if t.startswith("("):
        if ")" not in t:
            return ""
        t = KLAMMER_RE.sub("", t)
    return t


def parse_sitzung(nr: int) -> dict:
    stem = f"{config.WAHLPERIODE}{nr:03d}"
    zeilen = (config.TXT_DIR / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
    datum, quelle_datum = sitzungsdatum(nr)
    beitraege: list[dict] = []
    gesetze: list[dict] = []
    top_id, top_typ, top_punkte = None, None, []
    top_gesetze: dict[str, list[dict]] = {}
    top_block_ende = {}
    cur = None
    hauptredner = None
    marker = None
    fragesteller = None
    klammer = {}
    aktiv: list[dict] = []  # Gegenstand, dessen Block im Sammelaufruf gerade läuft ("Beschlussfassung ohne Aussprache")
    rueck = None  # Rücksprung zu einem früheren Punkt nur zur Ergebnisverkündung (20211:3435)
    namentlich = None  # (Zielgesetze, Restabsätze) nach "Ergebnis der namentlichen Abstimmung"
    i = 0
    while i < len(zeilen):
        t = zeilen[i]
        zeile = i + 1
        if not t.strip():
            i += 1
            continue
        kopf, skip = (t, 0) if t.endswith(":") and erkenne_kopf(t) else join_kopf(zeilen, i)
        info = erkenne_kopf(kopf) if kopf else None
        if info:
            if cur:
                cur["zeile_ende"] = zeile - 1
            if info["art"] != "praesidium" and rueck:
                if rueck["abstimmung"]:
                    top_id, top_typ, top_punkte = rueck["vorher"]
                rueck = None
            if info["art"] == "praesidium":
                redetyp = "praesidium"
            elif top_typ in ("Befragung", "Fragestunde"):
                redetyp = "befragung_antwort" if info["art"] == "regierung" else "befragung_frage"
            elif (marker == "zwischenfrage" or info["name"] == fragesteller) and info["name"] != hauptredner:
                # Rückfragen des Fragestellers im selben Dialog (20160:1797–1827 Aeffner/Stracke) bleiben Zwischenfrage
                redetyp, marker, fragesteller = "zwischenfrage", None, info["name"]
            elif marker == "kurzintervention" and info["name"] != hauptredner:
                redetyp, marker = "kurzintervention", "erwiderung"
            elif marker == "erwiderung" and info["name"] == hauptredner:
                redetyp, marker = "erwiderung", None
            else:
                redetyp = "plenarrede"
                if info["name"] != hauptredner:
                    marker, fragesteller = None, None
                hauptredner = info["name"]
            cur = {"sitzung": nr, "datum": datum, "zeile_start": zeile, "zeile_ende": zeile, "name": info["name"],
                   "ort": info["ort"], "fraktion_pdf": info["fraktion_pdf"], "rolle": info["rolle"],
                   "top_id": top_id, "top_typ": top_typ, "redetyp": redetyp, "_text": []}
            beitraege.append(cur)
            i += 1 + skip
            continue
        if cur and cur["redetyp"] == "praesidium":
            if NAECHSTER_REDNER.search(t) and not KURZINTERVENTION_WORT.search(t):
                marker = None
            if KURZINTERVENTION_WORT.search(t) and KURZINTERVENTION_ANKUENDIGUNG.search(t):
                marker = "kurzintervention"
            elif ERWIDERUNG_WORT.search(t):
                marker = "erwiderung"
            elif ZWISCHENFRAGE_WORT.search(t) and marker != "erwiderung":
                marker = "zwischenfrage"
            # Aufruf eines TOP (nur im Präsidiumstext; das Inhaltsverzeichnis steht nicht im txt)
            m = AUFRUF_RE.search(t)
            neu = bool(m) and not ABGESETZT_RE.search(t[m.end():m.end() + 40])
            if not neu:
                unterpunkte = {g["punkt"] for items in top_gesetze.values() for g in items}
                for alt in (WIEDER_RE.search(t), TOP_ENDE_RE.search(t)):
                    # "… komme nun zum Zusatzpunkt 7" (20164:1039) ist ein Unterpunkt des Sammelaufrufs "Zusatzpunkte 6 bis 8", kein neuer Aufruf
                    if alt and (pw := expand_punkte(alt["liste"])) and top_id_von(pw[0]) not in top_gesetze and pw[0] not in unterpunkte:
                        # „… kommen jetzt zum Tagesordnungspunkt 20:“ (20160:9471) ist ein Erstaufruf ohne "rufe"
                        m, neu = alt, True
                        break
                    # Unterpunkt eines früheren Sammelaufrufs: 20207 ruft in Z. 6263 TOP 26 b ff., ZP 28 und ZP 10 auf, danach
                    # laufen ZP 28 a und ZP 10 a als eigene Punkte. „Zusatzpunkt 10 g:“ (Z. 6963) wird laufender Gegenstand, sonst
                    # fallen die Abstimmungen über ZP 10 g bis 10 l an ZP 10 a. Gesucht wird in allen Gegenständen der Sitzung,
                    # weil ein zweiter Aufruf von TOP 26 (Z. 6389) den Eintrag in top_gesetze ersetzt. "zurück zu" bleibt beim
                    # Rücksprung unten (20211:3435)
                    if (alt and not neu and "zurück" not in t and (pw := expand_punkte(alt["liste"]))
                            and pw[0] not in {g["punkt"] for g in top_gesetze.get(top_id, [])}
                            and (h := [g for g in gesetze if g["punkt"] == pw[0]])):
                        aktiv = h
                        break
            frei = None
            if not neu and (mf := FREI_RE.search(t)) and mf["frei"] not in top_gesetze:
                frei, m = mf["frei"], mf
            if neu or frei:
                punkte = expand_punkte(m["liste"]) if neu else [frei]
                # Ergebnis im selben Absatz vor dem nächsten Aufruf: "… Damit ist die Beschlussempfehlung angenommen. Ich rufe
                # auf den Tagesordnungspunkt 27" (20195:4877) gehört zum alten Punkt
                vor = strip_zwischenruf(t[:m.start()])
                if punkte and top_id and top_gesetze.get(top_id) and ERGEBNIS_RE.search(vor):
                    for g in _ziel_items(top_gesetze[top_id], vor, aktiv):
                        if g["abgestimmt"] and not g["ergebnis_text"]:
                            g["ergebnis_text"] = ergebnis_fuer(vor, g)
                if punkte:
                    top_punkte = punkte
                    top_id = top_id_von(punkte[0]) if neu else frei
                    marker, fragesteller = None, None  # Marker aus dem Vor-TOP (20160:6939 Kretz als Kurzintervention) verfallen
                    rueck = None
                    block, j = aufruf_block(zeilen, zeile, t[m.end():].lstrip(":. "))
                    blocktext = " ".join(x for _, x in block)
                    top_typ = top_typ_von((frei or "") + " " + blocktext)
                    cur["_text"].append(text_ohne_zwischenruf(t[:m.end()], klammer))
                    cur["zeile_ende"] = zeile
                    items = parse_gesetze(block, punkte, tagesordnung(nr))
                    aktiv = []
                    for it in items:
                        it.update(sitzung=nr, datum=datum, top_id=top_id, top_typ=top_typ, aufruf_zeile=zeile,
                                  verbunden=punkte, abgestimmt=False, abstimmung_zeile=None, ergebnis_text=None,
                                  beschlussart="offen", _ueberwiesen=False, _vertagt=False)
                    _abstimmung_im_block(items, block)
                    top_gesetze[top_id] = items
                    gesetze.extend(items)
                    i = j - 1
                    continue
            elif (m := WIEDER_RE.search(t)) and (punkte := expand_punkte(m["liste"])):
                tid = top_id_von(punkte[0])
                if tid in top_gesetze and tid != top_id:
                    # Rücksprung ("… komme zurück zu Zusatzpunkt 35", 20211:3435) nur für diesen Präsidiumsblock, wenn er bloß ein
                    # Abstimmungsergebnis verkündet; sonst liefen 7 Reden zu TOP 18 (Afghanistan) unter dem Zustrombegrenzungsgesetz
                    rueck = {"vorher": (top_id, top_typ, top_punkte), "abstimmung": False}
                    top_id, top_punkte = tid, punkte
                    top_typ = top_gesetze[tid][0]["top_typ"] if top_gesetze[tid] else "Debatte"
            if rueck and (ABSTIMMUNG_RE.search(t) or ERGEBNIS_RE.search(t) or NAMENTLICH_ERGEBNIS_RE.search(t)
                          or re.search(r"Debatte fort|Aussprache fort|Abstimmung", t)):
                rueck["abstimmung"] = True
            # "… in dritter Beratung angenommen. … kommen zu Zusatzpunkt 7. Abstimmung über … 20/10722" (20164:1039): Teil vor dem
            # Punktwechsel gehört zum alten, Teil danach zum neuen Gegenstand
            teile = [t]
            if (mw := WIEDER_RE.search(t)) and mw.start() > 0:
                teile = [t[:mw.start()], t[mw.start():]]
            for tt in teile:
                if top_id and top_gesetze.get(top_id):
                    ziel0 = top_gesetze[top_id]
                    genannt = {"20/" + d for d in DRS_RE.findall(tt)}
                    pk = set(expand_punkte(tt)) if len(ziel0) > 1 else set()
                    if (h := [g for g in ziel0 if g["drucksache"] in genannt or set(g["drucksachen_alle"]) & genannt or g["punkt"] in pk]):
                        aktiv = h
                if top_id and (ma := ABSTIMMUNG_RE.search(tt)):
                    treffer = _ziel_items(top_gesetze.get(top_id, []), tt, aktiv)
                    for g in treffer:
                        if not g["abgestimmt"]:
                            g["abgestimmt"], g["abstimmung_zeile"] = True, zeile
                    if treffer and not aktiv:
                        aktiv = treffer
                if NAMENTLICH_ERGEBNIS_RE.search(tt):
                    genannt = {"20/" + d for d in DRS_RE.findall(tt)}
                    ziel = [g for g in gesetze if g["drucksache"] in genannt or set(g["drucksachen_alle"]) & genannt] \
                        or [g for g in top_gesetze.get(top_id, []) if g["abgestimmt"]]
                    if not ziel:
                        # Ergebnis ohne Drucksache, verkündet während eines anderen TOP (20211:3515 nach Rücksprung): jüngste
                        # noch ergebnislose namentliche Abstimmung der Sitzung passenden Typs
                        offen = [g for g in gesetze if g["abgestimmt"] and not g["ergebnis_text"]]
                        ziel = sorted(offen, key=lambda g: g["abstimmung_zeile"])[-1:]
                    namentlich = (ziel, 12) if ziel else None
                ziel = _ziel_items(top_gesetze[top_id], tt, aktiv) if top_id and top_gesetze.get(top_id) else []
                if namentlich:
                    ziel += [g for g in namentlich[0] if g not in ziel]
                    namentlich = (namentlich[0], namentlich[1] - 1) if namentlich[1] > 1 else None
                if UEBERWEISUNG_RE.search(tt) and SAMMEL_UEBERWEISUNG_RE.search(tt) and top_gesetze.get(top_id):
                    for g in top_gesetze[top_id]:
                        if not g["abgestimmt"]:
                            g["_ueberwiesen"] = True
                if ziel:
                    me = ERGEBNIS_RE.search(strip_zwischenruf(tt))
                    ueberw = UEBERWEISUNG_RE.search(tt)
                    genannt = {"20/" + d for d in DRS_RE.findall(tt)}
                    for g in ziel:
                        if ueberw and (not genannt or g["drucksache"] in genannt or set(g["drucksachen_alle"]) & genannt):
                            g["_ueberwiesen"] = True
                        if VERTAGT_RE.search(tt):
                            g["_vertagt"] = True
                        if not (g["abgestimmt"] or (namentlich and g in namentlich[0])):
                            continue
                        # ergebnis_fuer prüft den Typ: 20211 ZP 3 (FDP-Antrag) erbte sonst "Der Gesetzentwurf ist damit
                        # angenommen" von ZP 26. Erstes Endergebnis bleibt, nur ein Ergebnis "in zweiter Beratung" wird
                        # ersetzt; vorher gewann das letzte, und in Sammelaufrufen trugen alle Sammelübersichten das
                        # Ergebnis der letzten (20166 TOP 27)
                        satz = ergebnis_fuer(strip_zwischenruf(tt), g) if me else None
                        if satz and (not g["ergebnis_text"] or ZWEITE_BERATUNG in g["ergebnis_text"]):
                            g["ergebnis_text"] = satz
        if cur:
            cur["_text"].append(text_ohne_zwischenruf(t, klammer))
            cur["zeile_ende"] = zeile
        i += 1
    # Derselbe Gegenstand wird in 20147 zweimal aufgerufen (Ankündigung Z. 4169 ohne Drucksachen -> Inhaltsverzeichnis, Block Z. 4253):
    # Nachschlag-Zeilen weichen einer Aufruf-Zeile derselben Drucksache, Abstimmungsfelder werden zwischen Doppeln abgeglichen
    _nachtrag_namentlich(gesetze, zeilen)
    nach_drs: dict[str, list[dict]] = {}
    for g in gesetze:
        nach_drs.setdefault(g["drucksache"], []).append(g)
    for doppel in nach_drs.values():
        # Abgleich vor dem Entfernen der Nachschlag-Zeile: 20164 ZP 7 (20/10722) trug Abstimmung und Ergebnis nur auf dem Doppel
        quelle = next((g for g in doppel if g["ergebnis_text"]), None) or next((g for g in doppel if g["abgestimmt"]), None)
        if quelle and len(doppel) > 1:
            for g in doppel:
                if g is not quelle and (not g["abgestimmt"] or (not g["ergebnis_text"] and quelle["ergebnis_text"])):
                    g.update(abgestimmt=True, abstimmung_zeile=g["abstimmung_zeile"] or quelle["abstimmung_zeile"],
                             ergebnis_text=g["ergebnis_text"] or quelle["ergebnis_text"])
    aufruf_drs = {g["drucksache"] for g in gesetze if g["quelle_drucksache"] == "aufruf"}
    gesetze[:] = [g for g in gesetze if not (g["quelle_drucksache"] == "tagesordnung" and g["drucksache"] in aufruf_drs)]
    for tid, items in top_gesetze.items():
        items[:] = [g for g in items if g in gesetze]
    for g in gesetze:
        if g["ergebnis_text"] and ZURUECK_RE.search(g["ergebnis_text"]):
            g["beschlussart"] = "zurückverwiesen"
        elif g["ergebnis_text"]:
            g["beschlussart"] = "abgelehnt" if "abgelehnt" in g["ergebnis_text"] else "angenommen"
        elif g.pop("_ueberwiesen"):
            g["beschlussart"] = "überwiesen"
        elif g.pop("_vertagt"):
            g["beschlussart"] = "vertagt"
        g.pop("_ueberwiesen", None)
        g.pop("_vertagt", None)

    beitraege += parse_anlagen(nr, datum, top_gesetze)
    beginn = next((m for z in zeilen[:5] if (m := BEGIN_RE.match(z))), None)
    schluss = next((m for z in reversed(zeilen) if (m := END_RE.match(z))), None)
    meta = {"sitzung": nr, "datum": datum, "quelle_datum": quelle_datum,
            "beginn": f"{int(beginn.group(1)):02d}:{beginn.group(2)}" if beginn else None,
            "schluss": f"{int(schluss.group(1)):02d}:{schluss.group(2)}" if schluss else None,
            "n_zeilen": sum(1 for z in zeilen if z.strip()), "woerter": sum(zaehle_woerter(z) for z in zeilen)}
    for b in beitraege:
        b.setdefault("anlage", None)
        b.setdefault("anlage_titel", None)
        text = " ".join(x for x in b.pop("_text") if x).strip()
        b["text"] = text
        b["woerter"] = zaehle_woerter(text)
        if b["redetyp"] == "praesidium":
            res = {"partei": None, "fraktion": None, "gruppe": None, "quelle": "praesidium"}
        else:
            res = _lookup(b["name"], datum, b["fraktion_pdf"], b["ort"])
        # ist_mdb: Treffer in den MdB-Stammdaten (auch "kopf"/"mehrdeutig"); Restliste = Regierung ohne Mandat, Länder, Gäste;
        # Präsidium sind MdB
        b.update(partei=res["partei"], fraktion=res["fraktion"], gruppe=res["gruppe"], quelle_partei=res["quelle"],
                 phase=phase(nr), rolle_regierung=rolle_regierung(nr, res["fraktion"] or res["partei"]),
                 ist_mdb=res["quelle"] in ("stammdaten", "stammdaten+gruppe_vorgriff", "kopf", "nur_kopf", "mehrdeutig", "praesidium"))
    return {"beitraege": beitraege, "gesetze": gesetze, "sitzung": meta}


def _nachtrag_namentlich(gesetze: list[dict], zeilen: list[str]) -> None:
    """Ergebnis einer namentlichen Abstimmung, das erst in einem späteren TOP verkündet wird (20211: Abstimmung Z. 3235,
    Verkündung Z. 3515–3519 während TOP 18), nachträglich dem abgestimmten Gegenstand zuordnen."""
    for g in gesetze:
        if not g["abgestimmt"] or g["ergebnis_text"] or not g["abstimmung_zeile"]:
            continue
        for i in range(g["abstimmung_zeile"], len(zeilen)):
            t = zeilen[i]
            if not NAMENTLICH_ERGEBNIS_RE.search(t):
                continue
            genannt = {"20/" + d for d in DRS_RE.findall(t)}
            if genannt and not (g["drucksache"] in genannt or set(g["drucksachen_alle"]) & genannt):
                continue
            folge = [z for z in zeilen[i:i + 16] if z.strip()][:8]
            for z in folge:
                if (satz := ergebnis_fuer(strip_zwischenruf(z), g)):
                    g["ergebnis_text"] = satz
                    break
            break


def merge_reden(beitraege: list[dict]) -> list[dict]:
    """Rede-Regel: plenarrede-Beiträge derselben Person im selben TOP, nur durch praesidium/zwischenfrage getrennt,
    = eine Rede (z. B. Fortsetzung nach abgelehnter Zwischenfrage); so zählt auch das XML (Sitzung 160: 193 <rede> bei
    277 Köpfen)."""
    reden: list[dict] = []
    offen = None  # letzte plenarrede, an die angehängt werden darf
    for b in beitraege:
        b["rede_id"] = None
        if b["redetyp"] == "praesidium":
            continue
        if b.get("anlage") is not None:
            b["rede_id"] = f"{b['sitzung']}-A{b['zeile_start']:04d}"
            reden.append(_neue_rede(b))
            continue
        if b["redetyp"] == "zwischenfrage":
            b["rede_id"] = f"{b['sitzung']}-{b['zeile_start']}"
            reden.append(_neue_rede(b))
            continue
        if offen and offen["name"] == b["name"] and offen["top_id"] == b["top_id"] \
                and b["redetyp"] in ("plenarrede", "kurzintervention", "erwiderung"):
            offen["text"] = (offen["text"] + " " + b["text"]).strip()
            offen["woerter"] += b["woerter"]
            offen["zeile_ende"] = b["zeile_ende"]
            offen["n_beitraege"] += 1
            b["rede_id"] = offen["rede_id"]
            continue
        b["rede_id"] = f"{b['sitzung']}-{b['zeile_start']}"
        r = _neue_rede(b)
        reden.append(r)
        # Erwiderung wird vom Präsidium unterbrochen und läuft als plenarrede weiter (20160:593/617 Schmidt)
        offen = r if b["redetyp"] in ("plenarrede", "kurzintervention", "erwiderung") else None
    return reden


def _neue_rede(b: dict) -> dict:
    keys = ("rede_id", "sitzung", "datum", "phase", "name", "partei", "fraktion", "gruppe", "quelle_partei", "rolle",
            "rolle_regierung", "ist_mdb", "top_id", "top_typ", "redetyp", "anlage", "zeile_start", "zeile_ende", "text", "woerter")
    return {**{k: b[k] for k in keys}, "n_beitraege": 1}


def beteiligte(reden: list[dict], gesetze: list[dict]) -> list[dict]:
    """Jede Rede eines Tagesordnungspunkts zählt bei jeder dort aufgerufenen Vorlage. gegenstaende_im_top macht sichtbar,
    wann das eine gemeinsame Debatte mehrerer Vorlagen ist: Haushaltsgesetz 2024 und die Einzelpläne laufen in der
    Haushaltswoche unter einem TOP I, daher zählen dort alle Redner der Woche."""
    nach_top = {}
    for g in gesetze:
        nach_top.setdefault((g["sitzung"], g["top_id"]), []).append(g)
    out = []
    for r in reden:
        its = nach_top.get((r["sitzung"], r["top_id"]), [])
        n_vorlagen = len({g["drucksache"] for g in its if g["drucksache"]})
        for g in its:
            if g["drucksache"]:
                out.append({"rede_id": r["rede_id"], "sitzung": r["sitzung"], "drucksache": g["drucksache"],
                            "name": r["name"], "partei": r["partei"], "fraktion": r["fraktion"], "redetyp": r["redetyp"],
                            "gegenstaende_im_top": n_vorlagen})
    return out


def parse_all(sitzungen=None, log_path=None) -> dict[str, pd.DataFrame]:
    sitzungen = config.SITZUNGEN if sitzungen is None else sitzungen
    log_path = config.LOGS / "parse.log" if log_path is None else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    B, G, S = [], [], []
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("sitzung\tstatus\tbeitraege\treden\tgesetze\tohne_partei\tmeldung\n")
        for nr in sitzungen:
            try:
                res = parse_sitzung(nr)
                reden = merge_reden(res["beitraege"])
                ohne = sum(1 for b in res["beitraege"] if b["redetyp"] != "praesidium" and not b["partei"])
                n_pl = sum(1 for r in reden if r["redetyp"] == "plenarrede")
                status = "OK" if res["beitraege"] and n_pl else "WARN"
                row = (nr, status, len(res["beitraege"]), n_pl, len(res["gesetze"]), ohne, "")
                B += res["beitraege"]
                G += res["gesetze"]
                S.append(res["sitzung"])
            except Exception as exc:
                row = (nr, "ERROR", 0, 0, 0, 0, f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            f.write("\t".join(map(str, row)) + "\n")
            print(f"[parse] {row}", flush=True)
    beitraege = pd.DataFrame(B)
    reden = pd.DataFrame(merge_reden(B)) if B else pd.DataFrame()
    gesetze = pd.DataFrame(G)
    if not gesetze.empty:
        sitz = gesetze.groupby("drucksache")["sitzung"].agg(lambda s: sorted(set(s)))
        gesetze["sitzungen"] = gesetze["drucksache"].map(sitz)
    bet = pd.DataFrame(beteiligte(reden.to_dict("records"), G)) if B else pd.DataFrame()
    return {"beitraege": beitraege, "reden": reden, "gesetze": gesetze, "gesetz_beteiligte": bet,
            "sitzungen": pd.DataFrame(S), "gesetze_einheit": gesetze_einheit(gesetze, bet)}


def gesetze_einheit(gesetze: pd.DataFrame, bet: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je Gesetzentwurf-Drucksache über alle Lesungen (20/12804 ist genau ein Gesetz, Sitzungen 185 und 211)."""
    if gesetze.empty:
        return pd.DataFrame()
    ge = gesetze[gesetze.ist_gesetz].sort_values(["drucksache", "sitzung"])
    bet = bet[bet.redetyp == "plenarrede"] if not bet.empty else bet
    rows = []
    for d, grp in ge.groupby("drucksache"):
        arten = list(grp.beschlussart)
        # G: Ergebnis der letzten Sitzung mit Beschluss (20/10722: 160 überwiesen, 164 abgelehnt -> abgelehnt)
        final = next((a for a in reversed(arten) if a in ("angenommen", "abgelehnt", "zurückverwiesen")), None) \
            or next((a for a in reversed(arten) if a != "offen"), "offen")
        b = bet[bet.drucksache == d] if not bet.empty else bet
        kurz = next(iter(grp.kurztitel.dropna()), None) or next(iter(grp.kurztitel_ende.dropna()), None)
        rows.append({"drucksache": d, "titel": max(grp.titel, key=len), "kurztitel": kurz,
                     "gesetz_titel": max(grp.gesetz_titel, key=len),
                     "sitzungen": sorted(set(grp.sitzung)), "beratungsstufen": list(grp.beratungsstufe),
                     "beschlussempfehlungen": [x for x in grp.beschlussempfehlung_drucksache if isinstance(x, str)],
                     "beschlussart_final": final, "n_reden": int(len(b)), "n_redner": int(b.name.nunique()) if len(b) else 0,
                     # größte Zahl von Vorlagen, mit denen der Entwurf in einem TOP gemeinsam beraten wurde (1 = eigene Debatte)
                     "gegenstaende_im_top_max": int(b.gegenstaende_im_top.max()) if len(b) else 0,
                     "fraktionen": sorted(set(x for x in (b.fraktion.fillna(b.partei) if len(b) else []) if isinstance(x, str)))})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO)
    tabs = parse_all()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in tabs.items():
        df = df.copy()
        for c in df.columns:
            if df[c].map(lambda v: isinstance(v, list)).any():
                df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
        if "datum" in df:
            df["datum"] = df["datum"].astype(str)
        # atomar (temp + os.replace): ein parallel laufender Leser (validate.py, metrics.py) sieht nie eine halb geschriebene Datei
        for suffix, writer in ((".parquet", df.to_parquet), (".csv", df.to_csv)):
            ziel = config.PROCESSED / f"{name}{suffix}"
            tmp = config.PROCESSED / f".{name}{suffix}.tmp"
            writer(tmp, index=False)
            os.replace(tmp, ziel)
        print(f"[parse] {name}: {len(df)} Zeilen")


if __name__ == "__main__":
    main()
