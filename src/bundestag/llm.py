"""LLM-Schritt mit lokalen Ollama-Modellen.

1. gesetze: Aufruf-Blöcke → Beratungsgegenstände (JSON-Schema aus Pydantic), Vergleich mit dem Regex-Parser
2. vergleich: 30 Blöcke mit qwen2.5:3b, qwen2.5:7b und deepseek-r1:7b (Beleg für die Toolwahl)
3. zusammenfassung: Top-10-Gesetze, Kernpunkt und Haltung je Fraktion aus Redeanfängen

Aufruf:
  python -m bundestag.llm --schritt gesetze [--modell qwen2.5:7b]   → data/llm/llm_gesetze.parquet, divergenz.csv
  python -m bundestag.llm --schritt vergleich [--modell A B …]      → data/llm/modellvergleich.csv
  python -m bundestag.llm --schritt zusammenfassung                 → data/llm/top10_zusammenfassungen.json/.csv
Jede Antwort liegt je (Modell, Prompt-Hash) in data/llm/cache/; ein Neulauf kostet nur die fehlenden Aufrufe.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import time
from typing import Annotated, Literal

import pandas as pd
import requests
from pydantic import BaseModel, StringConstraints, ValidationError, field_validator

from . import config
from . import parse as P  # nur gelesen (parse_sitzung → aufruf_block): Blöcke wie beim Regex-Parser, Vergleich 1:1

OUT = config.LLM_DIR          # alle LLM-Ausgaben an einem Ort, im Teillauf data/teillauf/<von>-<bis>/llm (config.py)
CACHE_DIR = config.LLM_CACHE  # je (Modell, Prompt-Hash) eine JSON-Datei: Gesetze-Neulauf ≈ 1,3 s (laufzeiten_teil.csv) statt 2711 s ohne Cache
LAUFZEITEN = config.PROCESSED / "laufzeiten_teil.csv"  # (schritt, sekunden): echte Laufzeiten, die ein Cache-Neulauf nicht mehr misst
SEED = 82                     # Projektseed: Ollama-Seed, Blockreihenfolge, Stichprobe Modellvergleich
MODELL = config.OLLAMA_MODEL  # qwen2.5:7b: größtes lokal vorhandenes Modell (4,7 GB gegen 1,9 GB für 3b)
VERGLEICH_MODELLE = ["qwen2.5:3b", "qwen2.5:7b", "deepseek-r1:7b"]  # 3b gegen 7b: Modellgröße; R1: Kosten des Reasoning
VERGLEICH_GOLD = (211, "Zusatzpunkt 35")  # Beispiel der Aufgabenstellung, immer in der Stichprobe
# 5 Blöcke trennten schon (7b 5/5 gleich, 3b 0/5, R1 1/5 bei 34,6 s/Block), trugen aber keine Quote: 30 Blöcke
# geben je Modell ein Intervall von etwa ±15 Prozentpunkten; R1 braucht dafür etwa 20 min
VERGLEICH_N = 30

# Messung an 20211 ZP 35: 1968 Prompt-Zeichen → 687 prompt_eval_count = 2,86 Zeichen je Token (qwen2.5,
# deutscher Protokolltext); 2,8 rundet ab, damit 600 Token nicht überschritten werden.
ZEICHEN_JE_TOKEN = 2.8
# 600 Token ≈ 1680 Zeichen: 476 von 516 gefilterten Blöcken bleiben ungekürzt (Median 447 Zeichen);
# gekürzt werden nur Sammelaufrufe mit vielen Unterpunkten (max. 4919 Zeichen), die vorn Titel und Drucksachen tragen.
# Die Schranke hält Systemprompt + Block + Antwort in num_ctx und die Laufzeit je Block berechenbar.
BLOCK_MAX_TOKEN = 600
# temperature 0 + Seed 82: gleiche Eingabe → gleiche Antwort, Voraussetzung für Reproduzierbarkeit und Cache.
# num_ctx 4096: Systemprompt mit Beispiel ≈ 900 (gemessen: 1087 Token mit ZP 35) + Block ≤ 600 + Antwort
# ≤ 1200, bei der Wiederholung zusätzlich erste Antwort + Korrektur (≈ 3950). num_predict 1200: Obergrenze gegen
# Endlosschleifen; ein Gegenstand kostet ≈ 140 Token (ZP 35: 143).
OPT_GESETZE = {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": 1200}
# deepseek-r1 schreibt vor der Antwort Reasoning; mit 1200 Token würde die JSON-Antwort abgeschnitten.
# num_predict 4096 ≈ 200 s je Versuch bei ≈ 20 Token/s (7B, gemessen); num_ctx 6144 = Prompt ≈ 1100 + 4096 + Rest.
OPT_REASONING = {"temperature": 0, "seed": SEED, "num_ctx": 6144, "num_predict": 4096}
# Zusammenfassung: Kontext ≤ 2500 Token + Anweisung ≈ 400 + Antwort ≤ 700 → 4096.
OPT_ZUSAMMEN = {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": 700}
TIMEOUT_S = 240               # 1200 Token Antwort bei ≈ 20 Token/s ≈ 60 s, Faktor 4 Reserve; danach „ungültig“
SCHNITT_GESETZE_S = 30 * 60   # 30 min ≈ 158 Blöcke zu je 11,4 s; Fertiges bleibt für den Neulauf im Cache
SCHNITT_ZUSAMMEN_S = 15 * 60  # harter Schnitt nach 15 min; die 10 Zusammenfassungen brauchten 320 s
KONTEXT_MAX_TOKEN = 2500      # passt mit SYSTEM_ZUSAMMEN (549 Zeichen ≈ 200 Token), Titel und Antwort (≤ 700) in num_ctx 4096
# Reden < 50 Wörter bekommen kein Sentiment (sentiment.MIN_WOERTER); hier ebenso, sonst belegt Freis GO-Antrag auf
# Unterbrechung (20211, ≈ 45 Wörter) den einzigen CDU/CSU-Platz im Kontext zum Zustrombegrenzungsgesetz
MIN_WOERTER = 50
REDE_WOERTER = 120            # Redeanfang trägt die Position (These nach der Anrede)
# Anrede am Redeanfang trägt keine Position: 89,4 % der 6414 Plenarreden beginnen damit, Median 8 Wörter
ANREDE_RE = re.compile(r"^(?:(?:Sehr geehrte|Sehr verehrte|Liebe|Geehrte|Verehrte|Frau|Herr|Meine|Werte)[^!.?]{0,90}!\s*)+")
TITEL_GLEICH = 0.9            # SequenceMatcher-Quote; darunter fehlt z. B. der Klammer-Kurztitel (≈ 25 Zeichen)
TITEL_PAAR = 0.5              # ohne gemeinsame Drucksache erst ab halber Titelgleichheit paaren, sonst fremde Paare
FILTER_RE = re.compile(r"Entwurfs? eines|Drucksache 20/")  # wörtlich „Entwurf eines“ träfe 1 Block, der Genitiv 221
ENTWURF_RE = re.compile(r"Entwurfs? eines")  # Gesetzentwurf-Blöcke (224 von 516) laufen zuerst, siehe reihenfolge()
DRS_FORM = r"^20/\d{3,5}$"    # Form wie parse.DRS_RE ("Drucksache 20/N" 1569, Listen 434 Fälle)

Typ = Literal["gesetzentwurf", "beschlussempfehlung", "antrag", "bericht", "sonstiges"]
Stufe = Literal["erste", "zweite", "zweite und dritte", "dritte", "beratung", "unbekannt"]
Haltung = Literal["zustimmend", "ablehnend", "gemischt", "unklar"]


class Gegenstand(BaseModel):
    titel: str
    kurztitel: str | None
    drucksachen: list[Annotated[str, StringConstraints(pattern=DRS_FORM)]]
    typ: Typ
    beratungsstufe: Stufe


class Extraktion(BaseModel):
    gegenstaende: list[Gegenstand]


def _max_woerter(v: str, n: int) -> str:
    if len(v.split()) > n:
        raise ValueError(f"höchstens {n} Wörter, hat {len(v.split())}")
    return v


class Position(BaseModel):
    fraktion: str
    haltung: Haltung
    begruendung: str

    @field_validator("begruendung")
    @classmethod
    def _kurz(cls, v: str) -> str:
        return _max_woerter(v, 20)  # ≤ 20 Wörter wie in SYSTEM_ZUSAMMEN; ein Verstoß löst die Korrektur in frage() aus


class Zusammenfassung(BaseModel):
    kernpunkt: str
    positionen: list[Position]

    @field_validator("kernpunkt")
    @classmethod
    def _kurz(cls, v: str) -> str:
        return _max_woerter(v, 25)  # ≤ 25 Wörter wie in SYSTEM_ZUSAMMEN; ein Verstoß löst die Korrektur in frage() aus


SYSTEM_GESETZE = """Du extrahierst Beratungsgegenstände aus dem Aufruf eines Tagesordnungspunkts im Plenarprotokoll des Deutschen Bundestages (20. Wahlperiode). Antworte nur mit JSON nach dem Schema.
Ein Gegenstand beginnt mit „Erste/Zweite/Zweite und dritte/Dritte Beratung des …“, „Beratung des Antrags/der Beschlussempfehlung/des Berichts/der Unterrichtung …“, „Befragung der Bundesregierung“, „Fragestunde“, „Aktuelle Stunde“, „Vereinbarte Debatte“, „Wahl …“ oder „hier: Einzelplan …“. Aufzählungen „a)“, „b)“ trennen Gegenstände.
Felder je Gegenstand:
- titel: Wortlaut vom Beginn des Gegenstands bis vor die erste Drucksache, unverändert übernommen.
- kurztitel: Klammerausdruck mit „gesetz“ im Titel, z. B. „Zustrombegrenzungsgesetz“, sonst null.
- drucksachen: alle Nummern des Gegenstands im Format „20/12345“, auch die der Beschlussempfehlung, ohne Zusätze wie „Buchstabe a“.
- typ: gesetzentwurf (Entwurf eines Gesetzes, auch „hier: Einzelplan“ des Haushaltsgesetzes); beschlussempfehlung (nur wenn der Gegenstand selbst mit „Beratung der Beschlussempfehlung“ beginnt); antrag; bericht (Bericht oder Unterrichtung); sonstiges (Befragung, Fragestunde, Aktuelle Stunde, Debatte, Wahl).
- beratungsstufe: erste, zweite, zweite und dritte oder dritte laut Einleitung des Gegenstands; beratung bei Anträgen, Beschlussempfehlungen, Berichten und Sonstigem; unbekannt, wenn nicht erkennbar.
Steht „Beschlussempfehlung und Bericht des Ausschusses … Drucksache …“ ohne vorangestelltes „Beratung der“ innerhalb eines Gesetzentwurfs, ist das kein eigener Gegenstand: die Drucksache gehört zum Gesetzentwurf.
Beispiel (erfunden). Aufruf: „a) Zweite und dritte Beratung des von der Bundesregierung eingebrachten Entwurfs eines Gesetzes zur Änderung des Musterrechts (Mustergesetz) Drucksache 20/1111 Beschlussempfehlung und Bericht des Rechtsausschusses (6. Ausschuss) Drucksache 20/2222 b) Beratung der Beschlussempfehlung und des Berichts des Ausschusses für Arbeit und Soziales (11. Ausschuss) zu dem Antrag der Fraktion der AfD Musterantrag Drucksachen 20/3333, 20/4444“
Antwort: {"gegenstaende": [{"titel": "Zweite und dritte Beratung des von der Bundesregierung eingebrachten Entwurfs eines Gesetzes zur Änderung des Musterrechts (Mustergesetz)", "kurztitel": "Mustergesetz", "drucksachen": ["20/1111", "20/2222"], "typ": "gesetzentwurf", "beratungsstufe": "zweite und dritte"}, {"titel": "Beratung der Beschlussempfehlung und des Berichts des Ausschusses für Arbeit und Soziales (11. Ausschuss) zu dem Antrag der Fraktion der AfD Musterantrag", "kurztitel": null, "drucksachen": ["20/3333", "20/4444"], "typ": "beschlussempfehlung", "beratungsstufe": "beratung"}]}"""

SYSTEM_ZUSAMMEN = """Du fasst die Bundestagsdebatte zu einem Gesetzentwurf zusammen. Grundlage sind nur die Redeauszüge (je Rede die ersten Wörter), nach Fraktion gruppiert. Antworte nur mit JSON nach dem Schema.
- kernpunkt: worum es im Gesetzentwurf geht, höchstens 25 Wörter.
- positionen: je Fraktion aus den Auszügen genau ein Eintrag; fraktion wie in der Überschrift des Auszugs; haltung zum Gesetzentwurf: zustimmend, ablehnend, gemischt oder unklar (unklar, wenn die Auszüge keine Haltung erkennen lassen); begruendung: höchstens 20 Wörter, nur aus den Auszügen."""


# Cache vor dem HTTP-Aufruf: mit --nur-cache läuft der Schritt ohne Ollama aus data/llm/cache
def _modell_dir(modell: str) -> str:
    return re.sub(r"[^\w.-]", "_", modell)


def chat(messages: list[dict], schema: dict, modell: str, optionen: dict, think: bool | None = None,
         nur_cache: bool = False, timeout: float = TIMEOUT_S) -> dict | None:
    """Ein /api/chat-Aufruf mit Cache je (Modell, Prompt-Hash). None bei Netzfehler oder nur_cache ohne Treffer."""
    anfrage = {"model": modell, "messages": messages, "format": schema, "options": optionen, "stream": False}
    if think is not None:
        anfrage["think"] = think  # nur deepseek-r1 im Modellvergleich: 613–2572 Zeichen Reasoning je Block (modellvergleich.csv)
    schluessel = hashlib.sha256(json.dumps(anfrage, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    pfad = CACHE_DIR / _modell_dir(modell) / f"{schluessel}.json"
    if pfad.exists():
        return json.loads(pfad.read_text(encoding="utf-8")) | {"cache": True}
    if nur_cache:
        return None
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=anfrage, timeout=timeout)
        r.raise_for_status()
        d = r.json()
    except (requests.RequestException, ValueError) as e:
        return {"fehler": f"{type(e).__name__}: {str(e)[:200]}", "cache": False, "wand_s": time.perf_counter() - t0}
    msg = d.get("message", {})
    eintrag = {"modell": modell, "inhalt": msg.get("content", ""), "thinking": msg.get("thinking"),
               "eval_count": d.get("eval_count"), "prompt_eval_count": d.get("prompt_eval_count"),
               "total_duration": d.get("total_duration"), "load_duration": d.get("load_duration"),
               "wand_s": time.perf_counter() - t0, "prompt_zeichen": sum(len(m["content"]) for m in messages)}
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_suffix(".tmp")
    tmp.write_text(json.dumps(eintrag, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pfad)
    return eintrag | {"cache": False}


def _sekunden(a: dict) -> float:
    """Rechenzeit ohne Modell-Laden (load_duration), damit der erste Aufruf nach Modellwechsel vergleichbar bleibt."""
    if a.get("total_duration"):
        return (a["total_duration"] - (a.get("load_duration") or 0)) / 1e9
    return a.get("wand_s", 0.0)


def frage(system: str, user: str, schema_cls: type[BaseModel], modell: str, optionen: dict,
          think: bool | None = None, nur_cache: bool = False, timeout: float = TIMEOUT_S) -> dict:
    """Aufruf + Pydantic-Validierung; bei Fehler genau eine Wiederholung mit der Fehlermeldung als Korrektur.

    Mit temperature 0 und festem Seed lieferte eine identische Wiederholung dieselbe Antwort; deshalb hängt die
    Wiederholung Antwort und Validierungsfehler an den Verlauf (anderer Prompt, eigener Cache-Eintrag)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    schema = schema_cls.model_json_schema()
    versuche, fehler = [], None
    for _ in range(2):
        a = chat(msgs, schema, modell, optionen, think=think, nur_cache=nur_cache, timeout=timeout)
        if a is None:
            return {"objekt": None, "fehler": "nicht im Cache", "versuche": versuche, "offen": True}
        versuche.append(a)
        if "fehler" in a:
            fehler = a["fehler"]
            break  # Netz-/Zeitfehler: eine Wiederholung kostete erneut bis zu 240 s (TIMEOUT_S)
        try:
            return {"objekt": schema_cls.model_validate_json(a["inhalt"]), "fehler": None, "versuche": versuche,
                    "offen": False}
        except ValidationError as e:
            fehler = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
            msgs = msgs + [{"role": "assistant", "content": a["inhalt"]},
                           {"role": "user", "content": f"Die Antwort verletzt das Schema ({fehler}). "
                                                       "Antworte erneut, nur mit gültigem JSON nach dem Schema."}]
    return {"objekt": None, "fehler": fehler, "versuche": versuche, "offen": False}


def _messwerte(res: dict) -> dict:
    v = res["versuche"]
    return {"versuche": len(v), "sekunden": round(sum(_sekunden(a) for a in v), 2),
            "eval_count": sum(a.get("eval_count") or 0 for a in v),
            "prompt_eval_count": sum(a.get("prompt_eval_count") or 0 for a in v),
            "thinking_zeichen": sum(len(a.get("thinking") or "") for a in v),
            "cache": bool(v) and all(a.get("cache") for a in v)}


def laufzeit_anhaengen(schritt: str, sekunden: float) -> None:
    neu = not LAUFZEITEN.exists()
    with LAUFZEITEN.open("a", encoding="utf-8") as f:
        if neu:
            f.write("schritt,sekunden\n")
        f.write(f"{schritt},{sekunden:.1f}\n")


# Aufruf-Blöcke: gesetze.parquet speichert nur Titel und Drucksachen, den Blocktext liest blocktext() aus data/txt
def blocktext(zeilen: list[str], zeile: int) -> str:
    """Aufrufblock wie in parse.parse_sitzung vor aufruf_block(): Erstaufruf, Wiederaufnahme oder freier Aufruf."""
    t = zeilen[zeile - 1]
    m = P.AUFRUF_RE.search(t)
    if not (m and not P.ABGESETZT_RE.search(t[m.end():m.end() + 40])):
        m = next((a for a in (P.WIEDER_RE.search(t), P.TOP_ENDE_RE.search(t))
                  if a and P.expand_punkte(a["liste"])), None) or P.FREI_RE.search(t)
    if m is None:
        return ""
    block, _ = P.aufruf_block(zeilen, zeile, t[m.end():].lstrip(":. "))
    return " ".join(x for _, x in block)


def _liste(x) -> list:
    if isinstance(x, str):
        return json.loads(x)
    return list(x) if x is not None else []


def lade_bloecke() -> list[dict]:
    """Alle Aufruf-Blöcke aus gesetze.parquet mit rekonstruiertem Text und den Regex-Gegenständen."""
    g = pd.read_parquet(config.PROCESSED / "gesetze.parquet")
    texte: dict[int, list[str]] = {}
    bloecke = []
    for (nr, zeile), grp in g.groupby(["sitzung", "aufruf_zeile"], sort=True):
        # quelle_drucksache = tagesordnung (198 Gegenstände): aus dem Inhaltsverzeichnis
        # nachgeschlagen, steht nicht im Aufrufblock, für das LLM unsichtbar → nur „aufruf“-Gegenstände vergleichen
        toc = grp.quelle_drucksache.eq("tagesordnung") if "quelle_drucksache" in grp else pd.Series(False, grp.index)
        if nr not in texte:
            texte[nr] = (config.TXT_DIR / f"20{nr}.txt").read_text(encoding="utf-8").splitlines()
        text = blocktext(texte[nr], int(zeile))
        regex = [{"titel": r.titel, "kurztitel": r.kurztitel if isinstance(r.kurztitel, str) else None,
                  "drucksachen": _liste(r.drucksachen_alle), "typ": r.typ_fein, "beratungsstufe": r.beratungsstufe,
                  "punkt": r.punkt} for r in grp[~toc].itertuples()]
        im_text = set("20/" + d for d in P.DRS_RE.findall(text))
        ok = all(set(r["drucksachen"]) <= im_text for r in regex)
        gesetz = grp.ist_gesetz.any() if "ist_gesetz" in grp else grp.typ_fein.eq("gesetzentwurf").any()
        bloecke.append({"sitzung": int(nr), "aufruf_zeile": int(zeile), "top_id": grp.top_id.iloc[0],
                        "text": text, "regex": regex, "rekonstruktion_ok": ok, "n_regex_toc": int(toc.sum()),
                        "entwurf": bool(ENTWURF_RE.search(text)), "filter": bool(FILTER_RE.search(text)),
                        "gesetz": bool(gesetz)})
    return bloecke


def reihenfolge(bloecke: list[dict], auswahl: str = "filter") -> list[dict]:
    """Gefilterte Blöcke, Gesetzentwürfe zuerst, je Gruppe gemischt (Seed 82): ein Schnitt nach 30 min lässt dann
    eine Zufallsstichprobe statt nur der frühen Sitzungen zurück.

    auswahl = gesetzentwurf: alle Blöcke mit ≥ 1 Gegenstand ist_gesetz (249 von 610 Blöcken,
    159 davon schon im 30-min-Lauf); alle: jeder Block, für den Cache-Abgleich, damit jede vorhandene Antwort zählt, gleich wie gewählt."""
    rng = random.Random(SEED)
    if auswahl == "alle":
        return list(bloecke)
    if auswahl == "gesetzentwurf":
        a = [b for b in bloecke if b["gesetz"]]
        rng.shuffle(a)
        return a
    a = [b for b in bloecke if b["filter"] and b["entwurf"]]
    rest = [b for b in bloecke if b["filter"] and not b["entwurf"]]
    rng.shuffle(a)
    rng.shuffle(rest)
    return a + rest


def kuerze(text: str, max_token: int = BLOCK_MAX_TOKEN) -> tuple[str, bool]:
    grenze = int(max_token * ZEICHEN_JE_TOKEN)
    if len(text) <= grenze:
        return text, False
    return text[:grenze].rsplit(" ", 1)[0] + " …", True


def nutzer_gesetze(b: dict, text: str) -> str:
    return f"Aufruf {b['top_id']} (Sitzung {b['sitzung']}):\n{text}"


def extrahiere(b: dict, modell: str, optionen: dict = OPT_GESETZE, think: bool | None = None,
               timeout: float = TIMEOUT_S) -> dict:
    text, gekuerzt = kuerze(b["text"])
    res = frage(SYSTEM_GESETZE, nutzer_gesetze(b, text), Extraktion, modell, optionen, think=think, timeout=timeout)
    sichtbar = set("20/" + d for d in P.DRS_RE.findall(text))  # 40 Blöcke gekürzt: Vergleich nur über Gesehenes
    llm = [g.model_dump() for g in res["objekt"].gegenstaende] if res["objekt"] else None
    return {"res": res, "llm": llm, "gekuerzt": gekuerzt, "sichtbar": sichtbar}


# Vergleich LLM gegen Regex: das LLM prüft den Parser, so fiel der Ordinaltitel-Fehler auf
def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .,:;–-")


def _kurz_teile(s: str | None) -> set[str]:
    # Varianten statt Abweichung: „Haushaltsgesetz 2025 – HG 2025“ gegen „Haushaltsgesetz 2025“ (20186 TOP 1),
    # „… – GVSG“ gegen „GVSG“, „Umsetzungsund“ gegen „Umsetzungs- und“ (NIS-2); je Teil, nur Buchstaben/Ziffern
    return {t for t in (re.sub(r"\W", "", x.lower()) for x in re.split(r"\s+[–-]\s+", s or "")) if t}


def kurz_gleich(a: str | None, b: str | None) -> bool:
    ta, tb = _kurz_teile(a), _kurz_teile(b)
    return ta == tb or bool(ta & tb)


def titel_quote(a: str | None, b: str | None) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _bester(x: dict, kandidaten: list[dict]) -> tuple[dict | None, float]:
    """Partner mit den meisten gemeinsamen Drucksachen, bei Gleichstand bzw. ohne Drucksache nach Titelquote."""
    best, best_key = None, (0, TITEL_PAAR - 1e-9)
    for k in kandidaten:
        key = (len(set(x["drucksachen"]) & set(k["drucksachen"])), titel_quote(x["titel"], k["titel"]))
        if (key[0] > 0 or key[1] >= TITEL_PAAR) and key > best_key:
            best, best_key = k, key
    return best, best_key[1]


def vergleiche(regex: list[dict], llm: list[dict] | None, sichtbar: set[str] | None = None) -> dict:
    """Divergenz je Aufruf-Block. Paare beidseitig (jeder Regex- und jeder LLM-Gegenstand bekommt seinen besten
    Partner), damit zusammengelegte oder geteilte Gegenstände als Typabweichung sichtbar werden.
    Klasse nach fester Priorität: ungueltig > drucksache_fehlt_llm > drucksache_fehlt_regex > typ_abweichend >
    stufe_abweichend > nur_titelvariante (Titelquote < 0,9 oder anderer Kurztitel) > gleich."""
    if sichtbar is not None:  # gekürzter Block (40 von 516): nur vergleichen, was das LLM sehen konnte
        regex = [r for r in regex if not r["drucksachen"] or r["drucksachen"][0] in sichtbar]
    drs_r = set().union(*(set(r["drucksachen"]) for r in regex)) if regex else set()
    if sichtbar is not None:
        drs_r &= sichtbar
    out = {"n_regex": len(regex), "drs_regex": sorted(drs_r)}
    if llm is None:
        return out | {"klasse": "ungueltig", "n_llm": None, "drs_llm": None, "fehlt_llm": None,
                      "fehlt_regex": None, "drs_gleich": None, "titel_quote": None, "kurztitel_gleich": None,
                      "typ_gleich": None, "stufe_gleich": None, "typ_diff": None, "stufe_diff": None}
    drs_l = set().union(*(set(x["drucksachen"]) for x in llm)) if llm else set()
    paare = []
    for r in regex:
        k, q = _bester(r, llm)
        if k is not None:
            paare.append((r, k, q))
    n_r = len(paare)
    gepaart = {id(k) for _, k, _ in paare}
    for x in llm:
        if id(x) not in gepaart:
            r, q = _bester(x, regex)
            if r is not None:
                paare.append((r, x, q))
    n_ohne = len(regex) + len(llm) - len({id(r) for r, _, _ in paare}) - len({id(k) for _, k, _ in paare})
    typ_gleich = all(r["typ"] == k["typ"] for r, k, _ in paare) and n_ohne == 0
    # „Regex→LLM“ je abweichendem Paar: zeigt, wer irrt (20188 TOP 29: Regex „sonstiges“ bei „Entwurfs eines Dritten
    # Gesetzes“), ungepaarte Gegenstände als „–“
    typ_diff = sorted({f"{r['typ']}→{k['typ']}" for r, k, _ in paare if r["typ"] != k["typ"]}
                      | {f"{r['typ']}→–" for r in regex if id(r) not in {id(x) for x, _, _ in paare}}
                      | {f"–→{k['typ']}" for k in llm if id(k) not in {id(y) for _, y, _ in paare}})
    stufe_diff = sorted({f"{r['beratungsstufe']}→{k['beratungsstufe']}" for r, k, _ in paare
                         if r["beratungsstufe"] != k["beratungsstufe"]})
    stufe_gleich = all(r["beratungsstufe"] == k["beratungsstufe"] for r, k, _ in paare)
    kurz_ok = all(kurz_gleich(r["kurztitel"], k["kurztitel"]) for r, k, _ in paare)
    # Titelquote nur aus Sicht der Regex-Gegenstände: abgespaltene LLM-Gegenstände (20211 ZP 35, qwen2.5:3b) zählen
    # schon als Typabweichung
    quote = min((q for _, _, q in paare[:n_r]), default=0.0 if regex else 1.0)
    if drs_r - drs_l:
        klasse = "drucksache_fehlt_llm"
    elif drs_l - drs_r:
        klasse = "drucksache_fehlt_regex"
    elif not typ_gleich:
        klasse = "typ_abweichend"
    elif not stufe_gleich:
        klasse = "stufe_abweichend"
    elif quote < TITEL_GLEICH or not kurz_ok:
        klasse = "nur_titelvariante"
    else:
        klasse = "gleich"
    return out | {"klasse": klasse, "n_llm": len(llm), "drs_llm": sorted(drs_l), "fehlt_llm": sorted(drs_r - drs_l),
                  "fehlt_regex": sorted(drs_l - drs_r), "drs_gleich": drs_r == drs_l, "titel_quote": round(quote, 3),
                  "kurztitel_gleich": kurz_ok, "typ_gleich": typ_gleich, "stufe_gleich": stufe_gleich,
                  "typ_diff": typ_diff, "stufe_diff": stufe_diff}


# Schritt 1, Gesetze: 10,9 s je Block (Median 7,1 s), daher Zeitschnitt und Cache
def schritt_gesetze(modell: str = MODELL, schnitt_s: float = SCHNITT_GESETZE_S, auswahl: str = "filter") -> pd.DataFrame:
    t0 = time.perf_counter()
    alle = lade_bloecke()
    kaputt = [f"{b['sitzung']}:{b['aufruf_zeile']}" for b in alle if not b["rekonstruktion_ok"]]
    print(f"{len(alle)} Blöcke rekonstruiert, Drucksachen nicht im Text: {len(kaputt)} {kaputt[:5]}", flush=True)
    bloecke = reihenfolge(alle, "alle" if schnitt_s <= 0 else auswahl)
    zahlen = {"bloecke_gesamt": len(alle), "bloecke_filter": sum(b["filter"] for b in alle),
              "bloecke_gesetzentwurf": sum(b["gesetz"] for b in alle)}
    print(f"Auswahl {auswahl if schnitt_s > 0 else 'alle (nur Cache)'}: {len(bloecke)} Blöcke; {zahlen}", flush=True)
    zeilen, gegenstaende = [], []
    for i, b in enumerate(bloecke, 1):
        offen = time.perf_counter() - t0 > schnitt_s
        if offen:  # nach dem Zeitschnitt (SCHNITT_GESETZE_S) nur noch Cache-Treffer, der Rest bleibt „offen“
            res = frage(SYSTEM_GESETZE, nutzer_gesetze(b, kuerze(b["text"])[0]), Extraktion, modell, OPT_GESETZE,
                        nur_cache=True)
            if res["offen"]:
                continue
            e = extrahiere(b, modell)
        else:
            e = extrahiere(b, modell, timeout=min(TIMEOUT_S, max(30.0, schnitt_s - (time.perf_counter() - t0))))
        res = e["res"]
        v = vergleiche(b["regex"], e["llm"], e["sichtbar"])
        zeilen.append({"sitzung": b["sitzung"], "aufruf_zeile": b["aufruf_zeile"], "top_id": b["top_id"],
                       "entwurf": b["entwurf"], "gesetz": b["gesetz"], "gekuerzt": e["gekuerzt"],
                       "zeichen": len(b["text"]),
                       "n_regex_toc": b["n_regex_toc"],
                       "gueltig": e["llm"] is not None, "fehler": res["fehler"]} | _messwerte(res) | v)
        for j, g in enumerate(e["llm"] or []):
            gegenstaende.append({"sitzung": b["sitzung"], "aufruf_zeile": b["aufruf_zeile"], "top_id": b["top_id"],
                                 "nr": j, "titel": g["titel"], "kurztitel": g["kurztitel"],
                                 "drucksache": g["drucksachen"][0] if g["drucksachen"] else None,
                                 "drucksachen": json.dumps(g["drucksachen"]), "typ": g["typ"],
                                 "beratungsstufe": g["beratungsstufe"], "modell": modell})
        if i % 10 == 0:
            d = pd.DataFrame(zeilen)
            print(f"{time.strftime('%H:%M:%S')} {i}/{len(bloecke)} Blöcke, {time.perf_counter() - t0:.0f} s, "
                  f"gültig {d.gueltig.mean():.0%}, gleich {(d.klasse == 'gleich').mean():.0%}", flush=True)
    wand = time.perf_counter() - t0
    div = pd.DataFrame(zeilen)
    for c in ("drs_regex", "drs_llm", "fehlt_llm", "fehlt_regex", "typ_diff", "stufe_diff"):
        div[c] = div[c].map(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None)
    OUT.mkdir(parents=True, exist_ok=True)
    ges = pd.DataFrame(gegenstaende)
    ges.to_parquet(OUT / "llm_gesetze.parquet", index=False)
    ges.to_csv(OUT / "llm_gesetze.csv", index=False)
    div.to_csv(OUT / "divergenz.csv", index=False)
    if schnitt_s <= 0 and LAUFZEITEN.exists():  # Cache-Neulauf dauert ≈ 1,3 s; berichtet wird die Summe echter Läufe
        lz = pd.read_csv(LAUFZEITEN)
        echt = lz.schritt.isin(["llm_gesetze", "llm_gesetze_gesetzentwurf"])
        wand = float(lz[echt].sekunden.sum()) if echt.any() else wand
    kz = kennzahlen(div, zahlen, wand)
    kz.to_csv(OUT / "kennzahlen_gesetze.csv", index=False)
    print(kz.to_string(index=False), flush=True)
    schritt = ("llm_gesetze" if auswahl == "filter" else f"llm_gesetze_{auswahl}") if schnitt_s > 0 else "llm_gesetze_cache"
    laufzeit_anhaengen(schritt, time.perf_counter() - t0)
    return div


def kennzahlen(div: pd.DataFrame, zahlen: dict, wand: float) -> pd.DataFrame:
    g = div[div.gueltig]
    k = zahlen | {"bloecke_bearbeitet": len(div), "gesetzentwurf_abgedeckt": int(div.gesetz.sum()),
                  "gesetzentwurf_offen": zahlen["bloecke_gesetzentwurf"] - int(div.gesetz.sum()),
         "bloecke_gekuerzt": int(div.gekuerzt.sum()), "gueltig_anteil": div.gueltig.mean(),
         "gueltig_erst_nach_wiederholung": int((div.gueltig & (div.versuche > 1)).sum()),
         "quote_drucksachen": g.drs_gleich.mean(), "quote_titel_ab_0_9": (g.titel_quote >= TITEL_GLEICH).mean(),
         "quote_kurztitel": g.kurztitel_gleich.mean(), "quote_typ": g.typ_gleich.mean(),
         "quote_stufe": g.stufe_gleich.mean(), "quote_anzahl_gegenstaende": (g.n_llm == g.n_regex).mean(),
         "laufzeit_wand_s": wand, "llm_sekunden_summe": div.sekunden.sum(),
         "sekunden_je_block_mittel": div.sekunden.mean(), "sekunden_je_block_median": div.sekunden.median(),
         "eval_count_mittel": div.eval_count.mean()}
    k |= {f"klasse_{c}": int(n) for c, n in div.klasse.value_counts().items()}
    for spalte in ("typ_diff", "stufe_diff"):  # 6 häufigste Muster; vor dem Fix in parse.ENTWURF_RE: 22 + 16 Typmuster = Ordinalfehler
        muster = g[spalte].dropna().map(json.loads).explode().dropna().value_counts().head(6)
        k |= {f"{spalte}_{m}": int(n) for m, n in muster.items()}
    return pd.DataFrame({"kennzahl": list(k), "wert": [round(float(v), 4) for v in k.values()]})


# Schritt 2: der Modellvergleich belegt qwen2.5:7b (5/5 gleich gegen 0/5 und 1/5)
def vorhandene_modelle() -> set[str]:
    try:
        return {m["name"] for m in requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10).json()["models"]}
    except requests.RequestException:
        return set()


def stichprobe_vergleich(bloecke: list[dict]) -> list[dict]:
    gold = [b for b in bloecke if (b["sitzung"], b["top_id"]) == VERGLEICH_GOLD][:1]
    rest = sorted((b for b in bloecke if b["filter"] and b not in gold), key=lambda b: (b["sitzung"], b["aufruf_zeile"]))
    return gold + random.Random(SEED).sample(rest, VERGLEICH_N - len(gold))


def schritt_vergleich(modelle: list[str] | None = None) -> pd.DataFrame:
    t0 = time.perf_counter()
    modelle = modelle or VERGLEICH_MODELLE
    da = vorhandene_modelle()
    probe = stichprobe_vergleich(lade_bloecke())
    zeilen = []
    for modell in modelle:
        if modell not in da and not any(modell == n.removesuffix(":latest") for n in da):
            print(f"{modell} nicht installiert, übersprungen", flush=True)
            continue
        reasoning = modell.startswith("deepseek-r1")
        for b in probe:
            e = extrahiere(b, modell, OPT_REASONING if reasoning else OPT_GESETZE, think=True if reasoning else None,
                           timeout=600 if reasoning else TIMEOUT_S)  # 4096 Token bei ≈ 20 Token/s ≈ 200 s, Faktor 3
            v = vergleiche(b["regex"], e["llm"], e["sichtbar"])
            zeilen.append({"modell": modell, "sitzung": b["sitzung"], "top_id": b["top_id"],
                           "aufruf_zeile": b["aufruf_zeile"], "gueltig": e["llm"] is not None,
                           "fehler": e["res"]["fehler"]} | _messwerte(e["res"]) |
                          {k: v[k] for k in ("klasse", "drs_gleich", "typ_gleich", "stufe_gleich", "titel_quote")})
            print(f"{time.strftime('%H:%M:%S')} {modell} {b['sitzung']} {b['top_id']}: {zeilen[-1]['sekunden']} s, "
                  f"{zeilen[-1]['eval_count']} Token, {v['klasse']}", flush=True)
    df = pd.DataFrame(zeilen)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "modellvergleich.csv", index=False)
    if len(df):
        print(df.groupby("modell")[["sekunden", "eval_count", "gueltig", "drs_gleich", "typ_gleich", "stufe_gleich"]]
              .mean().round(2).to_string(), flush=True)
    laufzeit_anhaengen("llm_vergleich", time.perf_counter() - t0)
    return df


# Schritt 3: Top-10-Zusammenfassungen; ohne Gold prüft erwartung() die Haltungen gegen Einbringer und Beschlussart
# Fraktionsnamen der WP 20 (wie beitraege.fraktion_pdf); LLM-Ausgaben schreiben sie frei („Grüne“, „Union“)
FRAKTION_MUSTER = [("CDU/CSU", r"CDU|CSU|Union"), ("SPD", r"SPD|Sozialdemokrat"),
                   ("BÜNDNIS 90/DIE GRÜNEN", r"GRÜNE|Grüne|BÜNDNIS|Bündnis"), ("FDP", r"FDP|Freie Demokrat"),
                   ("AfD", r"AfD"), ("Die Linke", r"LINKE|Linke"), ("BSW", r"BSW"), ("fraktionslos", r"fraktionslos")]
KOALITION_AMPEL_BIS = 196  # Ampel-Bruch 06.11.2024, Sitzung 197 am 07.11. (parse.REGIERUNG)
# „des von der Fraktion der CDU/CSU eingebrachten Entwurfs“ (Aufgabenbeispiel 20211 ZP 35): Einbringer stimmt erwartbar zu
EINBRINGER_RE = re.compile(r"von (?:der|den) (?P<wer>[^()]{3,160}?) eingebrachten")


def norm_fraktion(s: str | None) -> str | None:
    for name, muster in FRAKTION_MUSTER:
        if s and re.search(muster, s):
            return name
    return None


def koalition(sitzung: int) -> set[str]:
    return {"SPD", "BÜNDNIS 90/DIE GRÜNEN"} | ({"FDP"} if sitzung <= KOALITION_AMPEL_BIS else set())


def einbringer(titel: str, sitzung: int) -> set[str]:
    m = EINBRINGER_RE.search(titel or "")
    if not m:
        return set()
    if "Bundesregierung" in m["wer"]:
        return koalition(sitzung)
    if "Fraktion" not in m["wer"]:
        return set()  # Bundesrat oder Gruppenentwurf (55 „von den Abgeordneten“, z. B. 20/10188): keine Fraktionslinie
    # „Fraktionen SPD, CDU/CSU, … sowie dem Abgeordneten Stefan Seidler“ (20/12977): nur die Fraktionen zählen
    return {f for f in (norm_fraktion(t) for t in re.split(r",| und | sowie ", m["wer"])) if f}


def erwartung(fraktion: str | None, titel: str, sitzung: int, beschlussart: str) -> tuple[str | None, str]:
    """Erwartete Haltung aus Einbringer und beschlussart_final; None, wenn die Daten nichts hergeben."""
    ein = einbringer(titel, sitzung)
    if fraktion in ein:
        return "zustimmend", "einbringer"
    if beschlussart == "abgelehnt" and ein and sitzung <= KOALITION_AMPEL_BIS and fraktion in koalition(sitzung):
        return "ablehnend", "abgelehnt_mit_koalitionsmehrheit"
    if beschlussart == "angenommen" and ein == koalition(sitzung) and fraktion in koalition(sitzung):
        return "zustimmend", "angenommen_regierungsentwurf"
    return None, "keine"


def top10() -> tuple[pd.DataFrame, str]:
    e = pd.read_parquet(config.PROCESSED / "gesetze_einheit.parquet")
    pfad = config.PROCESSED / "top10_gesetze.csv"
    if pfad.exists():
        t = pd.read_csv(pfad, dtype=str)
        return e.set_index("drucksache").loc[[d for d in t["drucksache"] if d in set(e.drucksache)]].reset_index(), \
            "top10_gesetze.csv"
    # Einzelpläne sind Teile des Haushaltsgesetzes (parse: „hier: Einzelplan“ als Gesetzentwurf); ohne Ausschluss
    # wären 9 der 10 Plätze Einzelpläne einer Haushaltswoche (Drucksache ≠ Gesetz).
    e = e[~e.titel.fillna("").str.startswith("hier: Einzelplan")]
    return e.sort_values(["n_reden", "drucksache"], ascending=[False, True]).head(10).reset_index(drop=True), \
        "gesetze_einheit (ohne Einzelpläne)"


def kontext(drucksache: str, reden: pd.DataFrame, bet: pd.DataFrame) -> tuple[str, int, list[str]]:
    """Redeanfänge je Fraktion, reihum ergänzt (erste Rede jeder Fraktion zuerst), bis KONTEXT_MAX_TOKEN erreicht.

    Jüngste Lesung zuerst: 2500 Token fassen ≈ 1 Rede je Fraktion (20/12804: 56 Reden), und beschlussart_final, gegen
    die die Haltung geprüft wird, stammt aus der letzten Lesung (20/12804: Sitzung 211, nicht 185)."""
    ids = bet[(bet.drucksache == drucksache) & (bet.redetyp == "plenarrede")].rede_id.unique()
    r = reden[reden.rede_id.isin(ids)].copy()
    r["frak"] = r.fraktion.where(r.fraktion.notna(), r.partei).map(norm_fraktion)
    r = r[r.frak.notna() & (r.woerter >= MIN_WOERTER)]
    r = r.assign(minus_sitzung=-r.sitzung).sort_values(["minus_sitzung", "zeile_start"])
    gruppen = {f: list(g.itertuples()) for f, g in r.groupby("frak", sort=False)}
    grenze = KONTEXT_MAX_TOKEN * ZEICHEN_JE_TOKEN
    teile: dict[str, list[str]] = {f: [] for f in gruppen}
    laenge, n, voll = 0, 0, False
    for runde in range(max((len(v) for v in gruppen.values()), default=0)):
        for f, liste in gruppen.items():
            if runde >= len(liste):
                continue
            x = liste[runde]
            worte = ANREDE_RE.sub("", str(x.text)).split()[:REDE_WOERTER]
            auszug = f"- {x.name} (Sitzung {x.sitzung}): " + " ".join(worte) + " …"
            kopf = 0 if teile[f] else len(f"### {f}\n") + 2  # Überschrift zählt mit, sonst > 2500 Token (7038 Z.)
            if laenge + kopf + len(auszug) > grenze:
                voll = True
                break
            teile[f].append(auszug)
            laenge += kopf + len(auszug) + 1
            n += 1
        if voll:
            break
    text = "\n\n".join(f"### {f}\n" + "\n".join(v) for f, v in teile.items() if v)
    return text, n, [f for f, v in teile.items() if v]


def schritt_zusammenfassung(modell: str = MODELL, schnitt_s: float = SCHNITT_ZUSAMMEN_S) -> list[dict]:
    t0 = time.perf_counter()
    gesetze, quelle = top10()
    reden = pd.read_parquet(config.PROCESSED / "reden.parquet")
    bet = pd.read_parquet(config.PROCESSED / "gesetz_beteiligte.parquet")
    print(f"Top 10 aus {quelle}: {list(gesetze.drucksache)}", flush=True)
    ergebnisse, zeilen = [], []
    for g in gesetze.itertuples():
        titel = g.titel if isinstance(g.titel, str) else ""
        kurz = g.kurztitel if isinstance(g.kurztitel, str) else None
        ktext, n_reden, fraktionen = kontext(g.drucksache, reden, bet)
        user = f"Gesetzentwurf Drucksache {g.drucksache}: {titel}\n\nRedeauszüge nach Fraktion:\n\n{ktext}"
        rest = schnitt_s - (time.perf_counter() - t0)
        res = frage(SYSTEM_ZUSAMMEN, user, Zusammenfassung, modell, OPT_ZUSAMMEN, nur_cache=rest <= 0,
                    timeout=max(30.0, min(TIMEOUT_S, rest)))
        sitzung = max(_liste(g.sitzungen)) if len(_liste(g.sitzungen)) else 0
        obj = res["objekt"]
        eintrag = {"drucksache": g.drucksache, "kurztitel": kurz, "titel": titel, "n_reden_gesamt": int(g.n_reden),
                   "n_reden_kontext": n_reden, "fraktionen_kontext": fraktionen,
                   "beschlussart_final": g.beschlussart_final, "status": "offen" if res["offen"] else
                   ("gueltig" if obj else "ungueltig"), "fehler": res["fehler"]} | _messwerte(res)
        positionen = []
        if obj:
            eintrag["kernpunkt"] = obj.kernpunkt
            for p in obj.positionen:
                f = norm_fraktion(p.fraktion) or p.fraktion
                erw, basis = erwartung(f, titel, sitzung, g.beschlussart_final)
                plaus = None if erw is None or p.haltung in ("unklar", "gemischt") else p.haltung == erw
                positionen.append({"fraktion": f, "haltung": p.haltung, "begruendung": p.begruendung,
                                   "erwartung": erw, "basis": basis, "plausibel": plaus})
                zeilen.append({k: eintrag[k] for k in ("drucksache", "kurztitel", "beschlussart_final", "kernpunkt")}
                              | positionen[-1])
        eintrag["positionen"] = positionen
        ergebnisse.append(eintrag)
        print(f"{time.strftime('%H:%M:%S')} {g.drucksache} {kurz or titel[:40]}: {eintrag['status']}, "
              f"{n_reden} Auszüge, {eintrag['sekunden']} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "top10_zusammenfassungen.json").write_text(
        json.dumps({"quelle_top10": quelle, "modell": modell, "gesetze": ergebnisse}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    pd.DataFrame(zeilen).to_csv(OUT / "top10_zusammenfassungen.csv", index=False)
    plaus = pd.Series([z["plausibel"] for z in zeilen], dtype=object)
    kz = {"gesetze": len(ergebnisse), "gueltig": sum(e["status"] == "gueltig" for e in ergebnisse),
          "neu_berechnet": sum(not e["cache"] and e["versuche"] > 0 for e in ergebnisse),
          "aus_cache": sum(e["cache"] for e in ergebnisse), "positionen": len(zeilen),
          "plausibel_ja": int((plaus == True).sum()), "plausibel_nein": int((plaus == False).sum()),  # noqa: E712
          "plausibel_ohne_erwartung": int(plaus.isna().sum()),
          "llm_sekunden_summe": round(sum(e["sekunden"] for e in ergebnisse), 1)}
    pd.DataFrame({"kennzahl": list(kz), "wert": list(kz.values())}).to_csv(OUT / "kennzahlen_zusammenfassung.csv",
                                                                         index=False)
    print(kz, flush=True)
    laufzeit_anhaengen("llm_zusammenfassung" if schnitt_s > 0 else "llm_zusammenfassung_cache",
                       time.perf_counter() - t0)
    return ergebnisse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schritt", required=True, choices=["gesetze", "vergleich", "zusammenfassung"])
    ap.add_argument("--modell", nargs="*", help="Modellname(n); vergleich nimmt mehrere, sonst der erste")
    ap.add_argument("--nur-cache", action="store_true", help="keine neuen LLM-Aufrufe, Ausgaben nur aus dem Cache")
    ap.add_argument("--auswahl", choices=["filter", "gesetzentwurf"], default="filter",
                    help="gesetze: Textfilter („Entwurf(s) eines“, „Drucksache 20/“) oder alle Blöcke mit Gesetzentwurf")
    ap.add_argument("--schnitt-min", type=float, default=SCHNITT_GESETZE_S / 60, help="gesetze: Schnitt in Minuten")
    a = ap.parse_args()
    modell = a.modell[0] if a.modell else MODELL
    if a.schritt == "gesetze":
        schritt_gesetze(modell, 0 if a.nur_cache else a.schnitt_min * 60, a.auswahl)
    elif a.schritt == "vergleich":
        schritt_vergleich(a.modell)
    else:
        schritt_zusammenfassung(modell, 0 if a.nur_cache else SCHNITT_ZUSAMMEN_S)


if __name__ == "__main__":
    main()
