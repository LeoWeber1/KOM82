"""Schritt 2b: Güte der PDF-Textextraktion gegen das amtliche XML messen.

Das XML dient ausschließlich als Validierungsreferenz; die Pipeline selbst
arbeitet nur mit dem PDF-Text. Drei Kennzahlen je Sitzung:

1. Absatz-Ordnungsquote: Anfang jedes Rede-Absatzes (6 Wörter) in der txt
   suchen; Anteil der Fundstellen in monoton steigender Reihenfolge
   (längste nichtfallende Teilfolge / gefunden). Deckt Spaltenvertauschung
   und Zwischenruf-Verschiebungen auf. Zusätzlich Fundquote.
2. 5-Gramm-Recall: Anteil der XML-Wort-5-Gramme, die in der txt vorkommen.
   Deckt lokale Fehler auf (Silbentrennung, fehlende Zeilen).
3. Token-Recall: Multiset-Recall der XML-Tokens. Deckt reine Textverluste auf.

Parser-Validierung (``--alle``): Beitrags- und Reden-F1 per Sequenz-Alignment der
Nachnamen gegen gold_beitraege/gold_reden, Partei gegen Stammdaten (ID-Join), Beginn/Schluss
gegen gold_zeiten, Drucksachen-Recall gegen den XML-Aufrufblock und gold_gesetze.csv.
Ausgaben: validierung.parquet/.csv (eine Zeile je Sitzung), fehler.csv (jede Abweichung).

CLI: ``python -m bundestag.validate --sitzungen 146 196`` | ``--alle`` | ``--stichprobe``
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from lxml import etree

from . import config, stammdaten

# Weiches Trennzeichen (U+00AD): bleibt nach NFKC erhalten, steht in manchen
# XML-Absätzen innerhalb von Wörtern ("Ver\xadant\xadwortung") -> Token zerfällt.
SOFT_HYPHEN = "­"
# Typografische Anführungszeichen des XML („“‚‘»«) vs. PDF-Extraktion, die
# teils gerade Zeichen liefert; beide werden zu Leerraum, damit "„Gesetz" == "Gesetz".
QUOTES = "„“”‚‘’«»‹›\"'`´"
# Gedankenstrich (–, —) und Bindestrich (-): PDF-Zeilenumbruch vor einem
# Gedankenstrich erzeugt "Wort–" ohne Leerzeichen; alle Strichvarianten
# werden Trenner, sodass "CDU/CSU-Fraktion" beidseitig in 3 Tokens zerfällt.
DASHES = "–—‐‑‒−-"
TOKEN_RE = re.compile(r"\w+")


def normalize(text: str) -> str:
    # NFKC: Ligaturen (ﬁ, ﬂ) aus dem PDF-Font und geschützte Leerzeichen
    # (U+00A0, U+202F in "20. Wahlperiode") auf ASCII-Äquivalente.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(SOFT_HYPHEN, "")
    table = {ord(c): " " for c in QUOTES + DASHES}
    return text.translate(table).lower()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def _stem(nr: int) -> str:
    return f"{config.WAHLPERIODE}{nr:03d}"


def load_txt_tokens(nr: int) -> list[str]:
    return tokens((config.TXT_DIR / f"{_stem(nr)}.txt").read_text(encoding="utf-8"))


def load_xml_reden(nr: int, mit_anlagen: bool = False) -> tuple[list[list[str]], int]:
    """Je <rede> die Absatztexte (ohne klasse='redner'); <kommentar> bleibt außen vor,
    weil die txt Zwischenrufe getrennt flusht und sie nicht Redetext sind.
    Zu Protokoll gegebene Reden stehen im XML als <rede> unter <anlagen>; die txt
    enthält nur section='body' (20146: 9 solche Reden, 5,6 % der 5-Gramme; 20196: 1) ->
    standardmäßig ausgeschlossen, Anzahl wird mitgeliefert."""
    root = etree.parse(str(config.XML_DIR / f"{_stem(nr)}.xml")).getroot()
    body, anlagen = reden_aus_root(root)
    return (body + anlagen if mit_anlagen else body), len(anlagen)


def reden_aus_root(root) -> tuple[list[list[str]], list[list[str]]]:
    """Absatztexte je <rede>, getrennt nach Sitzungsverlauf und <anlagen>."""
    body: list[list[str]] = []
    anlagen: list[list[str]] = []
    for rede in root.iter("rede"):
        paras = []
        for p in rede.findall("p"):
            if p.get("klasse") == "redner":
                continue
            txt = "".join(p.itertext()).strip()
            if txt:
                paras.append(txt)
        if not paras:
            continue
        if any(a.tag == "anlagen" for a in rede.iterancestors()):
            anlagen.append(paras)
        else:
            body.append(paras)
    return body, anlagen


def longest_nondecreasing(seq: list[int]) -> int:
    tails: list[int] = []
    for x in seq:
        i = bisect.bisect_right(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _ngrams(toks: list[str], n: int):
    return (tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def ordnungsquote(xml_reden: list[list[str]], txt_tokens: list[str], n: int = 6) -> dict:
    """Index aller 6-Gramme der txt; je Absatzanfang die erste Fundstelle ab der
    letzten Fundstelle (Formelabsätze wie 'Vielen Dank. – Nächster Redner ist'
    kommen dutzendfach vor; die erste Fundstelle insgesamt würde die Ordnung
    künstlich einbrechen lassen)."""
    index: dict[tuple, list[int]] = defaultdict(list)
    for i, g in enumerate(_ngrams(txt_tokens, n)):
        index[g].append(i)

    positions: list[int] = []
    per_rede: list[float] = []
    eligible = found = skipped = 0
    missing: list[str] = []
    last = -1
    for paras in xml_reden:
        rede_pos: list[int] = []
        for p in paras:
            toks = tokens(p)
            if len(toks) < n:
                skipped += 1
                continue
            eligible += 1
            hits = index.get(tuple(toks[:n]))
            if not hits:
                missing.append(" ".join(toks[:n]))
                continue
            found += 1
            j = bisect.bisect_left(hits, last)
            pos = hits[j] if j < len(hits) else hits[0]
            last = pos
            positions.append(pos)
            rede_pos.append(pos)
        if len(rede_pos) >= 2:
            per_rede.append(longest_nondecreasing(rede_pos) / len(rede_pos))
    return {
        "absaetze_xml": eligible,
        "absaetze_kurz": skipped,
        "gefunden": found,
        "fundquote": found / eligible if eligible else float("nan"),
        "ordnungsquote": longest_nondecreasing(positions) / found if found else float("nan"),
        "ordnungsquote_reden": sum(per_rede) / len(per_rede) if per_rede else float("nan"),
        "_missing": missing,
    }


def ngram_recall(xml_reden: list[list[str]], txt_tokens: list[str], n: int = 5) -> dict:
    """XML-5-Gramme nur innerhalb eines Absatzes; txt-5-Gramme über den gesamten
    Tokenstrom, weil to_paragraphs() bei jedem Zwischenruf flusht und ein
    XML-<p> dann in der txt auf zwei Absätze verteilt ist (kein Fehler)."""
    txt_set = set(_ngrams(txt_tokens, n))
    total = hit = 0
    missing: Counter = Counter()
    for paras in xml_reden:
        for p in paras:
            for g in _ngrams(tokens(p), n):
                total += 1
                if g in txt_set:
                    hit += 1
                else:
                    missing[" ".join(g)] += 1
    return {"ngramme_xml": total, "recall_5gramm": hit / total if total else float("nan"),
            "_missing_ngrams": missing}


def token_recall(xml_reden: list[list[str]], txt_tokens: list[str]) -> dict:
    xml_counts = Counter(t for paras in xml_reden for p in paras for t in tokens(p))
    txt_counts = Counter(txt_tokens)
    total = sum(xml_counts.values())
    hit = sum(min(c, txt_counts[t]) for t, c in xml_counts.items())
    return {"tokens_xml": total, "tokens_txt": len(txt_tokens),
            "recall_token": hit / total if total else float("nan")}


def validate_sitzung(nr: int, mit_anlagen: bool = False) -> dict:
    reden, n_anlagen = load_xml_reden(nr, mit_anlagen)
    txt = load_txt_tokens(nr)
    out = {"sitzung": nr, "reden_xml": len(reden), "reden_anlagen": n_anlagen}
    out.update(ordnungsquote(reden, txt))
    out.update(ngram_recall(reden, txt))
    out.update(token_recall(reden, txt))
    return out


REF_DIR = config.REFERENCE
# gold.py schlüsselt Sitzungen als 20146, parse.py als 146; Ausgaben folgen parse.py.
GOLD_OFFSET = 20000
# Parser-Typen, die je einem <rede>-Element des Sitzungsverlaufs entsprechen. Zwischenfragen
# stehen im XML innerhalb der <rede> des Hauptredners, Befragungsköpfe meist ohne <rede>.
REDE_TYPEN = ("plenarrede", "kurzintervention", "erwiderung")
ANLAGE_TYPEN = ("zu_protokoll", "erklaerung_go31")
# TOP-Typen, in denen das XML andere Einheiten bildet (20156: 140 Köpfe gegen 65 <rede>).
BEFRAGUNG_TYPEN = ("Befragung", "Fragestunde")
BEFRAGUNG_RE = re.compile(r"Befragung der Bundesregierung|Fragestunde")
# Gold-Titel sind leer, wenn ein TOP ohne T_-Absätze beginnt oder nach einer Unterbrechung als zweites
# Element mit derselben top-id fortgesetzt wird (20209 TOP 2: 46 + 58 <rede>); daher Prüfung im XML
# über Titelabsätze plus die ersten Prosa-Absätze, vereinigt über alle Elemente derselben top-id.
BEFRAGUNG_PROSA_ABSAETZE = 3
DRS_RE = re.compile(r"20/\d+")
# XML-top-id enthält teils NBSP ("Tagesordnungspunkt\xa07") oder Listen ("Zusatzpunkt 3, 4");
# der Parser führt den ersten Punkt des Aufrufs -> Vergleich über den ersten Punkt.
TOP_ID_RE = re.compile(r"^(Tagesordnungspunkt|Zusatzpunkt)\s+([0-9IVX.]+\s?[a-z]?)")
# Zeile 1 der txt ist Absatz 1, Absätze sind durch Leerzeilen getrennt ("\n\n".join).
ABSATZ_KEY_LEN = 25
# Schluss nach Mitternacht (20207: 09:00 -> 00:25) liegt am Folgetag.
TAG_MINUTEN = 1440


def nachname_key(name) -> str:
    """Letztes Namenswort nach stammdaten.norm_name: 'Dr. Katrin Göring-Eckardt' (Parser) und
    das XML-Feld 'Göring-Eckardt' werden beide 'eckardt', 'Beatrix von Storch' -> 'storch'."""
    toks = stammdaten.norm_name(str(name or "")).split()
    return toks[-1] if toks else ""


def top_key(top_id) -> str | None:
    if top_id is None or (isinstance(top_id, float) and pd.isna(top_id)):
        return None
    m = TOP_ID_RE.match(re.sub(r"\s+", " ", str(top_id)))
    return f"{m[1]} {m[2].replace(' ', '')}" if m else None


def lcs_align(a: list, b: list) -> list[tuple[int, int]]:
    """Längste gemeinsame Teilfolge mit Rückverfolgung -> monotone Paare (i, j).
    difflib.SequenceMatcher wäre greedy und stuft ab 200 Elementen jeden Namen mit > 1 %
    Anteil als Junk ein (Minister in Befragungen); die DP kostet je Sitzung <= 600 x 600."""
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai, Li, Ln = a[i], L[i], L[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                Li[j] = Ln[j + 1] + 1
            else:
                Li[j] = Ln[j] if Ln[j] >= Li[j + 1] else Li[j + 1]
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def prf_spalten(praefix: str, tp: int, n_parser: int, n_gold: int) -> dict:
    p, r, f1 = prf(tp, n_parser, n_gold)
    return {f"{praefix}_parser": n_parser, f"{praefix}_gold": n_gold, f"{praefix}_tp": tp,
            f"{praefix}_precision": p, f"{praefix}_recall": r, f"{praefix}_f1": f1}


def prf(tp: int, n_parser: int, n_gold: int) -> tuple[float, float, float]:
    if not n_parser and not n_gold:
        return float("nan"), float("nan"), float("nan")
    p = tp / n_parser if n_parser else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _json_list(v) -> list:
    if isinstance(v, str):
        return json.loads(v)
    return list(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else []


class Daten:
    """Alle Tabellen einmal laden; Gold-Sitzungsnummern auf den Parser-Schlüssel bringen."""

    def __init__(self):
        P = config.PROCESSED
        self.reden = pd.read_parquet(P / "reden.parquet")
        self.beitraege = pd.read_parquet(P / "beitraege.parquet")
        self.gesetze = pd.read_parquet(P / "gesetze.parquet")
        sp = P / "sitzungen.parquet"
        self.sitzungen = pd.read_parquet(sp) if sp.exists() else None
        self.fraktionen = pd.read_parquet(P / "fraktionen.parquet")
        gold = lambda n: pd.read_parquet(config.GOLD_DIR / f"{n}.parquet").assign(sitzung=lambda d: d.sitzung - GOLD_OFFSET)
        self.gold_reden = gold("gold_reden")
        self.gold_beitraege = gold("gold_beitraege")
        self.gold_tops = gold("gold_tops")
        self.gold_zeiten = gold("gold_zeiten")
        self.gold_gesetze = pd.read_csv(REF_DIR / "gold_gesetze.csv").assign(sitzung=lambda d: d.sitzung - GOLD_OFFSET)
        # Stammdaten je MdB-ID: Liste (von, bis, partei, fraktion) für den Datumsabgleich
        self.stamm: dict[str, list] = defaultdict(list)
        for z in self.fraktionen.itertuples():
            self.stamm[str(z.id)].append((z.von, z.bis, z.partei, z.fraktion))
        self._seiten: dict[int, dict[int, int]] = {}

    def partei_zum_datum(self, redner_id, datum: date) -> tuple[str | None, str | None] | None:
        """Stammdaten-Zeile zur ID am Sitzungstag; am Wechseltag gilt die später beginnende."""
        if redner_id is None or pd.isna(redner_id):
            return None
        treffer = [z for z in self.stamm.get(str(redner_id), ())
                   if z[0] <= datum and (z[1] is None or pd.isna(z[1]) or z[1] >= datum)]
        if not treffer:
            return None
        z = max(treffer, key=lambda z: z[0])
        return (None if pd.isna(z[2]) else z[2]), (None if pd.isna(z[3]) else z[3])

    def seite(self, nr: int, zeile) -> int | None:
        if zeile is None or pd.isna(zeile):
            return None
        if nr not in self._seiten:
            self._seiten[nr] = seiten_index(nr)
        return self._seiten[nr].get(int(zeile))


def seiten_index(nr: int) -> dict[int, int]:
    """txt-Zeile -> PDF-Seite (1-basiert) über den Absatzanfang (25 Zeichen) in data/lines;
    Absätze beginnen mit dem Text ihrer ersten Layoutzeile (20146: 2870/2870 gefunden)."""
    pfad = config.LINES_DIR / f"{_stem(nr)}.jsonl"
    txt = config.TXT_DIR / f"{_stem(nr)}.txt"
    if not pfad.exists() or not txt.exists():
        return {}
    recs = [json.loads(l) for l in pfad.read_text(encoding="utf-8").splitlines()[1:]]
    body = [r for r in recs if r.get("section") == "body"]
    index: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(body):
        index[r["text"][:ABSATZ_KEY_LEN]].append(i)
    out, ptr = {}, 0
    for i, p in enumerate(txt.read_text(encoding="utf-8").split("\n")):
        if not p.strip():
            continue
        kand = index.get(p[:ABSATZ_KEY_LEN], ())
        j = next((k for k in kand if k >= ptr), None)
        if j is not None:
            ptr = j
            out[i + 1] = body[j]["page"] + 1
    return out


def gold_positionen(root) -> dict[tuple, int]:
    """Lesefolge aller Sprecherköpfe (p[@klasse='redner'], <name>) außerhalb <anlagen>,
    verschlüsselt wie gold.py: (rede_id, segment_nr) in <rede>, sonst (top_id, segment_nr).
    Nötig, weil gold_beitraege Befragungsköpfe hinter alle <rede>-Segmente hängt."""
    pos: dict[tuple, int] = {}
    zaehler: Counter = Counter()
    for el in root.iter("p", "name"):
        parent = el.getparent()
        if parent.tag == "rede":
            container = ("rede", parent.get("id"))
        elif parent.tag in ("tagesordnungspunkt", "sitzungsbeginn"):
            container = (parent.tag, parent.get("top-id"))
        else:
            continue
        if el.tag == "p" and el.get("klasse") != "redner":
            continue
        if any(a.tag == "anlagen" for a in el.iterancestors()):
            continue
        pos.setdefault((container[1], zaehler[container]), len(pos))
        zaehler[container] += 1
    return pos


def gold_sortiert(gb: pd.DataFrame, pos: dict[tuple, int]) -> tuple[pd.DataFrame, int]:
    """gold_beitraege einer Sitzung in Lesefolge; Zeilen ohne Position bleiben hinten."""
    keys = [(r.top_id if pd.isna(r.rede_id) else r.rede_id, int(r.segment_nr)) for r in gb.itertuples()]
    p = [pos.get((None if isinstance(k[0], float) else k[0], k[1]), len(pos) + i) for i, k in enumerate(keys)]
    ohne = sum(1 for k in keys if (None if isinstance(k[0], float) else k[0], k[1]) not in pos)
    return gb.assign(_pos=p).sort_values("_pos"), ohne


def xml_befragung_tops(root) -> set[str]:
    """top-ids (roh, wie in gold_reden.top_id) aller TOPs mit Befragung der Bundesregierung oder Fragestunde."""
    out: set[str] = set()
    for top in root.iter("tagesordnungspunkt"):
        prosa = 0
        for p in top.findall("p"):
            klasse = p.get("klasse") or ""
            if klasse == "redner":
                break
            if not klasse.startswith("T_"):
                prosa += 1
                if prosa > BEFRAGUNG_PROSA_ABSAETZE:
                    break
            if BEFRAGUNG_RE.search("".join(p.itertext())):
                out.add(top.get("top-id"))
                break
    return out


def xml_aufruf_drucksachen(root) -> list[tuple[str | None, str, bool]]:
    """(top_id, Drucksache, debattiert) aus den T_Drs-Absätzen jedes <tagesordnungspunkt>:
    das ist der Aufrufblock des XML und entspricht dem "Ich rufe ... auf"-Block des Parsers;
    Drucksachen aus der Prosa (Entschließungsanträge bei der Abstimmung) bleiben außen vor."""
    out = []
    for top in root.iter("tagesordnungspunkt"):
        text = " ".join("".join(p.itertext()) for p in top.iter("p") if p.get("klasse") == "T_Drs")
        debattiert = top.find("rede") is not None
        for drs in dict.fromkeys(DRS_RE.findall(text)):
            out.append((top.get("top-id"), drs, debattiert))
    return out


# Titelabgleich: Titel eines Gegenstands = T_NaS-/T_fett-/T_ZP_NaS-Absätze vor seinem T_Drs-Absatz im Aufrufblock
# (20211 ZP 35: T_NaS "Zweite und dritte Beratung … (Zustrombegrenzungsgesetz)", dann T_Drs "Drucksache 20/12804").
TITEL_KLASSEN = {"T_NaS", "T_fett", "T_ZP_NaS"}
# Gliederung am Absatzanfang gehört nicht zum Titel und beginnt einen neuen Gegenstand: "b) Beratung" (20157 TOP 26 b),
# "ZP 14Beratung" (20182), "5 Beratung" (20207 TOP 5)
GLIEDERUNG_RE = re.compile(r"^(?:ZP ?\d+(?: ?[a-z]{1,2}\b)?|\d+(?: ?[a-z]{1,2}\b)?|[a-z]{1,2}\))\s*")
# Kurztitel im XML wie beim Parser (parse.KURZTITEL_RE): Klammer mit "gesetz" am Titelende
KURZ_XML_RE = re.compile(r"\(([^()]*gesetz[^()]*)\)\s*$", re.I)
# ab dieser Token-F1 gilt ein Titel als "bis auf einzelne Wörter gleich" (ein abweichendes Wort in 20 Wörtern)
TITEL_FAST = 0.95


def xml_aufruf_titel(root) -> dict[str, str]:
    """Drucksache -> Titeltext aus dem Aufrufblock des XML. Drucksachen, die in mehreren Aufrufen mit verschiedenem
    Titel stehen (20/8661: Bericht des Haushaltsausschusses zu allen Einzelplänen), bleiben außen vor."""
    titel: dict[str, set[str]] = defaultdict(set)
    for top in root.iter("tagesordnungspunkt"):
        puffer: list[str] = []
        for p in top.iter("p"):
            k, text = p.get("klasse"), " ".join("".join(p.itertext()).split())
            if k in TITEL_KLASSEN:
                rest = GLIEDERUNG_RE.sub("", text)
                if rest != text:
                    puffer = []
                puffer.append(rest)
            elif k == "T_Drs":
                for d in dict.fromkeys(DRS_RE.findall(text)):
                    if puffer:
                        titel[d].add(" ".join(puffer))
                puffer = []
            else:
                puffer = []
    return {d: next(iter(t)) for d, t in titel.items() if len(t) == 1}


def titel_tokens(s) -> list[str]:
    return TOKEN_RE.findall(str(s).replace("\u00ad", "").lower()) if isinstance(s, str) else []


def token_f1(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    tp = sum((Counter(a) & Counter(b)).values())
    return 2 * tp / (len(a) + len(b))


def titel_vergleich(nr: int, pg: pd.DataFrame, root, fehler: list[dict], D: "Daten") -> dict:
    """Titel und Kurztitel des Parsers gegen den Aufrufblock des XML, nur Gegenstände mit Drucksache aus dem Aufruf
    selbst (Sammelaufrufe holen Titel aus der Tagesordnung, dafür hat das XML keinen eigenen Titel)."""
    xt = xml_aufruf_titel(root)
    out = dict.fromkeys(("titel_n", "titel_gleich", "titel_fast", "titel_gesetz_n", "titel_gesetz_gleich",
                         "titel_gesetz_fast", "kurztitel_xml", "kurztitel_gleich", "kurztitel_nur_parser"), 0)
    for z in pg[pg.quelle_drucksache == "aufruf"].itertuples():
        x = xt.get(z.drucksache)
        if not x:
            continue
        a, b = titel_tokens(z.titel), titel_tokens(x)
        gleich, fast = a == b, token_f1(a, b) >= TITEL_FAST
        out["titel_n"] += 1
        out["titel_gleich"] += gleich
        out["titel_fast"] += fast
        if z.ist_gesetz:
            out["titel_gesetz_n"] += 1
            out["titel_gesetz_gleich"] += gleich
            out["titel_gesetz_fast"] += fast
        if not fast:
            fehler.append(_f("gesetze", nr, "titel_abweichung", z.aufruf_zeile, str(z.titel)[:160], x[:160], D))
        kx, kp = KURZ_XML_RE.search(x), z.kurztitel if isinstance(z.kurztitel, str) else None
        if kx:
            out["kurztitel_xml"] += 1
            out["kurztitel_gleich"] += titel_tokens(kp) == titel_tokens(kx.group(1))
        elif kp:
            out["kurztitel_nur_parser"] += 1
    return out


def _minuten(hhmm) -> int | None:
    if hhmm is None or (isinstance(hhmm, float) and pd.isna(hhmm)) or ":" not in str(hhmm):
        return None
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 60 + int(m)


def zeit_diff(p_beginn, p_schluss, g_start, g_ende) -> tuple[float, float]:
    """Abweichung Parser - Gold in Minuten; ein Schluss vor dem Beginn liegt am Folgetag."""
    pb, ps, gb, gs = map(_minuten, (p_beginn, p_schluss, g_start, g_ende))
    d_beginn = pb - gb if pb is not None and gb is not None else float("nan")
    if None in (pb, ps, gb, gs):
        return d_beginn, float("nan")
    ps += TAG_MINUTEN if ps < pb else 0
    gs += TAG_MINUTEN if gs < gb else 0
    return d_beginn, ps - gs


def parser_zeiten(nr: int, D: Daten) -> tuple[str | None, str | None]:
    """beginn/schluss aus sitzungen.parquet; davor aus der meta-Zeile von data/lines."""
    if D.sitzungen is not None:
        z = D.sitzungen[D.sitzungen.sitzung == nr]
        if not z.empty:
            return z.iloc[0].beginn, z.iloc[0].schluss
    pfad = config.LINES_DIR / f"{_stem(nr)}.jsonl"
    if pfad.exists():
        with pfad.open(encoding="utf-8") as f:
            meta = json.loads(f.readline()).get("meta", {})
        return meta.get("beginn"), meta.get("schluss")
    return None, None


def _f(name: str, nr: int, art: str, zeile, parser_wert, gold_wert, D: Daten | None = None) -> dict:
    return {"sitzung": nr, "seite": D.seite(nr, zeile) if D else None,
            "zeile": None if zeile is None or pd.isna(zeile) else int(zeile),
            "art": art, "parser_wert": parser_wert, "gold_wert": gold_wert, "_quelle": name}


def _naechste_zeile(pairs_gold: dict[int, int], j: int, zeilen: list) -> int | None:
    """Ungefähre txt-Zeile einer fehlenden Gold-Rede: Zeile der letzten davor gematchten Parser-Rede."""
    vorher = [g for g in pairs_gold if g < j]
    if vorher:
        return zeilen[pairs_gold[max(vorher)]]
    nachher = [g for g in pairs_gold if g > j]
    return zeilen[pairs_gold[min(nachher)]] if nachher else None


def reden_vergleich(nr: int, D: Daten, pr_main: pd.DataFrame, gr_main: pd.DataFrame, datum: date,
                    fehler: list, gold_top_keys: set[str]) -> dict:
    """Hauptvergleich: LCS über die Nachnamen; Partei der Treffer gegen Stammdaten zur Gold-ID;
    Parser-TOPs ohne XML-Gegenstück und ohne Treffer markieren ein unvollständiges XML."""
    pk = [nachname_key(n) for n in pr_main.name]
    gk = [nachname_key(n) for n in gr_main.nachname]
    pairs = lcs_align(pk, gk)
    matched_p = {i for i, _ in pairs}
    pairs_gold = {j: i for i, j in pairs}
    zeilen = pr_main.zeile_start.tolist()
    out = prf_spalten("reden", len(pairs), len(pk), len(gk))
    out.update({"rede_zuviel": len(pk) - len(pairs), "rede_fehlt": len(gk) - len(pairs)})
    for j in range(len(gk)):
        if j not in pairs_gold:
            g = gr_main.iloc[j]
            fehler.append(_f("reden", nr, "rede_fehlt", _naechste_zeile(pairs_gold, j, zeilen), None,
                             f"{g.vorname} {g.nachname} ({g.fraktion or g.rolle}) {g.top_id} {g.rede_id}", D))
    for i in range(len(pk)):
        if i not in matched_p:
            z = pr_main.iloc[i]
            fehler.append(_f("reden", nr, "rede_zuviel", z.zeile_start,
                             f"{z['name']} ({z.partei}) {z.redetyp} {z.top_id}", None, D))
    # Partei: ID-Join Gold-redner_id -> Stammdaten am Sitzungstag, nur wenn die ID dort steht
    gleich = pruefbar = 0
    for i, j in pairs:
        st = D.partei_zum_datum(gr_main.iloc[j].redner_id, datum)
        if st is None or st[0] is None:
            continue
        pruefbar += 1
        pp = pr_main.iloc[i].partei
        if pp == st[0]:
            gleich += 1
        else:
            z = pr_main.iloc[i]
            fehler.append(_f("partei", nr, "partei_abweichung", z.zeile_start, f"{z['name']}: {pp}",
                             f"{st[0]} ({st[1]}) id={gr_main.iloc[j].redner_id}", D))
    out.update({"partei_pruefbar": pruefbar, "partei_gleich": gleich,
                "partei_quote": gleich / pruefbar if pruefbar else float("nan")})
    # XML-Vollständigkeit: Parser-TOP mit >= 3 Reden, dessen erster Punkt im XML fehlt und dessen
    # Reden höchstens zur Hälfte gematcht wurden (20194: XML endet nach TOP 17, Parser hat TOP 16-25).
    fehlende_tops, reden_fehlend = [], 0
    if len(pr_main):
        tmp = pr_main.assign(m_=[i in matched_p for i in range(len(pk))], k_=pr_main.top_id.map(top_key))
        for k, grp in tmp.groupby("k_", dropna=True):
            if len(grp) >= 3 and k not in gold_top_keys and grp.m_.mean() <= 0.5:
                fehlende_tops.append(k)
                reden_fehlend += int((~grp.m_).sum())
    out.update({"xml_fehlende_tops": len(fehlende_tops), "xml_fehlende_reden": reden_fehlend,
                "xml_fehlende_top_ids": ";".join(fehlende_tops)})
    return out


def validate_parser_sitzung(nr: int, D: Daten, root) -> tuple[dict, list[dict]]:
    fehler: list[dict] = []
    out: dict = {}
    gz = D.gold_zeiten[D.gold_zeiten.sitzung == nr]
    datum = date.fromisoformat(gz.iloc[0].datum) if len(gz) else None
    gold_tops = D.gold_tops[D.gold_tops.sitzung == nr]
    gold_top_keys = {k for k in gold_tops.top_id.map(top_key) if k}
    befr = xml_befragung_tops(root)

    # Reden-F1 ohne Befragung/Fragestunde (dort zählt unten der Kopf, s. BEFRAGUNG_TYPEN) und ohne die nicht
    # gesprochenen Anlagen-Reden (zu Protokoll gegeben); diese zählen nur in reden_inkl_anlagen
    gr = D.gold_reden[D.gold_reden.sitzung == nr]
    gr_body = gr[gr.anlage_typ.isna()]
    gr_befr = gr_body[gr_body.top_id.isin(befr)]
    gr_main = gr_body[~gr_body.top_id.isin(befr)]
    gr_anl = gr[gr.anlage_typ.notna()]
    pr = D.reden[D.reden.sitzung == nr]
    in_befr = pr.top_typ.isin(BEFRAGUNG_TYPEN)
    pr_main = pr[pr.redetyp.isin(REDE_TYPEN) & ~in_befr].sort_values("zeile_start")
    sort_anl = ["anlage", "zeile_start"] if "anlage" in pr.columns else ["zeile_start"]
    pr_anl = pr[pr.redetyp.isin(ANLAGE_TYPEN)].sort_values(sort_anl)
    out.update(reden_vergleich(nr, D, pr_main, gr_main, datum, fehler, gold_top_keys))

    pk2 = [nachname_key(n) for n in pd.concat([pr_main.name, pr_anl.name])]
    gk2 = [nachname_key(n) for n in pd.concat([gr_main.nachname, gr_anl.nachname])]
    out.update({"reden_anlagen_parser": len(pr_anl), "reden_anlagen_gold": len(gr_anl)})
    out.update(prf_spalten("reden_inkl_anlagen", len(lcs_align(pk2, gk2)), len(pk2), len(gk2)))

    # Einheit: ein Sprecherkopf = ein Beitrag. Parser ohne praesidium und Anlagen-Typen; Gold zaehlt_als_beitrag
    # (Redebeginn, Wiederaufnahme nach Präsidiumseinschub, Zwischenfrage, Befragungskopf) ohne anlage_typ.
    pos = gold_positionen(root)
    gb = D.gold_beitraege[(D.gold_beitraege.sitzung == nr) & D.gold_beitraege.zaehlt_als_beitrag
                          & D.gold_beitraege.anlage_typ.isna()]
    gb, ohne_pos = gold_sortiert(gb, pos)
    pb = D.beitraege[(D.beitraege.sitzung == nr) & ~D.beitraege.redetyp.isin(("praesidium",) + ANLAGE_TYPEN)]
    pb = pb.sort_values("zeile_start")
    pk = [nachname_key(n) for n in pb.name]
    gk = [nachname_key(n) for n in gb.nachname]
    pairs = lcs_align(pk, gk)
    out.update(prf_spalten("beitraege", len(pairs), len(pk), len(gk)))
    out["beitraege_gold_ohne_pos"] = ohne_pos
    mp = {i for i, _ in pairs}
    mg = {j for _, j in pairs}
    pairs_gold = {j: i for i, j in pairs}
    zeilen = pb.zeile_start.tolist()
    for i in range(len(pk)):
        if i not in mp:
            z = pb.iloc[i]
            fehler.append(_f("beitraege", nr, "beitrag_zuviel", z.zeile_start, f"{z['name']} {z.redetyp}", None, D))
    for j in range(len(gk)):
        if j not in mg:
            g = gb.iloc[j]
            fehler.append(_f("beitraege", nr, "beitrag_fehlt", _naechste_zeile(pairs_gold, j, zeilen), None,
                             f"{g.vorname} {g.nachname} {g.redetyp} {g.top_id}", D))

    # Befragung/Fragestunde auf Kopf-Ebene, weil das XML <rede> dort anders schneidet (20156: 140 Köpfe gegen 65 <rede>)
    gb_befr = gb[gb.top_id.isin(befr)]
    pb_befr = pb[pb.top_typ.isin(BEFRAGUNG_TYPEN)]
    pk = [nachname_key(n) for n in pb_befr.name]
    gk = [nachname_key(n) for n in gb_befr.nachname]
    out.update(prf_spalten("befragung", len(lcs_align(pk, gk)), len(pk), len(gk)))
    out.update({"befragung_reden_gold": len(gr_befr), "befragung_reden_parser": int(in_befr.sum())})

    pb_, ps_ = parser_zeiten(nr, D)
    gs, ge = (gz.iloc[0].start, gz.iloc[0].ende) if len(gz) else (None, None)
    db, ds = zeit_diff(pb_, ps_, gs, ge)
    out.update({"beginn_parser": pb_, "beginn_gold": gs, "schluss_parser": ps_, "schluss_gold": ge,
                "beginn_diff_min": db, "schluss_diff_min": ds})
    for art, d, pw, gw in (("beginn", db, pb_, gs), ("schluss", ds, ps_, ge)):
        if pd.isna(d) or d != 0:
            fehler.append(_f("zeiten", nr, "zeit_abweichung", None, f"{art} {pw}", f"{art} {gw}", D))

    pg = D.gesetze[D.gesetze.sitzung == nr]
    parser_drs: set[str] = set()
    for z in pg.itertuples():
        parser_drs.update(_json_list(z.drucksachen_alle))
        if isinstance(z.drucksache, str):
            parser_drs.add(z.drucksache)
    aufruf = xml_aufruf_drucksachen(root)
    alle = {d for t, d, _ in aufruf}
    deb = {d for t, d, debattiert in aufruf if debattiert}
    nennungen = {d for lst in gold_tops.drucksachen for d in _json_list(lst)}
    out.update({"drs_gold": len(alle), "drs_gefunden": len(alle & parser_drs),
                "drs_recall": len(alle & parser_drs) / len(alle) if alle else float("nan"),
                "drs_recall_debattiert": len(deb & parser_drs) / len(deb) if deb else float("nan"),
                "drs_recall_nennungen": len(nennungen & parser_drs) / len(nennungen) if nennungen else float("nan"),
                "drs_parser": len(parser_drs)})
    aufruf_zeile = {top_key(z.top_id): z.aufruf_zeile for z in pg.itertuples()}
    for t, d, _ in aufruf:
        if d not in parser_drs:
            fehler.append(_f("gesetze", nr, "drucksache_fehlt", aufruf_zeile.get(top_key(t)), None, f"{d} ({t})", D))
    out.update(titel_vergleich(nr, pg, root, fehler, D))
    return out, fehler


def gold_gesetze_vergleich(D: Daten, vollstaendig: tuple[int, ...] = (211,)) -> dict:
    """Recall je Gold-Zeile (gefunden, wenn eine ihrer Drucksachen oder - ohne Drucksache - ihr Punkt
    beim Parser in derselben Sitzung steht); Precision nur für Sitzungen, die Gold vollständig
    annotiert hat (20211; 20160/20185 listen nur 2/7 von 36/22 Aufrufen)."""
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
    pg = D.gesetze.assign(
        drs_set=D.gesetze.apply(lambda z: set(_json_list(z.drucksachen_alle))
                                | ({z.drucksache} if isinstance(z.drucksache, str) else set()), axis=1),
        punkt_norm=D.gesetze.punkt.map(norm))
    gg = D.gold_gesetze.assign(
        drs_set=D.gold_gesetze.drucksachen.map(lambda s: set(DRS_RE.findall(str(s))) if isinstance(s, str) else set()),
        punkt_norm=D.gold_gesetze.top.map(norm))
    treffer = []
    for z in gg.itertuples():
        p = pg[pg.sitzung == z.sitzung]
        drs = set().union(*p.drs_set) if len(p) else set()
        treffer.append(bool(z.drs_set & drs) if z.drs_set else z.punkt_norm in set(p.punkt_norm))
    gg = gg.assign(treffer=treffer)
    sicher, mit_drs = gg[gg.unsicher == 0], gg[gg.drs_set.map(bool)]
    out = {"gold_gesetze_n": len(gg), "gold_gesetze_recall": gg.treffer.mean(),
           "gold_gesetze_recall_sicher": sicher.treffer.mean(),
           # 7 Gold-Zeilen ohne Drucksache (Aktuelle Stunde, Regierungserklärung, Einspruch) sind keine Gegenstände der Gesetze-Tabelle
           "gold_gesetze_recall_mit_drs": mit_drs.treffer.mean(),
           "gold_gesetze_recall_mit_drs_sicher": mit_drs[mit_drs.unsicher == 0].treffer.mean(),
           "gold_gesetze_fehlend": [f"{z.sitzung} {z.top} {z.drucksachen}" for z in gg[~gg.treffer].itertuples()]}
    for s in vollstaendig:
        p, g = pg[pg.sitzung == s], gg[gg.sitzung == s]
        gold_drs = set().union(*g.drs_set) if len(g) else set()
        gold_punkte = set(g.punkt_norm)
        # Gold führt Punkte eines Sammelaufrufs teils nur unter verbunden_mit (20211 ZP 28, Drs. 20/13957 steht im Aufruf Z. 39)
        verbunden = gold_punkte | {norm(v) for lst in g.verbunden_mit.dropna() for v in str(lst).split(";")}
        tp = [bool(z.drs_set & gold_drs) or z.punkt_norm in gold_punkte for z in p.itertuples()]
        tp_v = [ok or z.punkt_norm in verbunden for z, ok in zip(p.itertuples(), tp)]
        out[f"gold_gesetze_precision_{s}"] = sum(tp) / len(tp) if tp else float("nan")
        out[f"gold_gesetze_precision_verbunden_{s}"] = sum(tp_v) / len(tp_v) if tp_v else float("nan")
        out[f"gold_gesetze_fp_{s}"] = [f"{z.punkt} {z.drucksache}" for z, ok in zip(p.itertuples(), tp) if not ok]
    return out


# Abstimmungserkennung: Precision auf einer Stichprobe, Urteil je Fall in data/reference/abstimmung_urteile.csv;
# die Wörter unten zeigen bei abgestimmt=False, ob eine Abstimmung fehlt
ABSTIMMUNG_WORTE = re.compile(r"Abstimmung|Wer stimmt|angenommen|abgelehnt|überwiesen|Überweisung|Stimmenthaltung", re.I)


# Stichprobe v3 (Prüfung nach der Korrektur der Abstimmungserkennung): 50 + 25 Fälle, ohne die 60 Fälle der
# Stichproben v1 und v2, an denen die Regeln entwickelt wurden (sonst wäre die Precision an den eigenen Testfällen gemessen)
STICHPROBE_V3 = {"n_true": 50, "n_false": 25, "datei": "abstimmung_stichprobe_v3.md", "urteile": "abstimmung_urteile_v3.csv",
                 "ohne": ("abstimmung_stichprobe.md", "abstimmung_stichprobe_v1.md")}
# Stichprobe v4 (24.10.2026): prüft die vier Beschlussart-Regeln, die an den Fehlern von v3 entstanden, an neuen Fällen
# (ohne v1 bis v3), gleicher Aufbau wie v3, Seed 83
STICHPROBE_V4 = {"n_true": 50, "n_false": 25, "seed": 83, "datei": "abstimmung_stichprobe_v4.md",
                 "urteile": "abstimmung_urteile_v4.csv",
                 "ohne": ("abstimmung_stichprobe.md", "abstimmung_stichprobe_v1.md", "abstimmung_stichprobe_v3.md")}
FALL_RE = re.compile(r"^## Fall \d+: Sitzung (\d+),.*\nGegenstand: .*?\| Drs\. (20/\d+)", re.M)


def fruehere_faelle(dateien) -> set[tuple[int, str]]:
    """(Sitzung, Drucksache) aller Fälle früherer Stichproben."""
    out = set()
    for name in dateien:
        if (REF_DIR / name).exists():
            out |= {(int(s), d) for s, d in FALL_RE.findall((REF_DIR / name).read_text(encoding="utf-8"))}
    return out


def schreibe_abstimmung_stichprobe(D: Daten, pfad: Path, seed: int = 82, n_true: int = 20, n_false: int = 10,
                                   kontext: int = 3, ohne: set[tuple[int, str]] | None = None,
                                   urteile_name: str = "abstimmung_urteile.csv") -> pd.DataFrame:
    """n_true Fälle abgestimmt=True (Kontext um abstimmung_zeile) und n_false Fälle False (Kontext um den Aufruf
    plus alle Abstimmungs-/Überweisungssätze bis zum nächsten Aufruf, damit sich ein Fehlen prüfen lässt)."""
    g = D.gesetze
    if ohne:
        g = g[~pd.Series([(int(s), d) in ohne for s, d in zip(g.sitzung, g.drucksache)], index=g.index)]
    faelle = pd.concat([g[g.abgestimmt].sample(n_true, random_state=seed),
                        g[~g.abgestimmt].sample(n_false, random_state=seed)]).reset_index(drop=True)
    txt_cache: dict[int, list[str]] = {}
    zeilen = lambda nr: txt_cache.setdefault(nr, (config.TXT_DIR / f"{_stem(nr)}.txt").read_text(encoding="utf-8").split("\n"))
    out = [f"# Abstimmungs-Stichprobe (Seed {seed}: {n_true} x abgestimmt=True, {n_false} x False)", "",
           f"Zeilen beziehen sich 1-basiert auf data/txt/20NNN.txt. Urteile stehen in {urteile_name}.", ""]
    for k, z in enumerate(faelle.itertuples(), 1):
        L = zeilen(int(z.sitzung))
        zeile = int(z.abstimmung_zeile) if z.abgestimmt and not pd.isna(z.abstimmung_zeile) else int(z.aufruf_zeile)
        out.append(f"## Fall {k}: Sitzung {z.sitzung}, Zeile {zeile}, abgestimmt={z.abgestimmt}")
        titel = next((t for t in (z.kurztitel, z.titel) if isinstance(t, str) and t), "")
        out.append(f"Gegenstand: {z.punkt} | {z.typ} | {z.beratungsstufe} | Drs. {z.drucksache} | "
                   f"{titel[:110]} | ergebnis_text: {z.ergebnis_text} | beschlussart: {z.beschlussart}")
        for i in range(max(1, zeile - kontext), min(len(L), zeile + kontext) + 1):
            if L[i - 1].strip():
                out.append(f"  {i:>6}: {L[i - 1][:220]}")
        if not z.abgestimmt:
            folgende = g[(g.sitzung == z.sitzung) & (g.aufruf_zeile > z.aufruf_zeile)].aufruf_zeile
            ende = int(folgende.min()) if len(folgende) else len(L)
            hits = [(i, L[i - 1]) for i in range(zeile + 1, ende) if ABSTIMMUNG_WORTE.search(L[i - 1])]
            out.append(f"  Treffer Abstimmungswörter bis zum nächsten Aufruf (Zeile {ende}): {len(hits)}")
            for i, t in hits[:8]:
                out.append(f"  {i:>6}: {t[:220]}")
        out.append("")
    pfad.write_text("\n".join(out), encoding="utf-8")
    return faelle


def abstimmung_precision(pfad: Path) -> dict | None:
    if not pfad.exists():
        return None
    u = pd.read_csv(pfad)
    t, f = u[u.abgestimmt], u[~u.abgestimmt]
    return {"n": len(u), "precision": (t.urteil == "richtig").mean() if len(t) else float("nan"),
            "fehlrate_false": (f.urteil == "falsch").mean() if len(f) else float("nan"), "n_true": len(t), "n_false": len(f)}


def abstimmung_recall(precision: float, fehlrate_false: float, n_abgestimmt: int, n_nicht: int) -> float:
    """Geschätzter Recall der Abstimmungserkennung aus der geschichteten Stichprobe: richtig erkannte Abstimmungen
    (Precision × erkannte) geteilt durch diese plus die übersehenen (Fehlrate der False-Schicht × nicht erkannte).
    Gemessen ist der Recall nicht, er folgt aus den beiden Schichten."""
    tp, fn = precision * n_abgestimmt, fehlrate_false * n_nicht
    return tp / (tp + fn) if tp + fn else float("nan")


def extraktion_sitzung(nr: int, root) -> dict:
    """Extraktionskennzahlen (Sitzungsverlauf) plus dieselben Kennzahlen für Anlagen-Reden gegen
    data/txt/<nr>_anlagen.txt, sofern diese Datei existiert (461 der 8558 XML-Reden stehen in Anlagen)."""
    body, anl = reden_aus_root(root)
    out = {"sitzung": nr, "reden_xml": len(body), "reden_anlagen": len(anl)}
    txt = load_txt_tokens(nr)
    out.update(ordnungsquote(body, txt))
    out.update(ngram_recall(body, txt))
    out.update(token_recall(body, txt))
    anl_txt = config.TXT_DIR / f"{_stem(nr)}_anlagen.txt"
    if anl and anl_txt.exists():
        t = tokens(anl_txt.read_text(encoding="utf-8"))
        o = ordnungsquote(anl, t)
        out.update({"fundquote_anlagen": o["fundquote"], "ordnungsquote_anlagen": o["ordnungsquote"],
                    "recall_5gramm_anlagen": ngram_recall(anl, t)["recall_5gramm"],
                    "recall_token_anlagen": token_recall(anl, t)["recall_token"]})
    return out


# Sitzungen gelten als im XML unvollständig, wenn mindestens 10 Parser-Reden in TOPs liegen, die das XML nicht
# führt und deren Reden höchstens zur Hälfte gematcht sind. Lauf 67 Sitzungen: 9 Sitzungen mit 13-58 solchen
# Reden (147, 175, 178, 191, 194, 203, 207, 209, 210; 20194 endet im XML nach TOP 17, 20210 nach TOP 12),
# alle übrigen 58 Sitzungen haben 0.
XML_UNVOLLSTAENDIG_MIN_REDEN = 10


def validate_alle(nrs, D: Daten | None = None, ausgabe: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    D = D or Daten()
    rows, fehler = [], []
    for nr in nrs:
        root = etree.parse(str(config.XML_DIR / f"{_stem(nr)}.xml")).getroot()
        r = extraktion_sitzung(nr, root)
        o, f = validate_parser_sitzung(nr, D, root)
        r.update(o)
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        fehler.extend(f)
    df = pd.DataFrame(rows)
    df["xml_unvollstaendig"] = df.xml_fehlende_reden >= XML_UNVOLLSTAENDIG_MIN_REDEN
    fe = pd.DataFrame(fehler, columns=["sitzung", "seite", "zeile", "art", "parser_wert", "gold_wert", "_quelle"])
    fe = fe.drop(columns="_quelle")
    extra = gold_gesetze_vergleich(D)
    if ausgabe:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED / "validierung.parquet", index=False)
        df.to_csv(config.PROCESSED / "validierung.csv", index=False)
        fe.to_csv(config.PROCESSED / "fehler.csv", index=False)
    return df, fe, extra


def mikro(df: pd.DataFrame, praefix: str) -> tuple[float, float, float]:
    return prf(int(df[f"{praefix}_tp"].sum()), int(df[f"{praefix}_parser"].sum()), int(df[f"{praefix}_gold"].sum()))


def ursache(z) -> str:
    if z.xml_unvollstaendig:
        return f"XML unvollständig ({z.xml_fehlende_tops} TOPs/{z.xml_fehlende_reden} Reden fehlen im XML)"
    if z.befragung_parser > 2 * max(z.befragung_gold, 1):
        return f"TOP-Typ Befragung/Fragestunde läuft weiter ({z.befragung_parser} Parser-Köpfe gegen {z.befragung_gold} im XML)"
    if z.rede_fehlt > 2 * z.rede_zuviel:
        return f"Reden fehlen ({z.rede_fehlt}): Köpfe nicht erkannt oder TOP-Typ falsch"
    if z.rede_zuviel > 2 * z.rede_fehlt:
        return f"Reden zu viel ({z.rede_zuviel}): Beiträge nicht zusammengeführt"
    return f"gemischt ({z.rede_fehlt} fehlen, {z.rede_zuviel} zu viel): Namensabweichungen"


def zusammenfassung(df: pd.DataFrame, fe: pd.DataFrame, extra: dict, urteile: dict | None) -> list[str]:
    ok = df[~df.xml_unvollstaendig]
    lines = []
    p, r, f = mikro(df, "reden")
    p2, r2, f2 = mikro(ok, "reden")
    lines.append(f"Reden-F1 Hauptvergleich mikro {f:.3f} (P {p:.3f}, R {r:.3f}, {int(df.reden_parser.sum())} Parser / "
                 f"{int(df.reden_gold.sum())} Gold); ohne {int(df.xml_unvollstaendig.sum())} Sitzungen mit unvollständigem XML "
                 f"({', '.join(map(str, df.sitzung[df.xml_unvollstaendig]))}): F1 {f2:.3f} (P {p2:.3f}, R {r2:.3f})")
    lines.append(f"Reden-F1 je Sitzung Median {df.reden_f1.median():.3f}, Min {df.reden_f1.min():.3f}; "
                 f"inkl. Anlagen mikro F1 {mikro(df, 'reden_inkl_anlagen')[2]:.3f} "
                 f"({int(df.reden_anlagen_parser.sum())} Parser / {int(df.reden_anlagen_gold.sum())} Gold Anlagen-Reden)")
    bp, br, bf = mikro(df, "beitraege")
    lines.append(f"Beitrags-F1 mikro {bf:.3f} (P {bp:.3f}, R {br:.3f}); Median {df.beitraege_f1.median():.3f}, Min {df.beitraege_f1.min():.3f}")
    fp, fr, ff = mikro(df, "befragung")
    lines.append(f"Befragung/Fragestunde Kopf-F1 mikro {ff:.3f} (P {fp:.3f}, R {fr:.3f}); XML dort {int(df.befragung_reden_gold.sum())} <rede> "
                 f"gegen {int(df.befragung_gold.sum())} Köpfe")
    lines.append(f"Partei gleich {df.partei_gleich.sum() / max(df.partei_pruefbar.sum(), 1):.4f} "
                 f"({int(df.partei_gleich.sum())}/{int(df.partei_pruefbar.sum())} prüfbar); Median je Sitzung {df.partei_quote.median():.3f}, Min {df.partei_quote.min():.3f}")
    lines.append(f"Extraktion Median/Min: Ordnung {df.ordnungsquote.median():.3f}/{df.ordnungsquote.min():.3f}, Fund {df.fundquote.median():.3f}/{df.fundquote.min():.3f}, "
                 f"5-Gramm {df.recall_5gramm.median():.3f}/{df.recall_5gramm.min():.3f}, Token {df.recall_token.median():.3f}/{df.recall_token.min():.3f}")
    if "recall_5gramm_anlagen" in df:
        a = df.dropna(subset=["recall_5gramm_anlagen"])
        lines.append(f"Anlagen-Extraktion ({len(a)} Sitzungen) Median/Min: Ordnung {a.ordnungsquote_anlagen.median():.3f}/{a.ordnungsquote_anlagen.min():.3f}, "
                     f"5-Gramm {a.recall_5gramm_anlagen.median():.3f}/{a.recall_5gramm_anlagen.min():.3f}, Token {a.recall_token_anlagen.median():.3f}/{a.recall_token_anlagen.min():.3f}")
    lines.append(f"Beginn/Schluss: |Abweichung| Median {df.beginn_diff_min.abs().median():.0f}/{df.schluss_diff_min.abs().median():.0f} min, "
                 f"Max {df.beginn_diff_min.abs().max():.0f}/{df.schluss_diff_min.abs().max():.0f} min, "
                 f"Sitzungen mit Abweichung {int(((df.beginn_diff_min != 0) | (df.schluss_diff_min != 0)).sum())}")
    lines.append(f"Drucksachen-Recall (XML-Aufrufblock) mikro {df.drs_gefunden.sum() / max(df.drs_gold.sum(), 1):.3f} "
                 f"({int(df.drs_gefunden.sum())}/{int(df.drs_gold.sum())}); Median {df.drs_recall.median():.3f}, Min {df.drs_recall.min():.3f}; "
                 f"nur debattierte TOPs Median {df.drs_recall_debattiert.median():.3f}; alle Nennungen Median {df.drs_recall_nennungen.median():.3f}")
    tn, tg = max(df.titel_n.sum(), 1), max(df.titel_gesetz_n.sum(), 1)
    lines.append(f"Titel gegen XML-Aufrufblock: gleich {df.titel_gleich.sum() / tn:.3f}, Token-F1 >= {TITEL_FAST} "
                 f"{df.titel_fast.sum() / tn:.3f} ({int(df.titel_n.sum())} Gegenstände); Gesetzentwürfe gleich "
                 f"{df.titel_gesetz_gleich.sum() / tg:.3f}, F1 >= {TITEL_FAST} {df.titel_gesetz_fast.sum() / tg:.3f} "
                 f"({int(df.titel_gesetz_n.sum())}); Kurztitel gleich {int(df.kurztitel_gleich.sum())}/"
                 f"{int(df.kurztitel_xml.sum())}, nur Parser {int(df.kurztitel_nur_parser.sum())}")
    lines.append(f"gold_gesetze.csv ({extra['gold_gesetze_n']} Zeilen): Recall {extra['gold_gesetze_recall']:.3f} (unsicher=0: {extra['gold_gesetze_recall_sicher']:.3f}), "
                 f"nur Zeilen mit Drucksache {extra['gold_gesetze_recall_mit_drs']:.3f} (unsicher=0: {extra['gold_gesetze_recall_mit_drs_sicher']:.3f}); "
                 f"Precision 20211 {extra.get('gold_gesetze_precision_211', float('nan')):.3f}, inkl. verbundener Punkte {extra.get('gold_gesetze_precision_verbunden_211', float('nan')):.3f}; "
                 f"fehlend: {'; '.join(extra['gold_gesetze_fehlend']) or '-'}")
    if urteile:
        lines.append(f"Abstimmung: Precision {urteile['precision']:.2f} ({urteile['n_true']} Fälle True), Fehlrate bei False {urteile['fehlrate_false']:.2f} ({urteile['n_false']} Fälle)")
    arten = fe.art.value_counts()
    lines.append("Fehlerarten: " + ", ".join(f"{a} {n}" for a, n in arten.items()))
    lines.append("Schlechteste Sitzungen nach Reden-F1:")
    for z in df.sort_values("reden_f1").head(3).itertuples():
        lines.append(f"  {z.sitzung}: F1 {z.reden_f1:.3f} (P {z.reden_precision:.3f}, R {z.reden_recall:.3f}, "
                     f"{z.reden_parser}/{z.reden_gold}) - {ursache(z)}")
    return lines


CSV_COLS = ["sitzung", "reden_xml", "reden_anlagen", "absaetze_xml", "absaetze_kurz", "gefunden", "fundquote",
            "ordnungsquote", "ordnungsquote_reden", "ngramme_xml", "recall_5gramm",
            "tokens_xml", "tokens_txt", "recall_token"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitzungen", nargs="+", type=int, default=[146, 196])
    ap.add_argument("--alle", action="store_true", help="alle 67 Sitzungen, Parser-Validierung, validierung.parquet + fehler.csv")
    ap.add_argument("--stichprobe", action="store_true", help="Abstimmungs-Stichprobe neu schreiben")
    ap.add_argument("--stichprobe-v4", action="store_true", help="Stichprobe v4 (Beschlussart nach den neuen Regeln) schreiben")
    ap.add_argument("--beispiele", type=int, default=2, help="fehlende Beispiele je Sitzung")
    ap.add_argument("--mit-anlagen", action="store_true", help="zu Protokoll gegebene Reden einbeziehen")
    args = ap.parse_args(argv)

    if args.alle or args.stichprobe or args.stichprobe_v4:
        D = Daten()
        if args.stichprobe_v4:
            v4 = STICHPROBE_V4
            pfad = REF_DIR / v4["datei"]
            schreibe_abstimmung_stichprobe(D, pfad, seed=v4["seed"], n_true=v4["n_true"], n_false=v4["n_false"],
                                           ohne=fruehere_faelle(v4["ohne"]), urteile_name=v4["urteile"])
            print(f"-> {pfad.relative_to(config.ROOT)}")
        if args.stichprobe:
            v3 = STICHPROBE_V3
            pfad = REF_DIR / v3["datei"]
            schreibe_abstimmung_stichprobe(D, pfad, n_true=v3["n_true"], n_false=v3["n_false"],
                                           ohne=fruehere_faelle(v3["ohne"]), urteile_name=v3["urteile"])
            print(f"-> {pfad.relative_to(config.ROOT)}")
        if args.alle:
            nrs = list(config.SITZUNGEN) if args.sitzungen == [146, 196] else args.sitzungen
            df, fe, extra = validate_alle(nrs, D)
            for line in zusammenfassung(df, fe, extra, abstimmung_precision(REF_DIR / STICHPROBE_V3["urteile"])):
                print(line)
            print(f"-> data/processed/validierung.parquet (+csv, {len(df)} Sitzungen), fehler.csv ({len(fe)} Zeilen)")
        return

    rows = []
    for nr in args.sitzungen:
        r = validate_sitzung(nr, args.mit_anlagen)
        rows.append(r)
        flag = "OK " if min(r["ordnungsquote"], r["recall_5gramm"], r["recall_token"]) > 0.9 else "!! "
        print(f"{flag}{nr}: Ordnung {r['ordnungsquote']:.3f} (Reden {r['ordnungsquote_reden']:.3f}, "
              f"Fund {r['fundquote']:.3f}, {r['gefunden']}/{r['absaetze_xml']}) | "
              f"5-Gramm {r['recall_5gramm']:.3f} | Token {r['recall_token']:.3f} "
              f"({r['tokens_xml']} xml / {r['tokens_txt']} txt)")
        for ex in r["_missing"][: args.beispiele]:
            print(f"    Absatz nicht gefunden: {ex}")
        for g, c in r["_missing_ngrams"].most_common(args.beispiele):
            print(f"    5-Gramm fehlt ({c}x): {g}")

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[CSV_COLS]
    path = config.PROCESSED / "validierung_extraktion.csv"
    df.to_csv(path, index=False)
    print(f"-> {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
