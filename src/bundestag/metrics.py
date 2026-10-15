"""Kennzahlen der Redebeiträge (Aufgabe „Analyse von Redebeiträgen“): je Abgeordnetem, je Fraktion und Phase, WPM,
Top-10-Gesetze.

Die Definitionen stehen an den Funktionen. Eingaben: data/processed/{reden, beitraege,
sitzungen, fraktionen, gesetze, gesetze_einheit, gesetz_beteiligte}.parquet, optional sentiment.parquet.
Aufruf: python -m bundestag.metrics
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import PROCESSED, TXT_DIR
from .stammdaten import norm_name

# Hauptstatistik nur in Debatten-TOPs: Befragung (2382 Reden-Zeilen) und Fragestunde (1135) sind Frage-Antwort-Formate
DEBATTE = ("Debatte", "Aktuelle Stunde", "Regierungserklärung")
PHASEN = ("koalition", "minderheit", "nach_vertrauensfrage")
# Ampel-Bruch am Abend des 06.11.2024 (Sitzung 196), Vertrauensfrage am 16.12.2024 (Sitzung 205)
KOALITION_BIS, MINDERHEIT_BIS = date(2024, 11, 6), date(2024, 12, 15)
FRAKTIONEN = ("SPD", "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN", "FDP", "AfD", "Die Linke", "BSW", "fraktionslos")
REG_MIT, REG_OHNE = "Regierungsbank mit Mandat", "Regierungsbank ohne Mandat"
LAENDER = "Bundesrat und Wehrbeauftragte"
GRUPPEN = FRAKTIONEN + (REG_MIT, REG_OHNE, LAENDER)
# Regierungsmitglieder ohne Mandat (Pistorius, Faeser, Geywitz, Kukies) haben nur eine Partei; für die Variante
# „Regierungsredezeit wird der Fraktion angerechnet" (H1) wird die Partei auf ihre Fraktion abgebildet
PARTEI_FRAKTION = {"SPD": "SPD", "CDU": "CDU/CSU", "CSU": "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
                   "FDP": "FDP"}
# LaTeX-Makronamen erlauben nur Buchstaben (zahlen.tex)
KURZ = {"SPD": "SPD", "CDU/CSU": "Union", "BÜNDNIS 90/DIE GRÜNEN": "Gruene", "FDP": "FDP", "AfD": "AfD",
        "Die Linke": "Linke", "BSW": "BSW", "fraktionslos": "Fraktionslos", REG_MIT: "RegMitMandat",
        REG_OHNE: "RegOhneMandat", LAENDER: "Laender", "alle": "Alle"}
PHASE_KURZ = {"gesamt": "", "koalition": "Koalition", "minderheit": "Minderheit",
              "nach_vertrauensfrage": "NachVertrauensfrage"}
SPALTEN = ["rede_id", "sitzung", "datum", "name", "partei", "fraktion", "rolle", "ist_mdb", "quelle_partei",
           "top_id", "top_typ", "redetyp", "woerter"]
# sentiment.parquet des BERT-Laufs heißt netto/label, die Metrik-Definition netto_rede/label_rede
SENTIMENT_ALIAS = {"netto": "netto_rede", "label": "label_rede"}
KENNZAHLEN_CSV = PROCESSED / "kennzahlen.csv"
LAUFZEITEN_CSV = PROCESSED / "laufzeiten_teil.csv"


@contextmanager
def laufzeit(schritt: str):
    """Misst einen Schritt und ersetzt seine Zeile in laufzeiten_teil.csv (Wiederholungen überschreiben)."""
    t0 = time.perf_counter()
    yield
    sek = round(time.perf_counter() - t0, 2)
    alt = pd.read_csv(LAUFZEITEN_CSV) if LAUFZEITEN_CSV.exists() else pd.DataFrame(columns=["schritt", "sekunden"])
    neu = pd.concat([alt[alt.schritt != schritt], pd.DataFrame([{"schritt": schritt, "sekunden": sek}])])
    neu.to_csv(LAUFZEITEN_CSV, index=False)


def kennzahl(schluessel: str, wert, einheit: str, quelle: str, ki=(np.nan, np.nan)) -> dict:
    return {"schluessel": schluessel, "wert": wert, "ki_unten": ki[0], "ki_oben": ki[1], "einheit": einheit,
            "quelle": quelle}


def schreibe_kennzahlen(zeilen: list[dict], skript: str) -> pd.DataFrame:
    """Ersetzt alle Zeilen des Skripts in kennzahlen.csv; Schlüssel nur aus Buchstaben und eindeutig."""
    neu = pd.DataFrame(zeilen, columns=["schluessel", "wert", "ki_unten", "ki_oben", "einheit", "quelle"])
    falsch = neu.schluessel[~neu.schluessel.str.fullmatch(r"[A-Za-z]+")]
    if len(falsch) or not neu.schluessel.is_unique:
        raise ValueError(f"Schlüssel ungültig oder doppelt: {list(falsch)[:5]} {list(neu.schluessel[neu.schluessel.duplicated()])[:5]}")
    alt = pd.read_csv(KENNZAHLEN_CSV) if KENNZAHLEN_CSV.exists() else neu.iloc[:0]
    alt = alt[~alt.quelle.str.startswith(skript + ":") & ~alt.schluessel.isin(neu.schluessel)]
    out = pd.concat([alt, neu], ignore_index=True)
    # ganze Zahlen ohne „.0", damit zahlen.tex sie direkt setzen kann
    out["wert"] = pd.Series([int(v) if pd.notna(v) and float(v).is_integer() else v for v in out.wert],
                            index=out.index, dtype=object)
    out.to_csv(KENNZAHLEN_CSV, index=False)
    return out


def speichere(df: pd.DataFrame, name: str, csv: bool = True) -> None:
    df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    if csv:
        df.to_csv(PROCESSED / f"{name}.csv", index=False)


def quartile(w: pd.Series) -> dict:
    """Mittel, Median, Q1, Q3, IQR einer Längenverteilung (Redelängen sind rechtsschief, daher Median + IQR)."""
    if w.empty:
        return {"woerter_mittel": np.nan, "woerter_median": np.nan, "woerter_q1": np.nan, "woerter_q3": np.nan,
                "woerter_iqr": np.nan}
    q1, q3 = w.quantile([.25, .75])
    return {"woerter_mittel": w.mean(), "woerter_median": w.median(), "woerter_q1": q1, "woerter_q3": q3,
            "woerter_iqr": q3 - q1}


def phase_von_datum(d) -> str:
    d = pd.Timestamp(d).date()
    return "koalition" if d <= KOALITION_BIS else "minderheit" if d <= MINDERHEIT_BIS else "nach_vertrauensfrage"


def bank(rolle: str, ist_mdb: bool, fraktion) -> str | None:
    """Sitzgruppe einer Rede: Fraktion bei Reden als MdB, sonst Regierungsbank (mit/ohne Mandat) oder Länder."""
    if rolle == "MdB":
        return fraktion if isinstance(fraktion, str) else None
    # Ländervertreter tragen das Land in Klammern („Ministerpräsident (Bayern)", 18 Plenarreden), die Wehrbeauftragte
    # spricht als Hilfsorgan des Bundestages (2) – beide weder Fraktion noch Bundesregierung
    if "(" in rolle or "Wehrbeauftragt" in rolle:
        return LAENDER
    return REG_MIT if ist_mdb else REG_OHNE


def _lies(name: str, spalten: list[str]) -> pd.DataFrame:
    """Nur benötigte Spalten lesen: reden.parquet hat 17 MB, fast alles davon Redetext."""
    p = PROCESSED / f"{name}.parquet"
    vorhanden = set(pq.read_schema(p).names)
    return pd.read_parquet(p, columns=[c for c in spalten if c in vorhanden])


def lade_reden() -> pd.DataFrame:
    r = _lies("reden", SPALTEN)
    if "ist_mdb" not in r:
        # reden.parquet älterer Parser-Läufe ohne ist_mdb: Restliste = Personen ohne Mandat (Regierung, Länder)
        r["ist_mdb"] = r.quelle_partei.ne("restliste")
    r["phase"] = r.datum.map(phase_von_datum)
    r["person"] = r.name.map(norm_name)
    r["bank"] = [bank(ro, m, f) for ro, m, f in zip(r.rolle, r.ist_mdb, r.fraktion)]
    r["haupt"] = r.redetyp.eq("plenarrede") & r.top_typ.isin(DEBATTE)
    return r


def lade_sitzungen() -> pd.DataFrame:
    s = pd.read_parquet(PROCESSED / "sitzungen.parquet")
    s["phase"] = s.datum.map(phase_von_datum)
    return s


def sitze(sitzungen: pd.DataFrame) -> pd.DataFrame:
    """Sitzzahl je Sitzung und Fraktion/Gruppe aus den Mitgliedschaftsintervallen in fraktionen.parquet."""
    f = pd.read_parquet(PROCESSED / "fraktionen.parquet", columns=["id", "fraktion", "von", "bis"])
    zeilen = []
    for nr, d in zip(sitzungen.sitzung, pd.to_datetime(sitzungen.datum).dt.date):
        z = f[(f.von <= d) & (f.bis.isna() | (f.bis >= d))]
        # Wechseltag 02.02.2024: „fraktionslos (Vorgriff)" endet an dem Tag, an dem die Gruppe beginnt; spätere gilt
        z = z.sort_values("von").groupby("id").tail(1)
        zeilen += [{"sitzung": nr, "fraktion": fr, "sitze": n} for fr, n in z.fraktion.value_counts().items()]
    return pd.DataFrame(zeilen)


def sitze_mittel(sitze_df: pd.DataFrame, sitzungen: pd.Series) -> pd.Series:
    """Mittlere Sitzzahl über die Sitzungsdaten = Sitz-Sitzungen / Zahl der Sitzungen (Exposure)."""
    return sitze_df[sitze_df.sitzung.isin(sitzungen)].groupby("fraktion").sitze.sum() / len(sitzungen)


def lade_sentiment(r: pd.DataFrame) -> pd.DataFrame | None:
    """netto_rede je rede_id aus sentiment.parquet (BERT-Lauf); None, solange die Datei fehlt."""
    p = PROCESSED / "sentiment.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p).rename(columns=SENTIMENT_ALIAS)
    if "netto_rede" not in s:
        s["netto_rede"] = s.p_pos - s.p_neg
    s = s.drop_duplicates("rede_id")[["rede_id", "netto_rede"]]
    return r.merge(s, on="rede_id", how="inner")


# Minuten sind keine gemessene Zeit, sondern Wörter / WPM; WPM stammt aus den angesagten Dauern
# („Für die Aussprache … Minuten“) der Aufruf-Blöcke
SATZ_RE = re.compile(r"[^.!?]*\b(?:Aussprache|Debatte)\b[^.!?]*")
VERB_RE = re.compile(r"vereinbart|vorgesehen|beschlossen|verabredet")
ZAHL = r"\d+|eine[rn]?|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|anderthalb|eineinhalb"
# „vier Stunden und 30 Minuten" (20186:25) zusammen lesen; „Drei-Minuten-Runde" (20160:4191, 20203:4069) nennt die
# Redezeit je Redner, keine Gesamtdauer: \s+ zwischen Zahl und Einheit und (?!-) schließen sie aus
DAUER_RE = re.compile(rf"\b(?:(?P<h>{ZAHL})\s+Stunden?(?:\s+und\s+(?P<hm>\d+)\s+Minuten)?|(?P<m>\d+)\s+Minuten)\b(?!-)",
                      re.I)
ZAHLWORT = {"eine": 1, "einer": 1, "einen": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
            "acht": 8, "neun": 9, "zehn": 10, "anderthalb": 1.5, "eineinhalb": 1.5}
# Haushaltswoche: „Für die heutige Aussprache … insgesamt acht Stunden" (20183:137, 20184:23, 20185:1487, 20186:25)
# gilt für alle Einzelpläne des Tages, nicht für einen Aufruf-Block
TAGESANSAGE_RE = re.compile(r"heutige|für Mittwoch|für Donnerstag")


def _zahl(x: str) -> float:
    return float(x) if x.isdigit() else ZAHLWORT[x.lower()]


def _praesidium(b: pd.DataFrame):
    """(sitzung, zeile_start, Zeilen) je Präsidiumsbeitrag des Sitzungsverlaufs; Klammern (Zwischenrufe) geleert."""
    pr = b[(b.redetyp == "praesidium") & b.anlage.isna()]
    for nr, g in pr.groupby("sitzung"):
        lines = (TXT_DIR / f"20{nr:03d}.txt").read_text(encoding="utf-8").splitlines()
        for zs, ze in zip(g.zeile_start, g.zeile_ende):
            yield nr, zs, ["" if t.startswith("(") else t for t in lines[zs - 1: ze]]


def ansagen(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dauer-Ansagen im Präsidiumstext (Fundstelle Sitzung + Zeile in data/txt/20NNN.txt) und Aufruf-Zeilen."""
    from .parse import AUFRUF_RE, FREI_RE, TOP_ENDE_RE, WIEDER_RE
    out, aufrufe = [], []
    for nr, zs, teile in _praesidium(b):
        for i, t in enumerate(teile):
            if AUFRUF_RE.search(t) or WIEDER_RE.search(t) or FREI_RE.match(t) or TOP_ENDE_RE.search(t):
                aufrufe.append({"sitzung": nr, "zeile": zs + i})
        # Absätze verbinden: 20193:279 bricht die Ansage nach „Für die Aussprache" um
        text = " ".join(teile)
        grenzen = np.cumsum([len(t) + 1 for t in teile])
        for s in SATZ_RE.finditer(text):
            m = DAUER_RE.search(s.group(0))
            if not m or not VERB_RE.search(s.group(0)):
                continue
            minuten = _zahl(m["h"]) * 60 + float(m["hm"] or 0) if m["h"] else float(m["m"])
            zeile = zs + int(np.searchsorted(grenzen, s.start() + m.start(), side="right"))
            kontext = " ".join(s.group(0)[max(0, m.start() - 70): m.end() + 30].split())
            out.append({"sitzung": nr, "zeile": zeile, "minuten_angesagt": minuten, "ansage": kontext,
                        "tagesansage": bool(TAGESANSAGE_RE.search(s.group(0)))})
    return pd.DataFrame(out), pd.DataFrame(aufrufe)


def wpm_bloecke(b: pd.DataFrame) -> pd.DataFrame:
    """Je Ansage: Aufruf-Block bis zum nächsten Aufruf, zum ersten Redner eines anderen TOP oder zur nächsten
    Ansage; WPM = Wörter der plenarrede-Beiträge / angesagte Minuten."""
    ans, aufrufe = ansagen(b)
    verlauf = b[b.anlage.isna()].sort_values(["sitzung", "zeile_start"])
    # TOPs mit zu Protokoll gegebenen Reden schöpfen die Ansage nicht aus: 114 Blöcke, WPM-Median 81 statt 138
    zu_protokoll = set(zip(*b[b.redetyp == "zu_protokoll"][["sitzung", "top_id"]].dropna().to_numpy().T))
    zeilen = []
    for nr, g in ans.groupby("sitzung"):
        bs = verlauf[verlauf.sitzung == nr]
        grenzen = sorted(set(g.zeile) | set(aufrufe.zeile[aufrufe.sitzung == nr]))
        for z, minuten, text, tag in zip(g.zeile, g.minuten_angesagt, g.ansage, g.tagesansage):
            nach = bs[bs.zeile_start > z]
            # 411 von 419 Ansagen stehen im Präsidiumsbeitrag, der noch die top_id des vorigen TOP trägt
            redner = nach[nach.redetyp != "praesidium"]
            row = {"sitzung": nr, "zeile": z, "minuten_angesagt": minuten, "ansage": text}
            if redner.empty:
                zeilen.append(row | {"grund": "kein Redner danach"})
                continue
            top, typ = redner.top_id.iloc[0], redner.top_typ.iloc[0]
            anders = redner[redner.top_id.ne(top) | redner.top_id.isna()]
            # 20181:1441 (68 Minuten) lief ohne Aufruf-Grenze über 46 Plenarreden desselben top_id weiter
            ende = min(anders.zeile_start.iloc[0] if len(anders) else np.inf,
                       min((a for a in grenzen if a > z), default=np.inf))
            blk = nach[(nach.zeile_start < ende) & (nach.redetyp == "plenarrede")]
            woerter = int(blk.woerter.sum())
            grund = ("Tagesansage Haushaltswoche" if tag else "ohne TOP" if pd.isna(top) else
                     # Regierungserklärung (7 Blöcke) steht vor der angesagten Aussprache und gehört nicht zur Dauer
                     "Regierungserklärung" if typ == "Regierungserklärung" else
                     "Reden zu Protokoll im TOP" if (nr, top) in zu_protokoll else
                     "keine Plenarrede" if woerter == 0 else "")
            zeilen.append(row | {"top_id": top, "top_typ": typ, "erster_kopf": int(redner.zeile_start.iloc[0]),
                                 "ende_zeile": None if ende == np.inf else int(ende), "reden_n": blk.rede_id.nunique(),
                                 "woerter": woerter, "wpm": woerter / minuten, "grund": grund})
    w = pd.DataFrame(zeilen)
    # mehrere Ansagen vor demselben ersten Redner („erneut", „wiederum"): gleiche Dauer zählt einmal, abweichende gar nicht
    dup = w.erster_kopf.notna() & w.duplicated(["sitzung", "erster_kopf"], keep=False)
    for _, g in w[dup].groupby(["sitzung", "erster_kopf"]):
        konflikt = g.minuten_angesagt.nunique() > 1
        w.loc[g.index if konflikt else g.index[1:], "grund"] = "abweichende Doppelansage" if konflikt else "Doppelansage"
    w["verwendet"] = w.grund.eq("")
    return w


def wpm_sitzungen(sitzungen: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Untergrenze: gesprochene Wörter je Sitzung / (Schluss − Beginn); enthält Abstimmungen und Pausen."""
    gesprochen = b[b.anlage.isna()].groupby("sitzung").woerter.sum()
    beginn = pd.to_datetime(sitzungen.beginn, format="%H:%M")
    schluss = pd.to_datetime(sitzungen.schluss, format="%H:%M")
    dauer = (schluss - beginn).dt.total_seconds() / 60
    # Sitzung 210 endet nach Mitternacht (09:00 bis 02:19)
    dauer = dauer.where(dauer > 0, dauer + 24 * 60)
    s = sitzungen[["sitzung", "datum", "beginn", "schluss"]].assign(dauer_min=dauer.to_numpy())
    s["woerter_gesprochen"] = s.sitzung.map(gesprochen).fillna(0).astype(int)
    s["wpm_untergrenze"] = s.woerter_gesprochen / s.dauer_min
    return s


def partei_perioden(stamm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stammdaten-Perioden (name_key, partei, von) mit Partei wie in der Ausgabe ("Plos" → parteilos)."""
    if stamm is None:
        p = PROCESSED / "fraktionen.parquet"
        stamm = (pd.read_parquet(p, columns=["name_key", "partei", "von"]) if p.exists()
                 else pd.DataFrame(columns=["name_key", "partei", "von"]))
    return (stamm.dropna(subset=["partei", "von"])
            .assign(partei=lambda d: d.partei.replace({"Plos": "parteilos"}), von=lambda d: d.von.astype(str).str[:10]))


def abgeordnete(r: pd.DataFrame, b: pd.DataFrame, wpm: float, sent: pd.DataFrame | None,
                stamm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Eine Zeile je Person (norm_name): Häufigkeit, Länge (Hauptstatistik), Minuten, Sentiment."""
    h = r[r.haupt]
    g = r.groupby("person")
    out = pd.DataFrame({"name": g.name.agg(lambda s: s.mode().iloc[0]),
                        # Stammdaten-Code "Plos" (5 Personen in reden.parquet, u. a. Wissing nach dem FDP-Austritt)
                        # heißt in der Ausgabe parteilos wie bei Badenberg, sonst stünden zwei Codes für dieselbe Angabe
                        "partei": g.partei.agg(lambda s: s.replace({"Plos": "parteilos"}).mode().iloc[0] if s.notna().any() else None),
                        "ist_mdb": g.ist_mdb.any(),
                        "regierungsamt": g.bank.agg(lambda s: s.isin([REG_MIT, REG_OHNE]).any())})
    fr = (r.dropna(subset=["fraktion"]).groupby(["person", "fraktion"])
          .agg(h=("haupt", "sum"), n=("haupt", "size"), erste=("sitzung", "min")).reset_index())
    # Hauptfraktion = meiste Hauptstatistik-Reden, bei Gleichstand meiste Beiträge (Wissing: FDP bis 196, dann fraktionslos)
    out["fraktion"] = fr.sort_values(["h", "n"], ascending=False).groupby("person").fraktion.first()
    out["fraktion_wechsel"] = (fr.sort_values("erste").groupby("person").fraktion
                               .agg(lambda s: " → ".join(s) if len(s) > 1 else ""))
    # Parteiwechsel mit Datum, weil partei nur den häufigsten Wert zeigt ("FDP → parteilos ab 2024-11-08"). "ab" ist der
    # Beginn der Stammdaten-Periode, in die die erste Rede mit der neuen Partei fällt, nicht diese Rede selbst
    # (Wissing sprach erst am 2024-12-19 wieder); ohne Stammdaten-Periode bleibt das Redendatum.
    pw = (r.dropna(subset=["partei"]).assign(partei=lambda d: d.partei.replace({"Plos": "parteilos"}),
                                             datum=lambda d: d.datum.astype(str).str[:10])
          .groupby(["person", "partei"]).datum.min().reset_index())
    m = pw.merge(partei_perioden(stamm), left_on=["person", "partei"], right_on=["name_key", "partei"], how="left")
    ab = m[m.von <= m.datum].groupby(["person", "partei"]).von.max()
    pw["ab"] = [ab.get((p, q), d) for p, q, d in zip(pw.person, pw.partei, pw.datum)]
    pw = pw.sort_values("datum")
    out["partei_wechsel"] = pw.groupby("person").apply(
        lambda d: " → ".join([d.partei.iloc[0]] + [f"{p} ab {t}" for p, t in zip(d.partei.iloc[1:], d.ab.iloc[1:])])
        if len(d) > 1 else "", include_groups=False)
    out["partei_wechsel"] = out["partei_wechsel"].fillna("")
    zaehle = {"reden_n": h, "kurzinterventionen_n": r[r.redetyp == "kurzintervention"],
              "erwiderungen_n": r[r.redetyp == "erwiderung"], "zwischenfragen_n": r[r.redetyp == "zwischenfrage"],
              "befragung_n": r[r.redetyp.isin(["befragung_frage", "befragung_antwort"])],
              "zu_protokoll_n": r[r.redetyp.isin(["zu_protokoll", "erklaerung_go31"])]}
    for spalte, d in zaehle.items():
        out[spalte] = d.groupby("person").size()
    bb = b[b.redetyp != "praesidium"]
    out["beitraege_n"] = bb.name.map(norm_name).value_counts()
    w = h.groupby("person").woerter
    out["woerter_gesamt"] = w.sum()
    out["woerter_mittel"], out["woerter_median"] = w.mean(), w.median()
    out["woerter_iqr"] = w.quantile(.75) - w.quantile(.25)
    zahl = [c for c in out if c.endswith("_n")] + ["woerter_gesamt"]
    out[zahl] = out[zahl].fillna(0).astype(int)
    out["minuten_gesamt"] = out.woerter_gesamt / wpm
    out["minuten_mittel"] = out.woerter_mittel / wpm
    out["sentiment_netto_mittel"] = (sent[sent.haupt].groupby("person").netto_rede.mean()
                                     if sent is not None else np.nan)
    out = out.reset_index().sort_values(["reden_n", "woerter_gesamt"], ascending=False)
    spalten = ["name", "partei", "partei_wechsel", "fraktion", "fraktion_wechsel", "ist_mdb", "regierungsamt", "reden_n",
               "kurzinterventionen_n", "erwiderungen_n", "zwischenfragen_n", "befragung_n", "zu_protokoll_n",
               "beitraege_n", "woerter_gesamt", "woerter_mittel", "woerter_median", "woerter_iqr", "minuten_gesamt",
               "minuten_mittel", "sentiment_netto_mittel", "person"]
    return out[spalten]


def fraktionen_kennzahlen(r: pd.DataFrame, sitze_df: pd.DataFrame, sitzungen: pd.DataFrame,
                          wpm: float) -> pd.DataFrame:
    """Fraktionstabelle: je Phase und Sitzgruppe Reden, Wörter, Minuten roh und pro Sitz, Länge, Kurzinterventionen."""
    zeilen = []
    for ph in ("gesamt",) + PHASEN:
        rr = r if ph == "gesamt" else r[r.phase == ph]
        sz = sitzungen.sitzung if ph == "gesamt" else sitzungen.sitzung[sitzungen.phase == ph]
        if sz.empty:
            continue
        sm = sitze_mittel(sitze_df, sz)
        h, ki = rr[rr.haupt], rr[rr.redetyp == "kurzintervention"]
        w_fr = h[h.bank.isin(FRAKTIONEN)].woerter.sum()
        for gr in GRUPPEN + ("alle",):
            hg = h if gr == "alle" else h[h.bank == gr]
            # Summenzeile ohne pro-Sitz-Werte: ihr Zähler enthält Regierungsbank und Länder ohne Sitz
            s = np.nan if gr not in FRAKTIONEN else sm.get(gr, np.nan)
            n_ki = len(ki) if gr == "alle" else int((ki.bank == gr).sum())
            zeilen.append({"phase": ph, "gruppe": gr, "sitzungen_n": len(sz), "sitze_mittel": s,
                           "sitzanteil": s / sm.sum() if gr in FRAKTIONEN else np.nan,
                           "redner_n": hg.person.nunique(), "reden_n": len(hg), "woerter": int(hg.woerter.sum()),
                           "minuten": hg.woerter.sum() / wpm,
                           "woerteranteil": hg.woerter.sum() / w_fr if gr in FRAKTIONEN else np.nan,
                           "reden_pro_sitz": len(hg) / s, "woerter_pro_sitz": hg.woerter.sum() / s,
                           "minuten_pro_sitz": hg.woerter.sum() / wpm / s,
                           # Phasen sind 51, 8 und 8 Sitzungen lang; je Sitzung macht sie vergleichbar
                           "reden_pro_sitz_je_sitzung": len(hg) / s / len(sz),
                           **quartile(hg.woerter),
                           "kurzinterventionen_n": n_ki, "kurzinterventionen_anteil": n_ki / len(ki) if len(ki) else np.nan,
                           "kurzinterventionen_pro_sitz": n_ki / s})
    return pd.DataFrame(zeilen)


# Top-10-Gesetze nach Rednerzahl beantworten die Leitfrage der Aufgabenstellung „Welche Gesetze wurden diskutiert
# und wer hat daran mitgewirkt?“
LESUNG_23 = {"zweite", "dritte", "zweite und dritte"}
ZUSTROM = "20/12804"  # Beispiel der Aufgabenstellung (Sitzung 211, ZP 35), Zeile immer ausgeben


def top10_gesetze(r: pd.DataFrame) -> pd.DataFrame:
    e = pd.read_parquet(PROCESSED / "gesetze_einheit.parquet")
    g = _lies("gesetze", ["drucksache", "sitzung", "beratungsstufe"])
    gb = _lies("gesetz_beteiligte", ["rede_id", "drucksache", "redetyp"])
    gb = gb[gb.redetyp == "plenarrede"].merge(r[["rede_id", "fraktion", "partei"]], on="rede_id", how="left")
    gb["fr"] = gb.fraktion.fillna(gb.partei.map(PARTEI_FRAKTION)).fillna("ohne Fraktion")
    # Beschlussempfehlungen „hier: Einzelplan 23 …" der Haushaltswoche 149–152 führt parse.py als Gesetzentwurf; sie
    # gehören zum Haushaltsgesetz 2024 (20/7800) und belegten sonst 8 der Top-10-Plätze
    e = e[~e.titel.fillna("").str.startswith("hier: Einzelplan")].copy()
    # verbundene Gegenstände teilen alle Reden (Aufruf-Erbe): eine Zeile je identischer Redenmenge
    e["sig"] = e.drucksache.map(gb.groupby("drucksache").rede_id.agg(lambda x: tuple(sorted(set(x)))))
    e["verbunden_mit"] = e.apply(lambda x: ", ".join(e.drucksache[(e.sig == x.sig) & (e.drucksache != x.drucksache)])
                                 if isinstance(x.sig, tuple) else "", axis=1)
    e = e.sort_values(["n_redner", "n_reden", "drucksache"], ascending=[False, False, True])
    e = e[~(e.sig.map(lambda x: isinstance(x, tuple)) & e.sig.duplicated() & e.drucksache.ne(ZUSTROM))]
    e = e.reset_index(drop=True)
    e["rang"] = np.arange(1, len(e) + 1)
    sel = e[(e.rang <= 10) | (e.drucksache == ZUSTROM)].copy()

    def lesung(d: str, stufen: set) -> str:
        s = sorted(g[(g.drucksache == d) & g.beratungsstufe.isin(stufen)].sitzung.unique())
        return ", ".join(map(str, s))

    def fraktionen(d: str) -> str:
        n = gb[gb.drucksache == d].fr.value_counts()
        return "; ".join(f"{k} {v}" for k, v in n.items())

    sel["sitzungen_erste_lesung"] = sel.drucksache.map(lambda d: lesung(d, {"erste"}))
    sel["sitzungen_zweite_dritte"] = sel.drucksache.map(lambda d: lesung(d, LESUNG_23))
    sel["fraktionen_reden"] = sel.drucksache.map(fraktionen)
    sel["beispiel"] = sel.drucksache.eq(ZUSTROM)
    sel["sitzungen"] = sel.sitzungen.map(lambda s: ", ".join(map(str, json.loads(s))))
    return sel[["rang", "drucksache", "kurztitel", "titel", "sitzungen", "sitzungen_erste_lesung",
                "sitzungen_zweite_dritte", "beschlussart_final", "n_redner", "n_reden", "fraktionen_reden",
                "verbunden_mit", "gegenstaende_im_top_max", "beispiel"]]


def kennzahlen_metrics(r, abg, fk, w, ws, top) -> list[dict]:
    q = "metrics.py"
    h = r[r.haupt]
    v = w[w.verwendet]
    einheit = pd.read_parquet(PROCESSED / "gesetze_einheit.parquet", columns=["drucksache", "titel"])
    z = [kennzahl("redenGesamt", len(h), "Reden", f"{q}:reden"),
         kennzahl("plenarredenGesamt", int((r.redetyp == "plenarrede").sum()), "Reden", f"{q}:reden"),
         kennzahl("plenarredenOhneDebatte", int(((r.redetyp == "plenarrede") & ~r.haupt).sum()), "Reden",
                  f"{q}:reden"),
         kennzahl("kurzinterventionenGesamt", int((r.redetyp == "kurzintervention").sum()), "Beiträge", f"{q}:reden"),
         kennzahl("erwiderungenGesamt", int((r.redetyp == "erwiderung").sum()), "Beiträge", f"{q}:reden"),
         kennzahl("zwischenfragenGesamt", int((r.redetyp == "zwischenfrage").sum()), "Beiträge", f"{q}:reden"),
         kennzahl("befragungGesamt", int(r.redetyp.isin(["befragung_frage", "befragung_antwort"]).sum()), "Beiträge",
                  f"{q}:reden"),
         kennzahl("zuProtokollGesamt", int(r.redetyp.isin(["zu_protokoll", "erklaerung_go31"]).sum()), "Reden",
                  f"{q}:reden"),
         kennzahl("beitraegeGesamt", int(abg.beitraege_n.sum()), "Beiträge", f"{q}:abgeordnete"),
         kennzahl("personenGesamt", len(abg), "Personen", f"{q}:abgeordnete"),
         kennzahl("rednerMitRede", int((abg.reden_n > 0).sum()), "Personen", f"{q}:abgeordnete"),
         kennzahl("abgeordneteMitRede", int(((abg.reden_n > 0) & abg.ist_mdb).sum()), "Personen", f"{q}:abgeordnete"),
         kennzahl("abgeordneteGesamt", int(abg.ist_mdb.sum()), "Personen", f"{q}:abgeordnete"),
         kennzahl("ohneMandatMitRede", int(((abg.reden_n > 0) & ~abg.ist_mdb).sum()), "Personen", f"{q}:abgeordnete"),
         kennzahl("woerterGesamt", int(h.woerter.sum()), "Wörter", f"{q}:reden"),
         kennzahl("woerterMittel", h.woerter.mean(), "Wörter", f"{q}:reden"),
         kennzahl("woerterMedian", h.woerter.median(), "Wörter", f"{q}:reden"),
         kennzahl("woerterQEins", h.woerter.quantile(.25), "Wörter", f"{q}:reden"),
         kennzahl("woerterQDrei", h.woerter.quantile(.75), "Wörter", f"{q}:reden"),
         kennzahl("minutenGesamt", h.woerter.sum() / v.wpm.median(), "Minuten", f"{q}:reden"),
         kennzahl("wpmMedian", v.wpm.median(), "Wörter/Minute", f"{q}:wpm"),
         kennzahl("wpmQEins", v.wpm.quantile(.25), "Wörter/Minute", f"{q}:wpm"),
         kennzahl("wpmQDrei", v.wpm.quantile(.75), "Wörter/Minute", f"{q}:wpm"),
         kennzahl("wpmIqr", v.wpm.quantile(.75) - v.wpm.quantile(.25), "Wörter/Minute", f"{q}:wpm"),
         kennzahl("wpmN", len(v), "Blöcke", f"{q}:wpm"),
         kennzahl("wpmMedianMitZuProtokoll", w[w.grund.isin(["", "Reden zu Protokoll im TOP"])].wpm.median(),
                  "Wörter/Minute", f"{q}:wpm"),
         kennzahl("wpmZuProtokollAusgeschlossen", int((w.grund == "Reden zu Protokoll im TOP").sum()), "Blöcke",
                  f"{q}:wpm"),
         kennzahl("wpmAnsagen", len(w), "Ansagen", f"{q}:wpm"),
         kennzahl("wpmUntergrenzeMedian", ws.wpm_untergrenze.median(), "Wörter/Minute", f"{q}:wpm_sitzungen"),
         kennzahl("wpmUntergrenzeMin", ws.wpm_untergrenze.min(), "Wörter/Minute", f"{q}:wpm_sitzungen"),
         kennzahl("wpmUntergrenzeMax", ws.wpm_untergrenze.max(), "Wörter/Minute", f"{q}:wpm_sitzungen"),
         kennzahl("sitzungenGesamt", len(ws), "Sitzungen", f"{q}:sitzungen"),
         kennzahl("gesetzeEinheitGesamt", len(einheit), "Gesetzentwürfe", f"{q}:gesetze_einheit"),
         kennzahl("gesetzeEinzelplanAusgeschlossen", int(einheit.titel.fillna("").str.startswith("hier: Einzelplan")
                                                         .sum()), "Drucksachen", f"{q}:top10_gesetze")]
    if top.beispiel.any():
        zs = top[top.beispiel].iloc[0]
        z += [kennzahl("zustromRang", int(zs.rang), "Rang", f"{q}:top10_gesetze"),
              kennzahl("zustromRedner", int(zs.n_redner), "Personen", f"{q}:top10_gesetze"),
              kennzahl("zustromReden", int(zs.n_reden), "Reden", f"{q}:top10_gesetze")]
    else:   # Teillauf ohne Sitzung 211 (z. B. 146–147): IndexError statt Kennzahlen
        print(f"Hinweis: Beispielgesetz {ZUSTROM} nicht im Bereich, zustromRang/-Redner/-Reden entfallen")
    for _, x in fk[fk.phase == "gesamt"].iterrows():
        k = KURZ[x.gruppe]
        z += [kennzahl(f"reden{k}", int(x.reden_n), "Reden", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"woerter{k}", int(x.woerter), "Wörter", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"minuten{k}", x.minuten, "Minuten", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"woerterMedian{k}", x.woerter_median, "Wörter", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"woerterQEins{k}", x.woerter_q1, "Wörter", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"woerterQDrei{k}", x.woerter_q3, "Wörter", f"{q}:fraktionen_kennzahlen"),
              kennzahl(f"kurzinterventionen{k}", int(x.kurzinterventionen_n), "Beiträge", f"{q}:fraktionen_kennzahlen")]
        if x.gruppe in FRAKTIONEN:
            z += [kennzahl(f"sitze{k}", x.sitze_mittel, "Sitze", f"{q}:fraktionen_kennzahlen"),
                  kennzahl(f"minutenProSitz{k}", x.minuten_pro_sitz, "Minuten/Sitz", f"{q}:fraktionen_kennzahlen"),
                  kennzahl(f"kurzinterventionenProSitz{k}", x.kurzinterventionen_pro_sitz, "Beiträge/Sitz",
                           f"{q}:fraktionen_kennzahlen")]
    return z


def main() -> None:
    with laufzeit("metrics_laden"):
        r = lade_reden()
        b = _lies("beitraege", ["sitzung", "zeile_start", "zeile_ende", "name", "redetyp", "top_id", "top_typ",
                                "anlage", "rede_id", "woerter"])
        sitzungen = lade_sitzungen()
        sitze_df = sitze(sitzungen)
        sent = lade_sentiment(r)
    with laufzeit("metrics_wpm"):
        w = wpm_bloecke(b)
        ws = wpm_sitzungen(sitzungen, b)
        w.to_csv(PROCESSED / "wpm.csv", index=False)
        ws.to_csv(PROCESSED / "wpm_sitzungen.csv", index=False)
        wpm = w[w.verwendet].wpm.median()
    with laufzeit("metrics_abgeordnete"):
        abg = abgeordnete(r, b, wpm, sent)
        speichere(abg, "abgeordnete")
        top20 = abg.head(20)[["name", "partei", "fraktion", "reden_n", "woerter_gesamt", "woerter_mittel",
                              "woerter_median", "minuten_gesamt"]]
        top20.insert(0, "rang", range(1, len(top20) + 1))
        top20.to_csv(PROCESSED / "top20_redner.csv", index=False)
    with laufzeit("metrics_fraktionen"):
        fk = fraktionen_kennzahlen(r, sitze_df, sitzungen, wpm)
        speichere(fk, "fraktionen_kennzahlen")
        sitze_df.to_parquet(PROCESSED / "sitze_je_sitzung.parquet", index=False)
    with laufzeit("metrics_top10_gesetze"):
        top = top10_gesetze(r)
        top.to_csv(PROCESSED / "top10_gesetze.csv", index=False)
    schreibe_kennzahlen(kennzahlen_metrics(r, abg, fk, w, ws, top), "metrics.py")
    v = w[w.verwendet].wpm
    print(f"Reden (Hauptstatistik) {int(r.haupt.sum())}, Personen {len(abg)}, MdB mit Rede "
          f"{int(((abg.reden_n > 0) & abg.ist_mdb).sum())}")
    print(f"WPM Median {v.median():.1f}, IQR {v.quantile(.25):.1f}–{v.quantile(.75):.1f}, n {len(v)} von {len(w)} "
          f"Ansagen; Untergrenze Sitzung Median {ws.wpm_untergrenze.median():.1f}")
    print(w[~w.verwendet].grund.value_counts().to_string())


if __name__ == "__main__":
    main()
