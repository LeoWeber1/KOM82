"""Partei/Fraktion/Gruppe je Redner und Sitzungsdatum (MdB-Stammdaten + Restliste)."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from lxml import etree

from .config import PROCESSED, STAMMDATEN_XML, TXT_DIR, XML_DIR, WAHLPERIODE

FRAKTIONEN_PQ = PROCESSED / "fraktionen.parquet"
RESTLISTE_CSV = PROCESSED / "restliste.csv"

# INS_LANG-Werte der 9 Fraktion/Gruppe-Institutionen in WP 20 (Stammdaten-Stand 29.04.2025)
INS_KURZ = {
    "Fraktion der Sozialdemokratischen Partei Deutschlands": "SPD",
    "Fraktion der Christlich Demokratischen Union/Christlich - Sozialen Union": "CDU/CSU",
    "Fraktion BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
    "Fraktion der Freien Demokratischen Partei": "FDP",
    "Fraktion Alternative für Deutschland": "AfD",
    "Fraktion DIE LINKE.": "Die Linke",
    "Gruppe Die Linke": "Die Linke",
    "Gruppe BSW - Bündnis Sahra Wagenknecht - Vernunft und Gerechtigkeit": "BSW",
    "Fraktionslos": "fraktionslos",
}
GRUPPEN = {"Gruppe Die Linke": "Die Linke",
           "Gruppe BSW - Bündnis Sahra Wagenknecht - Vernunft und Gerechtigkeit": "BSW"}

# Die Fraktion DIE LINKE. endet laut Stammdaten am 05.12.2023 (MDBINS_BIS), die
# Gruppen Die Linke und BSW werden erst ab 02.02.2024 (MDBINS_VON) geführt. Dazwischen
# stehen alle 38 Betroffenen als "Fraktionslos" (06.12.2023–02.02.2024); das trifft die
# Sitzungen 146–149 (17.01.–30.01.2024), deren Köpfe "(fraktionslos)" lauten. Für die
# Analyse wird diese Lücke der späteren Gruppe zugeschlagen (quelle "...+gruppe_vorgriff").
GRUPPEN_START = date(2024, 2, 2)
LINKE_FRAKTION_ENDE = date(2023, 12, 5)

FRAKTION_KOPF = {
    "cdu/csu": "CDU/CSU", "spd": "SPD", "afd": "AfD", "fdp": "FDP",
    "bündnis 90/die grünen": "BÜNDNIS 90/DIE GRÜNEN", "die linke": "Die Linke",
    "die linke.": "Die Linke", "bsw": "BSW", "fraktionslos": "fraktionslos",
}
PARTEI_VON_FRAKTION = {"SPD": "SPD", "BÜNDNIS 90/DIE GRÜNEN": "BÜNDNIS 90/DIE GRÜNEN",
                       "FDP": "FDP", "AfD": "AfD", "Die Linke": "Die Linke", "BSW": "BSW"}

TITEL = re.compile(r"\b(?:Dr\.-Ing|Dr|Prof|h\.\s?c|Dipl\.-\w+|mult)\.?(?=\s|$)", re.I)
# Köpfe wie "Carsten Müller (Braunschweig) (CDU/CSU):" tragen den Ortszusatz der Stammdaten
KOPF_ABG = re.compile(r"^(?P<name>[^():,\[\]]{3,70}?)(?: \((?P<ort>[^()]{2,30})\))? \((?P<fr>[^()]{2,40})\):$")
KOPF_REG = re.compile(r"^(?P<name>[^():,\[\]]{3,70}?), (?P<rolle>(?:Bundes|Staats|Parl\. Staats|Minister|Senator|"
                      r"Wehrbeauftragt|Erste|Regierende|Bürgermeister)[^:]{0,160}):$")


ROLLE = r"(?:Bundes|Staats|Parl\. Staats|Minister|Senator|Wehrbeauftragt|Erste|Regierende|Bürgermeister)"
KOPF_REG_START = re.compile(r"^[^():,\[\]]{3,70}?, " + ROLLE)


def norm_name(s: str) -> str:
    s = TITEL.sub(" ", s)
    s = unicodedata.normalize("NFKC", s).lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[-‐–]", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def _d(s: str | None) -> date | None:
    return datetime.strptime(s, "%d.%m.%Y").date() if s else None


def _parse() -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    root = etree.parse(str(STAMMDATEN_XML), parser).getroot()
    rows, alias = [], []
    for mdb in root.iter("MDB"):
        wp20 = [w for w in mdb.iter("WAHLPERIODE") if w.findtext("WP") == str(WAHLPERIODE)]
        if not wp20:
            continue
        mid = mdb.findtext("ID")
        partei_kurz = mdb.findtext("BIOGRAFISCHE_ANGABEN/PARTEI_KURZ")
        namen = list(mdb.iter("NAME"))
        for n in namen:
            teile = [n.findtext(t) for t in ("VORNAME", "ADEL", "PRAEFIX", "NACHNAME")]
            voll = " ".join(t for t in teile if t)
            alias.append({"id": mid, "name": voll, "ort": n.findtext("ORTSZUSATZ") or "",
                          "key": norm_name(voll)})
        akt = namen[-1]
        name = " ".join(t for t in (akt.findtext(x) for x in ("VORNAME", "ADEL", "PRAEFIX", "NACHNAME")) if t)
        for w in wp20:
            for ins in w.iter("INSTITUTION"):
                if ins.findtext("INSART_LANG") != "Fraktion/Gruppe":
                    continue
                lang = ins.findtext("INS_LANG")
                fr = INS_KURZ.get(lang, lang)
                partei = PARTEI_VON_FRAKTION.get(fr)
                if fr == "CDU/CSU":
                    partei = partei_kurz if partei_kurz in ("CDU", "CSU") else "CDU/CSU"
                if partei is None:
                    # fraktionslos: nur aktueller PARTEI_KURZ verfügbar (z. B. Wissing "Plos")
                    partei = partei_kurz
                rows.append({"id": mid, "nachname": akt.findtext("NACHNAME"), "vorname": akt.findtext("VORNAME"),
                             "name": name, "partei": partei, "fraktion": fr, "gruppe": GRUPPEN.get(lang),
                             "ins_lang": lang, "von": _d(ins.findtext("MDBINS_VON")),
                             "bis": _d(ins.findtext("MDBINS_BIS")), "vorgriff": False})
    df = pd.DataFrame(rows)
    # Lücke 06.12.2023–02.02.2024 der späteren Gruppe zuordnen (siehe GRUPPEN_START)
    for mid, g in df.groupby("id"):
        grp = g[g.gruppe.notna() & (g.von == GRUPPEN_START)]
        if grp.empty:
            continue
        lose = g[(g.fraktion == "fraktionslos") & (g.von == LINKE_FRAKTION_ENDE.replace(day=6))]
        for i in lose.index:
            gr = grp.iloc[0]
            df.loc[i, ["fraktion", "gruppe", "partei", "vorgriff"]] = [gr.fraktion, gr.gruppe, gr.partei, True]
    df["name_key"] = df["name"].map(norm_name)
    return df, pd.DataFrame(alias).drop_duplicates()


# Personen ohne Mandat in WP 20, die in den Sitzungen 146–212 als Redner auftreten (aus
# measure(): Köpfe ohne Stammdaten-Treffer). Partei: öffentlich bekannte Zugehörigkeit,
# Funktion laut Rednerkopf.
RESTLISTE = [
    # name, partei, funktion
    ("Boris Pistorius", "SPD", "Bundesminister der Verteidigung"),
    ("Nancy Faeser", "SPD", "Bundesministerin des Innern und für Heimat"),
    ("Klara Geywitz", "SPD", "Bundesministerin für Wohnen, Stadtentwicklung und Bauwesen"),
    ("Jörg Kukies", "SPD", "Bundesminister der Finanzen"),
    ("Bodo Ramelow", "Die Linke", "Ministerpräsident (Thüringen)"),
    ("Anke Rehlinger", "SPD", "Ministerpräsidentin (Saarland)"),
    ("Stephan Weil", "SPD", "Ministerpräsident (Niedersachsen)"),
    ("Markus Söder", "CSU", "Ministerpräsident (Bayern)"),
    ("Boris Rhein", "CDU", "Ministerpräsident (Hessen)"),
    ("Judith Gerlach", "CSU", "Staatsministerin (Bayern)"),
    ("Armin Schuster", "CDU", "Staatsminister (Sachsen)"),
    ("Thomas Strobl", "CDU", "Minister (Baden-Württemberg)"),
    ("Marion Gentges", "CDU", "Ministerin (Baden-Württemberg)"),
    ("Ingmar Jung", "CDU", "Staatsminister (Hessen)"),
    ("Wolfgang Schmidt", "SPD", "Bundesminister für besondere Aufgaben"),
    ("Karl-Josef Laumann", "CDU", "Minister (Nordrhein-Westfalen)"),
    ("Alexander Schweitzer", "SPD", "Ministerpräsident (Rheinland-Pfalz)"),
    ("Christian Pegel", "SPD", "Minister (Mecklenburg-Vorpommern)"),
    ("Eva Högl", "SPD", "Wehrbeauftragte des Deutschen Bundestages"),
    # UNSICHER: Badenberg ist parteilos, auf CDU-Vorschlag Justizsenatorin
    ("Felor Badenberg", "parteilos", "Senatorin (Berlin)"),
]


def build() -> None:
    df, alias = _parse()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FRAKTIONEN_PQ, index=False)
    # CSV neben dem Parquet wie bei allen Abgabetabellen, damit die Daten-ZIP ohne pyarrow lesbar ist
    df.to_csv(FRAKTIONEN_PQ.with_suffix(".csv"), index=False)
    alias.to_parquet(PROCESSED / "stammdaten_namen.parquet", index=False)
    pd.DataFrame(RESTLISTE, columns=["name", "partei", "funktion"]).assign(
        name_key=lambda d: d.name.map(norm_name), quelle="manuell").to_csv(RESTLISTE_CSV, index=False)
    _tables.cache_clear()


@lru_cache(maxsize=1)
def _tables():
    if not FRAKTIONEN_PQ.exists():
        build()
    df = pd.read_parquet(FRAKTIONEN_PQ)
    alias = pd.read_parquet(PROCESSED / "stammdaten_namen.parquet")
    rest = pd.read_csv(RESTLISTE_CSV)
    return df, alias, rest


def _kandidaten(name: str, ort: str | None) -> list[str]:
    _, alias, _ = _tables()
    key = norm_name(name)
    hit = alias[alias.key == key]
    if hit.empty:
        # Köpfe nennen Doppelnamen gelegentlich verkürzt; nur Vor- und Nachname exakt gleich
        tk = key.split()
        if len(tk) >= 2:
            hit = alias[alias.key.map(lambda k: k.split()[:1] == tk[:1] and k.split()[-1:] == tk[-1:])]
    if ort and hit.id.nunique() > 1:
        o = hit[hit.ort == f"({ort})"]
        if not o.empty:
            hit = o
    return list(dict.fromkeys(hit.id))


def lookup(name: str, datum: date, fraktion_pdf: str | None = None, ort: str | None = None) -> dict:
    df, _, rest = _tables()
    fr_kopf = FRAKTION_KOPF.get((fraktion_pdf or "").strip().lower())
    ids = _kandidaten(name, ort)
    zeilen = df[df.id.isin(ids) & (df.von <= datum) & (df.bis.isna() | (df.bis >= datum))]
    # Überlappung am Wechseltag (z. B. fraktionslos bis 02.02.2024, Gruppe ab 02.02.2024): spätere gilt
    zeilen = zeilen.sort_values("von").groupby("id").tail(1)
    if zeilen.id.nunique() > 1 and fr_kopf:
        z = zeilen[zeilen.fraktion == fr_kopf]
        if z.empty and fr_kopf == "fraktionslos":
            z = zeilen[zeilen.vorgriff]
        zeilen = z if not z.empty else zeilen
    if zeilen.id.nunique() > 1:
        return {"partei": None, "fraktion": fr_kopf, "gruppe": None, "quelle": "mehrdeutig"}
    if not zeilen.empty:
        z = zeilen.iloc[0]
        quelle = "stammdaten+gruppe_vorgriff" if z.vorgriff else "stammdaten"
        # Sitzung 152 (02.02.2024, Tag der Gruppenanerkennung) führt 5 Mitglieder der Gruppe Die Linke noch als "(fraktionslos)"
        if fr_kopf and fr_kopf != z.fraktion and not (fr_kopf == "fraktionslos" and isinstance(z.gruppe, str)):
            # Kopf hat Vorrang; Partei aus Kopf ableiten, wenn eindeutig
            return {"partei": PARTEI_VON_FRAKTION.get(fr_kopf, z.partei), "fraktion": fr_kopf,
                    "gruppe": fr_kopf if fr_kopf in ("Die Linke", "BSW") and datum >= GRUPPEN_START else None,
                    "quelle": "kopf"}
        return {"partei": z.partei, "fraktion": z.fraktion,
                "gruppe": z.gruppe if isinstance(z.gruppe, str) else None, "quelle": quelle}
    r = rest[rest.name_key == norm_name(name)]
    if not r.empty:
        return {"partei": r.iloc[0].partei, "fraktion": None, "gruppe": None, "quelle": "restliste"}
    if fr_kopf:
        return {"partei": PARTEI_VON_FRAKTION.get(fr_kopf), "fraktion": fr_kopf, "gruppe": None, "quelle": "nur_kopf"}
    return {"partei": None, "fraktion": None, "gruppe": None, "quelle": "kein_treffer"}


@lru_cache(maxsize=1)
def sitzungsdaten() -> dict[int, date]:
    # download_manifest.csv enthält kein Datum; Quelle ist das Attribut sitzung-datum der XML
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    out = {}
    for p in sorted(XML_DIR.glob("20*.xml")):
        r = etree.parse(str(p), parser).getroot()
        out[int(p.stem[2:])] = _d(r.get("sitzung-datum"))
    return out


def koepfe(txt_dir: Path = TXT_DIR):
    for p in sorted(txt_dir.glob("20*.txt")):
        if not p.stem.isdigit():
            continue  # 20NNN_anlagen.txt (zu Protokoll gegebene Reden, extract_text.process) gehört nicht zur Kopf-Messung
        nr = int(p.stem[2:])
        lines = p.read_text(encoding="utf-8").splitlines()
        skip = 0
        for i, l in enumerate(lines):
            if skip:
                skip -= 1
                continue
            # Regierungsköpfe brechen um, teils mit Leerzeile dazwischen (Faeser: 95 Köpfe
            # "…des Innern und für" / "" / "Heimat:"; Geywitz: "Stadt-" / "" / "entwicklung…:")
            if not l.endswith(":") and len(l) < 120 and KOPF_REG_START.match(l):
                for j in (i + 1, i + 2):
                    if j < len(lines) and lines[j].strip():
                        nxt = lines[j]
                        joined = l[:-1] + nxt if l.endswith("-") and nxt[:1].islower() else l + " " + nxt
                        if nxt.endswith(":") and len(nxt) < 60 and not KOPF_REG.match(nxt) \
                                and not KOPF_ABG.match(nxt) and KOPF_REG.match(joined):
                            l, skip = joined, j - i
                        break
            m = KOPF_ABG.match(l)
            if m:
                yield nr, m["name"], m["fr"], m["ort"]
                continue
            m = KOPF_REG.match(l)
            if m:
                yield nr, m["name"], None, None


def measure() -> pd.DataFrame:
    daten = sitzungsdaten()
    rows = []
    for nr, name, fr, ort in koepfe():
        if fr and fr.lower() not in FRAKTION_KOPF:
            # "(Bayern)" usw.: Ländervertreter, Fraktionsfeld ist Land
            res = lookup(name, daten[nr], None, ort)
        else:
            res = lookup(name, daten[nr], fr, ort)
        rows.append({"sitzung": nr, "name": name, "fraktion_kopf": fr, **res})
    return pd.DataFrame(rows)
