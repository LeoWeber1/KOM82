"""Tests zu den Metrik-Definitionen: Wortzählung, ist_mdb, Gesetz-Einheit, beschlussart."""
import random
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ERLAUBT = {"überwiesen", "angenommen", "abgelehnt", "zurückverwiesen", "vertagt", "offen"}


@lru_cache(maxsize=None)
def reden(nr: int) -> list[dict]:
    return parse.merge_reden(parse.parse_sitzung(nr)["beitraege"])


def test_zaehle_woerter_definition():
    # Whitespace-Token mit Buchstabe/Ziffer: "–" und "…" zählen nicht, "1,5" und "Herr/Frau" je einmal (7 Wörter)
    assert parse.zaehle_woerter("Das ist – wie gesagt – 1,5 Prozent. Herr/Frau …") == 7
    assert parse.zaehle_woerter("") == 0


@pytest.mark.rohdaten
def test_wortzahl_gegen_xml_20_reden():
    """20 Plenarreden (Seed 82) gegen gold_reden.text (XML ohne Kommentare), gleiche Zählfunktion, MAD < 2 %."""
    gold = pd.read_parquet(ROOT / "data/reference/gold_reden.parquet")
    g = gold[gold.anlage_typ.isna() & ~gold.top_titel.fillna("").str.contains("Befragung|Fragestunde")].copy()
    g["nn"] = g.nachname.fillna("").str.split().str[-1]
    g["k"] = g.groupby(["sitzung", "nn"]).cumcount()
    rng = random.Random(82)
    reihenfolge = list(g.index)
    rng.shuffle(reihenfolge)
    abw, uebersprungen = [], 0
    for idx in reihenfolge:
        if len(abw) == 20:
            break
        z = g.loc[idx]
        meine = [r for r in reden(int(z.sitzung) - 20000)
                 if r["redetyp"] in ("plenarrede", "kurzintervention", "erwiderung") and r["name"].split()[-1] == z.nn]
        if z.k >= len(meine) or meine[z.k]["redetyp"] != "plenarrede":
            uebersprungen += 1
            continue
        r = meine[z.k]
        # Zuordnung prüfen: die ersten drei Wörter des XML-Textes stehen im PDF-Text
        if " ".join(z.text.split()[:3]) not in r["text"]:
            uebersprungen += 1
            continue
        wg = parse.zaehle_woerter(z.text)
        abw.append(abs(r["woerter"] - wg) / max(wg, 1))
    mad = sum(abw) / len(abw)
    print(f"Wortabweichung MAD {100 * mad:.2f} % über {len(abw)} Reden, {uebersprungen} übersprungen")
    assert len(abw) == 20
    assert mad < 0.02, mad


@pytest.mark.rohdaten
def test_ist_mdb():
    r = reden(211)
    faeser = [x for x in r if x["name"] == "Nancy Faeser"]
    merz = [x for x in r if x["name"] == "Friedrich Merz"]
    assert faeser and all(x["ist_mdb"] is False for x in faeser)  # Bundesministerin ohne Mandat (Restliste)
    assert merz and all(x["ist_mdb"] is True for x in merz)
    assert all(x["ist_mdb"] for x in r if x["redetyp"] == "praesidium")


@pytest.mark.rohdaten
def test_beschlussart_und_typ_fein_20160():
    res = parse.parse_sitzung(160)
    g = {x["drucksache"]: x for x in res["gesetze"]}
    assert {x["beschlussart"] for x in res["gesetze"]} <= ERLAUBT
    assert g["20/10722"]["beratungsstufe"] == "erste" and g["20/10722"]["beschlussart"] == "überwiesen"
    assert g["20/10031"]["beschlussart"] == "angenommen" and g["20/10031"]["beschlussempfehlung_drucksache"] == "20/10755"
    assert g["20/10031"]["typ_fein"] == "gesetzentwurf" and g["20/10031"]["ist_gesetz"]
    be = [x for x in res["gesetze"] if x["typ_fein"] == "beschlussempfehlung"]
    assert be and all(not x["ist_gesetz"] for x in be)
    # 20160:2229 "Beratung der Beschlussempfehlung … Drucksachen 20/9312, 20/9880": Bezug = beratene Vorlage
    assert g["20/9312"]["typ_fein"] == "beschlussempfehlung" and g["20/9312"]["bezug_drucksache"] == "20/9312"


def test_gesetze_einheit_zustrom():
    pq = ROOT / "data/processed/gesetze_einheit.parquet"
    if not pq.exists():
        pytest.skip("gesetze_einheit.parquet fehlt")
    e = pd.read_parquet(pq)
    z = e[e.drucksache == "20/12804"]
    assert len(z) == 1
    z = z.iloc[0]
    assert z.sitzungen == "[185, 211]" and z.beschlussart_final == "abgelehnt"
    assert z.kurztitel == "Zustrombegrenzungsgesetz" and z.n_reden >= 20 and z.n_redner >= 15
    assert "20/13648" in z.beschlussempfehlungen
    assert e.drucksache.is_unique
    g = pd.read_parquet(ROOT / "data/processed/gesetze.parquet")
    assert set(g.beschlussart) <= ERLAUBT
    assert (g.ist_gesetz == (g.typ_fein == "gesetzentwurf")).all()
    assert len(e) == g[g.ist_gesetz].drucksache.nunique()
