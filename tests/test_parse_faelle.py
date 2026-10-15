"""Parser-Fälle aus der Validierung: 284 von 552 im XML-Aufrufblock fehlenden Drucksachen stehen in Sammelaufrufen mit
Buchstabenbereich (20203 ZP 31 a bis 31 bb) und kommen aus dem Inhaltsverzeichnis; dazu eindeutige rede_id,
angekündigte Kurzinterventionen, Gesetzestitel, Regel reden_n und Ordinaltitel."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_buchstabenbereich_zweistellig():
    # 20203:3811 "Zusatzpunkte 31 a bis 31 bb sowie 34" = 28 Buchstaben (a…z, aa, bb) + ZP 34
    pk = parse.expand_punkte("die Zusatzpunkte 31 a bis 31 bb sowie 34")
    assert len(pk) == 29 and pk[25:28] == ["ZP 31 z", "ZP 31 aa", "ZP 31 bb"] and pk[-1] == "ZP 34"


@pytest.mark.rohdaten
def test_tagesordnung_nachschlag_20203():
    to = parse.tagesordnung(203)
    assert "ZP 31 a" in to and to["ZP 31 a"]["drucksachen"] == ["20/13954"]
    assert to["ZP 31 a"]["titel"].startswith("Erste Beratung des von der Bundesregierung eingebrachten Entwurfs")
    res = parse.parse_sitzung(203)
    zp31 = [g for g in res["gesetze"] if g["top_id"] == "Zusatzpunkt 31"]
    assert len(zp31) >= 25
    assert sum(g["quelle_drucksache"] == "tagesordnung" for g in zp31) >= 10
    assert any(g["drucksache"] == "20/13954" and g["beratungsstufe"] == "erste" for g in zp31)
    # Aufrufblock mit Drucksachen bleibt Quelle "aufruf"
    assert any(g["quelle_drucksache"] == "aufruf" for g in res["gesetze"])


@pytest.mark.rohdaten
def test_abstimmung_nur_laufender_gegenstand_20207():
    """20207: Abstimmung über TOP 6 a darf nicht auf den überwiesenen ZP 7 desselben Aufrufs übergehen."""
    res = parse.parse_sitzung(207)
    g = {x["punkt"]: x for x in res["gesetze"]}
    if "TOP 6 a" in g and "ZP 7" in g and g["TOP 6 a"]["top_id"] == g["ZP 7"]["top_id"]:
        assert g["ZP 7"]["beschlussart"] != "angenommen"
        assert g["TOP 6 a"]["abgestimmt"]


def test_rede_id_eindeutig():
    pq = ROOT / "data/processed/reden.parquet"
    if not pq.exists():
        pytest.skip("reden.parquet fehlt")
    r = pd.read_parquet(pq)
    assert r.rede_id.is_unique
    anl = r[r.redetyp.isin(["zu_protokoll", "erklaerung_go31"])]
    assert anl.rede_id.str.contains(r"-A\d{4}$").all()
    assert not r[~r.redetyp.isin(["zu_protokoll", "erklaerung_go31"])].rede_id.str.contains("A").any()
    # Hauptteil-IDs bleiben stabil: Schema sitzung-zeile
    assert r[r.redetyp == "plenarrede"].rede_id.str.match(r"^\d+-\d+$").all()


@pytest.mark.rohdaten
def test_angekuendigte_kurzintervention_20211():
    """Kurzinterventionen, die nur die Zeile davor ankündigt, liefen als plenarrede: 20211:1889 Zwischenruf
    "Kurzintervention!" vor Frohnmaier (1891), 20211:2405 "bittet um eine Kurzintervention" vor Dröge (2407)."""
    r = parse.merge_reden(parse.parse_sitzung(211)["beitraege"])
    typ = {x["zeile_start"]: x["redetyp"] for x in r}
    assert typ.get(1891) == "kurzintervention", typ.get(1891)
    assert typ.get(2407) == "kurzintervention", typ.get(2407)
    # offen: 20211:1925 Baerbock nach "Frau Baerbock, möchten Sie antworten?" (1923) läuft noch als plenarrede, nicht als
    # Erwiderung


@pytest.mark.rohdaten
def test_gesetz_titel_zustrom():
    res = parse.parse_sitzung(211)
    g = [x for x in res["gesetze"] if x["drucksache"] == "20/12804"][0]
    assert g["gesetz_titel"].startswith("Entwurf eines Gesetzes zur Begrenzung des illegalen Zustroms")
    assert g["gesetz_titel"].endswith("(Zustrombegrenzungsgesetz)") and g["kurztitel_ende"] == "Zustrombegrenzungsgesetz"
    pq = ROOT / "data/processed/gesetze_einheit.parquet"
    if pq.exists():
        e = pd.read_parquet(pq)
        z = e[e.drucksache == "20/12804"].iloc[0]
        assert z.gesetz_titel.startswith("Entwurf eines Gesetzes") and z.kurztitel == "Zustrombegrenzungsgesetz"
        assert (e.gesetz_titel.str.startswith("Entwurf")).mean() > 0.8


def test_regel_reden_n():
    """Regel reden_n (Hauptstatistik): nur plenarrede in Debatten-TOPs (Debatte, Aktuelle Stunde,
    Regierungserklärung); Befragung, Fragestunde und Wahl/GO zählen nicht."""
    pq = ROOT / "data/processed/reden.parquet"
    if not pq.exists():
        pytest.skip("reden.parquet fehlt")
    r = pd.read_parquet(pq)
    haupt = r[(r.redetyp == "plenarrede") & r.top_typ.isin(["Debatte", "Aktuelle Stunde", "Regierungserklärung"])]
    assert 5000 < len(haupt) < len(r[r.redetyp == "plenarrede"]) + 1
    assert not haupt.top_typ.isin(["Befragung", "Fragestunde", "Wahl/GO"]).any()


@pytest.mark.rohdaten
def test_ordinaltitel_als_gesetzentwurf():
    """Belege aus data/llm/divergenz.csv (typ_abweichend): 20170 ZP 12 (Aufruf Z. 9) und 20147 TOP 29 (Aufruf Z. 10993)."""
    for probe in ("Entwurfs eines Zweiten Gesetzes zur Änderung des Energiewirtschaftsgesetzes", "Entwurfs eines 2. Gesetzes zur",
                  "Entwurfs eines Zweiten Haushaltsfinanzierungsgesetzes 2024", "Entwurfs eines … Gesetzes zur Änderung"):
        assert parse.ENTWURF_RE.search(probe), probe
    assert not parse.ENTWURF_RE.search("Beratung des Antrags der Fraktion der CDU/CSU Gesetz der Vernunft")
    for nr, zeile in ((170, 9), (147, 10993)):
        res = parse.parse_sitzung(nr)
        items = [g for g in res["gesetze"] if g["aufruf_zeile"] == zeile and "Entwurf" in g["titel"]]
        assert items and all(g["typ"] == "gesetzentwurf" and g["ist_gesetz"] for g in items), (nr, zeile)
