import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/reference/gold_gesetze.csv"


@pytest.fixture(scope="module")
def s211():
    res = parse.parse_sitzung(211)
    return res, parse.merge_reden(res["beitraege"])


@pytest.mark.rohdaten
def test_zustrom_zp35_20211(s211):
    res, reden = s211
    g = [x for x in res["gesetze"] if x["drucksache"] == "20/12804"]
    assert len(g) == 1
    g = g[0]
    assert "Zustrombegrenzungsgesetz" in g["titel"] and g["kurztitel"] == "Zustrombegrenzungsgesetz"
    assert g["drucksachen_alle"] == ["20/12804", "20/13648"]
    assert g["beratungsstufe"] == "zweite und dritte" and g["typ"] == "gesetzentwurf"
    assert g["top_id"] == "Zusatzpunkt 35" and g["aufruf_zeile"] == 1065
    assert g["abgestimmt"] is True and g["abstimmung_zeile"]
    assert g["ergebnis_text"] and "abgelehnt" in g["ergebnis_text"]
    redner = [r for r in reden if r["top_id"] == "Zusatzpunkt 35" and r["redetyp"] == "plenarrede"]
    assert len(redner) >= 18 and len({r["name"] for r in redner}) == 18  # 18 Redner in 211 (7 Afghanistan-Redner gehören zu TOP 18)
    assert all(r["partei"] for r in redner), [r["name"] for r in redner if not r["partei"]]
    namen = {r["name"] for r in redner}
    assert "Friedrich Merz" in namen and "Nancy Faeser" in namen
    bet = parse.beteiligte(reden, res["gesetze"])
    assert {b["name"] for b in bet if b["drucksache"] == "20/12804"} >= namen


@pytest.mark.rohdaten
def test_zustrom_erste_beratung_20185():
    """Zweitfall laut data/reference/gold_gesetze.csv: 20185 ZP 3 = erste Beratung Drs. 20/12804 (20209 hat keinen mündlich aufgerufenen Erstfall)."""
    res = parse.parse_sitzung(185)
    g = [x for x in res["gesetze"] if x["drucksache"] == "20/12804"]
    assert len(g) == 1 and g[0]["beratungsstufe"] == "erste" and g[0]["top_id"].startswith("Zusatzpunkt")
    assert "Zustrom" in g[0]["titel"]


@pytest.mark.rohdaten
def test_20209_kopfueberweisung_kein_gesetz():
    """Überweisungen im vereinfachten Verfahren am Sitzungsanfang (20209 ZP 10 a–c) sind kein Aufruf."""
    res = parse.parse_sitzung(209)
    gold = pd.read_csv(GOLD)
    gold = gold[(gold.sitzung == 20209) & gold.drucksachen.notna()]
    kopf = gold[gold.top.str.startswith("ZP 10")]
    assert len(kopf) == 3 and set(kopf.drucksachen) == {"20/13963", "20/13964", "20/14704"}
    meine = {x["drucksache"] for x in res["gesetze"]}
    assert not (set(kopf.drucksachen) & meine)
    assert "20/12786" in meine  # ZP 1, zweite Beratung, mündlich aufgerufen (gold_gesetze Z. 589)


def _pr(gold: pd.DataFrame, mine: pd.DataFrame):
    gset = {(x.sitzung, x.drucksachen.split(";")[0].strip()) for x in gold.itertuples()}
    mset = {(20000 + s, d) for s, d in zip(mine.sitzung, mine.drucksache) if 20000 + s in set(gold.sitzung) and d}
    tp = len(gset & mset)
    return tp / max(len(mset), 1), tp / max(len(gset), 1)


def test_precision_recall_gold_gesetze():
    pq = ROOT / "data/processed/gesetze.parquet"
    if not pq.exists():
        pytest.skip("gesetze.parquet fehlt")
    gold = pd.read_csv(GOLD)
    gold = gold[gold.drucksachen.notna()]
    mine = pd.read_parquet(pq)
    p, r = _pr(gold, mine)
    ps, rs = _pr(gold[gold.unsicher == 0], mine)
    # Precision nur für 20211: dort listet das Gold alle Aufrufe (21 Zeilen); 20160/20185 enthält es nur Stichproben (2 bzw. 7)
    p211, r211 = _pr(gold[gold.sitzung == 20211], mine)
    print(f"P/R alle {p:.3f}/{r:.3f}, sicher {ps:.3f}/{rs:.3f}, 20211 {p211:.3f}/{r211:.3f}")
    assert rs >= 0.9, (ps, rs)
    assert r >= 0.75, (p, r)  # 20209 ZP 10 a–c sind Kopf-Überweisungen ohne Aufruf (unsicher=1) und fehlen absichtlich
    assert p211 >= 0.7, (p211, r211)


AFGHANISTAN = {"Sauter", "Nolte", "Türk-Nachbaur", "Gerschau", "Vogler", "Schmid", "Hierl"}


@pytest.mark.rohdaten
def test_ruecksprung_nur_ergebnis_20211(s211):
    """20211:3435 "Ich komme zurück zu Zusatzpunkt 35" verkündet nur das Ende der namentlichen Abstimmung; die Debatte zu TOP 18 läuft weiter."""
    res, reden = s211
    # Debatte zu TOP 18 (Aufruf Z. 3273) läuft von der Wiederaufnahme Z. 3435 bis zum nächsten Aufruf Z. 4057
    afg = [r for r in reden if r["redetyp"] == "plenarrede" and r["name"].split()[-1] in AFGHANISTAN and 3400 < r["zeile_start"] < 4057]
    assert len(afg) >= 7 and {r["top_id"] for r in afg} == {"Tagesordnungspunkt 18"}, [(r["name"], r["top_id"]) for r in afg]
    g = [x for x in res["gesetze"] if x["drucksache"] == "20/12804"][0]
    assert g["beschlussart"] == "abgelehnt"
    pq = ROOT / "data/processed/gesetze_einheit.parquet"
    if pq.exists():
        e = pd.read_parquet(pq)
        assert int(e[e.drucksache == "20/12804"].n_redner.iloc[0]) == 35


@pytest.mark.rohdaten
def test_bezahlkarte_20_10722_abgelehnt():
    """20164:1045 "Damit ist der Gesetzentwurf in zweiter Beratung abgelehnt" gilt für 20/10722 (ZP 7), nicht für den Entwurf 20/9470 (ZP 6)."""
    res = parse.parse_sitzung(164)
    g = {x["drucksache"]: x for x in res["gesetze"]}
    assert g["20/10722"]["beschlussart"] == "abgelehnt" and g["20/9470"]["beschlussart"] == "angenommen"
    pq = ROOT / "data/processed/gesetze_einheit.parquet"
    if pq.exists():
        e = pd.read_parquet(pq)
        assert e[e.drucksache == "20/10722"].beschlussart_final.iloc[0] == "abgelehnt"
