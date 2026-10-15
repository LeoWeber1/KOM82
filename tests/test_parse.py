import collections
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import config, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def s160():
    res = parse.parse_sitzung(160)
    return res["beitraege"], parse.merge_reden(res["beitraege"])


@pytest.mark.rohdaten
def test_reden_20160(s160):
    """gold_reden 20160 = 193 = 188 Sitzungsverlauf + 5 Anlagen (nicht im txt). XML zählt Kurzintervention und
    Erwiderung als <rede>, daher plenarrede + kurzintervention; Toleranz ±3 für Dialog-Zuordnung (ZP 4 Aeffner/Stracke)."""
    _, reden = s160
    gold = pd.read_parquet(ROOT / "data/reference/gold_reden.parquet")
    g = gold[gold.sitzung == 20160]
    assert len(g) == 193
    body = int(g.anlage_typ.isna().sum())
    n = sum(1 for r in reden if r["redetyp"] in ("plenarrede", "kurzintervention", "erwiderung"))
    assert abs(n - body) <= 3, (n, body)
    # inkl. Anlagen: 5 zu Protokoll gegebene Reden -> 193
    n_alle = n + sum(1 for r in reden if r["redetyp"] in ("zu_protokoll", "erklaerung_go31"))
    assert abs(n_alle - 193) <= 3, n_alle


@pytest.mark.rohdaten
def test_zu_protokoll_anlagen(s160):
    _, reden = s160
    zp = [r for r in reden if r["redetyp"] == "zu_protokoll"]
    assert len(zp) == 5 and all(r["anlage"] and r["top_id"] and r["woerter"] > 100 for r in zp)
    assert {r["top_id"] for r in zp} == {"Tagesordnungspunkt 20", "Tagesordnungspunkt 18",
                                         "Tagesordnungspunkt 23", "Tagesordnungspunkt 24"}
    # 20146: gold 177 = 112 Befragung + 56 Sitzungsverlauf + 9 Anlagen (1 zu Protokoll, 8 Erklärungen nach § 31 GO)
    res = parse.parse_sitzung(146)
    r146 = parse.merge_reden(res["beitraege"])
    gold = pd.read_parquet(ROOT / "data/reference/gold_reden.parquet")
    g = gold[gold.sitzung == 20146]
    assert len(g) == 177
    anl = [r for r in r146 if r["redetyp"] in ("zu_protokoll", "erklaerung_go31")]
    assert len(anl) == int(g.anlage_typ.notna().sum()) == 9
    ohne_befragung = int((g.top_id.str.replace("\xa0", " ") != "Tagesordnungspunkt 1").sum())
    n = sum(1 for r in r146 if r["redetyp"] in ("plenarrede", "kurzintervention", "erwiderung")) + len(anl)
    assert abs(n - ohne_befragung) <= 3, (n, ohne_befragung)


@pytest.mark.rohdaten
def test_erwiderung_eigener_typ(s160):
    _, reden = s160
    erw = [r for r in reden if r["redetyp"] == "erwiderung"]
    kurz = [r for r in reden if r["redetyp"] == "kurzintervention"]
    # 20160 TOP 7: Kurzinterventionen von Storch (Z. 433, 711) und Mordhorst (575); Erwiderungen Schmidt (593), Funke-Kaiser (737)
    assert {r["name"].split()[-1] for r in erw} >= {"Schmidt", "Funke-Kaiser"}
    assert {r["name"].split()[-1] for r in kurz} >= {"Storch", "Mordhorst"}
    assert not any(r["name"].endswith("Funke-Kaiser") for r in kurz)


@pytest.mark.rohdaten
def test_zwischenfrage_keine_rede(s160):
    beitraege, reden = s160
    zf = [r for r in reden if r["redetyp"] == "zwischenfrage"]
    assert 8 <= len(zf) <= 15  # gold_beitraege 20160: 12 Zwischenfrage-Segmente
    # Hauptredner setzt nach abgelehnter oder angenommener Zwischenfrage dieselbe Rede fort: Aeffner ZP 4 ist eine Rede
    aeffner = [r for r in reden if r["name"].endswith("Aeffner") and r["redetyp"] == "plenarrede"]
    assert len(aeffner) == 1 and aeffner[0]["n_beitraege"] >= 3


def test_faeser_kopf_zwei_zeilen():
    zeilen = ["Nancy Faeser, Bundesministerin des Innern und für", "", "Heimat:", "Sehr geehrte Frau Präsidentin!"]
    kopf, skip = parse.join_kopf(zeilen, 0)
    assert kopf == "Nancy Faeser, Bundesministerin des Innern und für Heimat:" and skip == 2
    info = parse.erkenne_kopf(kopf)
    assert info["name"] == "Nancy Faeser" and info["rolle"].startswith("Bundesministerin")
    # Silbentrennung im Funktionstitel eines Regierungskopfs: Trennstrich vor Kleinbuchstabe entfällt
    kopf, _ = parse.join_kopf(["Annalena Baerbock, Bundesministerin des Auswär-", "tigen:"], 0)
    assert kopf == "Annalena Baerbock, Bundesministerin des Auswärtigen:"


def test_praesidium_false_positive():
    assert parse.erkenne_kopf("Vizepräsident hier im Hohen Haus, wörtlich:") is None  # Präsidiumswort mitten im Satz
    assert parse.erkenne_kopf("Vizepräsidentin Magwas:")["art"] == "praesidium"
    assert parse.erkenne_kopf("Präsident Wolfgang Kubicki:")["name"] == "Wolfgang Kubicki"


@pytest.mark.skipif(not config.STAMMDATEN_XML.exists(), reason="MdB-Stammdaten fehlen: python run_all.py lädt sie")
def test_wissing_partei_wechsel():
    vor = parse._lookup("Dr. Volker Wissing", date(2024, 10, 9), None, None)
    nach = parse._lookup("Dr. Volker Wissing", date(2024, 12, 19), None, None)
    assert vor["partei"] == "FDP"
    assert nach["partei"] != "FDP" and nach["fraktion"] == "fraktionslos"


def test_expand_punkte():
    assert parse.expand_punkte("die Tagesordnungspunkte 36 a bis 36 h sowie die Zusatzpunkte 9 a und 9 b") == \
        [f"TOP 36 {c}" for c in "abcdefgh"] + ["ZP 9 a", "ZP 9 b"]
    assert parse.expand_punkte("die Zusatzpunkte 29 bis 34 und Tagesordnungspunkt 5") == \
        [f"ZP {n}" for n in range(29, 35)] + ["TOP 5"]
    assert parse.expand_punkte("Tagesordnungspunkt 9 und Zusatzpunkt 5") == ["TOP 9", "ZP 5"]


def test_ohne_partei_unter_1_prozent():
    pq = ROOT / "data/processed/beitraege.parquet"
    if not pq.exists():
        pytest.skip("beitraege.parquet fehlt; `python -m bundestag.parse` ausführen")
    b = pd.read_parquet(pq)
    b = b[b.redetyp != "praesidium"]
    assert b.sitzung.nunique() == 67
    anteil = b.partei.isna().mean()
    assert anteil < 0.01, anteil
