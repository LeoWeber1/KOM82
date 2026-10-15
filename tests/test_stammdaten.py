import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import stammdaten as s  # noqa: E402


@pytest.fixture(scope="module")
def daten():
    return s.sitzungsdaten()


@pytest.mark.rohdaten
def test_wissing_fdp_bis_07_11_2024(daten):
    # Stammdaten: FDP 26.10.2021–07.11.2024, fraktionslos ab 08.11.2024
    assert s.lookup("Dr. Volker Wissing", daten[196], None)["fraktion"] == "FDP"
    assert s.lookup("Dr. Volker Wissing", date(2024, 11, 7), None)["fraktion"] == "FDP"


@pytest.mark.rohdaten
def test_wissing_danach_fraktionslos(daten):
    for nr in (198, 200, 212):
        r = s.lookup("Dr. Volker Wissing", daten[nr], None)
        assert r["fraktion"] == "fraktionslos" and r["gruppe"] is None


@pytest.mark.parametrize("nr", range(146, 213))
@pytest.mark.rohdaten
def test_wagenknecht_gruppe_bsw(daten, nr):
    assert s.lookup("Dr. Sahra Wagenknecht", daten[nr], "fraktionslos")["gruppe"] == "BSW"
    assert s.lookup("Sahra Wagenknecht", daten[nr], None)["gruppe"] == "BSW"


@pytest.mark.parametrize("name,gruppe", [
    ("Dr. Sahra Wagenknecht", "BSW"), ("Klaus Ernst", "BSW"), ("Sevim Dağdelen", "BSW"),
    ("Dr. Gregor Gysi", "Die Linke"), ("Heidi Reichinnek", "Die Linke"), ("Pascal Meiser", "Die Linke"),
])
@pytest.mark.rohdaten
def test_fraktionslos_koepfe_146_149(daten, name, gruppe):
    for nr in (146, 147, 148, 149):
        r = s.lookup(name, daten[nr], "fraktionslos")
        assert r["gruppe"] == gruppe and r["fraktion"] == gruppe, (name, nr, r)
        assert r["quelle"] == "stammdaten+gruppe_vorgriff"


@pytest.mark.rohdaten
def test_echte_fraktionslose_bleiben(daten):
    r = s.lookup("Stefan Seidler", daten[146], "fraktionslos")
    assert r["fraktion"] == "fraktionslos" and r["gruppe"] is None


def test_namensnormalisierung():
    assert s.norm_name("Dr.-Ing. Zoe Mayer") == s.norm_name("Zoe Mayer")
    assert s.norm_name("Sevim Dağdelen") == "sevim dagdelen"
    assert s.norm_name("Prof. Dr. Armin Grau") == "armin grau"


@pytest.mark.rohdaten
def test_restliste(daten):
    r = s.lookup("Boris Pistorius", daten[150], None)
    assert r == {"partei": "SPD", "fraktion": None, "gruppe": None, "quelle": "restliste"}


@pytest.mark.rohdaten
def test_kein_fuzzy_match(daten):
    assert s.lookup("Volker Wissinger", daten[150], None)["quelle"] == "kein_treffer"


@pytest.mark.rohdaten
def test_trefferquote():
    m = s.measure()
    ohne = (~m.quelle.isin(["stammdaten", "stammdaten+gruppe_vorgriff", "restliste", "kopf"])).mean()
    assert len(m) > 10000 and ohne < 0.02
