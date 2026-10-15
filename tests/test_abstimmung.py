"""Abstimmungserkennung: die Fehlerfälle der Stichprobe (data/reference/abstimmung_urteile.csv) als Regressionstests.
Fall 2 und 12: Hinweis auf eine laufende namentliche Abstimmung ist keine Abstimmung über den aktuellen Punkt.
Fall 21: "Wir kommen zur Abstimmung über den Einzelplan 25" ist eine Abstimmung, kein neuer Aufruf.
Fall 28: "Wer stimmt gegen die Zurückverweisung?" ist eine Abstimmungsformel, die Beschlussart heißt "zurückverwiesen"."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bundestag import parse  # noqa: E402


@pytest.mark.parametrize("satz", [
    "Liebe Kolleginnen und Kollegen, denken Sie daran, dass die namentliche Abstimmung bald endet.",  # 20178:8715
    "Sehr geehrte Kolleginnen und Kollegen, bevor wir hier weiter fortfahren, mache ich darauf aufmerksam, dass die Zeit "
    "für die namentliche Abstimmung demnächst vorbei ist, exakt um 17.07 Uhr.",  # 20178:5735
    "Bevor ich den nächsten Redner aufrufe, möchte ich die namentliche Abstimmung schließen.",  # 20157:1841
])
def test_hinweis_auf_namentliche_abstimmung_ist_keine_formel(satz):
    assert parse.ABSTIMMUNG_RE.search(satz) is None


@pytest.mark.parametrize("satz", [
    "Wer stimmt gegen die Zurückverweisung? – Das sind die AfD-Fraktion und Die Linke.",  # 20207:7023
    "Wer stimmt gegen die Überweisung? – Die AfD-Fraktion.",  # 20203:4599
    "Die Fraktion der CDU/CSU hat namentliche Abstimmung verlangt.",  # 20152:1445
    "Dann eröffne ich jetzt die namentliche Abstimmung über die Beschlussempfehlung.",  # 20147:4125
    "Ich eröffne die dritte namentliche Abstimmung, und zwar über den Entschließungsantrag",  # 20152:2619
    "Wir kommen zur Abstimmung über den Einzelplan 25 – Bundesministerium für Wohnen",  # 20149:3867
    "Es handelt sich um 75 Petitionen. Wer stimmt dafür? – Das sind alle Fraktionen.",  # 20147:4201
    "Das sind sechs Petitionen. Wer stimmt hier dafür? – CDU/CSU, FDP, Bündnis 90/Die Grünen, SPD.",  # 20207:6787
    "Die Fraktionen von SPD und Bündnis 90/Die Grünen haben jeweils namentliche Abstimmungen verlangt",  # 20210:10509
])
def test_abstimmungsformeln(satz):
    assert parse.ABSTIMMUNG_RE.search(satz)


def test_einzelplan_abstimmung_ist_kein_aufruf():
    assert parse.FREI_RE.search("Wir kommen zur Abstimmung über den Einzelplan 25 – Bundesministerium für Wohnen, "
                                "Stadtentwicklung und Bauwesen – in der Ausschussfassung.") is None
    assert parse.FREI_RE.search("Wir kommen zu den Abstimmungen, und zwar zunächst über den Einzelplan 08") is None
    # Aufruf ohne Abstimmung bleibt ein Aufruf (20184:24)
    assert parse.FREI_RE.search("Wir beginnen mit dem Geschäftsbereich des Bundeskanzlers und des Bundeskanzleramtes, "
                                "Einzelplan 04, sowie des Unabhängigen Kontrollrates, Einzelplan 22.")


def test_zurueckverweisung_passt_zu_jedem_typ():
    assert parse.ergebnis_passt("Der Antrag auf Zurückverweisung ist angenommen", "bericht")
    # abgelehnte Zurückverweisung beendet die Abstimmung nicht: das Ergebnis über die Vorlage folgt danach
    assert not parse.ergebnis_passt("Der Antrag auf Zurückverweisung ist abgelehnt", "antrag")


@pytest.mark.rohdaten
def test_einzelplan_abgestimmt_20149():
    res = parse.parse_sitzung(149)
    ep25 = [g for g in res["gesetze"] if "Einzelplan 25" in (g["titel"] or "")]
    assert ep25 and all(g["abgestimmt"] and g["beschlussart"] == "angenommen" for g in ep25)
    # die Abstimmungszeile ist kein Aufruf mehr: kein Punkt "Einzelplan 25" mit eigener top_id
    assert not any(g["top_id"] == "Einzelplan 25" for g in res["gesetze"])


@pytest.mark.rohdaten
def test_hinweis_namentlich_nicht_abgestimmt_20178():
    g = {x["drucksache"]: x for x in parse.parse_sitzung(178)["gesetze"]}
    assert not g["20/11952"]["abgestimmt"] and g["20/11952"]["beschlussart"] == "überwiesen"
    assert not g["20/8413"]["abgestimmt"]


@pytest.mark.rohdaten
def test_zurueckverwiesen_20207():
    g = {x["drucksache"]: x for x in parse.parse_sitzung(207)["gesetze"]}
    for drs in ("20/10372", "20/6901"):  # ZP 10 l und 10 k
        assert g[drs]["abgestimmt"] and g[drs]["beschlussart"] == "zurückverwiesen"


def test_titel_ergaenzungsstrich_bleibt():
    it = parse._item("Erste Beratung des von der Bundesregierung eingebrachten Entwurfs eines Gesetzes zur Modernisierung "
                     "und zum Bürokratieabbau im Strom- und Energiesteuerrecht Drucksache 20/12784",
                     ["20/12784"], "TOP 30", "aufruf")
    assert "Strom- und Energiesteuerrecht" in it["titel"]
    it = parse._item("Zweite und dritte Beratung des von der Bundes- regierung eingebrachten Entwurfs Drucksache 20/1",
                     ["20/1"], "TOP 1", "aufruf")
    assert "Bundesregierung" in it["titel"]


@pytest.mark.parametrize("titel, kurz", [
    ("Entwurf eines Gesetzes zur Stärkung der Herzgesundheit (Gesundes-Herz-Gesetz – GHG)", "Gesundes-Herz-Gesetz – GHG"),
    ("Entwurf eines Gesetzes über die Feststellung des Wirtschaftsplans (ERP-Wirtschaftsplangesetz 2025 – ERPWiPlanG 2025)",
     "ERP-Wirtschaftsplangesetz 2025 – ERPWiPlanG 2025"),
    ("Entwurf eines Gesetzes zur Begrenzung (Zustrombegrenzungsgesetz)", "Zustrombegrenzungsgesetz"),
])
def test_kurztitel(titel, kurz):
    assert parse.KURZTITEL_RE.search(titel).group(1) == kurz


def test_ergebnis_passt_nummer_und_nebenabstimmung():
    assert not parse.ergebnis_passt("Die Sammelübersicht 566 ist angenommen", "bericht", "Sammelübersicht 563 zu Petitionen")
    assert parse.ergebnis_passt("Die Sammelübersicht 563 ist angenommen", "bericht", "Sammelübersicht 563 zu Petitionen")
    assert not parse.ergebnis_passt("Die Sammelübersicht 564 ist angenommen", "bericht", "Beratung der Beschlussempfehlung")
    assert not parse.ergebnis_passt("Der Änderungsantrag ist damit abgelehnt", "antrag", "Beratung des Antrags")
    g = {"typ": "antrag", "titel": "Beratung des Antrags"}
    assert parse.ergebnis_fuer("Der Änderungsantrag ist damit abgelehnt. Damit ist der Antrag angenommen.", g) \
        == "Damit ist der Antrag angenommen"


@pytest.mark.rohdaten
def test_sammeluebersichten_eigenes_ergebnis_20166():
    g = {x["punkt"]: x for x in parse.parse_sitzung(166)["gesetze"] if x["top_id"] == "Tagesordnungspunkt 27"}
    assert "563" in g["TOP 27 h"]["ergebnis_text"] and "559" in g["TOP 27 d"]["ergebnis_text"]


@pytest.mark.parametrize("satz, regex", [
    ("Damit ist auch dieser Antrag mit großer Mehrheit angenommen.", "ERGEBNIS_RE"),  # 20210:9079
    ("Damit ist der Überweisungsvorschlag angenommen.", "UEBERWEISUNG_RE"),  # 20181:3225
    ("Ich kann das ermittelte Ergebnis der namentlichen Schlussabstimmung über den Gesetzentwurf bekannt geben:",
     "NAMENTLICH_ERGEBNIS_RE"),  # 20167:1313
])
def test_beschlussart_formeln(satz, regex):
    assert getattr(parse, regex).search(satz)


def test_zuruecküberweisung():
    assert parse.ergebnis_passt("Der Antrag auf Zurücküberweisung ist damit angenommen", "bericht")  # 20203:4049


@pytest.mark.rohdaten
def test_sammelueberweisung_20172():
    g = [x for x in parse.parse_sitzung(172)["gesetze"] if x["aufruf_zeile"] == 5877]
    assert len(g) > 5 and all(x["beschlussart"] == "überwiesen" for x in g if not x["abgestimmt"])
