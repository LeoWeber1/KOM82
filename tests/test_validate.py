import math

import pandas as pd

from bundestag import validate as v


def test_identischer_text_ergibt_eins():
    reden = [["Meine Damen und Herren, die Sitzung ist eröffnet und wir beginnen.",
              "Wolfgang Schäuble hat sein Leben in den Dienst der Demokratie gestellt."]]
    txt = v.tokens(" ".join(reden[0]))
    o = v.ordnungsquote(reden, txt)
    assert o["fundquote"] == 1.0 and o["ordnungsquote"] == 1.0
    assert v.ngram_recall(reden, txt)["recall_5gramm"] == 1.0
    assert v.token_recall(reden, txt)["recall_token"] == 1.0


def test_vertauschte_absaetze_senken_nur_ordnung():
    a = "Erster Absatz handelt von der Energieversorgung in diesem Winter."
    b = "Zweiter Absatz spricht über den Haushalt des kommenden Jahres."
    c = "Dritter Absatz bedankt sich bei allen Kolleginnen und Kollegen."
    reden = [[a, b, c]]
    txt = v.tokens(" ".join([a, c, b]))  # Spaltenvertauschung: c vor b
    o = v.ordnungsquote(reden, txt)
    assert o["fundquote"] == 1.0
    assert abs(o["ordnungsquote"] - 2 / 3) < 1e-9
    assert v.token_recall(reden, txt)["recall_token"] == 1.0


def test_normalisierung_gleicht_typografie_aus():
    xml = "„Wir schaffen“ das – Verant­wortung für die CDU/CSU-Fraktion"
    pdf = '"Wir schaffen" das - Verantwortung für die CDU/CSU-Fraktion'
    assert v.tokens(xml) == v.tokens(pdf)
    assert v.tokens("ﬁnanziell") == ["finanziell"]
    assert v.longest_nondecreasing([1, 3, 2, 2, 5]) == 4


def test_nachname_key_parser_gegen_xml():
    assert v.nachname_key("Dr. Katrin Göring-Eckardt") == v.nachname_key("Göring-Eckardt") == "eckardt"
    assert v.nachname_key("Beatrix von Storch") == v.nachname_key("von Storch") == "storch"
    assert v.nachname_key("Dr. h. c. Thomas Sattelberger") == "sattelberger"
    assert v.nachname_key(None) == ""


def test_lcs_align_bei_einfuegung_und_loeschung():
    parser = ["merz", "scholz", "weidel", "lindner", "habeck"]
    gold = ["merz", "weidel", "lindner", "wagenknecht", "habeck"]  # scholz zu viel, wagenknecht fehlt
    pairs = v.lcs_align(parser, gold)
    assert pairs == [(0, 0), (2, 1), (3, 2), (4, 4)]
    p, r, f1 = v.prf(len(pairs), len(parser), len(gold))
    assert p == 0.8 and r == 0.8 and abs(f1 - 0.8) < 1e-12
    # wiederholte Namen (Minister in Befragungen) bleiben in Reihenfolge matchbar
    assert len(v.lcs_align(["a", "b", "a", "b", "a"], ["a", "a", "b", "a"])) == 4


def test_prf_randfaelle():
    assert all(math.isnan(x) for x in v.prf(0, 0, 0))
    assert v.prf(0, 3, 0) == (0.0, 0.0, 0.0)
    assert v.prf(2, 2, 4) == (1.0, 0.5, 2 / 3)


def test_zeit_diff_ueber_mitternacht():
    # 20207: Beginn 09:00, Schluss 00:25 am Folgetag; XML liefert dieselben Werte
    assert v.zeit_diff("09:00", "00:25", "09:00", "00:25") == (0, 0)
    assert v.zeit_diff("09:00", "00:25", "9:00", "23:55") == (0, 30)
    assert v.zeit_diff("13:05", "20:06", "13:00", "20:06") == (5, 0)
    db, ds = v.zeit_diff(None, "20:06", "13:00", "20:06")
    assert math.isnan(db) and math.isnan(ds)


def test_top_key_normalisiert_xml_schreibweisen():
    assert v.top_key("Tagesordnungspunkt\xa07") == "Tagesordnungspunkt 7"
    assert v.top_key("Zusatzpunkt 3, 4") == "Zusatzpunkt 3"
    assert v.top_key("Tagesordnungspunkt 31 a") == "Tagesordnungspunkt 31a"
    assert v.top_key("Zur Geschäftsordnung") is None and v.top_key(float("nan")) is None


def test_gold_sortiert_stellt_lesefolge_her():
    gb = pd.DataFrame({"rede_id": [None, "R1", None], "top_id": ["TOP 1", None, "TOP 1"],
                       "segment_nr": [1, 0, 0], "nachname": ["Frage2", "Rede", "Frage1"]})
    pos = {("TOP 1", 0): 0, ("TOP 1", 1): 1, ("R1", 0): 2}
    sortiert, ohne = v.gold_sortiert(gb, pos)
    assert sortiert.nachname.tolist() == ["Frage1", "Frage2", "Rede"] and ohne == 0
    _, ohne = v.gold_sortiert(gb, {("R1", 0): 0})
    assert ohne == 2


def test_xml_hilfen_aus_minimalem_protokoll():
    from lxml import etree
    xml = """<dbtplenarprotokoll><sitzungsverlauf>
      <tagesordnungspunkt top-id="Tagesordnungspunkt 1">
        <p klasse="T_NaS">Beratung des Antrags</p>
        <p klasse="T_Drs">Drucksachen 20/100, 20/200</p>
        <rede id="ID1"><p klasse="redner"><redner id="1"><name><nachname>Merz</nachname></name></redner>Merz:</p>
          <p klasse="J_1">Text.</p><name>Präsidentin:</name><p klasse="J_1">Bitte.</p>
          <p klasse="redner"><redner id="1"><name><nachname>Merz</nachname></name></redner>Merz:</p></rede>
      </tagesordnungspunkt>
      <tagesordnungspunkt top-id="Tagesordnungspunkt 2">
        <p klasse="J">Ich rufe den Tagesordnungspunkt 2 auf:</p><p klasse="T_fett">Befragung der Bundesregierung</p>
        <p klasse="redner"><redner id="2"><name><nachname>Scholz</nachname></name></redner>Scholz:</p>
        <p klasse="T_Drs">Drucksache 20/300</p>
      </tagesordnungspunkt>
      <tagesordnungspunkt top-id="Tagesordnungspunkt 2"><name>Vizepräsident:</name></tagesordnungspunkt>
    </sitzungsverlauf></dbtplenarprotokoll>"""
    root = etree.fromstring(xml)
    assert v.xml_befragung_tops(root) == {"Tagesordnungspunkt 2"}
    drs = v.xml_aufruf_drucksachen(root)
    assert [(d, deb) for _, d, deb in drs] == [("20/100", True), ("20/200", True), ("20/300", False)]
    pos = v.gold_positionen(root)
    # Lesefolge: Merz (Rede-Segment 0), Präsidentin (1), Merz (2), Scholz (TOP 2, Segment 0), Vizepräsident (TOP 2 zweites Element)
    assert pos[("ID1", 0)] == 0 and pos[("ID1", 2)] == 2 and pos[("Tagesordnungspunkt 2", 0)] == 3
