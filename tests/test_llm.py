"""LLM-Schritt ohne Netz: Pydantic-Validierung, Cache-Treffer, Wiederholung, Divergenzklassen (Ollama gemockt)."""
import json

import pytest
from pydantic import ValidationError

from bundestag import llm

GUELTIG = {"gegenstaende": [{"titel": "Zweite und dritte Beratung des von der Fraktion der CDU/CSU eingebrachten "
                                       "Entwurfs eines Gesetzes zur Begrenzung des illegalen Zustroms von "
                                       "Drittstaatsangehörigen nach Deutschland (Zustrombegrenzungsgesetz)",
                             "kurztitel": "Zustrombegrenzungsgesetz", "drucksachen": ["20/12804", "20/13648"],
                             "typ": "gesetzentwurf", "beratungsstufe": "zweite und dritte"}]}


class Antwort:
    def __init__(self, inhalt):
        self.inhalt = inhalt

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.inhalt}, "eval_count": 12, "prompt_eval_count": 34,
                "total_duration": 2_000_000_000, "load_duration": 500_000_000}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    return tmp_path


def test_extraktion_gueltig():
    e = llm.Extraktion.model_validate_json(json.dumps(GUELTIG))
    assert e.gegenstaende[0].drucksachen == ["20/12804", "20/13648"]


@pytest.mark.parametrize("feld, wert", [("typ", "gesetz"), ("beratungsstufe", "vierte"),
                                        ("drucksachen", ["20/13648A"]), ("drucksachen", ["Drucksache 20/1"])])
def test_extraktion_ungueltig(feld, wert):
    d = json.loads(json.dumps(GUELTIG))
    d["gegenstaende"][0][feld] = wert
    with pytest.raises(ValidationError):
        llm.Extraktion.model_validate(d)


def test_zusammenfassung_wortgrenzen():
    ok = {"kernpunkt": "Begrenzung der Zuwanderung", "positionen": [
        {"fraktion": "SPD", "haltung": "ablehnend", "begruendung": "verfassungsrechtlich bedenklich"}]}
    llm.Zusammenfassung.model_validate(ok)
    zu_lang = json.loads(json.dumps(ok))
    zu_lang["positionen"][0]["begruendung"] = " ".join(["Wort"] * 21)
    with pytest.raises(ValidationError):
        llm.Zusammenfassung.model_validate(zu_lang)
    with pytest.raises(ValidationError):
        llm.Zusammenfassung.model_validate(ok | {"positionen": [ok["positionen"][0] | {"haltung": "dafür"}]})


def test_cache_treffer_ohne_netz(cache, monkeypatch):
    aufrufe = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: aufrufe.append(1) or Antwort(json.dumps(GUELTIG)))
    r1 = llm.frage("s", "u", llm.Extraktion, "qwen2.5:7b", llm.OPT_GESETZE)
    assert r1["objekt"] is not None and not r1["versuche"][0]["cache"] and len(aufrufe) == 1

    def kein_netz(*a, **k):
        raise AssertionError("Netzaufruf trotz Cache")
    monkeypatch.setattr(llm.requests, "post", kein_netz)
    r2 = llm.frage("s", "u", llm.Extraktion, "qwen2.5:7b", llm.OPT_GESETZE)
    assert r2["versuche"][0]["cache"] and r2["objekt"] == r1["objekt"]
    assert llm._messwerte(r2)["sekunden"] == 1.5  # gespeicherte Dauer ohne load_duration, nicht 0
    assert len(list(cache.rglob("*.json"))) == 1
    # anderes Modell = anderer Schlüssel, nur_cache liefert dann „offen“ statt eines Netzaufrufs
    assert llm.frage("s", "u", llm.Extraktion, "qwen2.5:3b", llm.OPT_GESETZE, nur_cache=True)["offen"]


def test_wiederholung_mit_korrektur(cache, monkeypatch):
    kaputt = json.loads(json.dumps(GUELTIG))
    kaputt["gegenstaende"][0]["drucksachen"] = ["20/13648A"]
    gesendet = []

    def post(url, json=None, timeout=None):
        gesendet.append(json["messages"])
        return Antwort(__import__("json").dumps(kaputt if len(gesendet) == 1 else GUELTIG))
    monkeypatch.setattr(llm.requests, "post", post)
    r = llm.frage("s", "u", llm.Extraktion, "m", llm.OPT_GESETZE)
    assert r["objekt"] is not None and len(r["versuche"]) == 2
    assert len(gesendet[1]) == 4 and "verletzt das Schema" in gesendet[1][3]["content"]

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Antwort("kein JSON"))
    r = llm.frage("s", "anders", llm.Extraktion, "m", llm.OPT_GESETZE)
    assert r["objekt"] is None and r["fehler"] and len(r["versuche"]) == 2


def test_netzfehler_ungueltig_ohne_cache(cache, monkeypatch):
    def fehler(*a, **k):
        raise llm.requests.ConnectionError("aus")
    monkeypatch.setattr(llm.requests, "post", fehler)
    r = llm.frage("s", "u", llm.Extraktion, "m", llm.OPT_GESETZE)
    assert r["objekt"] is None and "ConnectionError" in r["fehler"] and not list(cache.rglob("*.json"))


def test_vergleiche_klassen():
    regex = [{"titel": GUELTIG["gegenstaende"][0]["titel"], "kurztitel": "Zustrombegrenzungsgesetz",
              "drucksachen": ["20/12804", "20/13648"], "typ": "gesetzentwurf", "beratungsstufe": "zweite und dritte"}]
    g = GUELTIG["gegenstaende"][0]
    assert llm.vergleiche(regex, [g])["klasse"] == "gleich"
    assert llm.vergleiche(regex, None)["klasse"] == "ungueltig"
    assert llm.vergleiche(regex, [g | {"drucksachen": ["20/12804"]}])["klasse"] == "drucksache_fehlt_llm"
    assert llm.vergleiche(regex, [g | {"drucksachen": ["20/12804", "20/13648", "20/1"]}])["klasse"] == \
        "drucksache_fehlt_regex"
    assert llm.vergleiche(regex, [g | {"typ": "antrag"}])["klasse"] == "typ_abweichend"
    assert llm.vergleiche(regex, [g | {"beratungsstufe": "erste"}])["klasse"] == "stufe_abweichend"
    assert llm.vergleiche(regex, [g | {"kurztitel": None}])["klasse"] == "nur_titelvariante"
    # abgespaltene Beschlussempfehlung: gleiche Drucksachen, aber Typabweichung
    geteilt = [g | {"drucksachen": ["20/12804"]},
               {"titel": "Beschlussempfehlung", "kurztitel": None, "drucksachen": ["20/13648"],
                "typ": "beschlussempfehlung", "beratungsstufe": "beratung"}]
    v = llm.vergleiche(regex, geteilt)
    assert v["klasse"] == "typ_abweichend" and v["drs_gleich"] and v["titel_quote"] == 1.0
    # gekürzter Block: nicht sichtbare Drucksachen zählen nicht als fehlend
    assert llm.vergleiche(regex, [g | {"drucksachen": ["20/12804"]}], sichtbar={"20/12804"})["klasse"] == "gleich"


def test_kuerze_und_erwartung():
    text, gekuerzt = llm.kuerze("wort " * 1000)
    assert gekuerzt and len(text) <= llm.BLOCK_MAX_TOKEN * llm.ZEICHEN_JE_TOKEN + 2
    assert llm.kuerze("kurz") == ("kurz", False)
    titel = "Zweite und dritte Beratung des von der Fraktion der CDU/CSU eingebrachten Entwurfs eines Gesetzes"
    assert llm.erwartung("CDU/CSU", titel, 211, "abgelehnt") == ("zustimmend", "einbringer")
    assert llm.erwartung("SPD", titel, 211, "abgelehnt")[0] is None  # nach dem Ampel-Bruch keine Mehrheitsannahme
    reg = "Erste Beratung des von der Bundesregierung eingebrachten Entwurfs eines Gesetzes"
    assert llm.einbringer(reg, 150) == {"SPD", "BÜNDNIS 90/DIE GRÜNEN", "FDP"}
    assert llm.einbringer(reg, 200) == {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}
