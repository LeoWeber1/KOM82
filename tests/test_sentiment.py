import pandas as pd
import pytest

from bundestag.sentiment import (FENSTER_WOERTER, clean_text, fenster_grenzen, fenster_texte, je_rede, label_aus,
                                 spalten_je_index)


def _text(n):
    return " ".join(f"wort{i}" for i in range(n))


@pytest.mark.parametrize("n, erwartet", [
    (49, []),
    (50, [50]),
    (150, [150]),
    (199, [199]),
    (200, [150, 50]),
    (340, [150, 190]),
    (350, [150, 150, 50]),
])
def test_fenster_groessen(n, erwartet):
    assert FENSTER_WOERTER == 150
    assert [bis - von for von, bis in fenster_grenzen(n)] == erwartet
    assert [w for _, w in fenster_texte(_text(n))] == erwartet


@pytest.mark.parametrize("n, erwartet", [(49, []), (50, [50]), (249, [249]), (250, [200, 50])])
def test_fenster_groessen_200_gold_v1(n, erwartet):
    assert [bis - von for von, bis in fenster_grenzen(n, 200)] == erwartet


def test_gedankenstrich_zaehlt_nicht_als_wort():
    text = "– " + _text(49) + " –"
    assert fenster_texte(text) == []
    teile = fenster_texte(text + " ende")
    assert [w for _, w in teile] == [50]
    assert teile[0][0].startswith("– wort0") and teile[0][0].endswith("– ende")


def test_fenster_decken_text_lueckenlos_ab():
    text = _text(450).replace("wort150 ", "– wort150 ")
    teile = fenster_texte(text)
    assert " ".join(t for t, _ in teile) == text
    assert teile[1][0].startswith("wort150")


def test_clean_text_wie_germansentiment():
    roh = "Am 7. Mai 2024 – siehe https://bundestag.de @user!\nGrüße"
    assert clean_text(roh) == "am sieben mai zwei null zwei vier siehe grüße"


def test_label_mapping_folgt_id2label():
    assert spalten_je_index({0: "positive", 1: "negative", 2: "neutral"}) == ["p_pos", "p_neg", "p_neu"]
    assert spalten_je_index({"0": "NEUTRAL", "1": "positive", "2": "negative"}) == ["p_neu", "p_pos", "p_neg"]


def test_label_argmax_deutsch():
    p = pd.DataFrame({"p_pos": [0.7, 0.1, 0.2], "p_neu": [0.2, 0.8, 0.3], "p_neg": [0.1, 0.1, 0.5]})
    assert label_aus(p).tolist() == ["positiv", "neutral", "negativ"]


def test_rede_laengengewichtet():
    f = pd.DataFrame({"rede_id": ["a", "a"], "redetyp": ["plenarrede"] * 2, "woerter": [200, 50],
                      "p_pos": [1.0, 0.0], "p_neu": [0.0, 1.0], "p_neg": [0.0, 0.0]})
    r = je_rede(f).iloc[0]
    assert (r.n_fenster, r.woerter, r.label) == (2, 250, "positiv")
    assert r.p_pos == pytest.approx(0.8) and r.netto == pytest.approx(0.8)


def test_schluessel_rede_id_und_redetyp():
    # reden.parquet: 146-1009 ist befragung_antwort (Hauptteil) und erklaerung_go31 (Anlage, eigene Zeilenzählung).
    f = pd.DataFrame({"rede_id": ["146-1009"] * 2, "redetyp": ["befragung_antwort", "erklaerung_go31"],
                      "woerter": [60, 60], "p_pos": [1.0, 0.0], "p_neu": [0.0, 0.0], "p_neg": [0.0, 1.0]})
    assert je_rede(f).label.tolist() == ["positiv", "negativ"]


def test_snapshot_laedt_vom_hub_wenn_lokal_fehlt(monkeypatch, tmp_path):
    """Frischer Klon ohne Hugging-Face-Cache: fehlt das Modell lokal, folgt genau ein Hub-Abruf mit fester Revision."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from bundestag import sentiment
    aufrufe = []

    def fake(repo, revision=None, local_files_only=False):
        aufrufe.append((repo, revision, local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("nicht im Cache")
        return str(tmp_path / revision)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert sentiment._snapshot() == tmp_path / sentiment.SENTIMENT_REVISION
    assert aufrufe == [(sentiment.SENTIMENT_MODEL, sentiment.SENTIMENT_REVISION, True),
                       (sentiment.SENTIMENT_MODEL, sentiment.SENTIMENT_REVISION, False)]


def test_snapshot_ohne_netz_nennt_abhilfe(monkeypatch):
    import huggingface_hub
    import pytest
    from huggingface_hub.errors import LocalEntryNotFoundError
    from bundestag import sentiment

    def fake(repo, revision=None, local_files_only=False):
        if local_files_only:
            raise LocalEntryNotFoundError("nicht im Cache")
        raise OSError("kein Netz")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    with pytest.raises(SystemExit) as fehler:
        sentiment._snapshot()
    assert "hf download oliverguhr/german-sentiment-bert --revision" in str(fehler.value)
    assert "kein Netz" in str(fehler.value)
