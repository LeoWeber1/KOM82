import logging

import pytest

from bundestag import extract_text as et
from bundestag.extract_text import Line, assign_sections, join_hyphenated, to_paragraphs


@pytest.mark.parametrize("parts, expected", [
    (["die Gemeinsame Außen-", "und Sicherheitspolitik"], "die Gemeinsame Außen- und Sicherheitspolitik"),
    (["Klima-, Energie-", "und Verkehrspolitik"], "Klima-, Energie- und Verkehrspolitik"),
    (["Ehe-", "oder Lebenspartner"], "Ehe- oder Lebenspartner"),
    (["Schwach-", "bzw. Leerstelle"], "Schwach- bzw. Leerstelle"),
    (["Gesetz-", "sowie Verordnungsentwürfe"], "Gesetz- sowie Verordnungsentwürfe"),
    (["Kinder-", "und/oder Jugendhilfe"], "Kinder- und/oder Jugendhilfe"),
    (["sowohl die Verhaltens-", "als auch die Verhältnisprävention"],
     "sowohl die Verhaltens- als auch die Verhältnisprävention"),
    (["EEG-", "und KWK-Umlage"], "EEG- und KWK-Umlage"),
    # Silbenfragmente, die wie Konjunktionen aussehen, werden weiter verbunden
    (["so-", "wie Weggefährten"], "sowie Weggefährten"),
    (["an-", "statt die Mehreinnahmen"], "anstatt die Mehreinnahmen"),
    (["den-", "noch heißt es"], "dennoch heißt es"),
    (["Canna-", "bis als Medizin"], "Cannabis als Medizin"),
    (["Grund-", "und"], "Grund- und"),
    (["Bun-", "destag"], "Bundestag"),
    (["CDU/CSU-", "Fraktion"], "CDU/CSU-Fraktion"),
    (["BÜNDNIS 90/", "DIE GRÜNEN"], "BÜNDNIS 90/DIE GRÜNEN"),
    (["Baerbock (BÜNDNIS 90/DIE GRÜ-", "NEN):"], "Baerbock (BÜNDNIS 90/DIE GRÜNEN):"),
    (["[BÜND-", "NIS 90/DIE GRÜNEN]"], "[BÜNDNIS 90/DIE GRÜNEN]"),
    (["des BÜNDNIS-", "SES 90/DIE GRÜNEN"], "des BÜNDNISSES 90/DIE GRÜNEN"),
    (["die GRÜNEN-", "Fraktion"], "die GRÜNEN-Fraktion"),
    (["EU-", "SILC"], "EU-SILC"),
])
def test_join_hyphenated(parts, expected):
    assert join_hyphenated(parts) == expected


def _ln(text, page=0, y0=100.0, speaker=False):
    return Line(page=page, col=0, x0=64.0, y0=y0, x1=290.0, text=text,
                speaker_font=speaker, bold_chars=0, size=10.0)


def test_head_continuation_in_times_joined():
    lines = [_ln("Annalena Baerbock (BÜNDNIS 90/DIE GRÜ-", y0=100, speaker=True),
             _ln("NEN):", y0=111),
             _ln("Sehr geehrte Frau Präsidentin!", y0=125)]
    assert to_paragraphs(lines) == ["Annalena Baerbock (BÜNDNIS 90/DIE GRÜNEN):",
                                    "Sehr geehrte Frau Präsidentin!"]


def test_sections_regular():
    lines = [_ln("Inhalt"), _ln("Beginn: 9.00 Uhr"), _ln("Rede"),
             _ln("(Schluss: 11.19 Uhr)"), _ln("Anlage 1")]
    meta = assign_sections(lines)
    assert meta == {"beginn": "09:00", "schluss": "11:19"}
    assert [ln.section for ln in lines] == ["toc", "body", "body", "body", "annex"]


def test_sections_fallback_anlage(caplog):
    lines = [_ln("Inhalt"), _ln("Anlage 1"), _ln("Beginn: 9.00 Uhr"), _ln("Rede"),
             _ln("Anlage 1"), _ln("Namensliste")]
    with caplog.at_level(logging.WARNING, logger="bundestag.extract_text"):
        meta = assign_sections(lines, source="20200.pdf")
    assert meta["schluss"] is None
    assert meta["warnung"].startswith("Schluss-Zeile fehlt")
    assert [ln.section for ln in lines] == ["toc", "toc", "body", "body", "annex", "annex"]
    assert "20200.pdf" in caplog.text


def test_sections_fallback_ohne_anlage(caplog):
    lines = [_ln("Beginn: 9.00 Uhr"), _ln("Rede")]
    with caplog.at_level(logging.WARNING):
        meta = assign_sections(lines)
    assert meta["warnung"] == "Schluss-Zeile und Anlagen-Beginn fehlen"
    assert [ln.section for ln in lines] == ["body", "body"]


def test_extract_all_logs_error_and_continues(tmp_path, monkeypatch):
    def fake_process(nr):
        if nr == 1:
            raise ValueError("kaputt")
        return {"beginn": "09:00", "schluss": None, "zeilen": 5, "woerter": 40,
                "warnung": "Schluss-Zeile fehlt, Anlagen-Beginn als Ende"} if nr == 3 else \
               {"beginn": "09:00", "schluss": "10:00", "zeilen": 5, "woerter": 40}
    monkeypatch.setattr(et, "process", fake_process)
    log_path = tmp_path / "extract_log.txt"
    counts = et.extract_all(sitzungen=[1, 2, 3], log_path=log_path, worker=1)  # Pool-Worker sähen den Patch nicht
    rows = log_path.read_text(encoding="utf-8").splitlines()[1:]
    assert counts == {"OK": 1, "WARN": 1, "ERROR": 1}
    assert [r.split("\t")[:2] for r in rows] == [["1", "ERROR"], ["2", "OK"], ["3", "WARN"]]
    assert "ValueError: kaputt" in rows[0]
