"""Synthetische Fälle für metrics.py, stats.py und stats_sentiment.py (Median/IQR, pro Sitz, Bootstrap, Schlüssel)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bundestag import metrics, stats, stats_sentiment
from bundestag.metrics import LAENDER, REG_MIT, REG_OHNE

ROOT = metrics.PROCESSED.parents[1]


def test_quartile_median_iqr():
    q = metrics.quartile(pd.Series([100, 200, 300, 400, 1000]))
    assert q["woerter_median"] == 300
    assert q["woerter_mittel"] == 400
    # lineare Interpolation (pandas-Standard): Q1 = 200, Q3 = 400
    assert (q["woerter_q1"], q["woerter_q3"], q["woerter_iqr"]) == (200, 400, 200)
    assert np.isnan(metrics.quartile(pd.Series([], dtype=float))["woerter_median"])


def test_phase_grenzen_aus_datum():
    assert metrics.phase_von_datum(date(2024, 11, 6)) == "koalition"
    assert metrics.phase_von_datum("2024-11-07") == "minderheit"
    assert metrics.phase_von_datum("2024-12-15") == "minderheit"
    assert metrics.phase_von_datum("2024-12-16") == "nach_vertrauensfrage"


def test_phase_datum_passt_zu_sitzungsnummer():
    p = metrics.PROCESSED / "sitzungen.parquet"
    if not p.exists():
        pytest.skip("sitzungen.parquet fehlt")
    s = pd.read_parquet(p).set_index("sitzung").datum
    assert (s[196], s[197], s[205]) == ("2024-11-06", "2024-11-07", "2024-12-16")
    nummer = s.index.map(lambda n: "koalition" if n <= 196 else "minderheit" if n <= 204 else "nach_vertrauensfrage")
    assert (s.map(metrics.phase_von_datum).to_numpy() == np.asarray(nummer)).all()


def test_bank_trennt_regierung_laender_fraktion():
    assert metrics.bank("MdB", True, "SPD") == "SPD"
    assert metrics.bank("Bundesminister der Verteidigung", False, None) == REG_OHNE
    assert metrics.bank("Bundesminister für Wirtschaft und Klimaschutz", True, "BÜNDNIS 90/DIE GRÜNEN") == REG_MIT
    assert metrics.bank("Ministerpräsident (Bayern)", False, None) == LAENDER
    assert metrics.bank("Wehrbeauftragte des Deutschen Bundestages", False, None) == LAENDER


def _reden():
    """Zwei Sitzungen, SPD 10 Sitze, AfD 5 Sitze; je eine Regierungsrede ohne Sitz."""
    zeilen = [("a", "SPD", "SPD", 100, 146), ("a", "SPD", "SPD", 300, 146), ("b", "SPD", "SPD", 200, 197),
              ("c", "AfD", "AfD", 400, 146), ("m", REG_OHNE, None, 900, 146)]
    r = pd.DataFrame(zeilen, columns=["person", "bank", "fraktion", "woerter", "sitzung"])
    r["phase"] = np.where(r.sitzung <= 196, "koalition", "minderheit")
    r["haupt"], r["redetyp"], r["partei"], r["ist_mdb"] = True, "plenarrede", "SPD", r.bank.ne(REG_OHNE)
    r["rede_id"] = [f"{s}-{i}" for i, s in enumerate(r.sitzung)]
    ki = r.iloc[[3]].assign(redetyp="kurzintervention", haupt=False, woerter=50, rede_id="146-99")
    sitzungen = pd.DataFrame({"sitzung": [146, 197], "phase": ["koalition", "minderheit"]})
    sitze_df = pd.DataFrame({"sitzung": [146, 146, 197, 197], "fraktion": ["SPD", "AfD", "SPD", "AfD"],
                             "sitze": [10, 5, 10, 5]})
    return pd.concat([r, ki], ignore_index=True), sitzungen, sitze_df


def test_pro_sitz_ohne_regierungsbank():
    r, sitzungen, sitze_df = _reden()
    assert metrics.sitze_mittel(sitze_df, sitzungen.sitzung)["SPD"] == 10
    fk = metrics.fraktionen_kennzahlen(r, sitze_df, sitzungen, wpm=100.0).set_index(["phase", "gruppe"])
    spd = fk.loc[("gesamt", "SPD")]
    assert (spd.reden_n, spd.reden_pro_sitz, spd.woerter_pro_sitz) == (3, 0.3, 60.0)
    assert spd.minuten == 6.0
    assert spd.woerteranteil == pytest.approx(600 / 1000)
    assert spd.sitzanteil == pytest.approx(10 / 15)
    afd = fk.loc[("gesamt", "AfD")]
    assert (afd.kurzinterventionen_n, afd.kurzinterventionen_pro_sitz, afd.kurzinterventionen_anteil) == (1, 0.2, 1.0)
    reg = fk.loc[("gesamt", REG_OHNE)]
    assert reg.reden_n == 1 and np.isnan(reg.reden_pro_sitz)
    assert fk.loc[("koalition", "SPD")].reden_pro_sitz_je_sitzung == 0.2


def test_bootstrap_seed_reproduzierbar():
    s = np.arange(40, dtype=float).reshape(20, 2)
    a, b = stats.bootstrap(s), stats.bootstrap(s)
    assert a.shape == (stats.N_BOOT, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, stats.bootstrap(s, seed=stats.SEED + 1))
    # jedes Resample zieht 20 Redner: Summen liegen zwischen 20 × Minimum und 20 × Maximum
    assert a[:, 0].min() >= 0 and a[:, 0].max() <= 20 * 38


def test_bootstrap_fraktion_reproduzierbar():
    r, sitzungen, sitze_df = _reden()
    a = stats.bootstrap_fraktion(r, sitze_df, sitzungen)
    b = stats.bootstrap_fraktion(r, sitze_df, sitzungen)
    pd.testing.assert_frame_equal(a, b)
    x = a[(a.phase == "gesamt") & (a.gruppe == "SPD") & (a.kennzahl == "woerter_mittel")].iloc[0]
    assert x.wert == 200 and x.ki_unten <= 200 <= x.ki_oben


def test_dauer_ansage():
    def minuten(t):
        m = metrics.DAUER_RE.search(t)
        return metrics._zahl(m["h"]) * 60 + float(m["hm"] or 0) if m["h"] else float(m["m"])
    assert minuten("Für die Aussprache ist eine Dauer von 39 Minuten vereinbart.") == 39
    assert minuten("eine Redezeit von insgesamt vier Stunden und 30 Minuten vereinbart") == 270
    assert minuten("eine Dauer von einer Stunde vereinbart") == 60
    assert metrics.DAUER_RE.search("Für die Aussprache wurde eine Drei-Minuten-Runde vereinbart.") is None


def test_kennzahlen_schluessel(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "KENNZAHLEN_CSV", tmp_path / "k.csv")
    metrics.schreibe_kennzahlen([metrics.kennzahl("redenGesamt", 1, "Reden", "x.py:t")], "x.py")
    out = metrics.schreibe_kennzahlen([metrics.kennzahl("wpmMedian", 2, "W", "y.py:t")], "y.py")
    assert list(out.schluessel) == ["redenGesamt", "wpmMedian"]
    with pytest.raises(ValueError):
        metrics.schreibe_kennzahlen([metrics.kennzahl("hEins-SPD", 1, "pp", "x.py:t")], "x.py")
    k = metrics.PROCESSED / "kennzahlen.csv"
    if k.exists():
        echt = pd.read_csv(k)
        assert echt.schluessel.str.fullmatch(r"[A-Za-z]+").all() and echt.schluessel.is_unique


def test_h2_differenz_cluster():
    """Redner p (FDP) vorher 2 Reden, beide mit SPD-Beifall; nachher 2 Reden ohne: Differenz je Rede = −1."""
    r = pd.DataFrame({"rede_id": ["1", "2", "3", "4"], "sitzung": [150, 160, 200, 210], "person": "p",
                      "bank": "FDP", "fraktion": "FDP", "partei": "FDP", "haupt": True, "woerter": 500})
    ev = pd.DataFrame({"sitzung": [150, 160], "rede_id": ["1", "2"], "zeile": [5, 9], "applaus_fraktion": "SPD",
                       "stark": True})
    h = stats.h2(r, ev)
    d = h[(h.variante == "stark") & (h.richtung == "SPD/Grüne→FDP-Reden") & (h.phase == "diff")].set_index("mass")
    assert d.loc["je_rede"].wert == -1.0
    assert d.loc["je_1000_woerter"].wert == -2.0


def test_stats_sentiment_synthetisch():
    rng = np.random.default_rng(0)
    zeilen = []
    for i in range(240):
        fr = ["FDP", "CDU/CSU", "SPD", "AfD"][i % 4]
        ph = ["koalition", "minderheit", "nach_vertrauensfrage"][i % 3]
        zeilen.append({"rede_id": str(i), "sitzung": 150 if ph == "koalition" else 200 if ph == "minderheit" else 210,
                       "top_id": f"TOP {i % 10}", "phase": ph, "bank": fr, "person": f"{fr}{i % 7}", "ist_mdb": True,
                       "haupt": True})
    r = pd.DataFrame(zeilen)
    # FDP nachher wie CDU/CSU (−0,3), vorher +0,2: H3 muss gestützt sein
    basis = np.where(r.bank == "CDU/CSU", -0.3, np.where((r.bank == "FDP") & (r.phase == "koalition"), 0.2,
                                                       np.where(r.bank == "FDP", -0.3, 0.0)))
    netto = basis + rng.normal(0, 0.05, len(r))
    s = pd.DataFrame({"rede_id": r.rede_id, "p_pos": (1 + netto) / 2, "p_neu": 0.0, "p_neg": (1 - netto) / 2})
    out = stats_sentiment.auswerten(r, s)
    h3 = out["h3"].set_index(["mass", "groesse"])
    assert h3.loc[("netto_rede", "abstandsdifferenz")].ki_unten > 0
    assert bool(h3.loc[("netto_rede", "abstandsdifferenz")].h3_gestuetzt)
    zentriert = out["sentiment_reden"].groupby(["sitzung", "top_id"]).netto_zentriert.mean()
    assert np.allclose(zentriert, 0)
    kw = out["kruskal_sentiment"].iloc[0]
    assert kw.epsilon2 == pytest.approx(kw.H / (kw.n_mdb - 1))
    a = stats_sentiment.auswerten(r, s)["sentiment_fraktion_phase"]
    pd.testing.assert_frame_equal(a, out["sentiment_fraktion_phase"])
    keys = pd.DataFrame(stats_sentiment.kennzahlen_sentiment(out)).schluessel
    assert keys.str.fullmatch(r"[A-Za-z]+").all() and keys.is_unique


def _wissing_reden(datum_nach: str) -> pd.DataFrame:
    """Wissing: bis zum FDP-Austritt FDP, danach Stammdaten-Code „Plos“, den metrics.abgeordnete als „parteilos“ ausgibt."""
    n = 3
    return pd.DataFrame({"person": [metrics.norm_name("Dr. Volker Wissing")] * n, "name": ["Dr. Volker Wissing"] * n,
                         "partei": ["FDP", "FDP", "Plos"], "ist_mdb": [True] * n, "bank": [REG_MIT] * n,
                         "fraktion": ["FDP", "FDP", "fraktionslos"], "haupt": [True] * n, "sitzung": [150, 190, 197],
                         "datum": ["2024-02-01", "2024-10-09", datum_nach], "redetyp": ["plenarrede"] * n,
                         "woerter": [500, 600, 700]})


def test_abgeordnete_gibt_partei_wechsel_aus():
    # erste Rede als parteilos erst am 2024-12-19, Austritt laut Stammdaten 2024-11-08
    r = _wissing_reden("2024-12-19")
    b = pd.DataFrame({"redetyp": ["plenarrede"] * 3, "name": ["Dr. Volker Wissing"] * 3})
    stamm = pd.DataFrame({"name_key": ["volker wissing"] * 2, "partei": ["FDP", "Plos"],
                          "von": [date(2021, 10, 26), date(2024, 11, 8)]})
    out = metrics.abgeordnete(r, b, 130.0, None, stamm)
    assert "partei_wechsel" in out.columns
    assert out.partei_wechsel.iloc[0] == "FDP → parteilos ab 2024-11-08"
    ohne = metrics.abgeordnete(r, b, 130.0, None, stamm.iloc[:0])   # keine Periode: Datum der ersten Rede
    assert ohne.partei_wechsel.iloc[0] == "FDP → parteilos ab 2024-12-19"


def test_abgeordnete_parquet_wissing():
    p = metrics.PROCESSED / "abgeordnete.parquet"
    if not p.exists():
        pytest.skip("abgeordnete.parquet fehlt: metrics.py noch nicht gelaufen")
    a = pd.read_parquet(p)
    w = a[a.name.str.contains("Wissing")]
    assert len(w) == 1 and w.partei_wechsel.iloc[0] == "FDP → parteilos ab 2024-11-08"
