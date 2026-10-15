"""Sentiment-Statistik: netto je Fraktion und Phase, Themenzentrierung, Anteil negativer Reden, Kruskal-Wallis, H3.

Läuft nur, wenn data/processed/sentiment.parquet existiert (BERT-Lauf). Spalten: rede_id, p_pos, p_neu, p_neg,
netto_rede oder netto (fehlt beides: p_pos − p_neg), label_rede oder label (fehlt beides: argmax), n_fenster.
Ob H3 im Bericht steht, entscheidet die vorab festgelegte Schwelle (κ BERT–Annotator-Mehrheit < 0,3 → nur
deskriptiv, κ aus sentiment_kappa.py), nicht dieses Skript.
Aufruf nach metrics.py und stats.py: python -m bundestag.stats_sentiment
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as st

from .config import PROCESSED
from .metrics import (FRAKTIONEN, KURZ, PHASE_KURZ, PHASEN, SENTIMENT_ALIAS, kennzahl, lade_reden, laufzeit,
                      schreibe_kennzahlen)
from .stats import METRICS, MIN_REDEN_KW, SEED, bootstrap, ki

SENTIMENT = PROCESSED / "sentiment.parquet"
WERTE = ["netto_rede", "netto_zentriert", "negativ"]
NACH = ("minderheit", "nach_vertrauensfrage")  # H3: nach Sitzung 196 (Ampel-Bruch)


def verbinde(r: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Hauptstatistik-Reden mit Sentiment; Reden < 50 Wörter haben keins (sentiment.MIN_WOERTER) und fallen heraus."""
    s = s.rename(columns=SENTIMENT_ALIAS).drop_duplicates("rede_id").copy()
    if "netto_rede" not in s:
        s["netto_rede"] = s.p_pos - s.p_neg
    if "label_rede" not in s:
        s["label_rede"] = s[["p_pos", "p_neu", "p_neg"]].idxmax(axis=1).str[2:]
    d = r[r.haupt].merge(s[["rede_id", "netto_rede", "label_rede"]], on="rede_id", how="inner")
    # Themenkontrolle: Abstand zum Mittel aller Reden desselben Aufruf-Blocks (Sitzung + erster Punkt des Aufrufs)
    d["netto_zentriert"] = d.netto_rede - d.groupby(["sitzung", "top_id"]).netto_rede.transform("mean")
    d["negativ"] = d.label_rede.astype(str).str.lower().str.startswith("neg").astype(float)
    return d


def _summen(d: pd.DataFrame, personen=None) -> np.ndarray:
    """Je Redner: Anzahl Reden und Summen von netto, netto_zentriert, negativ."""
    g = d.assign(eins=1.0).groupby("person")[["eins"] + WERTE].sum()
    return (g if personen is None else g.reindex(personen).fillna(0)).to_numpy(float)


def fraktion_phase(d: pd.DataFrame) -> pd.DataFrame:
    zeilen = []
    for ph in ("gesamt",) + PHASEN:
        dd = d if ph == "gesamt" else d[d.phase == ph]
        for fr in FRAKTIONEN:
            x = dd[dd.bank == fr]
            if x.empty:
                continue
            s = _summen(x)
            bs = bootstrap(s)
            punkt = s.sum(axis=0)
            for j, w in enumerate(WERTE, start=1):
                lo, hi = ki(bs[:, j] / bs[:, 0])
                zeilen.append({"phase": ph, "gruppe": fr, "mass": w, "wert": punkt[j] / punkt[0], "ki_unten": lo,
                               "ki_oben": hi, "n_reden": int(punkt[0]), "n_redner": len(s)})
    return pd.DataFrame(zeilen)


def bloecke(d: pd.DataFrame) -> pd.DataFrame:
    """Mittleres netto je Aufruf-Block (Bezug der Themenzentrierung)."""
    return (d.groupby(["sitzung", "top_id"]).netto_rede.agg(["size", "mean"])
            .rename(columns={"size": "n_reden", "mean": "netto_mittel"}).reset_index())


def kruskal_sentiment(d: pd.DataFrame) -> pd.DataFrame:
    """Ein Wert je MdB (Mittel netto, wie sentiment_netto_mittel) mit ≥ 3 Reden; fraktionslos ist keine Fraktion."""
    x = d[d.ist_mdb & d.bank.isin(FRAKTIONEN[:-1])]
    fr = x.groupby(["person", "bank"]).size().sort_values(ascending=False).reset_index().drop_duplicates("person")
    m = x.groupby("person").netto_rede.agg(["size", "mean"]).join(fr.set_index("person").bank)
    m = m[m["size"] >= MIN_REDEN_KW]
    gruppen = [g["mean"].to_numpy() for _, g in m.groupby("bank")]
    H, p = st.kruskal(*gruppen)
    n = len(m)
    return pd.DataFrame([{"gruppe": "alle", "n_mdb": n, "H": H, "df": len(gruppen) - 1, "p": p, "epsilon2": H / (n - 1)}]
                        + [{"gruppe": f, "n_mdb": len(g), "mittel": g["mean"].mean()} for f, g in m.groupby("bank")])


def h3(d: pd.DataFrame) -> pd.DataFrame:
    """FDP nach 196 näher an CDU/CSU derselben Phase als am eigenen Vorher-Wert: Abstandsdifferenz
    |FDP_nach − FDP_vor| − |FDP_nach − CDU_nach| > 0, Bootstrap getrennt über FDP- und CDU/CSU-Redner."""
    zeilen = []
    for mass in ("netto_rede", "netto_zentriert"):
        fdp, cdu = d[d.bank == "FDP"], d[(d.bank == "CDU/CSU") & d.phase.isin(NACH)]
        personen = fdp.person.unique()
        f_vor = _summen(fdp[fdp.phase == "koalition"], personen)
        f_nach = _summen(fdp[fdp.phase.isin(NACH)], personen)
        j = 1 + WERTE.index(mass)
        # derselbe FDP-Resample für vorher und nachher (Redner kommen in beiden Phasen vor)
        bs_f = bootstrap(np.hstack([f_vor[:, [0, j]], f_nach[:, [0, j]]]))
        bs_c = bootstrap(_summen(cdu)[:, [0, j]], seed=SEED + 1)
        fv, fn, cn = bs_f[:, 1] / bs_f[:, 0], bs_f[:, 3] / bs_f[:, 2], bs_c[:, 1] / bs_c[:, 0]
        p = {"fdp_vor": f_vor[:, j].sum() / f_vor[:, 0].sum(), "fdp_nach": f_nach[:, j].sum() / f_nach[:, 0].sum(),
             "cdu_nach": cdu[mass].mean()}
        werte = {"fdp_vor": (p["fdp_vor"], fv), "fdp_nach": (p["fdp_nach"], fn), "cdu_nach": (p["cdu_nach"], cn),
                 "fdp_nach_minus_cdu_nach": (p["fdp_nach"] - p["cdu_nach"], fn - cn),
                 "fdp_nach_minus_fdp_vor": (p["fdp_nach"] - p["fdp_vor"], fn - fv),
                 "abstandsdifferenz": (abs(p["fdp_nach"] - p["fdp_vor"]) - abs(p["fdp_nach"] - p["cdu_nach"]),
                                       np.abs(fn - fv) - np.abs(fn - cn))}
        for k, (w, b) in werte.items():
            lo, hi = ki(b)
            zeilen.append({"mass": mass, "groesse": k, "wert": w, "ki_unten": lo, "ki_oben": hi})
    out = pd.DataFrame(zeilen)
    out["h3_gestuetzt"] = (out.groesse == "abstandsdifferenz") & (out.ki_unten > 0)
    return out


def auswerten(r: pd.DataFrame, s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = verbinde(r, s)
    return {"sentiment_fraktion_phase": fraktion_phase(d), "sentiment_bloecke": bloecke(d),
            "kruskal_sentiment": kruskal_sentiment(d), "h3": h3(d),
            "sentiment_reden": d[["rede_id", "sitzung", "top_id", "phase", "bank", "person", "netto_rede",
                                  "netto_zentriert", "negativ"]]}


MASS_KURZ = {"netto_rede": "nettoMittel", "netto_zentriert": "nettoZentriert", "negativ": "anteilNegativ"}
H3_KURZ = {"fdp_vor": "FdpVor", "fdp_nach": "FdpNach", "cdu_nach": "UnionNach",
           "fdp_nach_minus_cdu_nach": "FdpNachMinusUnionNach", "fdp_nach_minus_fdp_vor": "FdpNachMinusFdpVor",
           "abstandsdifferenz": "Abstandsdifferenz"}


def kennzahlen_sentiment(out: dict[str, pd.DataFrame]) -> list[dict]:
    q = "stats_sentiment.py"
    z = [kennzahl(f"{MASS_KURZ[x.mass]}{KURZ[x.gruppe]}{PHASE_KURZ[x.phase]}", x.wert,
                  "Anteil" if x.mass == "negativ" else "netto", f"{q}:sentiment_fraktion_phase", (x.ki_unten, x.ki_oben))
         for x in out["sentiment_fraktion_phase"].itertuples()]
    a = out["kruskal_sentiment"].iloc[0]
    z += [kennzahl("kwSentimentH", a.H, "H", f"{q}:kruskal_sentiment"),
          kennzahl("kwSentimentP", a.p, "p", f"{q}:kruskal_sentiment"),
          kennzahl("kwSentimentEpsilon", a.epsilon2, "Epsilon-Quadrat", f"{q}:kruskal_sentiment"),
          kennzahl("kwSentimentN", int(a.n_mdb), "MdB", f"{q}:kruskal_sentiment")]
    for x in out["h3"].itertuples():
        v = "" if x.mass == "netto_rede" else "Zentriert"
        z.append(kennzahl(f"hDrei{H3_KURZ[x.groesse]}{v}", x.wert, "netto", f"{q}:h3", (x.ki_unten, x.ki_oben)))
    h = out["h3"]
    z.append(kennzahl("hDreiGestuetzt", int(h[(h.mass == "netto_rede")].h3_gestuetzt.any()), "ja=1", f"{q}:h3"))
    return z


def main() -> None:
    if not SENTIMENT.exists():
        print(f"{SENTIMENT} fehlt: Sentiment-Statistik übersprungen")
        sys.exit(0)
    METRICS.mkdir(exist_ok=True)
    with laufzeit("stats_sentiment"):
        out = auswerten(lade_reden(), pd.read_parquet(SENTIMENT))
        for name, df in out.items():
            df.to_parquet(METRICS / f"{name}.parquet", index=False)
    schreibe_kennzahlen(kennzahlen_sentiment(out), "stats_sentiment.py")
    print(out["h3"].round(4).to_string(index=False))
    print(out["kruskal_sentiment"].head(1).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
