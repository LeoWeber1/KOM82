"""Unsicherheit und Hypothesentests ohne Sentiment: Bootstrap-KI je Fraktion, Kruskal-Wallis, H1, H2.

Ausgaben in data/processed/metrics/: bootstrap_fraktion, kruskal_laenge, h1, h1_kurzinterventionen, h2 (Parquet).
Aufruf nach metrics.py: python -m bundestag.stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from . import applaus
from .config import PROCESSED
from .metrics import (FRAKTIONEN, KURZ, LAENDER, PARTEI_FRAKTION, PHASE_KURZ, PHASEN, REG_MIT, REG_OHNE, kennzahl,
                      lade_reden, lade_sitzungen, laufzeit, schreibe_kennzahlen, sitze, sitze_mittel)

METRICS = PROCESSED / "metrics"
SEED = 82            # Projektseed; jede Kennzahl startet einen eigenen Generator (reihenfolgeunabhängig)
N_BOOT = 1000
MIN_REDEN_KW = 3     # Kruskal-Wallis auf MdB-Ebene: ein Wert je MdB mit mindestens 3 Reden (Pseudoreplikation vermeiden)
TOLERANZ_PP = 3.0    # H1: Wörteranteil weicht höchstens 3 Prozentpunkte vom Sitzanteil ab
REGIERUNG = {"SPD", "BÜNDNIS 90/DIE GRÜNEN"}
GRENZE = 196         # H2: vor = Sitzung 146–196 (Koalition), nach = 197–212


def bootstrap(summen: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Cluster-Bootstrap über Redner: Zeilen (Redner) mit Zurücklegen ziehen, Spaltensummen je Resample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(summen), size=(n_boot, len(summen)))
    return summen[idx].sum(axis=1)


def ki(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanquantile(x, [.025, .975])
    return float(lo), float(hi)


def redner_summen(h: pd.DataFrame) -> np.ndarray:
    """Je Redner: Anzahl Reden und Wörter (Hauptstatistik)."""
    return h.groupby("person").woerter.agg(["size", "sum"]).to_numpy(float)


# Bootstrap über Redner, weil Reden derselben Person abhängig sind
def bootstrap_fraktion(r: pd.DataFrame, sitze_df: pd.DataFrame, sitzungen: pd.DataFrame) -> pd.DataFrame:
    zeilen = []
    for ph in ("gesamt",) + PHASEN:
        rr = r[r.haupt] if ph == "gesamt" else r[r.haupt & (r.phase == ph)]
        sz = sitzungen.sitzung if ph == "gesamt" else sitzungen.sitzung[sitzungen.phase == ph]
        if sz.empty:
            continue
        sm = sitze_mittel(sitze_df, sz)
        for fr in FRAKTIONEN:
            h = rr[rr.bank == fr]
            if h.empty:
                continue
            s = redner_summen(h)
            bs = bootstrap(s)
            punkt = s.sum(axis=0)
            werte = {"reden_pro_sitz": (punkt[0] / sm[fr], bs[:, 0] / sm[fr]),
                     "woerter_pro_sitz": (punkt[1] / sm[fr], bs[:, 1] / sm[fr]),
                     "woerter_mittel": (punkt[1] / punkt[0], bs[:, 1] / bs[:, 0])}
            for k, (w, b) in werte.items():
                lo, hi = ki(b)
                zeilen.append({"phase": ph, "gruppe": fr, "kennzahl": k, "wert": w, "ki_unten": lo, "ki_oben": hi,
                               "n_redner": len(s), "n_reden": int(punkt[0]), "sitze_mittel": sm[fr]})
    return pd.DataFrame(zeilen)


def kruskal_laenge(r: pd.DataFrame) -> pd.DataFrame:
    """Ein Wert je MdB (Median seiner Redelängen) mit ≥ 3 Reden als MdB; fraktionslos ist keine Fraktion."""
    h = r[r.haupt & r.ist_mdb & r.bank.isin(FRAKTIONEN[:-1])]
    # Hauptfraktion je MdB = Fraktion der meisten Reden (Wechsler zählen einmal)
    fr = h.groupby(["person", "bank"]).size().sort_values(ascending=False).reset_index().drop_duplicates("person")
    m = h.groupby("person").woerter.agg(["size", "median"]).join(fr.set_index("person").bank)
    m = m[m["size"] >= MIN_REDEN_KW]
    gruppen = [g["median"].to_numpy() for _, g in m.groupby("bank")]
    H, p = st.kruskal(*gruppen)
    n = len(m)
    zeilen = [{"gruppe": "alle", "n_mdb": n, "median_der_mdb_mediane": m["median"].median(), "H": H,
               "df": len(gruppen) - 1, "p": p, "epsilon2": H / (n - 1)}]
    zeilen += [{"gruppe": fr, "n_mdb": len(g), "median_der_mdb_mediane": g["median"].median()}
               for fr, g in m.groupby("bank")]
    return pd.DataFrame(zeilen)


def h1(r: pd.DataFrame, sitze_df: pd.DataFrame, sitzungen: pd.DataFrame) -> pd.DataFrame:
    """Wörteranteil je Fraktion (Debatten-TOPs) gegen Sitzanteil; Variante „mit_regierung" rechnet Reden der
    Regierungsbank der Fraktion der Partei zu (Anrechnung der Regierungsredezeit auf die Koalitionsfraktionen)."""
    zeilen = []
    for variante in ("mdb", "mit_regierung"):
        for ph in ("gesamt",) + PHASEN:
            h = r[r.haupt] if ph == "gesamt" else r[r.haupt & (r.phase == ph)]
            sz = sitzungen.sitzung if ph == "gesamt" else sitzungen.sitzung[sitzungen.phase == ph]
            if sz.empty:
                continue
            sm = sitze_mittel(sitze_df, sz).reindex(list(FRAKTIONEN)).fillna(0)
            if variante == "mdb":
                h = h.assign(ziel=h.bank)
            else:
                reg = h.bank.isin([REG_MIT, REG_OHNE])
                h = h.assign(ziel=h.bank.where(~reg, h.fraktion.fillna(h.partei.map(PARTEI_FRAKTION))))
            h = h[h.ziel.isin(FRAKTIONEN)]
            rng = np.random.default_rng(SEED)
            w_punkt, w_boot = {}, {}
            for fr in FRAKTIONEN:
                hw = h[h.ziel == fr].groupby("person").woerter.sum().to_numpy(float)
                w_punkt[fr] = hw.sum()
                w_boot[fr] = hw[rng.integers(0, len(hw), size=(N_BOOT, len(hw)))].sum(axis=1) if len(hw) else \
                    np.zeros(N_BOOT)
            tot_p, tot_b = sum(w_punkt.values()), sum(w_boot.values())
            for fr in FRAKTIONEN:
                sa = sm[fr] / sm.sum()
                wa = w_punkt[fr] / tot_p
                diff_b = (w_boot[fr] / tot_b - sa) * 100
                lo, hi = ki(diff_b)
                # relative Abweichung zusätzlich: 3 pp sind mehr als der ganze Sitzanteil von BSW (1,4 %) und Linke (3,8 %)
                zeilen.append({"variante": variante, "phase": ph, "gruppe": fr, "sitzanteil": sa, "woerteranteil": wa,
                               "diff_pp": (wa - sa) * 100, "diff_relativ": (wa - sa) / sa, "ki_unten": lo, "ki_oben": hi,
                               "erfuellt": abs(wa - sa) * 100 <= TOLERANZ_PP})
    return pd.DataFrame(zeilen)


def h1_kurzinterventionen(r: pd.DataFrame, sitze_df: pd.DataFrame) -> pd.DataFrame:
    """Kurzinterventionen pro Sitz Opposition vs. Regierungsfraktionen (FDP bis Sitzung 196 Regierung).
    Zählraten mit exaktem Poisson-KI, Verhältnis über das bedingte Binomial (Clopper-Pearson): bei 70 Ereignissen
    (h1_kurzinterventionen.parquet) trägt ein Redner-Bootstrap nicht; für Zählraten gilt daher das Poisson-KI."""
    s = sitze_df[sitze_df.fraktion.isin(FRAKTIONEN)].copy()
    s["seite"] = np.where(s.fraktion.isin(REGIERUNG) | ((s.fraktion == "FDP") & (s.sitzung <= GRENZE)),
                          "regierung", "opposition")
    exp = s.groupby("seite").sitze.sum() / s.sitzung.nunique()
    k = r[(r.redetyp == "kurzintervention") & r.bank.isin(FRAKTIONEN)]
    seite = np.where(k.bank.isin(REGIERUNG) | ((k.bank == "FDP") & (k.sitzung <= GRENZE)), "regierung", "opposition")
    n = pd.Series(seite).value_counts().reindex(["opposition", "regierung"]).fillna(0).astype(int)
    zeilen = []
    for si in ("opposition", "regierung"):
        x = n[si]
        lo = st.chi2.ppf(.025, 2 * x) / 2 if x else 0.0
        hi = st.chi2.ppf(.975, 2 * x + 2) / 2
        zeilen.append({"seite": si, "kurzinterventionen": x, "sitze_mittel": exp[si], "pro_sitz": x / exp[si],
                       "ki_unten": lo / exp[si], "ki_oben": hi / exp[si]})
    ko, kr = n["opposition"], n["regierung"]
    faktor = exp["regierung"] / exp["opposition"]
    p_lo = st.beta.ppf(.025, ko, kr + 1) if ko else 0.0
    p_hi = st.beta.ppf(.975, ko + 1, kr) if kr else 1.0
    zeilen.append({"seite": "verhaeltnis_opposition_regierung", "kurzinterventionen": ko + kr,
                   "pro_sitz": (ko / exp["opposition"]) / (kr / exp["regierung"]) if kr else np.inf,
                   "ki_unten": p_lo / (1 - p_lo) * faktor, "ki_oben": p_hi / (1 - p_hi) * faktor if p_hi < 1 else np.inf})
    return pd.DataFrame(zeilen)


# H2: Beifall zwischen FDP und SPD/Grünen fällt nach Sitzung 196, geprüft in beiden Richtungen
RICHTUNGEN = {"SPD/Grüne→FDP-Reden": (REGIERUNG, {"FDP"}), "FDP→SPD/Grüne-Reden": ({"FDP"}, REGIERUNG)}


def h2(r: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Rate je Rede (Anteil Reden mit Beifall der Quelle) und je 1000 Wörter (Beifall-Klammern der Quelle) für
    Reden der Zielfraktion; Differenz nachher − vorher innerhalb jedes Resamples (Cluster Redner).
    Hauptvariante „stark" = Beifall der Fraktion („bei der FDP"), „alle" schließt Teilbeifall
    („bei Abgeordneten der FDP") ein."""
    h = r[r.haupt & r.bank.ne(LAENDER)].copy()
    # Regierungsbank zählt zur Fraktion der Partei (Pistorius → SPD), wie in applaus.reden_und_klammern
    h["ziel"] = h.fraktion.fillna(h.partei.map(PARTEI_FRAKTION))
    h["seite"] = np.where(h.sitzung <= GRENZE, "vor", "nach")
    zeilen = []
    for variante in ("stark", "alle"):
        e = ev if variante == "alle" else ev[ev.stark]
        for lab, (quelle, ziel) in RICHTUNGEN.items():
            # „Beifall bei der SPD und dem BÜNDNIS 90/DIE GRÜNEN" ist ein Ereignis, nicht zwei
            k = e[e.applaus_fraktion.isin(quelle)].drop_duplicates(["sitzung", "zeile"]).groupby("rede_id").size()
            hz = h[h.ziel.isin(ziel)].assign(eins=1.0)
            hz["ereignisse"] = hz.rede_id.map(k).fillna(0)
            hz["mit"] = (hz.ereignisse > 0).astype(float)
            personen = hz.person.unique()
            teile = [hz[hz.seite == s].groupby("person")[["eins", "mit", "woerter", "ereignisse"]].sum()
                     .reindex(personen).fillna(0).to_numpy(float) for s in ("vor", "nach")]
            summen = np.hstack(teile)
            bs = bootstrap(summen)
            punkt = summen.sum(axis=0)
            for mass, (z, n, f) in {"je_rede": (1, 0, 1.0), "je_1000_woerter": (3, 2, 1000.0)}.items():
                rate_p = {s: punkt[4 * i + z] / punkt[4 * i + n] * f for i, s in enumerate(("vor", "nach"))}
                rate_b = {s: bs[:, 4 * i + z] / bs[:, 4 * i + n] * f for i, s in enumerate(("vor", "nach"))}
                for s in ("vor", "nach", "diff"):
                    w = rate_p["nach"] - rate_p["vor"] if s == "diff" else rate_p[s]
                    lo, hi = ki(rate_b["nach"] - rate_b["vor"] if s == "diff" else rate_b[s])
                    i = 0 if s == "vor" else 1
                    zeilen.append({"variante": variante, "richtung": lab, "mass": mass, "phase": s, "wert": w,
                                   "ki_unten": lo, "ki_oben": hi,
                                   "n_reden": int(punkt[0] + punkt[4]) if s == "diff" else int(punkt[4 * i]),
                                   "n_redner": len(personen)})
    return pd.DataFrame(zeilen)


# Bausteine der Makronamen in zahlen.tex, die nur Buchstaben erlauben (wie metrics.KURZ)
H2_KURZ = {"SPD/Grüne→FDP-Reden": "AnFdp", "FDP→SPD/Grüne-Reden": "VonFdp"}
MASS_KURZ = {"je_rede": "JeRede", "je_1000_woerter": "JeTausendWoerter"}
KZ_KURZ = {"reden_pro_sitz": "redenProSitz", "woerter_pro_sitz": "woerterProSitz", "woerter_mittel": "woerterMittel"}


def kennzahlen_stats(bf, kw, h1d, kid, h2d) -> list[dict]:
    q = "stats.py"
    z = []
    for x in bf.itertuples():
        z.append(kennzahl(f"{KZ_KURZ[x.kennzahl]}{KURZ[x.gruppe]}{PHASE_KURZ[x.phase]}", x.wert,
                          "Wörter" if x.kennzahl == "woerter_mittel" else "je Sitz", f"{q}:bootstrap_fraktion",
                          (x.ki_unten, x.ki_oben)))
    a = kw[kw.gruppe == "alle"].iloc[0]
    z += [kennzahl("kwLaengeH", a.H, "H", f"{q}:kruskal_laenge"), kennzahl("kwLaengeP", a.p, "p", f"{q}:kruskal_laenge"),
          kennzahl("kwLaengeEpsilon", a.epsilon2, "Epsilon-Quadrat", f"{q}:kruskal_laenge"),
          kennzahl("kwLaengeN", int(a.n_mdb), "MdB", f"{q}:kruskal_laenge"),
          kennzahl("kwLaengeDf", int(a.df), "Freiheitsgrade", f"{q}:kruskal_laenge")]
    for x in h1d[h1d.phase == "gesamt"].itertuples():
        v = "" if x.variante == "mdb" else "Reg"
        z += [kennzahl(f"hEins{v}{KURZ[x.gruppe]}", x.diff_pp, "Prozentpunkte", f"{q}:h1", (x.ki_unten, x.ki_oben)),
              kennzahl(f"hEins{v}{KURZ[x.gruppe]}Erfuellt", int(x.erfuellt), "ja=1", f"{q}:h1"),
              kennzahl(f"woerteranteil{v}{KURZ[x.gruppe]}", x.woerteranteil * 100, "Prozent", f"{q}:h1")]
        if x.variante == "mdb":
            z.append(kennzahl(f"sitzanteil{KURZ[x.gruppe]}", x.sitzanteil * 100, "Prozent", f"{q}:h1"))
    g = h1d[(h1d.phase == "gesamt")]
    z += [kennzahl("hEinsErfuelltN", int(g[g.variante == "mdb"].erfuellt.sum()), "Fraktionen", f"{q}:h1"),
          kennzahl("hEinsRegErfuelltN", int(g[g.variante == "mit_regierung"].erfuellt.sum()), "Fraktionen", f"{q}:h1")]
    name = {"opposition": "Opposition", "regierung": "Regierung", "verhaeltnis_opposition_regierung": "Verhaeltnis"}
    for x in kid.itertuples():
        z.append(kennzahl(f"kurzinterventionenProSitz{name[x.seite]}", x.pro_sitz,
                          "Verhältnis" if x.seite.startswith("verh") else "je Sitz", f"{q}:h1_kurzinterventionen",
                          (x.ki_unten, x.ki_oben)))
        if not x.seite.startswith("verh"):
            z.append(kennzahl(f"kurzinterventionen{name[x.seite]}", int(x.kurzinterventionen), "Beiträge",
                              f"{q}:h1_kurzinterventionen"))
    for x in h2d.itertuples():
        v = "" if x.variante == "stark" else "Alle"
        einheit = "Anteil Reden" if x.mass == "je_rede" else "je 1000 Wörter"
        z.append(kennzahl(f"hZwei{H2_KURZ[x.richtung]}{MASS_KURZ[x.mass]}{x.phase.capitalize()}{v}", x.wert, einheit,
                          f"{q}:h2", (x.ki_unten, x.ki_oben)))
    return z


def main() -> None:
    METRICS.mkdir(exist_ok=True)
    with laufzeit("stats_laden"):
        r = lade_reden()
        sitzungen = lade_sitzungen()
        sitze_df = sitze(sitzungen)
    with laufzeit("stats_bootstrap_fraktion"):
        bf = bootstrap_fraktion(r, sitze_df, sitzungen)
        bf.to_parquet(METRICS / "bootstrap_fraktion.parquet", index=False)
    with laufzeit("stats_kruskal_laenge"):
        kw = kruskal_laenge(r)
        kw.to_parquet(METRICS / "kruskal_laenge.parquet", index=False)
    with laufzeit("stats_h1"):
        h1d = h1(r, sitze_df, sitzungen)
        h1d.to_parquet(METRICS / "h1.parquet", index=False)
        kid = h1_kurzinterventionen(r, sitze_df)
        kid.to_parquet(METRICS / "h1_kurzinterventionen.parquet", index=False)
    with laufzeit("stats_h2_applaus_ereignisse"):
        ev = applaus.ereignisse_je_rede()  # liest applaus_ereignisse.parquet aus dem Schritt applaus
    with laufzeit("stats_h2_bootstrap"):
        h2d = h2(r, ev)
        h2d.to_parquet(METRICS / "h2.parquet", index=False)
    schreibe_kennzahlen(kennzahlen_stats(bf, kw, h1d, kid, h2d), "stats.py")
    a = kw[kw.gruppe == "alle"].iloc[0]
    print(f"KW Länge (MdB-Ebene): H={a.H:.1f}, df={int(a.df)}, p={a.p:.2g}, eps2={a.epsilon2:.3f}, n={int(a.n_mdb)}")
    g = h1d[h1d.phase == "gesamt"]
    for v in ("mdb", "mit_regierung"):
        print(f"H1 {v}: " + "; ".join(f"{KURZ[x.gruppe]} {x.diff_pp:+.1f} [{x.ki_unten:+.1f}, {x.ki_oben:+.1f}] "
                                      f"{'ja' if x.erfuellt else 'nein'}" for x in g[g.variante == v].itertuples()))
    print(kid.round(4).to_string(index=False))
    print(h2d[h2d.phase == "diff"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
