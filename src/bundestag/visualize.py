"""Abbildungen der Dokumentation als Vektor-PDF (nativ 13 cm breit, 8–9 pt, deutsche Beschriftung mit Dezimalkomma).

Eingaben: data/processed/metrics/{sentiment_reden, h1, bootstrap_fraktion}.parquet, top20_redner.csv,
applaus_heatmap.csv, reden.parquet (rolle_regierung), sitzungen.parquet. Jeder Titel ist eine Aussage, deren Zahl
aus den Daten berechnet wird; sie steht als Makro abbEinsTitel … abbSiebenTitel in LaTeX/abbildungen_titel.tex
für die Caption und nur mit --mit-titel zusätzlich im PDF.
Abb. 6 (Redelänge je Fraktion) und 7 (alle Redner) lesen reden.parquet bzw. abgeordnete.csv.
Ausgabe: data/figures/abb1_…pdf bis abb7_…pdf, data/figures/abbildungen.md, LaTeX/abbildungen_titel.tex.
Aufruf: python -m bundestag.visualize [--nur 1 4] [--mit-titel] [--breite-cm 13]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from .config import ABB_TITEL_TEX, FIGURES, PROCESSED, SITZUNGEN  # noqa: E402
from .metrics import FRAKTIONEN, laufzeit  # noqa: E402
from .stammdaten import norm_name  # noqa: E402
from .stats import METRICS, N_BOOT, SEED, TOLERANZ_PP, ki  # noqa: E402

CM = 1 / 2.54
# Kap. 2 setzt die Abbildungen auf 0,8–0,85 \textwidth: nativ 13 cm statt 16 cm, sonst schrumpft 8 pt auf ≈ 6,5 pt
BREITE_CM = 13.0
GROESSE: dict[str, tuple[float, float]] = {}
GRENZE = 196            # Ampel-Bruch am Abend des 06.11.2024 (Sitzung 196); Sitzung 197 am 07.11.2024
VERTRAUENSFRAGE = 205   # 16.12.2024
# die Aussage steht in der \caption (Seitenlimit); nur mit --mit-titel kommt eine 9-pt-Zeile samt Abstand hinzu
TITEL_CM = 0.6
MIT_TITEL = False
LATEX_TITEL = ABB_TITEL_TEX
TEX_NAMEN = ["abbEinsTitel", "abbZweiTitel", "abbDreiTitel", "abbVierTitel", "abbFuenfTitel", "abbSechsTitel",
             "abbSiebenTitel"]
TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
TEX_UNICODE = [("\u00a0", "~"), ("\u2009", r"\,"), ("−", "$-$"), ("→", r"$\rightarrow$"), ("±", r"$\pm$"),
               ("§ ", r"\S~")]

# Parteifarben, auf Weiß geprüft mit dataviz validate_palette.py --pairs all: CVD-ΔE ≥ 8,4 und
# Normalsicht-ΔE ≥ 15,8. Schwarz/Grau verfehlen das Chroma-Band (Konvention), FDP-Gold (2,86:1) und Grau (2,17:1)
# den 3:1-Kontrast; deshalb steht in jeder Abbildung der Fraktionsname als Text neben der Farbe
FARBE = {"SPD": "#E3000F", "CDU/CSU": "#2B2B2B", "BÜNDNIS 90/DIE GRÜNEN": "#1E7F3C", "FDP": "#C49000",
         "AfD": "#0B7FC0", "Die Linke": "#CC2E9A", "BSW": "#6A3399", "fraktionslos": "#B0B0B0"}
NAME = {"SPD": "SPD", "CDU/CSU": "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN": "Grüne", "FDP": "FDP", "AfD": "AfD",
        "Die Linke": "Linke", "BSW": "BSW", "fraktionslos": "fraktionslos"}
NOMINATIV = {"SPD": ("die SPD", False), "CDU/CSU": ("die Union", False), "BÜNDNIS 90/DIE GRÜNEN": ("die Grünen", True),
             "FDP": ("die FDP", False), "AfD": ("die AfD", False), "Die Linke": ("die Linke", False),
             "BSW": ("das BSW", False), "fraktionslos": ("Fraktionslose", True)}
INK, MUTED, GRID, BAND = "#1a1a1a", "#5a5a5a", "#e2e2e2", "#ebebeb"
# Regierung/Opposition sind keine Fraktionen: Blaugrün/Orange kollidieren mit keiner Parteifarbe
REGIERUNG, OPPOSITION = "#127E8A", "#D9661F"
# einfarbige Sequenz für Beträge: sieben Blaustufen von hell nach dunkel
BLAU = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MONATE = ["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."]
NBSP = " "
THIN = "\u2009"   # schmales Leerzeichen als Tausendertrenner; Arial hat die Glyphe, U+202F fehlt ihr


def de(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def proz(x: float, nd: int = 0) -> str:
    return f"{de(100 * x, nd)}{NBSP}%"


def tsd(n: float) -> str:
    """Ganzzahl wie im Text (DIN 5008, LaTeX zahlen.tex): ab fünf Stellen schmales Leerzeichen, vierstellig ohne
    Trennzeichen ("6368", "23 104"); vorher "4.917" und "23.104" gegen "12 583" im Text."""
    k = int(round(n))
    return f"{k:,}".replace(",", THIN) if abs(k) >= 10000 else str(k)


def satz(fr: str, verb: tuple[str, str], gross: bool = True) -> str:
    """„Die Linke spricht“ / „die Grünen sprechen“: Artikel und Numerus aus NOMINATIV."""
    n, plural = NOMINATIV[fr]
    return f"{n[0].upper() + n[1:] if gross else n} {verb[1] if plural else verb[0]}"


def formatiere(achse, nd: int = 1, prozent: bool = False) -> None:
    achse.set_major_formatter(FuncFormatter(
        (lambda v, _: f"{de(100 * v, 0)}{NBSP}%") if prozent else (lambda v, _: de(v, nd))))


def stil() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.titlesize": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "axes.axisbelow": True, "axes.titlelocation": "left",
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.05})


def breite() -> float:
    return BREITE_CM * CM


def hoehe(cm13: float) -> float:
    """Höhe in Zoll; cm13 gilt für 13 cm Breite ohne Titel und skaliert linear mit --breite-cm."""
    return (cm13 * BREITE_CM / 13 + (TITEL_CM if MIT_TITEL else 0)) * CM


def tex(text: str) -> str:
    """Titel als LaTeX-Text: erst Sonderzeichen escapen, dann Unicode (−, →, ±, §, geschütztes Leerzeichen) ersetzen."""
    s = "".join(TEX_ESC.get(c, c) for c in text)
    for alt, neu in TEX_UNICODE:
        s = s.replace(alt, neu)
    return s


def titel(fig, text: str) -> None:
    if not MIT_TITEL:
        return
    t = fig.suptitle(text, x=0.01, ha="left", fontsize=9, fontweight="bold")
    # zu lange Aussagen (Zahlen variieren mit den Daten) umbrechen statt abschneiden, bevorzugt nach „,“ „:“ „;“
    if t.get_window_extent(fig.canvas.get_renderer()).x1 > 0.99 * fig.bbox.width:
        stellen = [i + 1 for i, c in enumerate(text[:-1]) if c in ",:;" and text[i + 1] == " "] or \
            [i for i, c in enumerate(text) if c == " "]
        k = min(stellen, key=lambda i: abs(i - len(text) / 2))
        t.set_text(text[:k] + "\n" + text[k + 1:])


def speichern(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # ohne bbox_inches="tight": die Seitenbreite bleibt exakt BREITE_CM und wird in LaTeX nicht skaliert
    fig.savefig(FIGURES / name, metadata={"CreationDate": None})
    GROESSE[name] = tuple(float(v) * 2.54 for v in fig.get_size_inches())
    plt.close(fig)
    return name


def lade_sentiment() -> pd.DataFrame | None:
    """Hauptstatistik-Reden mit Sentiment (stats_sentiment) plus rolle_regierung zum Sitzungsdatum; None, wenn
    stats_sentiment mangels sentiment.parquet nichts geschrieben hat (Teillauf mit --skip-sentiment)."""
    p = METRICS / "sentiment_reden.parquet"
    if not p.exists():
        return None
    sr = pd.read_parquet(p)
    rr = pd.read_parquet(PROCESSED / "reden.parquet", columns=["rede_id", "rolle_regierung"])
    return sr.merge(rr, on="rede_id", how="left", validate="many_to_one")


def median_ki(x: pd.DataFrame) -> tuple[float, float, float]:
    """Median netto mit 95-%-KI per Cluster-Bootstrap über Redner (Schema von stats.bootstrap, Seed 82, 1000)."""
    if x.empty:   # leeres Panel im Teillauf (z. B. 146–147): np.concatenate bräche mit ValueError ab
        return np.nan, np.nan, np.nan
    gruppen = [g.to_numpy() for _, g in x.groupby("person").netto_rede]
    rng = np.random.default_rng(SEED)
    k = len(gruppen)
    boot = [np.median(np.concatenate([gruppen[i] for i in rng.integers(0, k, k)])) for _ in range(N_BOOT)]
    return float(x.netto_rede.median()), *ki(np.asarray(boot))


def abb1(sr: pd.DataFrame) -> dict:
    d = sr[sr.bank.isin(FRAKTIONEN)]
    med = d.groupby("bank").netto_rede.median()
    ordnung = med.sort_values(ascending=False).index.tolist()
    # fraktionslos sind Einzel-MdB ohne gemeinsame Linie (147 Reden von 7 Personen): auf der Achse, nicht im Titel
    kandidaten = [f for f in ordnung if f != "fraktionslos"]
    neg, pos = kandidaten[-1], kandidaten[0]
    t = (f"{satz(neg, ('spricht', 'sprechen'))} am negativsten: Median netto {de(med[neg])}; "
         f"{NOMINATIV[pos][0]} am wenigsten negativ ({de(med[pos])})")
    fig, axes = plt.subplots(1, 2, figsize=(breite(), hoehe(5.6)), sharey=True, layout="constrained")
    rng = np.random.default_rng(SEED)
    for ax, (name, maske) in zip(axes, [("bis Sitzung 196", d.sitzung <= GRENZE),
                                        ("ab Sitzung 197", d.sitzung > GRENZE)]):
        dd = d[maske]
        if dd.empty:   # Teillauf nur vor oder nur nach dem Ampel-Bruch, z. B. 146–147 ohne Sitzung ab 197
            ax.text(0.5, 0.5, f"keine Reden in Sitzung {SITZUNGEN[0]}–{SITZUNGEN[-1]}", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            print(f"abb1: Panel „{name}“ leer, keine Reden in Sitzung {SITZUNGEN[0]}–{SITZUNGEN[-1]}")
        for i, fr in enumerate(ordnung):
            x = dd[dd.bank == fr]
            if x.empty:
                continue
            ax.scatter(x.netto_rede, i + rng.uniform(-0.3, 0.3, len(x)), s=2.5, color=FARBE[fr], alpha=0.3,
                       linewidths=0)
            m, lo, hi = median_ki(x)
            ax.plot([lo, hi], [i, i], color=INK, lw=2, solid_capstyle="butt", zorder=4,
                    path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
            ax.plot(m, i, marker="D", ms=4.5, mfc="white", mec=INK, mew=1, zorder=5)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        ax.set_title(f"{name}: {tsd(len(dd))} Reden")
        ax.set_xlim(-1.02, 1.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        formatiere(ax.xaxis, 1)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.tick_params(axis="y", length=0)
    # ohne Legende: sie überlagerte die x-Beschriftung; Punkte, Raute und KI erklärt die Caption (02_ergebnisse.tex)
    fig.supxlabel("Netto-Sentiment je Rede (positiv minus negativ)", fontsize=8)
    axes[0].set_yticks(range(len(ordnung)), [NAME[f] for f in ordnung])
    axes[0].set_ylim(len(ordnung) - 0.5, -0.5)
    titel(fig, t)
    return {"datei": speichern(fig, "abb1_sentiment_fraktionen.pdf"), "titel": t,
            "kern": f"Median netto {NAME[neg]} {de(med[neg])}, {NAME[pos]} {de(med[pos])} (alle Sitzungen; "
                    f"fraktionslos {de(med.get('fraktionslos', np.nan))} nicht gewertet, keine Fraktion)",
            "quelle": "metrics/sentiment_reden.parquet (plenarrede, Debatten-TOPs, nur Fraktionen/Gruppen); "
                      "Median-KI hier berechnet (Cluster-Bootstrap über Redner, Seed 82, 1000)"}


def abb2(sr: pd.DataFrame) -> dict:
    t = pd.read_csv(PROCESSED / "top20_redner.csv").sort_values("rang").reset_index(drop=True)
    t["person"] = t.name.map(norm_name)
    t = t.join(sr.groupby("person").netto_rede.median().rename("median_netto"), on="person")
    fehlt = t.name[t.median_netto.isna()].tolist()
    if fehlt:
        print(f"abb2: kein Sentiment für {fehlt}")
    w = t.loc[t.woerter_gesamt.idxmax()]
    erst = t.iloc[0]
    tt = (f"{erst['name']} hält die meisten Reden ({erst.reden_n}), "
          f"{w['name']} spricht die meisten Wörter ({tsd(w.woerter_gesamt)})")
    # 9,2 statt 7,4 cm: bei 7,4 cm berührten sich die 20 Namen an der y-Achse
    fig, axes = plt.subplots(1, 3, figsize=(breite(), hoehe(9.2)), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    y = np.arange(len(t))
    farben = t.fraktion.map(FARBE).fillna(FARBE["fraktionslos"])
    for ax, spalte, faktor, name in [(axes[0], "reden_n", 1, "Reden (Hauptstatistik)"),
                                     (axes[1], "woerter_gesamt", 1e-3, "Wörter gesamt (Tsd.)")]:
        v = t[spalte] * faktor
        ax.hlines(y, 0, v, color=farben, lw=1.2)
        ax.scatter(v, y, s=24, color=farben, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlim(0, v.max() * 1.08)
        formatiere(ax.xaxis, 0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel(name)
    ax = axes[2]
    ax.scatter(t.median_netto, y, s=16, color=farben, edgecolors="white", linewidths=0.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(min(-0.8, t.median_netto.min() - 0.05), max(0.2, t.median_netto.max() + 0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    formatiere(ax.xaxis, 1)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlabel("Median netto")
    for a, name in zip(axes, ["Anzahl Reden", "Wörter gesamt", "Sentiment"]):
        a.set_title(name)
        a.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [f"{n} ({NAME.get(f, f)})" for n, f in zip(t.name, t.fraktion)])
    axes[0].set_ylim(len(t) - 0.5, -0.5)
    da = [f for f in FRAKTIONEN if f in set(t.fraktion)]
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FARBE[f], ms=4.5, label=NAME[f]) for f in da],
               loc="outside lower center", ncol=len(da), handlelength=1.0, handletextpad=0.2, columnspacing=0.8)
    titel(fig, tt)
    return {"datei": speichern(fig, "abb2_top20_redner.pdf"), "titel": tt,
            "kern": f"{erst['name']} {erst.reden_n} Reden; {w['name']} {tsd(w.woerter_gesamt)} Wörter",
            "quelle": "top20_redner.csv; Sentiment-Median je Redner aus metrics/sentiment_reden.parquet"}


def abb3(sr: pd.DataFrame) -> dict:
    sz = pd.read_parquet(PROCESSED / "sitzungen.parquet", columns=["sitzung", "datum"])
    sz["datum"] = pd.to_datetime(sz.datum)
    # nur Fraktionen/Gruppen: Regierungsbank (235 Reden) und Bundesrat sind keine Seite im Parlament
    d = sr[sr.bank.isin(FRAKTIONEN)].assign(seite=lambda x: np.where(x.rolle_regierung, "reg", "opp"))
    a = (d.groupby(["sitzung", "seite"]).negativ.mean().rename("anteil").reset_index()
         .merge(sz, on="sitzung").sort_values("sitzung"))
    vor, nach = d[d.sitzung <= GRENZE], d[d.sitzung > GRENZE]
    rv, rn = vor[vor.seite == "reg"].negativ.mean(), nach[nach.seite == "reg"].negativ.mean()
    ov, on = vor[vor.seite == "opp"].negativ.mean(), nach[nach.seite == "opp"].negativ.mean()
    t = (f"Vor dem Ampel-Bruch sind {proz(rv)} der Reden der Regierungsfraktionen negativ, danach {proz(rn)} "
         f"(Opposition: {proz(ov)} und {proz(on)})")
    kern = f"Regierung {proz(rv)} → {proz(rn)}, Opposition {proz(ov)} → {proz(on)} (bis 196 → ab 197)"
    if vor.empty or nach.empty:   # Teillauf nur vor oder nur nach 196/197 (146–147): Titel zeigte sonst „nan %“
        t = kern = (f"Anteil negativer Reden in Sitzung {SITZUNGEN[0]}–{SITZUNGEN[-1]}: "
                    "kein Vergleich vor/nach dem Ampel-Bruch, eine Phase ohne Reden")
    fig, ax = plt.subplots(figsize=(breite(), hoehe(4.4)), layout="constrained")
    # je Sitzung ein Wert als dünne Linie; Bruch bei 196/197, weil die Regierungsseite dort die FDP verliert
    stile = {"reg": (REGIERUNG, "Regierungsfraktionen"), "opp": (OPPOSITION, "Opposition")}
    for seite, (farbe, _) in stile.items():
        x = a[a.seite == seite]
        for _, seg in x.groupby(x.sitzung > GRENZE):
            ax.plot(seg.datum, seg.anteil, lw=0.9, color=farbe, marker="o", ms=2.2)
    datum = sz.set_index("sitzung").datum
    marken = []   # im Teillauf nur die Ereignisse, deren Sitzungen im Bereich liegen
    if {GRENZE, GRENZE + 1} <= set(datum.index):
        marken.append((datum[GRENZE] + (datum[GRENZE + 1] - datum[GRENZE]) / 2, "Ampel-Bruch ", "right"))
    if VERTRAUENSFRAGE in datum.index:
        marken.append((datum[VERTRAUENSFRAGE], " Vertrauensfrage", "left"))
        # bei 13 cm reicht der Platz rechts von Sitzung 205 (16.12.2024) nicht für das Etikett: Achse um 80 Tage verlängern
        ax.set_xlim(right=max(ax.get_xlim()[1], mdates.date2num(datum[VERTRAUENSFRAGE] + pd.Timedelta(days=80))))
    for x, text, ha in marken:
        ax.axvline(x, color=INK, lw=0.8, ls=":")
        ax.text(x, 1.0, text, transform=ax.get_xaxis_transform(), ha=ha, va="top", color=INK)
    # Sitzungen mit wenigen Reden erreichen 100 %: Luft über 1,0, damit der Punkt nicht am Rand halbiert wird
    ax.set_ylim(0, min(1.06, a.anteil.max() * 1.12))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    formatiere(ax.yaxis, prozent=True)
    ax.set_ylabel("Anteil negativer Reden")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (lambda dt: MONATE[dt.month - 1]
                                                             + (f" {dt.year}" if dt.month == 1 else ""))(mdates.num2date(v))))
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.legend(handles=[Line2D([], [], color=f, lw=0.9, marker="o", ms=2.2, label=lab) for f, lab in stile.values()],
               loc="outside lower center", ncol=2, borderaxespad=0.2)
    titel(fig, t)
    return {"datei": speichern(fig, "abb3_negativ_zeitreihe.pdf"), "titel": t,
            "kern": kern,
            "quelle": "metrics/sentiment_reden.parquet (label negativ) + reden.parquet (rolle_regierung), "
                      "sitzungen.parquet (Datum)"}


# Beschriftungsversatz in pt (dx, dy, ha), damit eng liegende Punkte (SPD/Union, FDP/AfD) lesbar bleiben
VERSATZ = {"SPD": (0, -22, "center"), "CDU/CSU": (-6, 5, "right"), "BÜNDNIS 90/DIE GRÜNEN": (-6, 5, "right"),
           "FDP": (4, -12, "left"), "AfD": (-6, -4, "right"), "Die Linke": (-6, 5, "right"),
           "BSW": (6, 0, "left"), "fraktionslos": (6, -7, "left")}


def abb4() -> dict:
    h = (pd.read_parquet(METRICS / "h1.parquet").query("variante == 'mdb' and phase == 'gesamt'")
         .set_index("gruppe").reindex(list(FRAKTIONEN)))
    b = (pd.read_parquet(METRICS / "bootstrap_fraktion.parquet").query("phase == 'gesamt' and kennzahl == 'reden_pro_sitz'")
         .set_index("gruppe").n_reden.reindex(list(FRAKTIONEN)))
    h["redenanteil"] = b / b.sum()
    gr = h.diff_pp.abs().idxmax()
    n_ok = int(h.erfuellt.sum())
    t = (f"{'Alle ' + str(len(h)) if n_ok == len(h) else str(n_ok) + ' von ' + str(len(h))} Fraktionen und Gruppen "
         f"reden nach Sitzanteil (± {de(TOLERANZ_PP, 0)} pp): größte Abweichung {NAME[gr]} {de(h.diff_pp[gr], 1)} pp")
    gross = h.index[h.sitzanteil >= 0.05]
    lim = np.ceil(h[["sitzanteil", "woerteranteil", "redenanteil"]].max().max() * 20) / 20
    zoom = np.ceil(h.loc[~h.index.isin(gross), ["sitzanteil", "woerteranteil", "redenanteil"]].max().max() * 100 + 0.5) / 100
    fig, axes = plt.subplots(1, 2, figsize=(breite(), hoehe(6.0)), layout="constrained")
    for ax, grenze, zeigen, name in [(axes[0], lim, gross, "alle Fraktionen"),
                                     (axes[1], zoom, h.index.difference(gross), f"Ausschnitt bis {de(zoom * 100, 0)}{NBSP}%")]:
        xs = np.linspace(0, grenze, 50)
        ax.fill_between(xs, xs - TOLERANZ_PP / 100, xs + TOLERANZ_PP / 100, color=BAND, lw=0, zorder=0)
        ax.plot([0, grenze], [0, grenze], color=MUTED, lw=0.8, zorder=1)
        for fr, z in h.iterrows():
            ax.plot([z.sitzanteil] * 2, [z.woerteranteil, z.redenanteil], color=FARBE[fr], lw=0.8, alpha=0.6, zorder=2)
            ax.scatter(z.sitzanteil, z.redenanteil, marker="D", s=22, facecolors="white", edgecolors=FARBE[fr],
                       linewidths=1.1, zorder=3)
            ax.scatter(z.sitzanteil, z.woerteranteil, marker="o", s=30, color=FARBE[fr], edgecolors="white",
                       linewidths=0.7, zorder=4)
            if fr in zeigen:
                dx, dy, ha = VERSATZ[fr]
                ax.annotate(NAME[fr], (z.sitzanteil, z.woerteranteil), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va="center", color=INK)
        ax.set_xlim(0, grenze)
        ax.set_ylim(0, grenze)
        ax.set_box_aspect(1)
        # x in 10er- bzw. 2er-Schritten: bei 5 bzw. 1 Prozentpunkt stießen die Beschriftungen „15 %20 %“ aneinander
        ax.xaxis.set_major_locator(MultipleLocator(0.1 if grenze > 0.1 else 0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.05 if grenze > 0.1 else 0.01))
        formatiere(ax.xaxis, prozent=True)
        formatiere(ax.yaxis, prozent=True)
        ax.grid(color=GRID, lw=0.5)
        ax.set_xlabel("Sitzanteil (Mittel über die Sitzungen)")
        # Tafelname in die leere Ecke über der Diagonale: ein Titel über dem quadratischen Feld würde bei 6,9 cm Höhe abgeschnitten
        ax.text(0.03, 0.97, name, transform=ax.transAxes, ha="left", va="top", color=INK)
    axes[0].set_ylabel("Anteil an Wörtern bzw. Reden")
    axes[0].add_patch(Rectangle((0, 0), zoom, zoom, fill=False, ec=MUTED, lw=0.6, zorder=5))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=5, label="Wörteranteil"),
                        Line2D([], [], marker="D", ls="", mfc="white", mec=MUTED, mew=1.1, ms=4.5, label="Redenanteil"),
                        Line2D([], [], color=MUTED, lw=0.8, label="y = x (§ 35 GO-BT)"),
                        Patch(color=BAND, label=f"± {de(TOLERANZ_PP, 0)} pp")],
               loc="outside lower center", ncol=4, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    titel(fig, t)
    ra = (h.redenanteil - h.sitzanteil) * 100
    rg = ra.abs().idxmax()
    return {"datei": speichern(fig, "abb4_h1_sitzanteil.pdf"), "titel": t,
            "kern": f"{n_ok}/{len(h)} innerhalb ± 3 pp; Wörter max. {NAME[gr]} {de(h.diff_pp[gr], 1)} pp, "
                    f"Reden max. {NAME[rg]} {de(ra[rg], 1)} pp",
            "quelle": "metrics/h1.parquet (variante mdb, gesamt); Redenanteil aus metrics/bootstrap_fraktion.parquet n_reden"}


def abb5() -> dict:
    a = pd.read_csv(PROCESSED / "applaus_heatmap.csv")
    m = {ph: a[a.phase == ph].set_index("redner_fraktion").reindex(list(FRAKTIONEN)) for ph in ("vor", "nach")}
    diff = m["nach"][list(FRAKTIONEN)] - m["vor"][list(FRAKTIONEN)]
    diff = diff.mask(np.eye(len(FRAKTIONEN), dtype=bool))
    if diff.notna().to_numpy().any():
        rf, af = diff.abs().stack().idxmax()
        v, n = m["vor"].loc[rf, af], m["nach"].loc[rf, af]
        t = (f"{satz(af, ('applaudiert', 'applaudieren'))} {NAME[rf]}-Reden nach dem Ampel-Bruch "
             f"{'seltener' if n < v else 'häufiger'}: {proz(v)} → {proz(n)} der Reden")
        kern = f"{NAME[af]}-Beifall bei {NAME[rf]}-Reden {proz(v)} → {proz(n)} (größte Änderung außerhalb der Diagonale)"
    else:   # Teillauf nur vor oder nur nach 196/197 (146–147): idxmax auf lauter NaN bräche ab
        rf = af = None
        t = kern = (f"Beifall je Fraktion in Sitzung {SITZUNGEN[0]}–{SITZUNGEN[-1]}: "
                    "kein Vergleich vor/nach dem Ampel-Bruch, eine Phase ohne Reden")
        print(f"abb5: {t}")
    cmap = LinearSegmentedColormap.from_list("blau", BLAU)
    vmax = np.ceil(np.nanmax([m[p][list(FRAKTIONEN)].max().max() for p in m]) * 10) / 10
    # 6,6 cm: die senkrechten Spaltennamen brauchen Höhe, bei 5,4 cm wurden die Zellen zu schmal für zwei Ziffern
    fig, axes = plt.subplots(1, 2, figsize=(breite(), hoehe(6.6)), sharey=True, layout="constrained")
    for ax, (ph, name) in zip(axes, [("vor", "bis Sitzung 196"), ("nach", "ab Sitzung 197")]):
        w = m[ph][list(FRAKTIONEN)].to_numpy(float)
        im = ax.imshow(w, cmap=cmap, vmin=0, vmax=vmax)
        if np.isnan(w).all():
            ax.text(0.5, 0.5, f"keine Reden in Sitzung {SITZUNGEN[0]}–{SITZUNGEN[-1]}", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if np.isnan(w[i, j]):
                    continue
                r, g, b_ = to_rgb(cmap(w[i, j] / vmax))
                hell = 0.2126 * r + 0.7152 * g + 0.0722 * b_
                ax.text(j, i, de(100 * w[i, j], 0), ha="center", va="center", color="white" if hell < 0.5 else INK)
        ax.set_xticks(np.arange(-0.5, len(FRAKTIONEN)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FRAKTIONEN)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(len(FRAKTIONEN)), [NAME[f] for f in FRAKTIONEN], rotation=90, ha="center",
                      rotation_mode="default")  # 90° statt 45°: schräg überlagerten sich „CDU/CSU“ und „Grüne“
        if rf is not None:
            i, j = list(FRAKTIONEN).index(rf), list(FRAKTIONEN).index(af)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=OPPOSITION, lw=1.6))
        ax.set_title(name)
        ax.set_xlabel("applaudierende Fraktion")
    # n beider Phasen in einer Beschriftung: die rechte Tafel braucht keine eigene Achse und die Zellen werden breiter
    axes[0].set_yticks(range(len(FRAKTIONEN)), [f"{NAME[f]} ({tsd(k1)}/{tsd(k2)})" for f, k1, k2 in zip(
        FRAKTIONEN, m["vor"].n_reden.fillna(0), m["nach"].n_reden.fillna(0))])
    axes[0].set_ylabel("Rednerfraktion\n(n Reden bis 196/ab 197)")
    cb = fig.colorbar(im, ax=list(axes), shrink=0.75, aspect=25, pad=0.02)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{de(100 * x, 0)}{NBSP}%"))
    cb.outline.set_visible(False)
    cb.set_label("Anteil der Reden mit Beifall")
    titel(fig, t)
    return {"datei": speichern(fig, "abb5_applaus_heatmap.pdf"), "titel": t,
            "kern": kern,
            "quelle": "applaus_heatmap.csv (Anteil der Reden der Rednerfraktion mit Beifall der Spaltenfraktion)"}


# Abb. 6 zeigt die Verteilung der Redelänge, die Tabelle 3 nur als Median und Quartile führt; Reden über X_MAX Wörter
# (Regierungserklärungen, Haushaltsreden; 0,3 % der Reden) stehen als Dreieck am rechten Rand statt die Achse zu dehnen
LAENGE_X_MAX = 1600


def abb6() -> dict:
    from .metrics import lade_reden   # Hauptstatistik und Sitzgruppe wie Tabelle 3 (metrics.fraktionen_kennzahlen)

    r = lade_reden()
    d = r[r.haupt & r.bank.isin(FRAKTIONEN)]
    med = d.groupby("bank").woerter.median()
    # wie Abb. 1: fraktionslos ist keine Fraktion (7 Personen ohne gemeinsame Linie), steht auf der Achse, nicht im Titel
    kand = med.drop("fraktionslos", errors="ignore")
    kurz, lang = kand.idxmin(), kand.idxmax()
    t = (f"{satz(kurz, ('redet', 'reden'))} am kürzesten (Median {tsd(med[kurz])} Wörter), "
         f"{NOMINATIV[lang][0]} am längsten ({tsd(med[lang])} Wörter)")
    fig, ax = plt.subplots(figsize=(breite(), hoehe(5.0)), layout="constrained")
    rng = np.random.default_rng(SEED)
    ueber = 0
    for i, fr in enumerate(FRAKTIONEN):
        w = d.loc[d.bank == fr, "woerter"].to_numpy()
        if not len(w):
            continue
        drin, raus = w[w <= LAENGE_X_MAX], w[w > LAENGE_X_MAX]
        ueber += len(raus)
        ax.scatter(drin, i + rng.uniform(-0.3, 0.3, len(drin)), s=2.5, color=FARBE[fr], alpha=0.3, linewidths=0)
        if len(raus):
            # ein Dreieck mit Anzahl statt eines je Rede: gestapelt deckten sich die Dreiecke am Rand gegenseitig
            ax.scatter([LAENGE_X_MAX], [i], s=14, marker=">", color=FARBE[fr], linewidths=0, clip_on=False, zorder=3)
            ax.annotate(str(len(raus)), (LAENGE_X_MAX, i), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK, annotation_clip=False)
        q1, q2, q3 = np.percentile(w, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK, lw=2, solid_capstyle="butt", zorder=4,
                path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
        ax.plot(q2, i, marker="D", ms=4.5, mfc="white", mec=INK, mew=1, zorder=5)
    ax.set_xlim(0, LAENGE_X_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: tsd(v)))
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlabel("Wörter je Rede (138 Wörter ≈ 1 Minute)")
    ax.set_yticks(range(len(FRAKTIONEN)), [f"{NAME[f]} ({tsd((d.bank == f).sum())})" for f in FRAKTIONEN])
    ax.set_ylim(len(FRAKTIONEN) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.6, ms=3, label="eine Rede"),
                        Line2D([], [], marker="D", ls="-", lw=2, color=INK, mfc="white", mec=INK, ms=4.5,
                               label="Median mit mittlerer Hälfte (Q1–Q3)"),
                        Line2D([], [], marker=">", ls="", color=MUTED, ms=4,
                               label=f"Reden über {tsd(LAENGE_X_MAX)} Wörter")],
               loc="outside lower center", ncol=3, handlelength=1.6, borderaxespad=0.2)
    titel(fig, t)
    return {"datei": speichern(fig, "abb6_redelaenge_fraktionen.pdf"), "titel": t,
            "kern": f"Median {NAME[kurz]} {tsd(med[kurz])}, {NAME[lang]} {tsd(med[lang])} Wörter; {len(d)} Reden, "
                    f"{ueber} über {LAENGE_X_MAX} Wörter am Rand",
            "quelle": "reden.parquet über metrics.lade_reden (Hauptstatistik, Sitzgruppe = Fraktion oder Gruppe)"}


def abb7() -> dict:
    """Alle Redner als kleine Mehrfachgrafik: acht Parteifarben in einem Streudiagramm trennt das Auge nicht
    (dataviz: Streudiagramme höchstens drei Farben), daher je Fraktion eine Tafel, die übrigen Redner grau."""
    a = pd.read_csv(PROCESSED / "abgeordnete.csv")
    a = a[a.reden_n > 0].copy()
    a["tsd"] = a.woerter_gesamt / 1000
    n_halb = int(a.reden_n.median())
    s = a.reden_n.sort_values(ascending=False)
    k = round(len(s) * 0.1)
    anteil = s.head(k).sum() / s.sum()
    t = (f"Die Hälfte der {len(a)} Redner hält höchstens {n_halb} Reden, "
         f"die {k} aktivsten halten {proz(anteil)} aller Reden")
    fig, axes = plt.subplots(2, 4, figsize=(breite(), hoehe(5.9)), sharex=True, sharey=True, layout="constrained")
    xmax, ymax = a.reden_n.max() * 1.05, a.tsd.max() * 1.08
    for ax, fr in zip(axes.flat, FRAKTIONEN):
        ax.scatter(a.reden_n, a.tsd, s=3, color=BAND, linewidths=0, zorder=1)
        x = a[a.fraktion == fr]
        # Grau der Fraktionslosen hebt sich vom grauen Hintergrund nur mit dunklem Rand ab
        rand = (INK, 0.4) if fr == "fraktionslos" else ("white", 0.3)
        ax.scatter(x.reden_n, x.tsd, s=7, color=FARBE[fr], edgecolors=rand[0], linewidths=rand[1], zorder=3)
        ax.set_title(f"{NAME[fr]} ({len(x)})")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        formatiere(ax.yaxis, 0)
        ax.grid(color=GRID, lw=0.5)
    fig.supxlabel("Reden (Hauptstatistik)", fontsize=8)
    fig.supylabel("Wörter gesamt (Tsd.)", fontsize=8)
    titel(fig, t)
    return {"datei": speichern(fig, "abb7_alle_redner.pdf"), "titel": t,
            "kern": f"{len(a)} Redner, Median {n_halb} Reden; Top {k}: {proz(anteil)} der Reden; "
                    f"{int(a.fraktion.isna().sum())} ohne Fraktion nur grau",
            "quelle": "abgeordnete.csv (reden_n > 0; Fraktion mit den meisten Reden der Person, metrics.abgeordnete)"}


def schreibe_md(info: list[dict]) -> None:
    zeilen = [f"# Abbildungen (erzeugt von `python -m bundestag.visualize`, Vektor-PDF, nativ {de(BREITE_CM, 1)} cm breit, "
              "Schrift 8–9 pt, in LaTeX unskaliert einbinden)", ""]
    zeilen += [f"- {x['datei']} ({de(GROESSE[x['datei']][0], 1)} × {de(GROESSE[x['datei']][1], 1)} cm): „{x['titel']}“. "
               f"Kernzahl: {x['kern']}. Quelle: {x['quelle']}." for x in info]
    zeilen += ["", "Farben: zentrales Dict FARBE in visualize.py (Parteifarben, FDP dunkles Gold); "
                   "Fraktionsnamen stehen immer als Text, Graustufen über Marker/Linienstile."]
    (FIGURES / "abbildungen.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")


def schreibe_tex(info: list[dict]) -> None:
    zeilen = ["% erzeugt von python -m bundestag.visualize: Titelaussagen der Abbildungen, Zahlen aus den Daten"]
    zeilen += [f"\\newcommand{{\\{TEX_NAMEN[x['nr'] - 1]}}}{{{tex(x['titel'])}}}" for x in info]
    LATEX_TITEL.parent.mkdir(parents=True, exist_ok=True)
    LATEX_TITEL.write_text("\n".join(zeilen) + "\n", encoding="utf-8")


def main() -> None:
    global MIT_TITEL, BREITE_CM
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nur", type=int, nargs="*", help="nur diese Abbildungen (1–7)")
    p.add_argument("--mit-titel", action="store_true", help="Titelaussage zusätzlich ins PDF (sonst nur in der Caption)")
    p.add_argument("--breite-cm", type=float, default=BREITE_CM, help="Abbildungsbreite in cm; Höhen skalieren mit")
    args = p.parse_args()
    nur, MIT_TITEL, BREITE_CM = set(args.nur or range(1, 8)), args.mit_titel, args.breite_cm
    alle = nur == set(range(1, 8))
    stil()
    with laufzeit("visualize"):
        sr = lade_sentiment()
        if sr is None and nur & {1, 2, 3}:
            print(f"Hinweis: {METRICS / 'sentiment_reden.parquet'} fehlt (kein sentiment.parquet), Abbildungen 1–3 entfallen")
            nur -= {1, 2, 3}
        bauer = {1: lambda: abb1(sr), 2: lambda: abb2(sr), 3: lambda: abb3(sr), 4: abb4, 5: abb5, 6: abb6, 7: abb7}
        info = [bauer[k]() | {"nr": k} for k in sorted(nur)]
    for x in info:
        print(f"{x['datei']}: {x['titel']}")
    if alle:
        schreibe_md(info)
        schreibe_tex(info)


if __name__ == "__main__":
    main()
