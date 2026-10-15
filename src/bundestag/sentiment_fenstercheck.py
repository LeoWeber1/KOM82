"""Prüfung der Fenstergröße: 200 gegen 150 Wörter je Rede und auf den 60 Gold-Fenstern.

Anlass: Guhr et al. (2020, S. 1631) trainierten mit höchstens 256 Tokens, die
200-Wort-Fenster liegen im Median bei 272. Beide Varianten laufen auf demselben Stand von reden.parquet;
sentiment.parquet bleibt unberührt, BERT-Ergebnisse kommen aus dem Fenster-Cache von bundestag.sentiment.
Ausgaben: Zeilen mit Quelle sentiment_fenstercheck.py in kennzahlen.csv, Zusammenfassung auf stdout.
Aufruf: python -m bundestag.sentiment_fenstercheck
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from .config import PROCESSED
from .metrics import kennzahl, schreibe_kennzahlen
from .sentiment import (FENSTER_WOERTER, MIN_WOERTER, REFERENCE, SEED, baue_fenster, bewerten, je_rede,
                        lies_reden, tokenizer)
from .sentiment_kappa import cohen_kappa
from .stats import N_BOOT, ki

N_REDEN = 1000            # klein, weil parallel ein Ollama-Lauf die GPU nutzte; genügt: Spearman-KI [0,85; 0,89] bleibt unter 0,9
TRAINING_TOKENS = 256     # Guhr et al. 2020, S. 1631: „maximal token length of 256“
Q = "sentiment_fenstercheck.py"
# 200 Wörter = Erstfassung und Gold v1; 150 = FENSTER_WOERTER seit der Umstellung nach diesem Check.
GROESSEN = {"Zweihundert": 200, "Hundertfuenfzig": FENSTER_WOERTER}


def mit_tokens(fenster, tok):
    fenster["tokens"] = [len(ids) for ids in tok(fenster.clean.tolist())["input_ids"]]
    return fenster


def token_kennzahlen(fenster, name):
    t = fenster.tokens
    werte = {"Median": t.median(), "Pfuenfundneunzig": t.quantile(.95), "Pneunundneunzig": t.quantile(.99)}
    z = [kennzahl(f"fensterCheckTok{k}{name}", float(v), "Tokens", f"{Q}:token_kennzahlen") for k, v in werte.items()]
    z.append(kennzahl(f"fensterCheckAnteilUeberTraining{name}", float((t > TRAINING_TOKENS).mean()), "Anteil",
                      f"{Q}:token_kennzahlen"))
    z.append(kennzahl(f"fensterCheckFenster{name}", len(t), "Fenster", f"{Q}:token_kennzahlen"))
    return z


def boot(idx, f):
    return ki(np.array([f(i) for i in idx]))


def reden_vergleich(stichprobe, tok):
    je_groesse = {}
    zeilen = []
    for name, groesse in GROESSEN.items():
        fenster = mit_tokens(baue_fenster(stichprobe, groesse), tok)
        zeilen += token_kennzahlen(fenster, name)
        je_groesse[name] = je_rede(bewerten(fenster, tok))
    v = je_groesse["Zweihundert"].merge(je_groesse["Hundertfuenfzig"], on=["rede_id", "redetyp"],
                                        suffixes=("_200", "_150"))
    v = v.merge(stichprobe[["rede_id", "redetyp", "fraktion"]], on=["rede_id", "redetyp"])
    a, b = v.netto_200.to_numpy(), v.netto_150.to_numpy()
    la, lb = v.label_200.to_numpy(), v.label_150.to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    q = f"{Q}:reden_vergleich"
    zeilen += [
        kennzahl("fensterCheckReden", len(v), "Reden", q),
        kennzahl("fensterCheckPearson", st.pearsonr(a, b)[0], "r", q, boot(idx, lambda i: st.pearsonr(a[i], b[i])[0])),
        kennzahl("fensterCheckSpearman", st.spearmanr(a, b)[0], "rho", q,
                 boot(idx, lambda i: st.spearmanr(a[i], b[i])[0])),
        kennzahl("fensterCheckLabelGleich", (la == lb).mean(), "Anteil", q, boot(idx, lambda i: (la[i] == lb[i]).mean())),
        kennzahl("fensterCheckKappaLabel", cohen_kappa(la, lb), "kappa", q,
                 boot(idx, lambda i: cohen_kappa(la[i], lb[i]))),
        kennzahl("fensterCheckNettoDiff", (b - a).mean(), "netto 150 minus 200", q, boot(idx, lambda i: (b[i] - a[i]).mean())),
    ]
    v["fraktion"] = v.fraktion.fillna("ohne Fraktion")
    fraktionen = v.groupby("fraktion").agg(n=("rede_id", "size"), netto_200=("netto_200", "median"),
                                           netto_150=("netto_150", "median"))
    labels = pd.crosstab(v.label_200, v.label_150)
    return zeilen, fraktionen, labels


def gold_vergleich(tok):
    gold = pd.read_csv(REFERENCE / "sentiment_gold_fenster.csv").rename(columns={"id": "rede_id"})
    gold = gold.assign(redetyp="gold", sitzung=0, fraktion=None)
    urteil = pd.read_csv(PROCESSED / "sentiment_gold_v1.csv").set_index("id")
    mehrheit = urteil.mehrheit.reindex(gold.rede_id).to_numpy()
    labels = {}
    for name, groesse in GROESSEN.items():
        r = je_rede(bewerten(mit_tokens(baue_fenster(gold, groesse), tok), tok)).set_index("rede_id")
        labels[name] = r.label.reindex(gold.rede_id).to_numpy()
    # Gegenprobe: ein Fenster von höchstens 249 Wörtern bleibt bei 200 Wörtern ganz, also gleich dem BERT-Label der Stichprobe.
    abweichend = int((labels["Zweihundert"] != urteil.bert.reindex(gold.rede_id).to_numpy()).sum())
    idx = np.random.default_rng(SEED).integers(0, len(gold), size=(N_BOOT, len(gold)))
    q = f"{Q}:gold_vergleich"
    zeilen = [kennzahl("fensterCheckGoldAbweichungZweihundert", abweichend, "Fenster", q),
              kennzahl("fensterCheckGoldUnentschieden", int((mehrheit == "unentschieden").sum()), "Fenster", q)]
    for name, lab in labels.items():
        ok = mehrheit != "unentschieden"
        zeilen += [kennzahl(f"fensterCheckKappa{name}", cohen_kappa(mehrheit, lab), "kappa", q,
                            boot(idx, lambda i: cohen_kappa(mehrheit[i], lab[i]))),
                   kennzahl(f"fensterCheckRoh{name}", float((mehrheit[ok] == lab[ok]).mean()), "Anteil", q)]
    wechsel = pd.crosstab(labels["Zweihundert"], labels["Hundertfuenfzig"])
    return zeilen, wechsel


def main():
    tok = tokenizer()
    reden = lies_reden()
    plenar = reden[(reden.redetyp == "plenarrede") & (reden.woerter >= MIN_WOERTER)]
    stichprobe = plenar.sample(N_REDEN, random_state=SEED)
    zeilen, fraktionen, labels = reden_vergleich(stichprobe, tok)
    gold_zeilen, wechsel = gold_vergleich(tok)
    zeilen += gold_zeilen
    schreibe_kennzahlen(zeilen, Q)
    pd.set_option("display.width", 160)
    print(pd.DataFrame(zeilen)[["schluessel", "wert", "ki_unten", "ki_oben"]].round(4).to_string(index=False))
    print(fraktionen.round(3).to_string())
    print("label_rede 200 (Zeilen) gegen 150 (Spalten):\n" + labels.to_string())
    print("Gold 200 (Zeilen) gegen 150 (Spalten):\n" + wechsel.to_string())


if __name__ == "__main__":
    main()
