"""Sentiment je Rede mit oliverguhr/german-sentiment-bert: Fenster zu 150 Wörtern (FENSTER_WOERTER), je Rede das
nach Wörtern gewichtete Mittel der Fenster (je_rede).

Aufruf:
  python -m bundestag.sentiment --gold-pool            Gold-Stichprobe 3 × 20 Fenster (data/reference)
  python -m bundestag.sentiment --voll                 alle Reden mit Text → sentiment(_fenster).parquet
  python -m bundestag.sentiment --sitzungen 160 161    Probelauf ohne Ausgabetabellen, füllt nur den Cache
"""
import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bundestag.config import BERT_CACHE, PROCESSED, REFERENCE, SENTIMENT_MAX_TOKENS, SENTIMENT_MODEL, SENTIMENT_REVISION

# Fenster-Hash → Wahrscheinlichkeiten: nach einem Parser-Neulauf werden nur geänderte Fenster neu gerechnet.
CACHE = BERT_CACHE
MIN_WOERTER = 50          # kürzere Reden ohne Sentiment, kürzeres Restfenster an das vorige
# Guhr et al. (2020, S. 1631) trainierten mit höchstens 256 Tokens. Darüber lagen im Fenstercheck an 1000 Reden
# (sentiment_fenstercheck.py) bei 200 Wörtern 73,9 % der Fenster, im Vollauf mit 150 Wörtern 871 von 31 065
# Fenstern = 2,8 %. Regel vorab: 200 Wörter nur bei Spearman ≥ 0,9; gemessen 0,87.
FENSTER_WOERTER = 150
TRAINING_TOKENS = 256     # Guhr et al. 2020, S. 1631: „maximal token length of 256“; der Anteil darüber steht im Log
SEED = 82                 # Projektseed: Gold-Stichprobe und Batch-Reihenfolge
GOLD_POOL = 1500          # liefert 57 positive Fenster (v2), genug für 20 je Label; sonst wächst der Pool
GOLD_JE_LABEL = 20        # 60 Fenster, je 20 mit BERT-Label positiv/neutral/negativ
# Gold v2 (G01–G60) in der Produktionseinheit 150 Wörter, damit κ die Vollauf-Fenster prüft; v1 (F01–F60) bleibt als Vergleich.
GOLD_VERSION, GOLD_ID = "_v2", "G"
# Messung auf M1 Pro (MPS, fp32, 320 nach Tokenlänge sortierte Fenster aus Sitzung 160): Batch 8 → 43,
# 16 → 45, 32 → 44, 64 → 41 Fenster/s bei 2,2 GB MPS-Speicher; sortierte Batches tragen kaum Padding, größere bringen nichts.
BATCH = 16
LOG_ALLE = 40             # Fortschrittszeile je 640 Fenster (≈ 13 s bei 48 Fenster/s), damit tail -n 5 den aktuellen Durchsatz zeigt
CACHE_ALLE = 50           # Cache-Flush je 50 Batches: Abbruch an der Zeitgrenze (< 15 Fenster/s oder > 60 min) verliert ≤ 800 Fenster

LABEL_DE = {"positive": "positiv", "neutral": "neutral", "negative": "negativ"}
SPALTE = {"positive": "p_pos", "neutral": "p_neu", "negative": "p_neg"}
P_SPALTEN = ["p_pos", "p_neu", "p_neg"]
LABEL_JE_SPALTE = {SPALTE[k]: v for k, v in LABEL_DE.items()}

# Nachbau von SentimentModel.clean_text aus germansentiment 1.1.0 (PyPI-sdist,
# germansentiment/sentimentmodel.py, Zeilen 17–19 und 42–56); das Modell wurde auf so bereinigtem Text trainiert.
# Kein Import des Pakets: es wählt nur CUDA oder CPU (Zeilen 8–11), auf dem M1 also nicht MPS.
_URL = re.compile(r"https*\S+", re.MULTILINE)
_AT = re.compile(r"@\S+", re.MULTILINE)
_NICHT_TEXT = re.compile(r"[^A-Za-züöäÖÜÄß ]", re.MULTILINE)
_ZIFFERN = [("0", " null"), ("1", " eins"), ("2", " zwei"), ("3", " drei"), ("4", " vier"),
            ("5", " fünf"), ("6", " sechs"), ("7", " sieben"), ("8", " acht"), ("9", " neun")]
# Wort wie in parse.py: Whitespace-Token mit mindestens einem Buchstaben oder einer Ziffer; „–“ zählt nicht.
_WORT = re.compile(r"[^\W_]")


def clean_text(text):
    text = text.replace("\n", " ")
    text = _URL.sub("", text)
    text = _AT.sub("", text)
    for ziffer, wort in _ZIFFERN:
        text = text.replace(ziffer, wort)
    text = _NICHT_TEXT.sub("", text)
    return " ".join(text.split()).strip().lower()


def fenster_grenzen(n_woerter, groesse=FENSTER_WOERTER):
    """Wortindizes [von, bis) je Fenster."""
    if n_woerter < MIN_WOERTER:
        return []
    grenzen = [[von, min(von + groesse, n_woerter)] for von in range(0, n_woerter, groesse)]
    if len(grenzen) > 1 and grenzen[-1][1] - grenzen[-1][0] < MIN_WOERTER:
        rest = grenzen.pop()
        grenzen[-1][1] = rest[1]
    return [tuple(g) for g in grenzen]


def fenster_texte(text, groesse=FENSTER_WOERTER):
    """(Originaltext, Wortzahl) je Fenster; Nicht-Wort-Token hängen am vorigen Wort."""
    token = text.split()
    wort_pos = [i for i, t in enumerate(token) if _WORT.search(t)]
    teile = []
    for von, bis in fenster_grenzen(len(wort_pos), groesse):
        start = wort_pos[von] if von else 0
        ende = wort_pos[bis] if bis < len(wort_pos) else len(token)
        teile.append((" ".join(token[start:ende]), bis - von))
    return teile


def spalten_je_index(id2label):
    """Logit-Index → p-Spalte nach model.config.id2label (oliverguhr: 0 positive, 1 negative, 2 neutral)."""
    return [SPALTE[str(id2label.get(i, id2label.get(str(i)))).lower()] for i in range(len(id2label))]


def label_aus(p):
    return p[P_SPALTEN].idxmax(axis=1).map(LABEL_JE_SPALTE)


def je_rede(fenster):
    gewichtet = fenster[P_SPALTEN].mul(fenster.woerter, axis=0)
    gewichtet[["rede_id", "redetyp", "woerter"]] = fenster[["rede_id", "redetyp", "woerter"]]
    gruppen = gewichtet.groupby(["rede_id", "redetyp"], sort=False)
    rede = gruppen[P_SPALTEN + ["woerter"]].sum()
    rede[P_SPALTEN] = rede[P_SPALTEN].div(rede.woerter, axis=0)
    rede.insert(0, "n_fenster", gruppen.size())
    rede["netto"] = rede.p_pos - rede.p_neg
    rede["label"] = label_aus(rede)
    return rede.reset_index()[["rede_id", "redetyp", "n_fenster", "woerter"] + P_SPALTEN + ["netto", "label"]]


def _snapshot():
    """Modellordner: erst der lokale Hugging-Face-Cache, sonst einmalig vom Hub (frischer Klon hat keinen Cache).

    Kein HF_HUB_OFFLINE mehr: huggingface_hub liest die Variable beim Import, danach wäre der Hub-Abruf in
    diesem Prozess gesperrt. Der erste Versuch bleibt durch local_files_only ohne Netz.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(snapshot_download(SENTIMENT_MODEL, revision=SENTIMENT_REVISION, local_files_only=True))
    except LocalEntryNotFoundError:
        pass
    print(f"{SENTIMENT_MODEL}@{SENTIMENT_REVISION[:7]} nicht im lokalen Cache, lade einmalig vom Hugging-Face-Hub "
          "(model.safetensors 436 MB)", flush=True)
    try:
        return Path(snapshot_download(SENTIMENT_MODEL, revision=SENTIMENT_REVISION))
    except Exception as exc:  # Netz, Proxy, gesetztes HF_HUB_OFFLINE: alle enden in derselben Handlungsanweisung
        raise SystemExit(
            f"BERT-Modell {SENTIMENT_MODEL} (Revision {SENTIMENT_REVISION}) weder lokal noch vom Hub ladbar: "
            f"{type(exc).__name__}: {exc}\nAbhilfe: Netzverbindung prüfen, HF_HUB_OFFLINE nicht setzen und einmalig "
            f".venv/bin/hf download {SENTIMENT_MODEL} --revision {SENTIMENT_REVISION} ausführen; danach "
            "python run_all.py --skip-download --ab sentiment") from exc


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_snapshot())


def lies_reden(sitzungen=None):
    reden = pd.read_parquet(PROCESSED / "reden.parquet")
    reden = reden[reden.text.fillna("").str.strip() != ""]
    if sitzungen:
        reden = reden[reden.sitzung.isin(sitzungen)]
    if reden.duplicated(["rede_id", "redetyp"]).any():
        raise SystemExit("(rede_id, redetyp) nicht eindeutig – Schlüssel der Sentiment-Tabellen")
    return reden


def baue_fenster(reden, groesse):
    zeilen = []
    for r in reden.itertuples(index=False):
        for nr, (text, n) in enumerate(fenster_texte(r.text, groesse), 1):
            zeilen.append((r.rede_id, r.redetyp, r.sitzung, r.fraktion, nr, n, text))
    fenster = pd.DataFrame(zeilen, columns=["rede_id", "redetyp", "sitzung", "fraktion", "fenster_nr",
                                            "woerter", "text"])
    fenster["clean"] = fenster.text.map(clean_text)
    schluessel = f"{SENTIMENT_MODEL}@{_snapshot().name}\n"
    fenster["hash"] = [hashlib.sha1((schluessel + c).encode()).hexdigest() for c in fenster.clean]
    return fenster


def fenster_mit_tokens(reden, tok, groesse=FENSTER_WOERTER):
    """Fenster mit Tokenzahl des bereinigten Texts inkl. [CLS]/[SEP]; über 512 schneidet BERT ab."""
    fenster = baue_fenster(reden, groesse)
    fenster["tokens"] = [len(ids) for ids in tok(fenster.clean.tolist())["input_ids"]]
    t = fenster.tokens
    print(f"Fenster {groesse} Wörter: {len(fenster)} Fenster, Tokens Median {t.median():.0f}, P95 {t.quantile(.95):.0f}, "
          f"P99 {t.quantile(.99):.0f}, max {t.max()}, über {TRAINING_TOKENS}: {(t > TRAINING_TOKENS).mean():.1%}, "
          f"über {SENTIMENT_MAX_TOKENS}: {int((t > SENTIMENT_MAX_TOKENS).sum())}", flush=True)
    return fenster


def lies_cache():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return pd.DataFrame(columns=P_SPALTEN, dtype="float64").rename_axis("hash")


def schreibe_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    cache.to_parquet(tmp)
    os.replace(tmp, CACHE)


def bewerten(fenster, tok, batch=BATCH):
    """Hängt p_pos/p_neu/p_neg und label an; gerechnet werden nur Hashes, die nicht im Cache stehen."""
    cache = lies_cache()
    offen = fenster.loc[~fenster.hash.isin(cache.index)].drop_duplicates("hash")
    print(f"{len(fenster)} Fenster, davon {len(fenster) - len(offen)} im Cache, {len(offen)} zu rechnen", flush=True)
    if len(offen):
        import torch
        from transformers import AutoModelForSequenceClassification
        geraet = "mps" if torch.backends.mps.is_available() else "cpu"
        modell = AutoModelForSequenceClassification.from_pretrained(_snapshot()).to(geraet).eval()
        spalten = spalten_je_index(modell.config.id2label)
        offen = offen.sort_values("tokens")
        batches = [offen.iloc[i:i + batch] for i in range(0, len(offen), batch)]
        # Zufällige Batch-Reihenfolge: sonst misst die 2-Minuten-Probe nur kurze Fenster und überschätzt den Durchsatz.
        reihenfolge = np.random.default_rng(SEED).permutation(len(batches))
        tok_gesamt = int(offen.tokens.clip(upper=SENTIMENT_MAX_TOKENS).sum())
        neu, fertig, tok_fertig, start = [], 0, 0, time.time()
        for k, idx in enumerate(reihenfolge, 1):
            b = batches[idx]
            enc = tok(b.clean.tolist(), padding=True, truncation=True, max_length=SENTIMENT_MAX_TOKENS,
                      return_tensors="pt").to(geraet)
            with torch.inference_mode():
                p = torch.softmax(modell(**enc).logits.float(), dim=-1).cpu().numpy()
            neu.append(pd.DataFrame(p, columns=spalten, index=pd.Index(b.hash.values, name="hash"))[P_SPALTEN])
            fertig += len(b)
            tok_fertig += int(b.tokens.clip(upper=SENTIMENT_MAX_TOKENS).sum())
            if k % CACHE_ALLE == 0 or k == len(batches):
                cache = pd.concat([cache] + neu)
                neu = []
                schreibe_cache(cache)
            if k % LOG_ALLE == 0 or k == len(batches):
                dauer = time.time() - start
                rest = (tok_gesamt - tok_fertig) / (tok_fertig / dauer) / 60
                print(f"{time.strftime('%H:%M:%S')} {fertig}/{len(offen)} Fenster, {dauer:.0f} s, "
                      f"{fertig / dauer:.1f} Fenster/s, Rest ≈ {rest:.1f} min", flush=True)
    p = cache.loc[fenster.hash, P_SPALTEN].to_numpy()
    fenster = fenster.assign(**{c: p[:, i] for i, c in enumerate(P_SPALTEN)})
    fenster["label"] = label_aus(fenster)
    return fenster


def gold_stichprobe(fenster, tok):
    plenar = fenster[fenster.redetyp == "plenarrede"].reset_index(drop=True)
    perm = np.random.default_rng(SEED).permutation(len(plenar))
    n = GOLD_POOL
    while True:
        pool = bewerten(plenar.iloc[perm[:n]], tok)
        anzahl = pool.label.value_counts()
        print(f"Pool {n}: {anzahl.to_dict()}", flush=True)
        if min(anzahl.get(lab, 0) for lab in LABEL_DE.values()) >= GOLD_JE_LABEL or n >= len(plenar):
            break
        n += GOLD_POOL
    rng = np.random.default_rng(SEED)
    teile = [pool[pool.label == lab].sample(GOLD_JE_LABEL, random_state=rng) for lab in LABEL_DE.values()]
    # Gemischte Reihenfolge: in Label-Blöcken ließe sich das BERT-Label an der Zeilennummer ablesen.
    gold = pd.concat(teile).sample(frac=1, random_state=rng).reset_index(drop=True)
    gold.insert(0, "id", [f"{GOLD_ID}{i:02d}" for i in range(1, len(gold) + 1)])
    REFERENCE.mkdir(parents=True, exist_ok=True)
    gold[["id", "text"]].to_csv(REFERENCE / f"sentiment_gold_fenster{GOLD_VERSION}.csv", index=False)
    schluessel = gold.rename(columns={"label": "bert_label"})
    schluessel[["id", "rede_id", "sitzung", "fraktion", "bert_label"] + P_SPALTEN].to_csv(
        REFERENCE / f"sentiment_gold_schluessel{GOLD_VERSION}.csv", index=False)
    print(f"Gold: {len(gold)} Fenster aus Pool {n}, Labels {gold.label.value_counts().to_dict()}", flush=True)


def schreibe_atomar(tabelle, pfad):
    tmp = pfad.with_suffix(".tmp")
    tabelle.to_parquet(tmp, index=False)
    os.replace(tmp, pfad)


def vollauf(fenster, tok, max_fenster=None):
    if max_fenster:
        print(f"Zeitgrenze: nur die ersten {max_fenster} Fenster je Rede ({len(fenster)} → "
              f"{int((fenster.fenster_nr <= max_fenster).sum())})", flush=True)
        fenster = fenster[fenster.fenster_nr <= max_fenster]
    start = time.time()
    fenster = bewerten(fenster, tok)
    spalten = ["rede_id", "redetyp", "fenster_nr", "woerter", "tokens"] + P_SPALTEN + ["label"]
    schreibe_atomar(fenster[spalten], PROCESSED / "sentiment_fenster.parquet")
    rede = je_rede(fenster)
    schreibe_atomar(rede, PROCESSED / "sentiment.parquet")
    print(f"Vollauf: {len(fenster)} Fenster, {len(rede)} Reden, {time.time() - start:.0f} s; Reden-Labels "
          f"{rede.label.value_counts().to_dict()}; plenarrede "
          f"{rede[rede.redetyp == 'plenarrede'].label.value_counts().to_dict()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modus = ap.add_mutually_exclusive_group(required=True)
    modus.add_argument("--gold-pool", action="store_true")
    modus.add_argument("--voll", action="store_true")
    modus.add_argument("--sitzungen", type=int, nargs="+")
    ap.add_argument("--max-fenster", type=int,
                    help="bei Überschreiten der Zeitgrenze: nur die ersten N Fenster je Rede")
    args = ap.parse_args()
    tok = tokenizer()
    reden = lies_reden(args.sitzungen)
    print(f"{len(reden)} Reden mit Text", flush=True)
    fenster = fenster_mit_tokens(reden, tok)
    print(f"Reden < {MIN_WOERTER} Wörter ohne Sentiment: {len(reden) - fenster.groupby(['rede_id', 'redetyp']).ngroups}",
          flush=True)
    if args.gold_pool:
        gold_stichprobe(fenster, tok)
    elif args.voll:
        vollauf(fenster, tok, args.max_fenster)
    else:
        start = time.time()
        bewertet = bewerten(fenster, tok)
        print(f"Probelauf {time.time() - start:.0f} s, Fenster-Labels {bewertet.label.value_counts().to_dict()}",
              flush=True)


if __name__ == "__main__":
    main()
