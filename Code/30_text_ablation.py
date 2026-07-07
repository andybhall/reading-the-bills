"""Redaction ablations: WHICH parts of the text carry the signal?

For every legislation-linked rollcall in the temporal-forecast holdout,
rebuild the exact 1,500-character text template the frozen champion's
MiniLM tower saw (question. description. pre-vote summary/title), apply
a redaction, re-embed with the same encoder, swap the embedding into the
frozen tower's in-memory lookup (the pickle on disk is never touched),
and re-score the identical member-vote cells. All variants pass through
the same tower with the same non-text features, so differences isolate
the redacted text component.

Variants:
  original       exact training template (internal baseline)
  no_summary     question + vote description only
  no_question    description + bill text only (question removed)
  no_propernouns capitalized sequences and acronyms masked ("entity")
  no_digits      digits removed (years, dollar amounts, bill numbers)
  shuffled_area  own question kept; description + bill text replaced by
                 those of a random different holdout rollcall in the
                 same policy area and question bucket (placebo: does
                 topic-matched but wrong-bill text suffice?)

Metrics per variant: log loss on legislation-linked holdout votes and
within-rollcall defection AUC (majority-yea passage-family rollcalls,
>=3 defections, equal-weight mean).

Outputs:
  Modified Data/results/measures/text_ablation.json
  Draft/tables/ablation_redaction.tex
  Draft/tables/ablation_macros.tex

Run: python3 Code/30_text_ablation.py   (approx. 10-20 min on CPU/MPS)
"""

import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
ART = RES / "frozen"
OUT = ROOT / "Draft" / "tables"

sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket  # noqa: E402
import importlib  # noqa: E402

embed_mod = importlib.import_module("08_embed_bills")

KEY = ["congress", "chamber", "rollnumber"]
EPS = 1e-6
SEED = 20260705

PROPER = re.compile(r"\b(?:[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+|[A-Z]{2,})\b")
DIGITS = re.compile(r"\d+")
GENERIC = "A bill to provide for certain matters"


def load_tower():
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"]
    with open(pkl, "rb") as f:
        blend = pickle.load(f)
    return blend.models[0]  # MiniLM-MLP component


def holdout_frame():
    """Test-window rollcalls with the pieces of the text template."""
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        KEY + ["date", "vote_question", "vote_desc"]]
    rc = rc.sort_values(["congress", "chamber", "date", "rollnumber"])
    rank = rc.groupby(["congress", "chamber"]).cumcount()
    size = rc.groupby(["congress", "chamber"])["rollnumber"].transform("size")
    rc["is_test"] = (rank + 0.5) / size >= 0.90
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        KEY + ["bill_type", "bill_no", "bill_category"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "title", "policy_area",
         "summaries"]]
    d = (rc[rc.is_test]
         .merge(links, on=KEY, how="inner")
         .merge(bills, on=["congress", "bill_type", "bill_no"], how="left"))
    d = d[d.bill_category == "legislation"].copy()
    d["vote_date"] = d["date"].astype(str)
    embed_mod.MAX_CHARS = 1500
    bill_texts = []
    for row in d.itertuples():
        if isinstance(row.summaries, str):
            bt, _ = embed_mod.pre_vote_text(row.summaries, row.title,
                                            row.vote_date)
        else:
            bt = (row.title or "")[:1500] if isinstance(row.title, str) else ""
        bill_texts.append(bt)
    d["bill_text"] = bill_texts
    d["question"] = d.vote_question.fillna("")
    d["desc"] = d.vote_desc.fillna("")
    d["qb"] = question_bucket(d["vote_question"])
    return d.reset_index(drop=True)


def compose(q, desc, bill):
    return ". ".join(p for p in (q, desc, bill) if p)[:1500]


def build_variants(d):
    rng = np.random.default_rng(SEED)
    v = {"original": [compose(r.question, r.desc, r.bill_text)
                      for r in d.itertuples()],
         "no_summary": [compose(r.question, r.desc, "")
                        for r in d.itertuples()],
         "no_question": [compose("", r.desc, r.bill_text)
                         for r in d.itertuples()],
         "no_propernouns": [
             compose(r.question, PROPER.sub("entity", r.desc),
                     PROPER.sub("entity", r.bill_text))
             for r in d.itertuples()],
         "no_digits": [
             compose(r.question, DIGITS.sub(" ", r.desc),
                     DIGITS.sub(" ", r.bill_text))
             for r in d.itertuples()],
         # generic title: the title-bearing fields (vote description and,
         # for title-only rollcalls, the bill text itself) replaced by
         # generic language; summaries retained. Pools the Table-5
         # single-bill title edit across the whole corpus.
         "generic_title": [
             compose(r.question, GENERIC,
                     GENERIC if len(r.bill_text) <= 300 else r.bill_text)
             for r in d.itertuples()]}
    # shuffled_area: donor drawn from a different bill, same policy area
    # and question bucket (falls back to same bucket anywhere)
    donor = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        pool = d.index[(d.policy_area == r.policy_area) & (d.qb == r.qb)
                       & (d.bill_no != r.bill_no)].to_numpy()
        if not len(pool):
            pool = d.index[(d.qb == r.qb) & (d.bill_no != r.bill_no)
                           ].to_numpy()
        donor[i] = rng.choice(pool)
    v["shuffled_area"] = [
        compose(d.question.iloc[i], d.desc.iloc[j], d.bill_text.iloc[j])
        for i, j in enumerate(donor)]
    return v


def score(tower, d, texts, cells):
    """Embed variant texts, swap into the tower lookup, predict, restore."""
    from sentence_transformers import SentenceTransformer
    if "encoder" not in score.__dict__:
        score.encoder = SentenceTransformer(embed_mod.MODEL, device="cpu")
    uniq = sorted(set(t for t in texts if t))
    emb = score.encoder.encode(uniq, batch_size=256,
                               normalize_embeddings=True)
    lut = {t: e for t, e in zip(uniq, emb)}
    dim = emb.shape[1]
    keys = [f"{r.congress}_{r.chamber}_{r.rollnumber}"
            for r in d.itertuples()]
    saved = {k: tower._emb_lookup.loc[k].copy()
             for k in keys if k in tower._emb_lookup.index}
    try:
        for k, t in zip(keys, texts):
            tower._emb_lookup.loc[k] = np.asarray(
                lut.get(t, np.zeros(dim)), dtype=np.float32)
        p = np.asarray(tower.predict_proba(cells))
    finally:
        for k, e in saved.items():
            tower._emb_lookup.loc[k] = e
    return p


def metrics(cells, p):
    q = np.clip(p, EPS, 1 - EPS)
    ll = float(-np.mean(np.where(cells.vote == 1, np.log(q), np.log(1 - q))))
    c = cells.assign(p_yea=p)
    c = c[c.qb.isin(["passage", "resolution", "cloture"])
          & c.party_code.isin([100.0, 200.0])]
    maj = (c.groupby(KEY + ["party_code"])["vote"].mean()
           .rename("party_rate").reset_index())
    c = c.merge(maj, on=KEY + ["party_code"])
    c = c[c.party_rate >= 0.5]
    c["defect"] = (c.vote == 0).astype(int)
    c["s"] = 1 - c.p_yea
    aucs = []
    for _, x in c.groupby(KEY):
        if x.defect.sum() < 3 or x.defect.sum() == len(x):
            continue
        r = x.s.rank().to_numpy()
        n1, n0 = int(x.defect.sum()), int((1 - x.defect).sum())
        aucs.append((r[x.defect == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return ll, float(np.mean(aucs)), len(aucs)


def main():
    d = holdout_frame()
    print(f"{len(d)} legislation-linked holdout rollcalls")
    variants = build_variants(d)

    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    cells = (votes.merge(d[KEY + ["vote_question", "bill_type", "bill_no",
                                  "qb"]], on=KEY, how="inner")
             .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                    suffixes=("_v", "")))
    cells["bill_category"] = "legislation"
    print(f"{len(cells):,} member-vote cells")

    tower = load_tower()
    res = {}
    for name, texts in variants.items():
        p = score(tower, d, texts, cells)
        ll, auc, n_rc = metrics(cells, p)
        res[name] = {"logloss": round(ll, 4), "defect_auc": round(auc, 3),
                     "n_auc_rollcalls": n_rc}
        print(f"{name:15s} LL {ll:.4f}  defection AUC {auc:.3f} ({n_rc} rc)")

    (RES / "measures" / "text_ablation.json").write_text(
        json.dumps(res, indent=2))

    label = {"original": "Original text",
             "no_summary": "Bill summary removed",
             "no_question": "Vote question removed",
             "generic_title": "Titles replaced by generic language",
             "no_propernouns": "Proper nouns masked",
             "no_digits": "Digits removed",
             "shuffled_area": "Text from another bill, same policy area"}
    base = res["original"]
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Text given to the frozen model & Log loss & "
             "$\\Delta$ & Defection AUC \\\\", "\\midrule"]
    for k in ("original", "no_summary", "no_question", "generic_title",
              "no_propernouns", "no_digits", "shuffled_area"):
        r = res[k]
        dl = r["logloss"] - base["logloss"]
        lines.append(f"{label[k]} & {r['logloss']:.3f} & "
                     f"{'' if k == 'original' else f'{dl:+.3f}'} & "
                     f"{r['defect_auc']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "ablation_redaction.tex").write_text("\n".join(lines) + "\n")

    m = {"ablOrigLL": f"{base['logloss']:.3f}",
         "ablOrigAUC": f"{base['defect_auc']:.2f}",
         "ablShuffLL": f"{res['shuffled_area']['logloss']:.3f}",
         "ablShuffAUC": f"{res['shuffled_area']['defect_auc']:.2f}",
         "ablNoSumLL": f"{res['no_summary']['logloss']:.3f}",
         "ablNoQLL": f"{res['no_question']['logloss']:.3f}",
         "ablNoPropLL": f"{res['no_propernouns']['logloss']:.3f}",
         "ablGenTitleLL": f"{res['generic_title']['logloss']:.3f}",
         "ablRC": f"{len(d):,}"}
    (OUT / "ablation_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                  for k, v in m.items()) + "\n")
    print("wrote ablation_redaction.tex, ablation_macros.tex")


if __name__ == "__main__":
    main()
