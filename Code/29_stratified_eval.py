"""Stratified evaluation of stored forecast predictions (review r3).

Three post-hoc strata over the temporal-forecast holdout, computed from
saved test predictions only (no model is refit):

1. Seen-bill vs fresh-bill: was the test rollcall's bill the subject of
   any rollcall in the training partition? If text gains concentrate on
   seen bills, the model may be recalling a bill family's coalition
   rather than reading content (review r3, major point 2).
2. Chamber: House vs Senate log loss per model (major point 9).
3. Policy area: the text gain (no-text minus model log loss) by CRS
   policy area, passage-family votes. policy_area is CRS-curated and
   enters only as an evaluation stratifier, never as a model input.

Outputs:
  Draft/tables/stratified_macros.tex   (macros for the paper)
  Draft/tables/policy_gain.tex         (policy-area table)
  Modified Data/results/measures/stratified_eval.json

Run: python3 Code/29_stratified_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"

sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket

MODELS = {"champ": "blend3_mlp_tfidf_emb3_tcal",
          "clean": "emb2_mlp_mq_16d_tcal",
          "notext": "notext_mq_16d_tcal"}
KEY = ["congress", "chamber", "rollnumber"]
EPS = 1e-6


def test_flag(rc):
    """Reproduce the forecast split: last 10% of each congress-chamber's
    rollcalls by date (ties by rollnumber)."""
    rc = rc.sort_values(["congress", "chamber", "date", "rollnumber"]).copy()
    rank = rc.groupby(["congress", "chamber"]).cumcount()
    size = rc.groupby(["congress", "chamber"])["rollnumber"].transform("size")
    rc["is_test"] = (rank + 0.5) / size >= 0.90
    return rc


def seen_bill_flags(rc):
    """For each test rollcall linked to a bill: does the same bill have a
    rollcall in the training partition, in either chamber?"""
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        KEY + ["bill_type", "bill_no"]]
    d = rc.merge(links, on=KEY, how="left")
    bid = ["congress", "bill_type", "bill_no"]
    train_bills = d[~d.is_test].dropna(subset=["bill_type"])[
        bid].drop_duplicates()
    train_bills["seen"] = True
    t = d[d.is_test].merge(train_bills, on=bid, how="left")
    t["seen"] = t.seen.eq(True) & t.bill_type.notna()
    t["linked"] = t.bill_type.notna()
    return t[KEY + ["seen", "linked"]].drop_duplicates(KEY)


def logloss(df):
    p = np.clip(df.p_yea, EPS, 1 - EPS)
    return float(-np.mean(np.where(df.vote == 1, np.log(p), np.log(1 - p))))


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        KEY + ["date", "vote_question"]]
    rc = test_flag(rc)
    rc["qb"] = question_bucket(rc["vote_question"])
    seen = seen_bill_flags(rc)

    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        KEY + ["bill_type", "bill_no"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "policy_area"]]
    pa = (rc.merge(links, on=KEY, how="left")
            .merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
          )[KEY + ["policy_area", "qb"]].drop_duplicates(KEY)

    res, preds = {}, {}
    for tag, model in MODELS.items():
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        p = p.dropna(subset=["vote"])
        p = p.merge(seen, on=KEY, how="left").merge(pa, on=KEY, how="left")
        preds[tag] = p
        r = {"ll_all": logloss(p)}
        for ch in ("House", "Senate"):
            r[f"ll_{ch}"] = logloss(p[p.chamber == ch])
            r[f"n_{ch}"] = int((p.chamber == ch).sum())
        lk = p[p.linked == True]  # noqa: E712 — bill-linked votes only
        for lab, mask in (("seen", lk.seen == True),      # noqa: E712
                          ("fresh", lk.seen == False)):   # noqa: E712
            r[f"ll_{lab}"] = logloss(lk[mask])
            r[f"n_{lab}"] = int(mask.sum())
            r[f"rc_{lab}"] = int(lk[mask][KEY].drop_duplicates().shape[0])
        res[tag] = {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()}
        print(tag, res[tag])

    # policy-area text gain: mean per-vote log loss, no-text minus model,
    # passage-family votes in areas with enough held-out rollcalls
    fam = ["passage", "resolution", "cloture"]
    base = preds["notext"]
    rows = []
    for area, g in base[base.qb.isin(fam) & base.policy_area.notna()].groupby(
            "policy_area"):
        n_rc = g[KEY].drop_duplicates().shape[0]
        if n_rc < 25:
            continue
        cells = g[KEY + ["icpsr"]]
        row = {"area": area, "n_rc": n_rc}
        ll_nt = logloss(g)
        for tag in ("champ", "clean"):
            m = preds[tag].merge(cells, on=KEY + ["icpsr"])
            row[f"gain_{tag}"] = ll_nt - logloss(m)
        rows.append(row)
    pa_tab = pd.DataFrame(rows).sort_values("gain_clean", ascending=False)
    print(pa_tab.to_string(index=False))

    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Policy area & Held-out rollcalls & "
             "\\multicolumn{2}{c}{Text gain in log loss} \\\\",
             " & & Ensemble & Clean tower \\\\", "\\midrule"]
    for _, r in pa_tab.iterrows():
        lines.append(f"{r.area} & {r.n_rc} & {r.gain_champ:+.3f} & "
                     f"{r.gain_clean:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "policy_gain.tex").write_text("\n".join(lines) + "\n")

    c, n, x = res["champ"], res["notext"], res["clean"]
    macros = {
        "seenLLchamp": f"{c['ll_seen']:.3f}",
        "freshLLchamp": f"{c['ll_fresh']:.3f}",
        "seenLLnotext": f"{n['ll_seen']:.3f}",
        "freshLLnotext": f"{n['ll_fresh']:.3f}",
        "seenLLclean": f"{x['ll_seen']:.3f}",
        "freshLLclean": f"{x['ll_fresh']:.3f}",
        "seenRC": f"{c['rc_seen']:,}",
        "freshRC": f"{c['rc_fresh']:,}",
        "houseLLchamp": f"{c['ll_House']:.3f}",
        "senateLLchamp": f"{c['ll_Senate']:.3f}",
        "houseLLnotext": f"{n['ll_House']:.3f}",
        "senateLLnotext": f"{n['ll_Senate']:.3f}",
    }
    (OUT / "stratified_macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                  for k, v in macros.items()) + "\n")
    res["policy_gain"] = pa_tab.to_dict("records")
    (RES / "measures" / "stratified_eval.json").write_text(
        json.dumps(res, indent=2, default=str))
    print("wrote stratified_macros.tex, policy_gain.tex")


if __name__ == "__main__":
    main()
