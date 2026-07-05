"""Majority-transition study (review r1, issue O7).

For each House test congress C in the text-era window, train the champion
architecture and the majority-status count table on congresses 108..C-1
and evaluate on C. Flip congresses (majority changed from C-1): 110, 112,
116, 118. Placebo congresses (no change): 113, 115, 117, 119. If the
transfer degradation is role-driven, it should concentrate in flip
transitions and, within them, in procedural votes.

Output: Modified Data/results/measures/transitions.json (overall and
by-question-bucket log loss per test congress x model), plus per-vote
predictions for the champion for any further breakdowns.

Run: python3 Code/25_transitions.py   (~45 min: 8 champion fits)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness import metrics
from models_forecast import MajorityQuestionRate, question_bucket
from models_texttower import TextTower

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
OUT = MOD / "results" / "measures"

FLIPS = {110: True, 112: True, 113: False, 115: False, 116: True,
         117: False, 118: True, 119: False}


def load_house():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[(votes.congress >= 108) & (votes.chamber == "House")]
    feat = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question", "bill_category",
         "bill_type", "bill_no"]]
    return votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                       how="left", validate="m:1")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_house()
    results = {}
    for cong, is_flip in FLIPS.items():
        train = panel[panel.congress < cong]
        test = panel[panel.congress == cong]
        if train.congress.nunique() < 2 or not len(test):
            continue
        print(f"== test congress {cong} ({'FLIP' if is_flip else 'placebo'}): "
              f"train {len(train):,}, test {len(test):,}")
        entry = {"flip": is_flip, "n_test": int(len(test)), "models": {}}
        y = test["vote"].to_numpy()
        qb = question_bucket(test["vote_question"]).to_numpy()

        for name, factory in (
            ("majority_question_rate", lambda: MajorityQuestionRate()),
            ("champion", lambda: TextTower(
                k=16, use_text=False, use_emb=True, mq_offset=True,
                calibrate=True, mlp_head=True, es_mode="temporal",
                name="champ_transfer")),
        ):
            model = factory().fit(train)
            p = np.asarray(model.predict_proba(test.drop(columns=["vote"])))
            m = metrics(y, p)
            by_q = {}
            for q in np.unique(qb):
                sel = qb == q
                if sel.sum() >= 500:
                    by_q[q] = round(metrics(y[sel], p[sel])["log_loss"], 4)
            entry["models"][name] = {
                "log_loss": round(m["log_loss"], 4),
                "accuracy": round(m["accuracy"], 4), "by_qbucket": by_q}
            print(f"   {name}: ll {m['log_loss']:.4f} acc {m['accuracy']:.4f}")
        results[str(cong)] = entry
        (OUT / "transitions.json").write_text(json.dumps(results, indent=2))
    print("done -> transitions.json")


if __name__ == "__main__":
    main()
