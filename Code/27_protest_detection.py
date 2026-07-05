"""Protest-vote detection: does reading the bill help? (P5v6, point 2)

Task: on held-out future rollcalls (forecast test window), among members
whose party's majority voted yea on a passage-family vote, identify who
defects (votes nay). Compares the champion (reads rollcall text) with
the same architecture stripped of all text (metadata + member history
only), using each model's saved test predictions.

Outputs:
  Modified Data/results/measures/protest_detection.json  (AUCs, counts)
  (figure built in make_figures.f12 from the same preds)

Run after both preds exist:
  run_benchmark --split forecast108_119 --models blend3... notext_mq_16d_tcal --save-preds
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "Modified Data" / "results"
OUT = RES / "measures"

import sys
sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket


def load(model):
    p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
    rc = pd.read_parquet(ROOT / "Modified Data" / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question"]]
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    p = (p.merge(rc, on=["congress", "chamber", "rollnumber"], how="left")
          .merge(mem, on=["congress", "chamber", "icpsr"], how="left"))
    p["qb"] = question_bucket(p["vote_question"])
    return p


def auc(y, s):
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    res = {}
    frames = {}
    for model in ("blend3_mlp_tfidf_emb3_tcal", "notext_mq_16d_tcal"):
        d = load(model)
        d = d[d.qb.isin(["passage", "resolution", "cloture"])]
        d = d[d.party_code.isin([100.0, 200.0])]
        # party majority side per rollcall x party, from realized votes
        maj = (d.groupby(["congress", "chamber", "rollnumber", "party_code"])
               ["vote"].mean().rename("party_rate").reset_index())
        d = d.merge(maj, on=["congress", "chamber", "rollnumber", "party_code"])
        d = d[d.party_rate >= 0.5]          # member's party majority is yea
        d["defect"] = (d.vote == 0).astype(int)
        d["score"] = 1 - d.p_yea            # predicted defection prob
        frames[model] = d
        res[model] = {
            "n_member_votes": int(len(d)),
            "n_defections": int(d.defect.sum()),
            "auc_pooled": round(auc(d.defect.to_numpy(), d.score.to_numpy()), 3),
        }
        # within-rollcall AUC (ranking members on each vote), averaged over
        # rollcalls with at least 3 defectors
        g = [auc(x.defect.to_numpy(), x.score.to_numpy())
             for _, x in d.groupby(["congress", "chamber", "rollnumber"])
             if x.defect.sum() >= 3]
        res[model]["auc_within_rollcall"] = round(float(np.nanmean(g)), 3)
        res[model]["n_rollcalls_3plus"] = int(np.sum(~np.isnan(g)))
        print(model, res[model])
    (OUT / "protest_detection.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
