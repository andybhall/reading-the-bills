"""Calibration metrics for the forecast probabilities (review r3, R10).

Brier score and expected calibration error (15 equal-count bins) on the
temporal-forecast holdout for the champion ensemble, the leakage-clean
MiniLM tower, and the no-text counterpart; plus rollcall-level
calibration of predicted majority-defection shares (predicted mean
defection probability per majority-yea passage-family rollcall vs the
realized share), reported as a decile correlation used by figure
f6_calibration's right panel.

Outputs:
  Modified Data/results/measures/calibration_metrics.json
  Draft/tables/calibration_macros.tex

Run: python3 Code/32_calibration_metrics.py
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


def ece(p, y, bins=15):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p) - 1, 0, bins - 1)
    err, n = 0.0, len(p)
    for b in range(bins):
        m = idx == b
        if m.sum():
            err += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(err)


def defection_frame(model):
    p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[KEY + ["vote_question"]]
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    d = (p.merge(rc, on=KEY, how="left")
          .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                 suffixes=("_v", "")))
    d["qb"] = question_bucket(d["vote_question"])
    d = d[d.qb.isin(["passage", "resolution", "cloture"])
          & d.party_code.isin([100.0, 200.0])]
    maj = (d.groupby(KEY + ["party_code"])["vote"].mean()
           .rename("party_rate").reset_index())
    d = d.merge(maj, on=KEY + ["party_code"])
    d = d[d.party_rate >= 0.5]
    g = d.groupby(KEY).agg(pred_share=("p_yea", lambda s: 1 - s.mean()),
                           real_share=("vote", lambda s: 1 - s.mean()),
                           n=("vote", "size"))
    return g[g.n >= 50]


def main():
    res, macros = {}, []
    for tag, model in MODELS.items():
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        p = p.dropna(subset=["vote"])
        pr, y = p.p_yea.to_numpy(), p.vote.to_numpy()
        res[tag] = {"brier": round(float(np.mean((pr - y) ** 2)), 4),
                    "ece": round(ece(pr, y), 4)}
        g = defection_frame(model)
        res[tag]["defshare_r"] = round(
            float(np.corrcoef(g.pred_share, g.real_share)[0, 1]), 3)
        res[tag]["defshare_n_rc"] = int(len(g))
        print(tag, res[tag])
        up = tag.capitalize()
        macros += [f"\\newcommand{{\\brier{up}}}{{{res[tag]['brier']:.3f}}}",
                   f"\\newcommand{{\\ece{up}}}{{{res[tag]['ece']:.3f}}}",
                   f"\\newcommand{{\\defShareR{up}}}"
                   f"{{{res[tag]['defshare_r']:.2f}}}"]
    macros.append(f"\\newcommand{{\\defShareNrc}}"
                  f"{{{res['champ']['defshare_n_rc']:,}}}")
    (RES / "measures" / "calibration_metrics.json").write_text(
        json.dumps(res, indent=2))
    (OUT / "calibration_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote calibration_macros.tex")


if __name__ == "__main__":
    main()
