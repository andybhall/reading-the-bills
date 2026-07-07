"""Do the same members drive the scaling gains and the defection gains?

Review r3, detailed comment 10: the paper claimed the members whose
scaling improves most (Figure 1) are the same members whose defections
the text model prices (Section 4). This computes the actual overlap:

- scaling gain: held-out GMP (ours minus DW-NOMINATE model), from
  member_fit.parquet (completion regime)
- defection-pricing gain: mean log-loss improvement (no-text minus
  champion) on the member's own defection votes in the forecast
  holdout (majority-yea passage-family votes where the member voted
  nay), members with >= 5 such defections

Reports the Spearman rank correlation between the two, over members
present in both panels. Output: Draft/tables/overlap_macros.tex

Run: python3 Code/33_member_overlap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"

sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket

KEY = ["congress", "chamber", "rollnumber"]
EPS = 1e-6


def defection_gain():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[KEY + ["vote_question"]]
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    frames = {}
    for tag, model in (("champ", "blend3_mlp_tfidf_emb3_tcal"),
                       ("notext", "notext_mq_16d_tcal")):
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        d = (p.merge(rc, on=KEY, how="left")
              .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                     suffixes=("_v", "")))
        d["qb"] = question_bucket(d["vote_question"])
        d = d[d.qb.isin(["passage", "resolution", "cloture"])
              & d.party_code.isin([100.0, 200.0])]
        maj = (d.groupby(KEY + ["party_code"])["vote"].mean()
               .rename("party_rate").reset_index())
        d = d.merge(maj, on=KEY + ["party_code"])
        d = d[(d.party_rate >= 0.5) & (d.vote == 0)]  # own defections only
        d["ll"] = -np.log(np.clip(1 - d.p_yea, EPS, None))
        frames[tag] = d.set_index(KEY + ["icpsr"])["ll"]
    j = pd.concat(frames, axis=1).dropna()
    g = (j["notext"] - j["champ"]).groupby("icpsr").agg(["mean", "size"])
    return g[g["size"] >= 5]["mean"].rename("defection_gain")


def main():
    mf = pd.read_parquet(RES / "measures" / "member_fit.parquet")
    mf = mf[mf.n >= 100].set_index("icpsr")
    scaling = (mf.gmp_ours - mf.gmp_nom).rename("scaling_gain")
    dg = defection_gain()
    j = pd.concat([scaling, dg], axis=1).dropna()
    rho, pval = spearmanr(j.scaling_gain, j.defection_gain)
    print(f"members in both panels: {len(j)}")
    print(f"Spearman rho = {rho:.3f} (p = {pval:.2g})")
    print("top-10 scaling gainers, their defection-gain percentile:")
    pct = j.defection_gain.rank(pct=True)
    for i in j.scaling_gain.nlargest(10).index:
        print(f"  icpsr {i}: defection-gain pctile {100*pct[i]:.0f}")
    lines = [f"\\newcommand{{\\overlapRho}}{{{rho:.2f}}}",
             f"\\newcommand{{\\overlapN}}{{{len(j):,}}}"]
    for tag, icpsr in (("paul", 14290.0), ("amash", 21143.0)):
        if icpsr in pct.index:
            lines.append(f"\\newcommand{{\\{tag}DefGainPct}}"
                         f"{{{100*pct[icpsr]:.0f}}}")
    (OUT / "overlap_macros.tex").write_text("\n".join(lines) + "\n")
    print("wrote overlap_macros.tex")


if __name__ == "__main__":
    main()
