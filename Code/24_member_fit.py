"""Member-level fit on held-out votes: the field's statistic, honestly.

Compares the benchmark's 8D spatial model against the classical model
(frozen DW-NOMINATE positions, estimated rollcall parameters) member by
member, on IDENTICAL held-out completion-regime cells — both models fit
on the same training cells, both scored on votes neither saw. Reports
each member's held-out geometric mean probability (GMP) and
classification error rate under each model.

(An earlier in-sample version of this comparison was discarded as
apples-to-oranges: a regularized forecaster against a saturated
in-sample scaler; see session log 2026-07-05.)

Requires: run_benchmark --split regimeA_seed42
          --models nominate_logit ideal_point_8d --save-preds
Output: Modified Data/results/measures/member_fit.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "Modified Data" / "results"
OUT = RES / "measures"
EPS = 1e-7


def per_member(preds: pd.DataFrame, tag: str) -> pd.DataFrame:
    q = np.clip(preds.p_yea, EPS, 1 - EPS)
    preds = preds.assign(
        ll=-(preds.vote * np.log(q) + (1 - preds.vote) * np.log(1 - q)),
        err=((preds.p_yea > 0.5) != (preds.vote == 1)).astype(float))
    g = preds.groupby("icpsr").agg(**{
        "n": ("ll", "size"), f"ll_{tag}": ("ll", "mean"),
        f"err_{tag}": ("err", "mean")}).reset_index()
    g[f"gmp_{tag}"] = np.exp(-g[f"ll_{tag}"])
    return g


def main():
    ours = pd.read_parquet(RES / "preds" / "regimeA_seed42_ideal_point_8d.parquet")
    nom = pd.read_parquet(RES / "preds" / "regimeA_seed42_nominate_logit.parquet")
    a, b = per_member(ours, "ours"), per_member(nom, "nom").drop(columns=["n"])
    mf = a.merge(b, on="icpsr")
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")
    names = (mem.sort_values("congress")
             .drop_duplicates("icpsr", keep="last")[
                 ["icpsr", "bioname", "party_code", "state_abbrev", "chamber"]])
    mf = mf.merge(names, on="icpsr", how="left")
    mf.to_parquet(OUT / "member_fit.parquet", index=False)
    sub = mf[mf.n >= 100]
    print(f"{len(mf)} members; held-out GMP higher under 8D for "
          f"{(sub.gmp_ours > sub.gmp_nom).mean():.1%} of {len(sub)} "
          f"members with >=100 held-out votes")
    print(f"median GMP: ours {sub.gmp_ours.median():.3f} "
          f"vs NOMINATE {sub.gmp_nom.median():.3f}")
    gain = (sub.assign(g=sub.gmp_ours - sub.gmp_nom)
            .nlargest(12, "g")[["bioname", "state_abbrev", "party_code",
                                "gmp_ours", "gmp_nom"]])
    print("\nlargest per-member gains (the members 1D fits worst):")
    print(gain.to_string(index=False))


if __name__ == "__main__":
    main()
