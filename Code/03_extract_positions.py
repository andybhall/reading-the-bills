"""Extract learned member positions and validate against DW-NOMINATE.

Fits the 1D ideal-point model PER CHAMBER on the benchmark training set.
Pooling chambers is fine for prediction but not for interpretation: House
and Senate share no rollcalls, so the relative rotation/scale of the two
blocks is identified only through the ~95 chamber-switchers — too weak, and
in practice the Senate block fit with inverted sign and inflated magnitudes
(diagnosed 2026-06-11; see logs/2026-06-11_session02.md). NOMINATE fits
chambers separately for the same reason.

Outputs:
  - Modified Data/results/member_positions_1d.parquet (one row per
    member x chamber, sign-aligned so higher = more conservative)
  - Modified Data/results/idealpoint_validation.json (correlations with
    DW-NOMINATE overall and within party, per chamber; top disagreements)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness import KEY
from models_idealpoint import IdealPoint

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
SPLIT_NAME = "regimeA_seed42"


def fit_chamber(train: pd.DataFrame, profile: pd.DataFrame, chamber: str):
    # party-sign init (+0.5 R / -0.5 D / 0 other) pins the axis orientation;
    # within-party ordering — what we validate — is still free
    party = profile["party_code"]
    init = pd.Series(np.where(party == 200, 0.5, np.where(party == 100, -0.5, 0.0)),
                     index=profile.index, dtype=float)
    model = IdealPoint(k=1).fit(train[train["chamber"] == chamber], init_positions=init)
    pos = model.member_positions()
    pos["chamber"] = chamber
    pos = pos.merge(profile, on="icpsr", how="left")

    # align sign within chamber: higher learned score = more conservative
    both = pos.dropna(subset=["nominate_dim1"])
    sign = np.sign(np.corrcoef(both["dim1"], both["nominate_dim1"])[0, 1])
    pos["dim1"] = pos["dim1"] * sign
    return pos


def validate(pos: pd.DataFrame) -> dict:
    both = pos.dropna(subset=["nominate_dim1"])
    stats = {
        "n_members": int(len(pos)),
        "n_with_nominate": int(len(both)),
        "corr_dim1_nominate": round(float(np.corrcoef(both["dim1"], both["nominate_dim1"])[0, 1]), 4),
        "spearman_dim1_nominate": round(float(both["dim1"].corr(both["nominate_dim1"], method="spearman")), 4),
        "share_abs_gt_5": round(float((both["dim1"].abs() > 5).mean()), 4),
    }
    for party, code in [("dem", 100), ("rep", 200)]:
        sub = both[both["party_code"] == code]
        stats[f"corr_within_{party}"] = round(float(np.corrcoef(sub["dim1"], sub["nominate_dim1"])[0, 1]), 4)
        stats[f"n_{party}"] = int(len(sub))
    return stats


def main():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    splits = pd.read_parquet(MOD / "splits" / f"{SPLIT_NAME}.parquet")
    df = votes.merge(splits, on=KEY, how="inner", validate="1:1")
    train = df[df["split"] == "train"]

    mem = pd.read_parquet(MOD / "members.parquet")
    profile = mem.sort_values("congress").groupby("icpsr").agg(
        bioname=("bioname", "last"), party_code=("party_code", "last"),
        nominate_dim1=("nominate_dim1", "last"), nominate_dim2=("nominate_dim2", "last"),
        last_congress=("congress", "last"))

    pos = pd.concat([fit_chamber(train, profile, ch) for ch in ("House", "Senate")],
                    ignore_index=True)
    stats = {ch: validate(pos[pos["chamber"] == ch]) for ch in ("House", "Senate")}

    # largest disagreements, computed within chamber (z-scores per chamber)
    both = pos.dropna(subset=["nominate_dim1"]).copy()
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    both["gap"] = (both.groupby("chamber")["dim1"].transform(z)
                   - both.groupby("chamber")["nominate_dim1"].transform(z)).abs()
    stats["top_disagreements"] = (
        both.nlargest(15, "gap")[["bioname", "chamber", "party_code", "last_congress",
                                  "nominate_dim1", "dim1", "gap"]]
        .round(3).to_dict(orient="records"))

    out = MOD / "results"
    out.mkdir(exist_ok=True)
    pos.to_parquet(out / "member_positions_1d.parquet", index=False)
    (out / "idealpoint_validation.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "top_disagreements"}, indent=2))
    print("\nTop disagreements with NOMINATE (z-score gap, within chamber):")
    for d in stats["top_disagreements"][:10]:
        print(f"  {d['bioname'][:32]:32s} {d['chamber'][:6]:6s} party {d['party_code']:>5} "
              f"cong {d['last_congress']} nom {d['nominate_dim1']:+.2f} "
              f"learned {d['dim1']:+.2f} gap {d['gap']:.2f}")


if __name__ == "__main__":
    main()
