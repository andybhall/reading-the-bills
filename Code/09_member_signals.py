"""Extract member signals: party-loyalty residual, orthogonal to ideology.

Method (memo: Notes/memo_party_loyalty.md):
1. Fit the validated 1D ideal-point model per chamber on Regime A training
   votes (party-sign init; see 03_extract_positions.py).
2. Identify PARTY-UNITY rollcalls in training data: a majority of voting
   Democrats opposes a majority of voting Republicans (each party >= 10
   voters on the rollcall).
3. For every train vote by a D or R member on those rollcalls:
     with_party    = voted with own party's majority side
     p_with_party  = model P(votes with party side)  [ideology-implied]
4. Per member-congress: loyalty residual = actual rate - model-implied rate.
   Positive = more loyal than ideology predicts ("party soldier");
   negative = less loyal ("maverick").

Caveats (documented in memo): predictions are in-sample (the 1D model is
deliberately too low-capacity to memorize individual votes: 2 params per
member); residual conflates party pressure with un-modeled dimensions.

Output: Modified Data/results/member_signals.parquet
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
MIN_PARTY_VOTERS = 10
RC = ["congress", "chamber", "rollnumber"]


def main():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    splits = pd.read_parquet(MOD / "splits" / f"{SPLIT_NAME}.parquet")
    df = votes.merge(splits, on=KEY, how="inner", validate="1:1")
    train = df[df["split"] == "train"].copy()
    train = train[train["party_code"].isin([100.0, 200.0])]

    # party majorities per rollcall (train votes only)
    pm = (train.groupby(RC + ["party_code"])["vote"]
          .agg(["mean", "size"]).reset_index())
    wide = pm.pivot_table(index=RC, columns="party_code",
                          values=["mean", "size"])
    wide.columns = [f"{a}_{int(b)}" for a, b in wide.columns]
    unity = wide[(wide["size_100"] >= MIN_PARTY_VOTERS)
                 & (wide["size_200"] >= MIN_PARTY_VOTERS)
                 & ((wide["mean_100"] >= 0.5) != (wide["mean_200"] >= 0.5))]
    party_side = (unity[["mean_100", "mean_200"]] >= 0.5).astype(float)
    party_side.columns = ["side_100", "side_200"]
    print(f"party-unity rollcalls: {len(unity):,} of {wide.shape[0]:,}")

    mem = pd.read_parquet(MOD / "members.parquet")
    profile = mem.drop_duplicates(["congress", "chamber", "icpsr"], keep="last")[
        ["congress", "chamber", "icpsr", "bioname", "party_code", "state_abbrev",
         "nominate_dim1"]]

    frames = []
    for chamber in ("House", "Senate"):
        tr = train[train["chamber"] == chamber]
        init = pd.Series(np.where(profile["party_code"] == 200, 0.5,
                                  np.where(profile["party_code"] == 100, -0.5, 0.0)),
                         index=pd.Index(profile["icpsr"], name="icpsr"))
        init = init.groupby(level=0).last()
        model = IdealPoint(k=1).fit(tr, init_positions=init)

        sub = tr.merge(party_side, left_on=RC, right_index=True, how="inner")
        p_yea = model.predict_proba(sub.drop(columns=["vote", "cast_code"],
                                             errors="ignore"))
        own_side = np.where(sub["party_code"] == 100.0, sub["side_100"], sub["side_200"])
        sub["with_party"] = (sub["vote"] == own_side).astype(float)
        sub["p_with_party"] = np.where(own_side == 1.0, p_yea, 1 - p_yea)

        g = sub.groupby(["congress", "chamber", "icpsr"]).agg(
            n_unity=("with_party", "size"),
            actual=("with_party", "mean"),
            expected=("p_with_party", "mean"))
        g["loyalty_residual"] = g["actual"] - g["expected"]

        pos = model.member_positions().set_index("icpsr")
        sign = 1.0  # party-sign init pins orientation; align to conservative=+
        merged = g.reset_index().merge(profile, on=["congress", "chamber", "icpsr"])
        if np.corrcoef(merged.dropna(subset=["nominate_dim1"])
                       .merge(pos["dim1"], on="icpsr")[["dim1", "nominate_dim1"]],
                       rowvar=False)[0, 1] < 0:
            sign = -1.0
        merged["ideal_1d"] = merged["icpsr"].map(pos["dim1"]) * sign
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(MOD / "results" / "member_signals.parquet", index=False)

    # validation summary
    o = out[out["n_unity"] >= 100]
    stats = {
        "member_congress_rows": int(len(out)),
        "rows_n_unity_ge_100": int(len(o)),
        "mean_actual_with_party": round(float(o["actual"].mean()), 4),
        "mean_expected_with_party": round(float(o["expected"].mean()), 4),
        "residual_sd": round(float(o["loyalty_residual"].std()), 4),
        "corr_residual_extremity_within_party": round(float(
            o.assign(ext=o.groupby(["party_code"])["ideal_1d"].transform(
                lambda s: (s - s.mean()).abs()))
            [["loyalty_residual", "ext"]].corr().iloc[0, 1]), 4),
    }
    (MOD / "results" / "member_signals_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    for congress, chamber in [(118, "Senate"), (118, "House")]:
        sub = o[(o.congress == congress) & (o.chamber == chamber)]
        print(f"\n=== {chamber} {congress}: LEAST loyal vs ideology-implied (mavericks)")
        cols = ["bioname", "party_code", "state_abbrev", "n_unity", "actual",
                "expected", "loyalty_residual"]
        print(sub.nsmallest(10, "loyalty_residual")[cols].round(3).to_string(index=False))
        print(f"=== {chamber} {congress}: MOST loyal vs ideology-implied")
        print(sub.nlargest(5, "loyalty_residual")[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
