"""Extract the measurement layer for Paper A's figures.

Three outputs, all to Modified Data/results/measures/:

1. surprises.parquet — every vote scored by the frozen v2 champion (fit
   on all data through its snapshot), with per-vote log loss. The top of
   this ranking is the "out-of-character vote" measure: decisions the
   best available model of a member's behavior gets confidently wrong
   even in sample. Metadata joined for presentation.

2. cutpoints_house118.parquet + members_house118.parquet — a 1D logistic
   ideal-point fit on the 118th House (party-sign initialized, following
   the identification convention in Notes/decisions.md), with each
   rollcall's implied cutpoint -b_r/a_r on the member scale, its policy
   area, question bucket, and yea share. Cutpoints are reported only
   where |a_r| is large enough for the location to be identified.

3. The same for the 118th Senate (smaller, for the appendix).

Run: python3 Code/22_extract_measures.py   (~10 min, mostly the 7M-vote
blend forward pass)
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models_forecast import question_bucket
from models_idealpoint import IdealPoint, NominateLogit

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
ART = MOD / "results" / "frozen"
OUT = MOD / "results" / "measures"

MIN_DISCRIM = 0.35  # below this |a|, a 1D cutpoint location is noise


def load_panel() -> pd.DataFrame:
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes["congress"] >= 108]
    feat = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question", "bill_category",
         "bill_type", "bill_no"]]
    return votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                       how="left", validate="m:1")


def surprises(panel: pd.DataFrame) -> None:
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"]
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    print(f"scoring {len(panel):,} votes with frozen v2 blend...")
    # per-congress chunks: a single 7M-row forward pass exceeds Metal's
    # matmul kernel limits (observed on the full panel; ~700k-row eval
    # sets are fine). predict_proba is row-independent, so chunking is
    # exact, not an approximation.
    parts = []
    for cong, chunk in panel.groupby("congress", sort=True):
        parts.append(pd.Series(
            np.asarray(model.predict_proba(chunk.drop(columns=["vote"]))),
            index=chunk.index))
        print(f"  congress {cong}: {len(chunk):,} votes scored")
    p = pd.concat(parts).reindex(panel.index).to_numpy()
    eps = 1e-7
    q = np.clip(p, eps, 1 - eps)
    y = panel["vote"].to_numpy()
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))

    out = panel[["congress", "chamber", "rollnumber", "icpsr", "date",
                 "vote_question", "vote", "party_code"]].copy()
    out["p_yea"], out["vote_ll"] = p, ll
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "bioname", "state_abbrev"]]
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "vote_desc", "bill_number"]]
    out = (out.merge(mem, on=["congress", "chamber", "icpsr"], how="left")
              .merge(rc, on=["congress", "chamber", "rollnumber"], how="left"))
    # NOTE: an in-sample member-fit comparison against NominateLogit was
    # removed here — a regularized forecaster against a saturated
    # in-sample scaler is apples-to-oranges. The honest member-level
    # comparison (identical held-out cells) lives in 24_member_fit.py.

    # keep the full distribution's summary + the informative tail
    out = out.sort_values("vote_ll", ascending=False)
    out.head(2000).to_parquet(OUT / "surprises.parquet", index=False)
    stats = {"n_scored": int(len(out)), "mean_ll": float(ll.mean()),
             "p99_ll": float(np.quantile(ll, 0.99)),
             "n_confident_wrong": int(((p > 0.9) & (y == 0)).sum()
                                      + ((p < 0.1) & (y == 1)).sum())}
    (OUT / "surprises_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cutpoints(panel: pd.DataFrame, congress: int, chamber: str) -> None:
    df = panel[(panel.congress == congress) & (panel.chamber == chamber)].copy()
    mem = pd.read_parquet(MOD / "members.parquet")
    mem = mem[(mem.congress == congress) & (mem.chamber == chamber)]
    init = pd.Series(np.where(mem["party_code"] == 200.0, 0.5,
                              np.where(mem["party_code"] == 100.0, -0.5, 0.0)),
                     index=mem["icpsr"].to_numpy())
    print(f"fitting 1D ideal points: {chamber} {congress} ({len(df):,} votes)")
    ip = IdealPoint(k=1, min_epochs=10).fit(df, init_positions=init)

    pos = ip.member_positions().rename(columns={"dim1": "x"})
    pos = pos.merge(mem[["icpsr", "bioname", "party_code", "state_abbrev"]],
                    on="icpsr", how="left")
    pos.to_parquet(OUT / f"members_{chamber.lower()}{congress}.parquet", index=False)

    inv = {i: r for r, i in ip.rollcall_index.items()}
    rc = pd.DataFrame({"rc_key": [inv[i] for i in range(len(inv))],
                       "a": ip._a[:, 0], "b": ip._b[:, 0]})
    key = rc["rc_key"].str.split("_", expand=True)
    rc["congress"], rc["chamber"] = key[0].astype(int), key[1]
    rc["rollnumber"] = key[2].astype(int)
    # cutpoint: where the average member is indifferent. c_m varies by
    # member; using its mean keeps this a 1-number-per-rollcall display
    c_bar = float(ip._c.mean())
    rc["cutpoint"] = -(rc["b"] + c_bar) / rc["a"]
    rc["identified"] = rc["a"].abs() >= MIN_DISCRIM

    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question",
         "bill_type", "bill_no"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "policy_area", "title"]]
    rcm = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "date", "vote_desc"]]
    yea = (panel.groupby(["congress", "chamber", "rollnumber"])["vote"]
           .mean().rename("yea_share").reset_index())
    rc = (rc.merge(links, on=["congress", "chamber", "rollnumber"], how="left")
            .merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
            .merge(rcm, on=["congress", "chamber", "rollnumber"], how="left")
            .merge(yea, on=["congress", "chamber", "rollnumber"], how="left"))
    rc["qbucket"] = question_bucket(rc["vote_question"])
    rc.to_parquet(OUT / f"cutpoints_{chamber.lower()}{congress}.parquet", index=False)
    n_id = int(rc["identified"].sum())
    print(f"  {len(rc)} rollcalls, {n_id} with identified cutpoints "
          f"({n_id / len(rc):.0%}); member x range "
          f"[{pos['x'].min():.2f}, {pos['x'].max():.2f}]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    surprises(panel)
    cutpoints(panel, 118, "House")
    cutpoints(panel, 118, "Senate")
    cutpoints(panel, 119, "House")
    cutpoints(panel, 119, "Senate")


if __name__ == "__main__":
    main()
