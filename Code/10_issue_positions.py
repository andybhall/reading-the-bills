"""Issue-specific ideal points: fit the 1D spatial model separately within
each policy area (top areas by rollcall count, plus Senate nominations),
congresses 108-119, Regime A training votes.

Each member gets a position PER TOPIC. Positions are z-scored within
chamber-topic (members with >= 50 topic votes) and sign-aligned to
DW-NOMINATE (+ = conservative), so cross-topic comparisons are in units of
within-chamber standard deviations. A pooled all-votes fit on the same
sample provides each member's "overall" position; deviation = topic z -
overall z isolates issue-specific positioning.

Output: Modified Data/results/issue_positions.parquet (long: chamber,
topic, icpsr, z, n_votes, overall_z, deviation) + summary stats JSON.
Memo: Notes/memo_issue_positions.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness import KEY
from models_idealpoint import IdealPoint

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RC = ["congress", "chamber", "rollnumber"]
N_TOPICS = 12
MIN_ROLLCALLS = 100
MIN_MEMBER_VOTES = 50


def fit_positions(train: pd.DataFrame, profile: pd.DataFrame, chamber: str,
                  label: str) -> pd.DataFrame | None:
    sub = train[train["chamber"] == chamber]
    n_rc = sub.groupby(RC).ngroups
    if n_rc < MIN_ROLLCALLS:
        return None
    init = pd.Series(np.where(profile["party_code"] == 200, 0.5,
                              np.where(profile["party_code"] == 100, -0.5, 0.0)),
                     index=pd.Index(profile["icpsr"], name="icpsr")).groupby(level=0).last()
    model = IdealPoint(k=1).fit(sub, init_positions=init)
    pos = model.member_positions()[["icpsr", "dim1"]]
    nv = sub.groupby("icpsr").size().rename("n_votes")
    pos = pos.merge(nv, on="icpsr").merge(
        profile.drop_duplicates("icpsr", keep="last")[["icpsr", "nominate_dim1", "party_code"]],
        on="icpsr", how="left")

    ok = pos[pos["n_votes"] >= MIN_MEMBER_VOTES].copy()
    both = ok.dropna(subset=["nominate_dim1"])
    if len(both) > 10 and np.corrcoef(both["dim1"], both["nominate_dim1"])[0, 1] < 0:
        ok["dim1"] *= -1
    ok["z"] = (ok["dim1"] - ok["dim1"].mean()) / ok["dim1"].std()
    ok["chamber"], ok["topic"] = chamber, label
    print(f"  {chamber:6s} {label[:38]:38s} rollcalls {n_rc:5d}  members {len(ok):4d}",
          flush=True)
    return ok[["chamber", "topic", "icpsr", "z", "n_votes", "party_code"]]


def main():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    splits = pd.read_parquet(MOD / "splits" / "regimeA_seed42.parquet")
    df = votes.merge(splits, on=KEY, how="inner", validate="1:1")
    train = df[(df["split"] == "train") & (df["congress"] >= 108)].copy()

    links = pd.read_parquet(MOD / "rollcall_bills.parquet")
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "policy_area"]]
    links = links.merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
    train = train.merge(links[RC + ["policy_area", "bill_category"]], on=RC,
                        how="left", validate="m:1")

    mem = pd.read_parquet(MOD / "members.parquet")
    profile = mem.sort_values("congress").drop_duplicates(
        ["icpsr"], keep="last")[["icpsr", "bioname", "party_code", "state_abbrev",
                                 "nominate_dim1"]]

    top_areas = (train[train.bill_category == "legislation"]
                 .drop_duplicates(RC)["policy_area"].value_counts()
                 .head(N_TOPICS).index.tolist())
    print("topics:", top_areas)

    frames = []
    for chamber in ("House", "Senate"):
        overall = fit_positions(train, profile, chamber, "OVERALL")
        frames.append(overall)
        for area in top_areas:
            r = fit_positions(train[train["policy_area"] == area], profile, chamber, area)
            if r is not None:
                frames.append(r)
        if chamber == "Senate":
            r = fit_positions(train[train["bill_category"] == "nomination"],
                              profile, chamber, "Nominations")
            if r is not None:
                frames.append(r)

    out = pd.concat([f for f in frames if f is not None], ignore_index=True)
    overall_z = out[out.topic == "OVERALL"].set_index(["chamber", "icpsr"])["z"]
    out["overall_z"] = out.set_index(["chamber", "icpsr"]).index.map(overall_z)
    out["deviation"] = out["z"] - out["overall_z"]
    out = out.merge(profile[["icpsr", "bioname", "state_abbrev"]], on="icpsr", how="left")
    out.to_parquet(MOD / "results" / "issue_positions.parquet", index=False)

    topics = out[out.topic != "OVERALL"]
    stats = {
        "rows": int(len(out)),
        "topics_fit": sorted(topics.groupby(["chamber", "topic"]).ngroups
                             for _ in [0])[0],
        "mean_abs_corr_topic_vs_overall": round(float(
            topics.groupby(["chamber", "topic"]).apply(
                lambda g: g[["z", "overall_z"]].corr().iloc[0, 1],
                include_groups=False).mean()), 4),
        "corr_by_topic": {f"{ch}|{t}": round(float(c), 3) for (ch, t), c in
                          topics.groupby(["chamber", "topic"]).apply(
                              lambda g: g[["z", "overall_z"]].corr().iloc[0, 1],
                              include_groups=False).items()},
    }
    (MOD / "results" / "issue_positions_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    for ch, topic in [("House", "Agriculture and Food"),
                      ("House", "Armed Forces and National Security"),
                      ("House", "Immigration"),
                      ("Senate", "International Affairs")]:
        sub = topics[(topics.chamber == ch) & (topics.topic == topic)]
        if len(sub) == 0:
            continue
        print(f"\n=== {ch} / {topic}: biggest LEFTWARD deviations from overall position")
        cols = ["bioname", "party_code", "state_abbrev", "n_votes", "z", "overall_z", "deviation"]
        print(sub.nsmallest(8, "deviation")[cols].round(2).to_string(index=False))
        print(f"=== {ch} / {topic}: biggest RIGHTWARD deviations")
        print(sub.nlargest(8, "deviation")[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
