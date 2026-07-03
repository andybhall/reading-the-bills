"""Build the clean analysis panel from raw Voteview files.

Input : Original Data/voteview/*.csv (never modified)
Output: Modified Data/votes.parquet     one row per member x rollcall
        Modified Data/rollcalls.parquet one row per rollcall
        Modified Data/members.parquet   one row per member-congress-chamber
        Modified Data/checks/build_checks.json  sanity-check record

Vote coding (Voteview cast_code):
  1,2,3 -> yea (vote=1);  4,5,6 -> nay (vote=0)
  7,8,9 -> present/abstain (vote=NaN, kept in panel, excluded from target)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Original Data" / "voteview"
OUT = ROOT / "Modified Data"
CONGRESSES = range(101, 120)

YEA_CODES = {1, 2, 3}
NAY_CODES = {4, 5, 6}
ABSTAIN_CODES = {7, 8, 9}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "checks").mkdir(exist_ok=True)
    checks: dict = {}

    # --- load votes -------------------------------------------------------
    frames, file_rows = [], {}
    for cong in CONGRESSES:
        for chamber in ("H", "S"):
            f = RAW / f"{chamber}{cong}_votes.csv"
            df = pd.read_csv(f, usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"])
            file_rows[f.name] = len(df)
            frames.append(df)
    votes = pd.concat(frames, ignore_index=True)
    checks["raw_vote_rows"] = int(len(votes))
    assert len(votes) == sum(file_rows.values()), "concat lost rows"

    # presidents (icpsr >= 99000) appear in votes files as announced positions;
    # they are not voting members, so drop them from the panel
    n_pres = int((votes["icpsr"] >= 99000).sum())
    votes = votes[votes["icpsr"] < 99000].copy()
    checks["president_rows_dropped"] = n_pres

    dupes = votes.duplicated(["congress", "chamber", "rollnumber", "icpsr"]).sum()
    checks["duplicate_member_rollcall_rows"] = int(dupes)
    assert dupes == 0, f"{dupes} duplicate member-rollcall rows"

    bad_codes = ~votes["cast_code"].isin(YEA_CODES | NAY_CODES | ABSTAIN_CODES | {0})
    checks["unexpected_cast_codes"] = int(bad_codes.sum())
    assert bad_codes.sum() == 0, "unexpected cast_code values"
    votes = votes[votes["cast_code"] != 0].copy()  # 0 = not a member at vote time

    votes["vote"] = np.where(
        votes["cast_code"].isin(YEA_CODES), 1.0,
        np.where(votes["cast_code"].isin(NAY_CODES), 0.0, np.nan),
    )
    checks["cast_code_distribution"] = votes["cast_code"].value_counts().sort_index().to_dict()

    # --- rollcalls --------------------------------------------------------
    rc = pd.read_csv(RAW / "HSall_rollcalls.csv", low_memory=False)
    rc = rc[rc["congress"].isin(CONGRESSES)].copy()
    rc["date"] = pd.to_datetime(rc["date"])
    keep_rc = ["congress", "chamber", "rollnumber", "date", "yea_count", "nay_count",
               "bill_number", "vote_question", "vote_result", "vote_desc", "dtl_desc",
               "nominate_mid_1", "nominate_mid_2", "nominate_spread_1", "nominate_spread_2"]
    rc = rc[keep_rc]
    assert not rc.duplicated(["congress", "chamber", "rollnumber"]).any()

    # --- members ----------------------------------------------------------
    mem = pd.read_csv(RAW / "HSall_members.csv", low_memory=False)
    mem = mem[(mem["congress"].isin(CONGRESSES)) & (mem["chamber"] != "President")].copy()
    keep_mem = ["congress", "chamber", "icpsr", "state_abbrev", "district_code",
                "party_code", "bioname", "bioguide_id", "born",
                "nominate_dim1", "nominate_dim2", "nokken_poole_dim1", "nokken_poole_dim2"]
    mem = mem[keep_mem]
    # a handful of members appear twice in a congress (e.g. party switch /
    # re-coding); keep the last record per member-congress-chamber
    n_mem_dupes = int(mem.duplicated(["congress", "chamber", "icpsr"]).sum())
    checks["member_dupes_collapsed"] = n_mem_dupes
    mem = mem.drop_duplicates(["congress", "chamber", "icpsr"], keep="last")

    # --- merges, with match verification ----------------------------------
    n0 = len(votes)
    votes = votes.merge(rc[["congress", "chamber", "rollnumber", "date"]],
                        on=["congress", "chamber", "rollnumber"], how="left", validate="m:1")
    checks["votes_missing_rollcall_info"] = int(votes["date"].isna().sum())

    votes = votes.merge(mem[["congress", "chamber", "icpsr", "party_code", "state_abbrev",
                             "nominate_dim1", "nominate_dim2"]],
                        on=["congress", "chamber", "icpsr"], how="left", validate="m:1")
    checks["votes_missing_member_info"] = int(votes["party_code"].isna().sum())
    assert len(votes) == n0, "merge changed row count"

    # --- sanity checks ----------------------------------------------------
    votes_yn = votes.dropna(subset=["vote"])
    checks["panel_rows"] = int(len(votes))
    checks["yea_nay_rows"] = int(len(votes_yn))
    checks["overall_yea_share"] = round(float(votes_yn["vote"].mean()), 4)
    by_cc = votes.groupby(["congress", "chamber"])["icpsr"].nunique()
    checks["members_per_chamber_congress"] = {f"{c}_{ch}": int(n) for (c, ch), n in by_cc.items()}
    house_sizes = by_cc.xs("House", level="chamber")
    senate_sizes = by_cc.xs("Senate", level="chamber")
    # unique members per congress exceeds seat counts (435/100) because of
    # mid-term replacements; e.g. 111 senators served in the 111th Congress
    assert house_sizes.between(430, 465).all(), f"odd House sizes: {house_sizes.to_dict()}"
    assert senate_sizes.between(99, 115).all(), f"odd Senate sizes: {senate_sizes.to_dict()}"
    checks["rollcalls_per_congress"] = {int(c): int(n) for c, n in
                                        rc.groupby("congress").size().items()}
    checks["date_range"] = [str(votes["date"].min().date()), str(votes["date"].max().date())]

    # recorded yea/nay counts in the rollcalls file should match our tallies
    tally = votes_yn.groupby(["congress", "chamber", "rollnumber"])["vote"].agg(["sum", "count"])
    tally = tally.join(rc.set_index(["congress", "chamber", "rollnumber"])[["yea_count", "nay_count"]])
    yea_match = (tally["sum"] == tally["yea_count"]).mean()
    checks["rollcall_yea_tally_match_rate"] = round(float(yea_match), 4)

    # --- write ------------------------------------------------------------
    votes.to_parquet(OUT / "votes.parquet", index=False)
    rc.to_parquet(OUT / "rollcalls.parquet", index=False)
    mem.to_parquet(OUT / "members.parquet", index=False)
    (OUT / "checks" / "build_checks.json").write_text(json.dumps(checks, indent=2))

    print(json.dumps({k: v for k, v in checks.items()
                      if k not in ("members_per_chamber_congress", "rollcalls_per_congress",
                                   "cast_code_distribution")}, indent=2))
    print(f"\nWrote {len(votes):,} panel rows, {len(rc):,} rollcalls, {len(mem):,} member-congress records")


if __name__ == "__main__":
    main()
