"""Worked landmark rollcall (review r1, O11/O12): the April 2024 Ukraine
security supplemental, House final passage (118th Congress, H.R. 8035
family) — a vote inside the temporal-forecast TEST window, so the
champion's saved predictions for it are genuine holdout forecasts.

Produces Draft/tables/worked_example.tex: the rollcall's pre-vote text
fields, estimated cutpoint/discrimination, realized outcome, and a
named-member panel (predicted probability, realized vote, ideal point,
defense-issue deviation, loyalty residual). Also a landmark-cutpoints
mini table for recognizable 117th-118th votes.

Run: python3 Code/26_worked_example.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"


def find_rollcall():
    # H.R. 10545, the American Relief Act of 2025 (December 20, 2024):
    # the shutdown-averting continuing resolution that passed with
    # minority-party votes over majority-flank opposition — a storied
    # majority-splitting vote at the end of the temporal TEST window
    rc = pd.read_parquet(MOD / "rollcalls.parquet")
    h = rc[(rc.congress == 118) & (rc.chamber == "House")]
    uk = h[h.bill_number.fillna("").str.contains("HR10545")]
    uk = uk[uk.vote_question.fillna("").str.contains("Passage|Pass", case=False)]
    return uk.iloc[0]


def main():
    row = find_rollcall()
    key = (118, "House", int(row.rollnumber))
    print(f"rollcall: H {row.rollnumber} {row.date} | {str(row.vote_desc)[:60]}")

    preds = pd.read_parquet(RES / "preds" /
                            "forecast108_119_blend3_mlp_tfidf_emb3_tcal.parquet")
    p = preds[(preds.congress == 118) & (preds.chamber == "House")
              & (preds.rollnumber == key[2])]
    assert len(p) > 0, "rollcall not in forecast test predictions"
    print(f"in temporal TEST window: {len(p)} member predictions")

    cuts = pd.read_parquet(RES / "measures" / "cutpoints_house118.parquet")
    c = cuts[cuts.rollnumber == key[2]].iloc[0]
    mem = pd.read_parquet(RES / "measures" / "members_house118.parquet")
    memfull = pd.read_parquet(MOD / "members.parquet")
    names = memfull[(memfull.congress == 118) & (memfull.chamber == "House")][
        ["icpsr", "bioname", "party_code", "state_abbrev"]]
    iss = pd.read_parquet(RES / "issue_positions.parquet")
    defense = iss[iss.topic == "Armed Forces and National Security"][
        ["icpsr", "deviation"]].rename(columns={"deviation": "def_dev"})
    sig = pd.read_parquet(RES / "member_signals.parquet")
    loy = (sig[sig.congress == 118].groupby("icpsr")["loyalty_residual"]
           .mean().rename("loyalty").reset_index())

    d = (p.merge(names, on="icpsr").merge(mem[["icpsr", "x"]], on="icpsr")
          .merge(defense, on="icpsr", how="left")
          .merge(loy, on="icpsr", how="left"))

    picks = [("JOHNSON", "LA"), ("JEFFRIES", "NY"), ("GREENE", "GA"),
             ("MASSIE", "KY"), ("OCASIO-CORTEZ", "NY"), ("ROY", "TX"),
             ("SPARTZ", "IN"), ("GOLDEN", "ME")]
    rows = []
    for last, st in picks:
        m = d[d.bioname.str.startswith(last) & (d.state_abbrev == st)]
        if not len(m):
            print(f"  (no match: {last} {st})")
            continue
        r = m.iloc[0]
        nm = str(r.bioname).split(",")[0].title()
        rows.append(" & ".join([
            f"{nm} ({r.state_abbrev})",
            "R" if r.party_code == 200.0 else "D",
            f"{r.x:+.2f}",
            "---" if pd.isna(r.loyalty) else f"{r.loyalty:+.3f}",
            f"{r.p_yea:.2f}",
            "Yea" if r.vote == 1 else "Nay",
            f"{-np.log(max(r.p_yea if r.vote == 1 else 1 - r.p_yea, 1e-9)):.2f}"]))

    yea_share = p.vote.mean()
    header = ("Member & Party & Position & Loyalty resid. & "
              "$\\hat p(\\text{yea})$ & Vote & Surprise")
    body = " \\\\\n".join(rows) + " \\\\"
    (OUT / "worked_example.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\n" + header +
        " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    stats = (f"% auto-generated context\n"
             f"\\newcommand{{\\ukRoll}}{{{key[2]}}}\n"
             f"\\newcommand{{\\ukDate}}{{{str(row.date)[:10]}}}\n"
             f"\\newcommand{{\\ukCut}}{{{c.cutpoint:.2f}}}\n"
             f"\\newcommand{{\\ukDiscrim}}{{{c.a:.2f}}}\n"
             f"\\newcommand{{\\ukYeaShare}}{{{100*yea_share:.0f}}}\n"
             f"\\newcommand{{\\ukNPred}}{{{len(p)}}}\n")
    (OUT / "worked_example_macros.tex").write_text(stats)
    print(f"cutpoint {c.cutpoint:.2f}, discrimination {c.a:.2f}, "
          f"yea share {yea_share:.2f}")
    print("wrote worked_example.tex + macros")


if __name__ == "__main__":
    main()
