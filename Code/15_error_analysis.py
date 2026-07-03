"""Error analysis: where does the best forecast model still fail?

Reads saved per-vote test predictions (run_benchmark --save-preds) and
breaks log loss / accuracy down by the dimensions that decide what to
build next: question bucket, text source (pre-vote summary vs title vs
desc-only), summary length, chamber, party, member tenure, vote closeness.

Usage: python3 Code/15_error_analysis.py [model_name]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from harness import metrics
from models_forecast import question_bucket

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
SPLIT = "forecast108_119"


def breakdown(df: pd.DataFrame, col: str, min_n: int = 2000) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(col, observed=True):
        if len(g) < min_n:
            continue
        m = metrics(g["vote"].to_numpy(), g["p_yea"].to_numpy())
        rows.append({col: val, "n": m["n"], "log_loss": round(m["log_loss"], 4),
                     "accuracy": round(m["accuracy"], 4),
                     "share_of_total_loss": round(
                         float(-np.sum(g["vote"] * np.log(np.clip(g["p_yea"], 1e-7, 1))
                               + (1 - g["vote"]) * np.log(np.clip(1 - g["p_yea"], 1e-7, 1)))
                               / total_loss), 4)})
    return pd.DataFrame(rows).sort_values("log_loss", ascending=False)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "emb2_mlp_mq_16d_tcal"
    preds = pd.read_parquet(MOD / "results" / "preds" / f"{SPLIT}_{model}.parquet")

    links = pd.read_parquet(MOD / "rollcall_bills.parquet")
    bills = pd.read_parquet(MOD / "bills.parquet")[["congress", "bill_type", "bill_no",
                                                    "summaries", "title"]]
    links = links.merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
    emb_src = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet",
                              columns=["congress", "chamber", "rollnumber", "text_source"])
    rckey = ["congress", "chamber", "rollnumber"]
    df = (preds.merge(links[rckey + ["vote_question", "bill_category", "summaries"]],
                      on=rckey, how="left")
          .merge(emb_src, on=rckey, how="left"))
    votes = pd.read_parquet(MOD / "votes.parquet",
                            columns=rckey + ["icpsr", "party_code"]).drop_duplicates()
    df = df.merge(votes, on=rckey + ["icpsr"], how="left")

    df["qbucket"] = question_bucket(df["vote_question"])
    df["summary_len"] = df["summaries"].fillna("[]").str.len()
    df["len_bin"] = pd.cut(df["summary_len"], [0, 3, 2000, 10000, np.inf],
                           labels=["no_summary", "short", "medium", "long"])
    # tenure: congresses served before this one (from members table)
    mem = pd.read_parquet(MOD / "members.parquet")[["icpsr", "congress"]]
    first = mem.groupby("icpsr")["congress"].min().rename("first_congress")
    df = df.merge(first, on="icpsr", how="left")
    df["tenure"] = pd.cut(df["congress"] - df["first_congress"], [-1, 0, 2, 100],
                          labels=["freshman", "junior", "senior"])
    # closeness from actual outcome (descriptive stratification only)
    yea_share = df.groupby(rckey)["vote"].transform("mean")
    df["closeness"] = pd.cut(np.minimum(yea_share, 1 - yea_share),
                             [-0.01, 0.05, 0.35, 0.5],
                             labels=["lopsided", "middling", "contested"])

    global total_loss
    p = np.clip(df["p_yea"], 1e-7, 1 - 1e-7)
    total_loss = float(-np.sum(df["vote"] * np.log(p) + (1 - df["vote"]) * np.log(1 - p)))

    out = {"model": model, "n": len(df),
           "overall": {k: round(v, 4) for k, v in
                       metrics(df["vote"].to_numpy(), df["p_yea"].to_numpy()).items()}}
    print(json.dumps(out["overall"], indent=2))
    report_lines = [json.dumps(out, indent=2)]
    for col in ["qbucket", "text_source", "len_bin", "bill_category", "closeness",
                "chamber", "party_code", "tenure"]:
        b = breakdown(df, col)
        report_lines.append(f"\n## by {col}\n{b.to_string(index=False)}")
        print(f"\n## by {col}")
        print(b.to_string(index=False))

    (MOD / "results" / f"error_analysis_{model}.txt").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
