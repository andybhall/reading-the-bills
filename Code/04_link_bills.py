"""Link rollcalls to bills: parse Voteview bill_number into (bill_type, number).

Scope: congresses 108-119 (GovInfo BILLSTATUS bulk data begins at the 108th).
Categories:
  - legislation (HR, S, HRES, SRES, HJRES, SJRES, HCONRES, SCONRES):
      fetchable from GovInfo BILLSTATUS
  - nomination (PN): no bill text; kept as a typed category
  - treaty (TREATYDOC), other/missing: typed, no text

Output: Modified Data/rollcall_bills.parquet (one row per rollcall with
parsed linkage) and a coverage report in Modified Data/checks/.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
FORECAST_CONGRESSES = range(108, 120)

GOVINFO_TYPES = {"HR": "hr", "S": "s", "HRES": "hres", "SRES": "sres",
                 "HJRES": "hjres", "SJRES": "sjres", "HCONRES": "hconres",
                 "SCONRES": "sconres"}


def parse_bill_number(raw):
    if pd.isna(raw):
        return pd.Series({"bill_type": None, "bill_no": None, "bill_category": "none"})
    m = re.fullmatch(r"([A-Z]+)\s*(\d+)", str(raw).strip().upper().replace(".", "").replace(" ", ""))
    if not m:
        return pd.Series({"bill_type": None, "bill_no": None, "bill_category": "unparseable"})
    prefix, num = m.group(1), int(m.group(2))
    if prefix in GOVINFO_TYPES:
        return pd.Series({"bill_type": GOVINFO_TYPES[prefix], "bill_no": num,
                          "bill_category": "legislation"})
    if prefix == "PN":
        return pd.Series({"bill_type": "pn", "bill_no": num, "bill_category": "nomination"})
    if prefix in ("TREATYDOC", "TREDOC", "TD"):
        return pd.Series({"bill_type": "treaty", "bill_no": num, "bill_category": "treaty"})
    return pd.Series({"bill_type": prefix.lower(), "bill_no": num, "bill_category": "other"})


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")
    rc = rc[rc["congress"].isin(FORECAST_CONGRESSES)].copy()
    parsed = rc["bill_number"].apply(parse_bill_number)
    parsed["bill_no"] = parsed["bill_no"].astype("Int64")  # keep integer despite NaNs
    out = pd.concat([rc[["congress", "chamber", "rollnumber", "date", "bill_number",
                         "vote_question"]], parsed], axis=1)

    report = {
        "n_rollcalls": int(len(out)),
        "by_category": out["bill_category"].value_counts().to_dict(),
        "category_share": out["bill_category"].value_counts(normalize=True).round(4).to_dict(),
        "unparseable_examples": out[out.bill_category == "unparseable"]["bill_number"]
                                .dropna().unique()[:20].tolist(),
        "unique_legislation_bills": int(out[out.bill_category == "legislation"]
                                        .groupby(["congress", "bill_type", "bill_no"]).ngroups),
    }

    out.to_parquet(MOD / "rollcall_bills.parquet", index=False)
    (MOD / "checks" / "bill_linkage_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
