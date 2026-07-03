"""Parse cached BILLSTATUS XML into Modified Data/bills.parquet.

One row per bill with: title, policy area, legislative subjects, sponsor
(bioguide id, party, state), introduced date, and ALL summary versions as a
list of (version_code, action_date, text) — multiple versions are kept so
downstream feature builders can enforce leakage discipline (only use the
latest summary whose action_date precedes the rollcall date).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Original Data" / "govinfo" / "billstatus"
MOD = ROOT / "Modified Data"


def parse_one(path: Path) -> dict | None:
    try:
        bill = ET.parse(path).getroot().find("bill")
        if bill is None:
            return None
    except ET.ParseError:
        return None

    sponsor = bill.find("sponsors/item")
    summaries = []
    # current schema uses summaries/summary; legacy v1 summaries/billSummaries/item
    for s in (bill.findall("summaries/summary")
              or bill.findall("summaries/billSummaries/item")):
        summaries.append({
            "version_code": s.findtext("versionCode", ""),
            "action_date": s.findtext("actionDate", ""),
            "text": s.findtext("text", "") or "",
        })
    subjects = [i.findtext("name", "") for i in
                bill.findall("subjects/legislativeSubjects/item")
                or bill.findall("subjects/billSubjects/legislativeSubjects/item")]

    return {
        "congress": int(bill.findtext("congress")),
        "bill_type": (bill.findtext("type") or bill.findtext("billType") or "").lower(),
        "bill_no": int(bill.findtext("number") or bill.findtext("billNumber")),
        "title": bill.findtext("title", ""),
        "policy_area": bill.findtext("policyArea/name", ""),
        "subjects": subjects,
        "sponsor_bioguide": sponsor.findtext("bioguideId", "") if sponsor is not None else "",
        "sponsor_party": sponsor.findtext("party", "") if sponsor is not None else "",
        "sponsor_state": sponsor.findtext("state", "") if sponsor is not None else "",
        "introduced_date": bill.findtext("introducedDate", ""),
        "summaries": summaries,
    }


def main():
    files = sorted(RAW.glob("BILLSTATUS-*.xml"))
    print(f"parsing {len(files)} XML files")
    rows, bad = [], []
    for i, f in enumerate(files, 1):
        rec = parse_one(f)
        (rows if rec else bad).append(rec or f.name)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    df["summaries"] = df["summaries"].apply(json.dumps)
    df["subjects"] = df["subjects"].apply(json.dumps)

    checks = {
        "parsed": len(rows), "failed": len(bad), "failed_files": bad[:20],
        "with_title": int((df.title.str.len() > 0).sum()),
        "with_policy_area": int((df.policy_area.str.len() > 0).sum()),
        "with_sponsor": int((df.sponsor_bioguide.str.len() > 0).sum()),
        "with_any_summary": int((df.summaries != "[]").sum()),
        "policy_area_top10": df.policy_area.value_counts().head(10).to_dict(),
    }
    df.to_parquet(MOD / "bills.parquet", index=False)
    (MOD / "checks" / "bills_parse_report.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
