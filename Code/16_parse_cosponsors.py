"""Extract cosponsors (with party and sponsorship DATE) from cached
BILLSTATUS XML. Dates matter: cosponsors accrue over a bill's life, so any
feature must count only cosponsorships before the rollcall date.

Output: Modified Data/cosponsors.parquet
        (congress, bill_type, bill_no, bioguide, party, sponsorship_date)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Original Data" / "govinfo" / "billstatus"
MOD = ROOT / "Modified Data"


def main():
    files = sorted(RAW.glob("BILLSTATUS-*.xml"))
    rows, bad = [], 0
    for i, f in enumerate(files, 1):
        try:
            bill = ET.parse(f).getroot().find("bill")
            if bill is None:
                bad += 1
                continue
        except ET.ParseError:
            bad += 1
            continue
        congress = int(bill.findtext("congress"))
        btype = (bill.findtext("type") or bill.findtext("billType") or "").lower()
        bno = int(bill.findtext("number") or bill.findtext("billNumber"))
        for item in bill.findall("cosponsors/item"):
            rows.append({
                "congress": congress, "bill_type": btype, "bill_no": bno,
                "bioguide": item.findtext("bioguideId", ""),
                "party": item.findtext("party", ""),
                "sponsorship_date": item.findtext("sponsorshipDate", ""),
            })
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(MOD / "cosponsors.parquet", index=False)
    report = {
        "bills_scanned": len(files), "parse_failures": bad,
        "cosponsor_rows": int(len(df)),
        "bills_with_cosponsors": int(df.groupby(["congress", "bill_type", "bill_no"]).ngroups),
        "party_counts": df["party"].value_counts().head(5).to_dict(),
        "missing_dates": int((df["sponsorship_date"] == "").sum()),
    }
    (MOD / "checks" / "cosponsors_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
