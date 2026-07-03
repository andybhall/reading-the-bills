"""Parse amendments out of the cached GovInfo BILLSTATUS XMLs.

For every <amendment> element in every cached bill: amendment identity,
purpose text, sponsor party, chamber, dates, the amended bill, and any
House roll-call numbers embedded in action text ("(Roll no. N)"). No
network access — the cache in Original Data/govinfo/billstatus is the
only input, so this is fully reproducible offline.

Output: Modified Data/amendments.parquet
        (one row per amendment x found roll number; roll_no = -1 when the
        amendment block contains no roll reference)
"""

import json
import re
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "Original Data" / "govinfo" / "billstatus"
OUT = ROOT / "Modified Data" / "amendments.parquet"

# House style: "(Roll no. 350)"; Senate style: "Record Vote Number: 123"
ROLL_PAT = re.compile(r"(?:Roll (?:no\.|Call) ?|Record Vote (?:Number|No\.?):? ?)(\d+)", re.I)
# identity from the filename — authoritative and uniform across BILLSTATUS
# schema versions (older files name the bill fields differently)
FNAME_PAT = re.compile(r"BILLSTATUS-(\d+)([a-z]+)(\d+)\.xml$")


def parse_file(path: Path) -> list[dict]:
    m = FNAME_PAT.search(path.name)
    if not m:
        return []
    congress, bill_type, bill_no = int(m.group(1)), m.group(2), int(m.group(3))
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError:
        return []
    root = tree.getroot()
    rows = []
    for am in root.iter("amendment"):
        # amendments can nest (amendment-to-amendment); each still carries
        # its own identity and purpose, so parse every element
        purpose = am.findtext("purpose") or am.findtext("description") or ""
        sponsor = am.find("sponsors/item")
        row = {
            "congress": congress,
            "bill_type": bill_type,
            "bill_no": bill_no,
            "amdt_type": am.findtext("type") or "",
            "amdt_number": int(am.findtext("number") or -1),
            "purpose": purpose.strip(),
            "sponsor_party": (sponsor.findtext("party") if sponsor is not None else "") or "",
            "chamber": am.findtext("chamber") or "",
            "proposed_date": am.findtext("proposedDate") or am.findtext("submittedDate") or "",
        }
        # walk in document order pairing each action <text> with the nearest
        # preceding <actionDate>. The date is REQUIRED downstream: clerk roll
        # numbers reset each session while Voteview rollnumbers are
        # congress-continuous, so a (roll_no, congress) join alone attaches
        # the wrong amendment ~40% of the time (diagnosed 2026-07-03 —
        # mismatch gaps clustered at ~367 days, the session offset).
        rolls = set()
        current_date = ""
        for t in am.iter():
            if t.tag == "actionDate" and t.text:
                current_date = t.text.strip()[:10]
            elif t.tag == "text" and t.text:
                for m in ROLL_PAT.finditer(t.text):
                    rolls.add((int(m.group(1)), current_date))
        if rolls:
            for r, d in sorted(rolls):
                rows.append(row | {"roll_no": r, "action_date": d})
        else:
            rows.append(row | {"roll_no": -1, "action_date": ""})
    return rows


def main():
    files = sorted(CACHE.glob("BILLSTATUS-*.xml"))
    print(f"parsing {len(files)} XML files for amendments")
    rows = []
    for i, f in enumerate(files, 1):
        rows.extend(parse_file(f))
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} ({len(rows)} amendment rows)")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    stats = {
        "files": len(files),
        "amendment_rows": int(len(df)),
        "unique_amendments": int(df.drop_duplicates(
            ["congress", "amdt_type", "amdt_number"]).shape[0]),
        "with_purpose": int((df["purpose"] != "").sum()),
        "with_roll_no": int((df["roll_no"] >= 0).sum()),
        "by_amdt_type": df["amdt_type"].value_counts().to_dict(),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
