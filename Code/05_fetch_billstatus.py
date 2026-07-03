"""Fetch GovInfo BILLSTATUS XML for every bill linked to a rollcall.

Source: https://www.govinfo.gov/bulkdata/BILLSTATUS/{congress}/{type}/
            BILLSTATUS-{congress}{type}{number}.xml
No API key required. Coverage: 108th congress onward.

Raw XML is cached in Original Data/govinfo/billstatus/ (never modified);
reruns skip cached files. Fetch log with failures goes to
Original Data/govinfo/fetch_log.json.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Original Data" / "govinfo" / "billstatus"
MOD = ROOT / "Modified Data"
SLEEP = 0.15
RETRIES = 3


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")
    bills = (links[links.bill_category == "legislation"]
             [["congress", "bill_type", "bill_no"]].drop_duplicates()
             .sort_values(["congress", "bill_type", "bill_no"]))
    print(f"{len(bills)} unique bills to fetch")

    session = requests.Session()
    session.headers["User-Agent"] = "academic-research-rollcall-prediction (andrewbenjaminhall@gmail.com)"
    ok = cached = failed = 0
    failures = []
    for i, row in enumerate(bills.itertuples(), 1):
        name = f"BILLSTATUS-{row.congress}{row.bill_type}{int(row.bill_no)}.xml"
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            cached += 1
            continue
        url = (f"https://www.govinfo.gov/bulkdata/BILLSTATUS/{row.congress}/"
               f"{row.bill_type}/{name}")
        for attempt in range(RETRIES):
            try:
                r = session.get(url, timeout=60)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    ok += 1
                    break
                if r.status_code == 404:
                    failures.append({"bill": name, "error": "404"})
                    failed += 1
                    break
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001 - retry then log any fetch error
                if attempt == RETRIES - 1:
                    failures.append({"bill": name, "error": str(e)})
                    failed += 1
                else:
                    time.sleep(2 ** attempt)
        time.sleep(SLEEP)
        if i % 250 == 0:
            print(f"  {i}/{len(bills)} (new {ok}, cached {cached}, failed {failed})", flush=True)

    log = {"fetched_utc": datetime.now(timezone.utc).isoformat(),
           "total": len(bills), "new": ok, "cached": cached, "failed": failed,
           "failures": failures}
    (RAW.parent / "fetch_log.json").write_text(json.dumps(log, indent=2))
    print(f"done: new {ok}, cached {cached}, failed {failed} / {len(bills)}")


if __name__ == "__main__":
    main()
