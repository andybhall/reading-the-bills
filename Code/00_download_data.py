"""Download raw roll-call data from Voteview into Original Data/voteview.

Source: https://voteview.com/data (Lewis et al., Voteview: Congressional
Roll-Call Votes Database). We download:
  - HSall_members.csv   : one row per member-congress (includes NOMINATE scores)
  - HSall_rollcalls.csv : one row per roll call (date, result, vote question)
  - HSall_parties.csv   : party summary per congress
  - {H,S}{congress}_votes.csv : individual member votes, congresses 101-119

Scope decision (flagged in Notes/decisions.md): congresses 101-119
(1989-present). Extending further back is a one-line change to CONGRESSES.

Writes Original Data/voteview/manifest.json recording URL, access time,
bytes, and sha256 for every file. Skips files already present with
matching size unless --force.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://voteview.com/static/data/out"
OUT = Path(__file__).resolve().parent.parent / "Original Data" / "voteview"
CONGRESSES = range(101, 120)  # 101st (1989) through 119th (current, partial)

FILES = [
    (f"{BASE}/members/HSall_members.csv", "HSall_members.csv"),
    (f"{BASE}/rollcalls/HSall_rollcalls.csv", "HSall_rollcalls.csv"),
    (f"{BASE}/parties/HSall_parties.csv", "HSall_parties.csv"),
]
for cong in CONGRESSES:
    for chamber in ("H", "S"):
        FILES.append((f"{BASE}/votes/{chamber}{cong}_votes.csv", f"{chamber}{cong}_votes.csv"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(force: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    failures = []
    for i, (url, name) in enumerate(FILES, 1):
        dest = OUT / name
        head = requests.head(url, timeout=30)
        if head.status_code != 200:
            failures.append((name, f"HTTP {head.status_code} on HEAD"))
            print(f"[{i}/{len(FILES)}] {name}: MISSING upstream (HTTP {head.status_code})")
            continue
        remote_bytes = int(head.headers.get("Content-Length", -1))

        if dest.exists() and not force and dest.stat().st_size == remote_bytes:
            print(f"[{i}/{len(FILES)}] {name}: already present ({remote_bytes:,} bytes), skipping")
        else:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"[{i}/{len(FILES)}] {name}: downloaded {len(r.content):,} bytes")
            time.sleep(0.5)  # be polite to voteview.com

        local_bytes = dest.stat().st_size
        if remote_bytes != -1 and local_bytes != remote_bytes:
            failures.append((name, f"size mismatch: local {local_bytes} vs remote {remote_bytes}"))
            continue
        manifest[name] = {
            "url": url,
            "accessed_utc": datetime.now(timezone.utc).isoformat(),
            "bytes": local_bytes,
            "sha256": sha256(dest),
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nManifest written: {manifest_path} ({len(manifest)} files)")
    if failures:
        print("FAILURES:")
        for name, why in failures:
            print(f"  {name}: {why}")
        sys.exit(1)
    print("All files downloaded and verified against remote sizes.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
