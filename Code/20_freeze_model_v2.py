"""Freeze the v2 prospective-forecast artifact (S5 winner).

Per Notes/prospective_protocol.md rule 4: a new frozen model gets a NEW
artifact + meta file and its own prospective clock; the v1 artifact keeps
being scored and is never replaced. Fits the S5 winner on ALL yea/nay
votes in congresses 108-119 up to the data snapshot.

Usage: python3 Code/20_freeze_model_v2.py [--model blend3_mlp_tfidf_emb3_tcal]
"""

import argparse
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models_texttower import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
ART = MOD / "results" / "frozen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="blend3_mlp_tfidf_emb3_tcal")
    args = ap.parse_args()

    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes["congress"] >= 108]
    feat = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question", "bill_category",
         "bill_type", "bill_no"]]
    votes = votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                        how="left", validate="m:1")
    snapshot = str(votes["date"].max().date())
    print(f"fitting {args.model} on {len(votes):,} votes through {snapshot}")

    model = REGISTRY[args.model]()
    model.fit(votes)

    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "prospective_model_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "none"
    meta = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "data_snapshot_max_vote_date": snapshot,
        "architecture": args.model,
        "n_train_votes": int(len(votes)),
        "pickle_sha256": sha,
        "git_rev": git_rev,
        "blend_weights": getattr(model, "blend_weights", None),
    }
    (ART / "prospective_model_v2_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
