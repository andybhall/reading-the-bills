"""Freeze the prospective-forecast model.

Fits the current best architecture (emb2_mlp_mq_16d_tcal) on ALL yea/nay
votes in congresses 108-119 up to the data snapshot, then pickles the
fitted model plus metadata (snapshot date, git rev, sha256 of the pickle).

The frozen artifact is scored against rollcalls dated AFTER the snapshot
(14_score_prospective.py) and must never be refit. See
Notes/prospective_protocol.md for the rules.

Calibration/early-stop uses the temporal internal dev slice (last 5% of
observed rollcalls), consistent with the benchmarked configuration.
"""

import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models_texttower import TextTower

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
ART = MOD / "results" / "frozen"


def main():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes["congress"] >= 108]
    feat = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question", "bill_category",
         "bill_type", "bill_no"]]
    votes = votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                        how="left", validate="m:1")
    snapshot = str(votes["date"].max().date())
    print(f"fitting on {len(votes):,} votes through {snapshot}")

    model = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                      calibrate=True, mlp_head=True, es_mode="temporal",
                      name="frozen_prospective")
    model.fit(votes)

    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "prospective_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "none"
    meta = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "data_snapshot_max_vote_date": snapshot,
        "architecture": "emb2_mlp_mq_16d_tcal (k=16, MLP head, mq offset, "
                        "temporal calibration), MiniLM v2 rollcall-level text",
        "n_train_votes": int(len(votes)),
        "pickle_sha256": sha,
        "git_rev": git_rev,
        "epochs_run": model.epochs_run,
        "internal_dev_log_loss": round(model.es_log_loss, 4),
        "temperature": round(model.temperature, 4),
    }
    (ART / "prospective_model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
