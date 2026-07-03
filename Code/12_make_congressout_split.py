"""Whole-congress-out split: how well does the model forecast a congress it
has never seen?

train = congresses 108-116, val = 117th, test = 118th. The 119th is
excluded entirely — it is the prospective holdout (see
Notes/prospective_protocol.md) and must never appear in any development
split. Unlike the within-congress forecast split, members' votes from the
evaluation congress are NOT in training, so freshmen are fully cold-start
and returning members may have drifted.

Output: Modified Data/splits/congressout118.parquet (+ stats), same cell
format as other splits so run_benchmark.py works unchanged.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Modified Data" / "splits"
KEY = ["congress", "chamber", "rollnumber", "icpsr"]
NAME = "congressout118"


def main():
    votes = pd.read_parquet(ROOT / "Modified Data" / "votes.parquet",
                            columns=KEY + ["vote"]).dropna(subset=["vote"])
    cells = votes[(votes.congress >= 108) & (votes.congress <= 118)][KEY].copy()
    cells["split"] = np.select(
        [cells.congress <= 116, cells.congress == 117], ["train", "val"], "test")

    train_members = set(cells.loc[cells.split == "train", "icpsr"])
    stats = {
        "design": "train 108-116, val 117, test 118; 119 reserved for prospective",
        "counts": cells["split"].value_counts().to_dict(),
        "test_members_unseen_in_train": int(
            (~cells.loc[cells.split == "test", "icpsr"].isin(train_members))
            .groupby(cells.loc[cells.split == "test", "icpsr"]).first().sum()),
        "test_members_total": int(cells.loc[cells.split == "test", "icpsr"].nunique()),
    }

    cells = cells.sort_values(KEY).reset_index(drop=True)
    cells.to_parquet(OUT / f"{NAME}.parquet", index=False)
    stats["content_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(cells, index=False).to_numpy().tobytes()).hexdigest()
    (OUT / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
