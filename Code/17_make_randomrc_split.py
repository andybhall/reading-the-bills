"""Create the random-rollcall holdout split, congresses 108-119.

ENTIRE rollcalls are assigned to train/val/test (80/10/10) by hashing the
rollcall key with a fixed seed, within each congress-chamber. Every vote on
a rollcall inherits its assignment. This is the evaluation regime of the
2011-2016 text-based vote-prediction literature (Gerrish-Blei 2011/2012,
Kraft-Jain-Rush 2016): held-out bills are interleaved in time with training
bills, so the model sees the surrounding legislative context — unlike the
temporal forecast split, where eval rollcalls all lie in the future.

Comparing the same models on this split vs forecast108_119 vs congressout118
vs the prospective ledger is Paper A's evaluation-regime audit.

Same congresses (108-119) and format as forecast108_119, so run_benchmark.py
and all registered models work unchanged:
Output: Modified Data/splits/randomrc108_119.parquet + stats.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
OUT = MOD / "splits"
KEY = ["congress", "chamber", "rollnumber", "icpsr"]
NAME = "randomrc108_119"
CONGRESSES = range(108, 120)
SEED = 42


def hash_unit(rc: pd.DataFrame, seed: int) -> np.ndarray:
    """Uniform [0,1) per rollcall from a stable hash of the rollcall key."""
    keys = (rc["congress"].astype(str) + "|" + rc["chamber"] + "|"
            + rc["rollnumber"].astype(str) + f"|rcseed{seed}")
    return keys.map(lambda k: int(hashlib.sha256(k.encode()).hexdigest()[:12], 16) / 16**12).to_numpy()


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")
    rc = rc[rc["congress"].isin(CONGRESSES)].copy()
    u = hash_unit(rc, SEED)
    rc["split"] = np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test"))

    votes = pd.read_parquet(MOD / "votes.parquet", columns=KEY + ["vote"])
    votes = votes.dropna(subset=["vote"])
    cells = votes.merge(rc[["congress", "chamber", "rollnumber", "split"]],
                        on=["congress", "chamber", "rollnumber"], how="inner")[KEY + ["split"]]

    stats = {
        "seed": SEED,
        "n": int(len(cells)),
        "shares": cells["split"].value_counts(normalize=True).round(4).to_dict(),
        "counts": cells["split"].value_counts().to_dict(),
        "rollcalls_per_split": rc["split"].value_counts().to_dict(),
        # per-congress-chamber share balance: hashing has no within-group
        # quota, so verify no congress-chamber drifted far from 80/10/10
        "worst_group_train_share": float(
            rc.groupby(["congress", "chamber"])["split"]
            .apply(lambda s: (s == "train").mean()).min()),
        # eval members with no training history (cold start), for parity
        # with the forecast split's reporting
        "members_unseen_in_train": int(
            len(set(map(tuple, cells.loc[cells.split != "train", ["congress", "chamber", "icpsr"]]
                        .drop_duplicates().itertuples(index=False)))
                - set(map(tuple, cells.loc[cells.split == "train", ["congress", "chamber", "icpsr"]]
                          .drop_duplicates().itertuples(index=False))))),
    }

    cells = cells.sort_values(KEY).reset_index(drop=True)
    cells.to_parquet(OUT / f"{NAME}.parquet", index=False)
    stats["content_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(cells, index=False).to_numpy().tobytes()).hexdigest()
    (OUT / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
