"""Create the canonical Regime A ("completion") split.

For every (member, rollcall) cell with a yea/nay vote, assign train/val/test
(80/10/10) by hashing the cell key with a fixed seed. Hash-based assignment
is deterministic and independent of row order, so the split can always be
reconstructed and any drift is detectable via the content hash.

Output: Modified Data/splits/regimeA_seed42.parquet  (keys + split label)
        Modified Data/splits/regimeA_seed42_stats.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Modified Data" / "splits"
SEED = 42
KEY = ["congress", "chamber", "rollnumber", "icpsr"]


def hash_unit(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Uniform [0,1) per row from a stable hash of the cell key."""
    keys = (df["congress"].astype(str) + "|" + df["chamber"] + "|"
            + df["rollnumber"].astype(str) + "|" + df["icpsr"].astype(str)
            + f"|seed{seed}")
    return keys.map(lambda k: int(hashlib.sha256(k.encode()).hexdigest()[:12], 16) / 16**12).to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    votes = pd.read_parquet(ROOT / "Modified Data" / "votes.parquet",
                            columns=KEY + ["vote"])
    cells = votes.dropna(subset=["vote"])[KEY].copy()

    u = hash_unit(cells, SEED)
    cells["split"] = np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test"))

    stats = {
        "seed": SEED,
        "n": int(len(cells)),
        "shares": cells["split"].value_counts(normalize=True).round(4).to_dict(),
        "counts": cells["split"].value_counts().to_dict(),
    }

    # coverage checks: every rollcall and member-congress should retain
    # training votes, otherwise some models can't form estimates for them
    rc_train = cells[cells.split == "train"].groupby(["congress", "chamber", "rollnumber"]).size()
    rc_all = cells.groupby(["congress", "chamber", "rollnumber"]).size()
    stats["rollcalls_total"] = int(len(rc_all))
    stats["rollcalls_no_train_votes"] = int(len(rc_all) - len(rc_train))
    mem_train = cells[cells.split == "train"].groupby(["congress", "chamber", "icpsr"]).size()
    mem_all = cells.groupby(["congress", "chamber", "icpsr"]).size()
    stats["member_congresses_total"] = int(len(mem_all))
    stats["member_congresses_no_train_votes"] = int(len(mem_all) - len(mem_train))

    cells = cells.sort_values(KEY).reset_index(drop=True)
    path = OUT / f"regimeA_seed{SEED}.parquet"
    cells.to_parquet(path, index=False)
    stats["content_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(cells, index=False).to_numpy().tobytes()
    ).hexdigest()

    (OUT / f"regimeA_seed{SEED}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
