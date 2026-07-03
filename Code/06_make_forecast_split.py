"""Create the canonical Regime B ("forecast") split, congresses 108-119.

ENTIRE rollcalls are assigned to train/val/test by date within each
congress-chamber: first 80% of rollcalls (by date, ties broken by
rollnumber) -> train, next 10% -> val, last 10% -> test. Every vote on a
rollcall inherits its assignment, so nothing contemporaneous with an eval
rollcall is ever observable at training time — the model must forecast
genuinely new rollcalls from member history and bill features alone.

Within-congress (rather than whole-congress-out) cutoffs keep member
composition stable; cross-congress generalization to the 119th+ can be a
separate, harder eval later.

Output matches the Regime A format (cell keys + split) so run_benchmark.py
works unchanged: Modified Data/splits/forecast108_119.parquet + stats.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
OUT = MOD / "splits"
KEY = ["congress", "chamber", "rollnumber", "icpsr"]
NAME = "forecast108_119"
CONGRESSES = range(108, 120)


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")
    rc = rc[rc["congress"].isin(CONGRESSES)].copy()
    rc = rc.sort_values(["congress", "chamber", "date", "rollnumber"])
    rank = rc.groupby(["congress", "chamber"]).cumcount()
    size = rc.groupby(["congress", "chamber"])["rollnumber"].transform("size")
    q = (rank + 0.5) / size
    rc["split"] = pd.cut(q, [0, 0.8, 0.9, 1.0], labels=["train", "val", "test"])

    votes = pd.read_parquet(MOD / "votes.parquet", columns=KEY + ["vote"])
    votes = votes.dropna(subset=["vote"])
    cells = votes.merge(rc[["congress", "chamber", "rollnumber", "split"]],
                        on=["congress", "chamber", "rollnumber"], how="inner")[KEY + ["split"]]
    cells["split"] = cells["split"].astype(str)

    stats = {
        "n": int(len(cells)),
        "shares": cells["split"].value_counts(normalize=True).round(4).to_dict(),
        "counts": cells["split"].value_counts().to_dict(),
        "rollcalls_per_split": rc["split"].value_counts().to_dict(),
        "val_test_date_ranges": {
            s: [str(rc[rc.split == s].date.min().date()), str(rc[rc.split == s].date.max().date())]
            for s in ("val", "test")},
        # members whose first-ever vote falls after the train window get no
        # history at all; count them so eval breakdowns can isolate cold-start
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
