"""Run registered models against the canonical benchmark.

Usage:
    python3 Code/run_benchmark.py                       # all registered models
    python3 Code/run_benchmark.py --models member_rate  # subset

Evaluates on the validation set (for model development) and the test set.
Appends one row per model x eval_set to Modified Data/results/leaderboard.csv
and writes full per-run detail JSON to Modified Data/results/runs/.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from baselines import REGISTRY
from harness import KEY, evaluate
from models_forecast import REGISTRY as FORECAST_REGISTRY

REGISTRY = REGISTRY | FORECAST_REGISTRY
try:  # torch models are optional; baselines must run without torch
    from models_idealpoint import REGISTRY as IDEAL_REGISTRY
    from models_texttower import REGISTRY as TEXT_REGISTRY
    from models_litbaselines import REGISTRY as LIT_REGISTRY
    REGISTRY = REGISTRY | IDEAL_REGISTRY | TEXT_REGISTRY | LIT_REGISTRY
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "none"
    except Exception:
        return "none"


def load_data(split_name: str):
    votes = pd.read_parquet(MOD / "votes.parquet")
    votes = votes.dropna(subset=["vote"])
    bills = MOD / "rollcall_bills.parquet"
    if bills.exists():  # rollcall-level features (question, bill linkage)
        feat = pd.read_parquet(bills)[["congress", "chamber", "rollnumber",
                                       "vote_question", "bill_category",
                                       "bill_type", "bill_no"]]
        votes = votes.merge(feat, on=["congress", "chamber", "rollnumber"],
                            how="left", validate="m:1")
    splits = pd.read_parquet(MOD / "splits" / f"{split_name}.parquet")
    df = votes.merge(splits, on=KEY, how="inner", validate="1:1")
    assert len(df) == len(splits), "votes/splits mismatch"
    return {name: df[df["split"] == name].drop(columns="split")
            for name in ("train", "val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(REGISTRY))
    ap.add_argument("--split", default="regimeA_seed42")
    ap.add_argument("--save-preds", action="store_true",
                    help="save per-vote test predictions to results/preds/")
    ap.add_argument("--eval-sets", nargs="*", default=["val", "test"],
                    choices=["val", "test"],
                    help="development sprints run '--eval-sets val' so the "
                         "test set stays unobserved until one adjudication run")
    args = ap.parse_args()

    data = load_data(args.split)
    print(f"train {len(data['train']):,} | val {len(data['val']):,} | test {len(data['test']):,}")

    runs_dir = MOD / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MOD / "results" / "leaderboard.csv"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []

    for name in args.models:
        model = REGISTRY[name]()
        t0 = time.time()
        model.fit(data["train"])
        fit_s = time.time() - t0
        for eval_set in args.eval_sets:
            res = evaluate(model, data["train"], data[eval_set], args.split, eval_set)
            row = res.flat() | {"fit_seconds": round(fit_s, 1), "git_rev": git_rev(),
                                "run_utc": stamp}
            rows.append(row)
            detail = {"flat": row, "lopsided": res.lopsided, "by_chamber": res.by_chamber}
            (runs_dir / f"{stamp}_{name}_{eval_set}.json").write_text(
                json.dumps(detail, indent=2))
            if args.save_preds and eval_set == "test":
                preds_dir = MOD / "results" / "preds"
                preds_dir.mkdir(parents=True, exist_ok=True)
                out = data[eval_set][KEY + ["vote"]].copy()
                out["p_yea"] = res.predictions
                out.to_parquet(preds_dir / f"{args.split}_{name}.parquet", index=False)
            print(f"{name:22s} {eval_set:4s}  logloss {row['log_loss']:.4f}  "
                  f"acc {row['accuracy']:.4f}  contested_acc {row['contested_accuracy']:.4f}  "
                  f"apre {row['apre']:.4f}")

    lb = pd.DataFrame(rows)
    if leaderboard_path.exists():
        lb = pd.concat([pd.read_csv(leaderboard_path), lb], ignore_index=True)
    lb.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard updated: {leaderboard_path}")


if __name__ == "__main__":
    main()
