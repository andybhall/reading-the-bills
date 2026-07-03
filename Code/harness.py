"""Evaluation harness: model protocol, metrics, and the evaluation loop.

Model protocol (duck-typed):
    model.fit(train_df)              train_df includes the 'vote' column
    model.predict_proba(eval_df)     eval_df does NOT include 'vote';
                                     returns np.ndarray of P(yea), same order

Structural leakage guard: the harness strips the label column (and cast_code)
before calling predict_proba, so a model cannot read the answers even by bug.

Primary metric: log loss. Contested-vote stratum is defined from TRAINING
votes only (minority share >= 0.35 on that rollcall in train data).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

KEY = ["congress", "chamber", "rollnumber", "icpsr"]
LABEL_COLS = ["vote", "cast_code"]
CONTESTED_MINORITY_SHARE = 0.35
EPS = 1e-7


class Model(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> "Model": ...
    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray: ...


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Core metric set for binary yea/nay prediction."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    if len(y) == 0:
        return {k: float("nan") for k in
                ("log_loss", "accuracy", "brier", "base_rate", "auc")} | {"n": 0}
    out = {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "base_rate": float(np.mean(y)),
    }
    # AUC only defined when both classes present
    if 0 < out["base_rate"] < 1:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p))
        ranks[order] = np.arange(1, len(p) + 1)
        # midranks for ties
        s = pd.Series(p)
        ranks = s.rank(method="average").to_numpy()
        n1, n0 = (y == 1).sum(), (y == 0).sum()
        out["auc"] = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        out["auc"] = float("nan")
    return out


def apre(y: np.ndarray, p: np.ndarray, rollcall_ids: np.ndarray) -> float:
    """Aggregate proportional reduction in error vs. predicting each
    rollcall's (training-free, eval-set) majority side — the NOMINATE
    literature's standard summary."""
    df = pd.DataFrame({"y": y, "correct": (np.asarray(p) >= 0.5) == (y == 1), "rc": rollcall_ids})
    g = df.groupby("rc").agg(n=("y", "size"), yea=("y", "sum"), right=("correct", "sum"))
    minority = np.minimum(g["yea"], g["n"] - g["yea"])
    errors = g["n"] - g["right"]
    denom = minority.sum()
    return float((minority.sum() - errors.sum()) / denom) if denom > 0 else float("nan")


def contested_rollcalls(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.Series:
    """Boolean per rollcall key-tuple: minority share >= threshold across all
    observed votes (train + eval). Purely a reporting stratum — labels never
    feed back into predictions — and using all votes keeps the definition
    valid for forecast splits, where eval rollcalls have no train votes."""
    pooled = pd.concat([train_df[["congress", "chamber", "rollnumber", "vote"]],
                        eval_df[["congress", "chamber", "rollnumber", "vote"]]])
    g = pooled.groupby(["congress", "chamber", "rollnumber"])["vote"].agg(["mean", "size"])
    minority_share = np.minimum(g["mean"], 1 - g["mean"])
    return minority_share >= CONTESTED_MINORITY_SHARE


@dataclass
class EvalResult:
    model_name: str
    split_name: str
    eval_set: str
    overall: dict
    contested: dict
    lopsided: dict
    apre: float
    by_chamber: dict
    predictions: np.ndarray | None = None  # P(yea), eval_df row order

    def flat(self) -> dict:
        return {
            "model": self.model_name, "split": self.split_name, "eval_set": self.eval_set,
            "log_loss": self.overall["log_loss"], "accuracy": self.overall["accuracy"],
            "auc": self.overall["auc"], "brier": self.overall["brier"],
            "apre": self.apre,
            "contested_log_loss": self.contested["log_loss"],
            "contested_accuracy": self.contested["accuracy"],
            "contested_n": self.contested["n"],
            "n": self.overall["n"],
        }


def evaluate(model, train_df: pd.DataFrame, eval_df: pd.DataFrame,
             split_name: str, eval_set: str) -> EvalResult:
    y = eval_df["vote"].to_numpy()
    features = eval_df.drop(columns=[c for c in LABEL_COLS if c in eval_df.columns])
    p = np.asarray(model.predict_proba(features), dtype=float)
    assert len(p) == len(eval_df), f"{model.name}: prediction length mismatch"
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), \
        f"{model.name}: predictions outside [0,1] or non-finite"

    rc_key = (eval_df["congress"].astype(str) + "_" + eval_df["chamber"]
              + "_" + eval_df["rollnumber"].astype(str))
    contested = contested_rollcalls(train_df, eval_df)
    contested_keys = {f"{c}_{ch}_{r}" for (c, ch, r), v in contested.items() if v}
    mask = rc_key.isin(contested_keys).to_numpy()

    return EvalResult(
        model_name=model.name, split_name=split_name, eval_set=eval_set,
        overall=metrics(y, p),
        contested=metrics(y[mask], p[mask]),
        lopsided=metrics(y[~mask], p[~mask]),
        apre=apre(y, p, rc_key.to_numpy()),
        by_chamber={ch: metrics(y[(eval_df["chamber"] == ch).to_numpy()],
                                p[(eval_df["chamber"] == ch).to_numpy()])
                    for ch in eval_df["chamber"].unique()},
        predictions=p,
    )
