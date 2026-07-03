"""Baseline models for the roll-call benchmark.

All baselines follow the harness Model protocol. Probabilities are smoothed
(add-k toward the parent rate) so log loss is well-behaved for thin cells.
Fallback chain for unseen keys: specific estimate -> parent rate -> global.
"""

import numpy as np
import pandas as pd

GLOBAL_FALLBACK = 0.5
SMOOTH_K = 5.0


def _smoothed_rate(df, group_cols, prior: pd.Series, prior_cols, k: float = SMOOTH_K) -> pd.Series:
    """Per-group P(yea) shrunk toward a coarser-level prior with pseudo-count k."""
    g = df.groupby(group_cols)["vote"].agg(["sum", "size"])
    pr = prior.reindex(pd.MultiIndex.from_frame(
        g.index.to_frame(index=False)[prior_cols])).to_numpy()
    return pd.Series((g["sum"] + k * pr) / (g["size"] + k), index=g.index)


class ConstantRate:
    """P(yea) = global training yea rate. The floor."""
    name = "constant_rate"

    def fit(self, train_df):
        self.rate = float(train_df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        return np.full(len(eval_df), self.rate)


class MemberRate:
    """P(yea) = this member-congress's training yea rate (smoothed toward
    the chamber-congress rate). Captures individual base rates only."""
    name = "member_rate"

    def fit(self, train_df):
        self.cc_rate = train_df.groupby(["congress", "chamber"])["vote"].mean()
        self.member_rate = _smoothed_rate(train_df, ["congress", "chamber", "icpsr"],
                                          self.cc_rate, ["congress", "chamber"])
        self.global_rate = float(train_df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        idx = pd.MultiIndex.from_frame(eval_df[["congress", "chamber", "icpsr"]])
        p = self.member_rate.reindex(idx).to_numpy()
        cc = self.cc_rate.reindex(
            pd.MultiIndex.from_frame(eval_df[["congress", "chamber"]])).to_numpy()
        p = np.where(np.isnan(p), cc, p)
        return np.where(np.isnan(p), self.global_rate, p)


class PartyRollcallRate:
    """P(yea) = share of the member's party voting yea on this rollcall in
    TRAINING data (smoothed toward the party-congress rate). 'Vote with your
    party's position on this bill' — the baseline to beat for Regime A."""
    name = "party_rollcall_rate"

    def fit(self, train_df):
        self.party_cc = train_df.groupby(["congress", "chamber", "party_code"])["vote"].mean()
        self.cell = _smoothed_rate(train_df, ["congress", "chamber", "rollnumber", "party_code"],
                                   self.party_cc, ["congress", "chamber", "party_code"])
        self.global_rate = float(train_df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        idx = pd.MultiIndex.from_frame(
            eval_df[["congress", "chamber", "rollnumber", "party_code"]])
        p = self.cell.reindex(idx).to_numpy()
        pcc = self.party_cc.reindex(pd.MultiIndex.from_frame(
            eval_df[["congress", "chamber", "party_code"]])).to_numpy()
        p = np.where(np.isnan(p), pcc, p)
        return np.where(np.isnan(p), self.global_rate, p)


REGISTRY = {m.name: m for m in [ConstantRate, MemberRate, PartyRollcallRate]}
