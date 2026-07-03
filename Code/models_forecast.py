"""No-text baselines for the Regime B forecast benchmark.

Under forecasting, nothing about the specific rollcall's voting is
observable — only member history and rollcall metadata (vote question,
bill linkage). These baselines set the bar that bill-text models must beat.
"""

import numpy as np
import pandas as pd

from baselines import SMOOTH_K, _smoothed_rate

# vote_question -> coarse bucket; order matters (first match wins)
QUESTION_BUCKETS = [
    ("nomination", r"Nomination"),
    ("passage", r"Passage|Pass(?:,| )|Suspend the Rules and Pass"),
    ("amendment", r"Amendment"),
    ("cloture", r"Cloture"),
    ("resolution", r"Agreeing to the Resolution|Concurrent Resolution"),
    ("conference", r"Conference Report"),
    ("procedural", r"Previous Question|Recommit|Table|Motion to Proceed|"
                   r"Adjourn|Quorum|Instruct|Postpone|Commit|Discharge"),
    ("veto", r"Veto"),
]


def question_bucket(q: pd.Series) -> pd.Series:
    out = pd.Series("other", index=q.index)
    unassigned = q.notna()
    for name, pattern in QUESTION_BUCKETS:
        hit = unassigned & q.str.contains(pattern, case=False, regex=True, na=False)
        out[hit] = name
        unassigned &= ~hit
    return out


class PartyCongressRate:
    """P(yea) = party x congress-chamber training rate. No rollcall info."""
    name = "party_congress_rate"

    def fit(self, train_df):
        self.rate = train_df.groupby(["congress", "chamber", "party_code"])["vote"].mean()
        self.global_rate = float(train_df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        idx = pd.MultiIndex.from_frame(eval_df[["congress", "chamber", "party_code"]])
        return self.rate.reindex(idx).fillna(self.global_rate).to_numpy()


class PartyQuestionRate:
    """P(yea) = party x congress-chamber x question-bucket training rate,
    smoothed toward the party-congress rate. Captures e.g. 'minority party
    nays procedural votes' without any bill content."""
    name = "party_question_rate"

    def fit(self, train_df):
        df = train_df.assign(qbucket=question_bucket(train_df["vote_question"]))
        self.party_cc = df.groupby(["congress", "chamber", "party_code"])["vote"].mean()
        self.cell = _smoothed_rate(df, ["congress", "chamber", "party_code", "qbucket"],
                                   self.party_cc, ["congress", "chamber", "party_code"])
        self.global_rate = float(df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        df = eval_df.assign(qbucket=question_bucket(eval_df["vote_question"]))
        idx = pd.MultiIndex.from_frame(df[["congress", "chamber", "party_code", "qbucket"]])
        p = self.cell.reindex(idx).to_numpy()
        pcc = self.party_cc.reindex(
            pd.MultiIndex.from_frame(df[["congress", "chamber", "party_code"]])).to_numpy()
        p = np.where(np.isnan(p), pcc, p)
        return np.where(np.isnan(p), self.global_rate, p)


class MemberQuestionRate:
    """P(yea) = member x question-bucket rate, smoothed toward the member's
    overall rate. Member history only — the individual-level floor."""
    name = "member_question_rate"

    def fit(self, train_df):
        df = train_df.assign(qbucket=question_bucket(train_df["vote_question"]))
        self.member = df.groupby(["congress", "chamber", "icpsr"])["vote"].mean()
        self.cell = _smoothed_rate(df, ["congress", "chamber", "icpsr", "qbucket"],
                                   self.member, ["congress", "chamber", "icpsr"])
        self.global_rate = float(df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        df = eval_df.assign(qbucket=question_bucket(eval_df["vote_question"]))
        idx = pd.MultiIndex.from_frame(df[["congress", "chamber", "icpsr", "qbucket"]])
        p = self.cell.reindex(idx).to_numpy()
        pm = self.member.reindex(
            pd.MultiIndex.from_frame(df[["congress", "chamber", "icpsr"]])).to_numpy()
        p = np.where(np.isnan(p), pm, p)
        return np.where(np.isnan(p), self.global_rate, p)


class MemberQuestionRateRecency:
    """member_question_rate with exponentially decayed vote weights,
    exp(-days_before_end_of_train / tau), tau fixed a priori. Recent
    behavior counts more; with tau -> inf this reduces exactly to
    MemberQuestionRate (weighted smoothing (sum wy + k*prior)/(sum w + k)
    equals _smoothed_rate at w = 1)."""
    name = "member_question_rate_recency"

    def __init__(self, tau_days: float = 365.0):
        self.tau = tau_days

    @staticmethod
    def _wrate(df, group_cols, prior, prior_cols, k=SMOOTH_K):
        g = df.groupby(group_cols)[["w", "wy"]].sum()
        pr = prior.reindex(pd.MultiIndex.from_frame(
            g.index.to_frame(index=False)[prior_cols])).to_numpy()
        return pd.Series((g["wy"] + k * pr) / (g["w"] + k), index=g.index)

    def fit(self, train_df):
        df = train_df.assign(qbucket=question_bucket(train_df["vote_question"]))
        w = np.exp(-(df["date"].max() - df["date"]).dt.days / self.tau)
        df = df.assign(w=w, wy=w * df["vote"])
        g = df.groupby(["congress", "chamber", "icpsr"])[["w", "wy"]].sum()
        self.member = g["wy"] / g["w"]
        self.cell = self._wrate(df, ["congress", "chamber", "icpsr", "qbucket"],
                                self.member, ["congress", "chamber", "icpsr"])
        self.global_rate = float(df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        df = eval_df.assign(qbucket=question_bucket(eval_df["vote_question"]))
        idx = pd.MultiIndex.from_frame(df[["congress", "chamber", "icpsr", "qbucket"]])
        p = self.cell.reindex(idx).to_numpy()
        pm = self.member.reindex(
            pd.MultiIndex.from_frame(df[["congress", "chamber", "icpsr"]])).to_numpy()
        p = np.where(np.isnan(p), pm, p)
        return np.where(np.isnan(p), self.global_rate, p)


def _majority_party(df: pd.DataFrame) -> pd.Series:
    """Majority party code per congress-chamber from MEMBER COMPOSITION
    (each member counted once; D=100 vs R=200 plurality). Composition is
    knowable before any vote, so this transfers to an unseen congress —
    unlike congress-keyed rates, which collapse to the global fallback."""
    members = df.drop_duplicates(["congress", "chamber", "icpsr"])
    counts = (members[members["party_code"].isin([100.0, 200.0])]
              .groupby(["congress", "chamber", "party_code"]).size())
    return counts.groupby(level=["congress", "chamber"]).idxmax().map(lambda t: t[2])


def _is_majority(df: pd.DataFrame, majority: pd.Series) -> np.ndarray:
    maj = majority.reindex(pd.MultiIndex.from_frame(df[["congress", "chamber"]]))
    return (df["party_code"].to_numpy() == maj.to_numpy()).astype(float)


class MajorityQuestionRate:
    """P(yea) = (party-is-majority x chamber x question-bucket) rate pooled
    ACROSS congresses. The fair party-level null for congress-out transfer:
    congress-keyed baselines collapse on an unseen congress, this cannot.
    Majority status for eval rows comes from the eval congress's own member
    composition (features, never labels)."""
    name = "majority_question_rate"

    def fit(self, train_df):
        df = train_df.assign(qbucket=question_bucket(train_df["vote_question"]),
                             in_maj=_is_majority(train_df, _majority_party(train_df)))
        self.chamber_rate = df.groupby(["chamber", "in_maj"])["vote"].mean()
        self.cell = _smoothed_rate(df, ["chamber", "in_maj", "qbucket"],
                                   self.chamber_rate, ["chamber", "in_maj"])
        self.global_rate = float(df["vote"].mean())
        return self

    def predict_proba(self, eval_df):
        df = eval_df.assign(qbucket=question_bucket(eval_df["vote_question"]),
                            in_maj=_is_majority(eval_df, _majority_party(eval_df)))
        idx = pd.MultiIndex.from_frame(df[["chamber", "in_maj", "qbucket"]])
        p = self.cell.reindex(idx).to_numpy()
        pc = self.chamber_rate.reindex(
            pd.MultiIndex.from_frame(df[["chamber", "in_maj"]])).to_numpy()
        p = np.where(np.isnan(p), pc, p)
        return np.where(np.isnan(p), self.global_rate, p)


class MemberPooledQuestionRate:
    """P(yea) = member x question-bucket rate pooled across congresses
    (no congress key), adjusted for nothing else. Returning members carry
    their history into an unseen congress; new members fall back to the
    majority x question table. The fair member-level null for transfer."""
    name = "member_pooled_question_rate"

    def fit(self, train_df):
        df = train_df.assign(qbucket=question_bucket(train_df["vote_question"]))
        self.member = df.groupby(["chamber", "icpsr"])["vote"].mean()
        self.cell = _smoothed_rate(df, ["chamber", "icpsr", "qbucket"],
                                   self.member, ["chamber", "icpsr"])
        self.majority = MajorityQuestionRate().fit(train_df)
        return self

    def predict_proba(self, eval_df):
        df = eval_df.assign(qbucket=question_bucket(eval_df["vote_question"]))
        idx = pd.MultiIndex.from_frame(df[["chamber", "icpsr", "qbucket"]])
        p = self.cell.reindex(idx).to_numpy()
        pm = self.member.reindex(
            pd.MultiIndex.from_frame(df[["chamber", "icpsr"]])).to_numpy()
        p = np.where(np.isnan(p), pm, p)
        return np.where(np.isnan(p), self.majority.predict_proba(eval_df), p)


REGISTRY = {m.name: m for m in [PartyCongressRate, PartyQuestionRate, MemberQuestionRate,
                                MajorityQuestionRate, MemberPooledQuestionRate]}
