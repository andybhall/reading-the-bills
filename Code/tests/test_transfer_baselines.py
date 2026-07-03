"""Test the majority-status baselines on a synthetic congress-out world.

World: three training congresses + one held-out congress. Voting propensity
depends on whether the member's party HOLDS THE MAJORITY (majority yeas
procedural votes, minority nays them; passage votes are majority-tilted) —
and the majority flips in the held-out congress. Congress-keyed baselines
cannot see the flip; majority-keyed ones transfer.

World A (majority-driven behavior): the flip means pooled member history
ANTI-transfers — a member who yea'd procedurals as majority is confidently
mispredicted as minority. This is the real documented mechanism of the
champion's congress-out collapse (decisions.md), so the test asserts it.
World B (stable personal propensities, no majority effect): member history
is exactly what transfers; majority status carries nothing.

Checks:
  A1. majority_question_rate transfers across the flip (near Bayes)
  A2. party_question_rate collapses (materially worse than majority)
  A3. member_pooled ANTI-transfers (worse than majority baseline) —
      pooled member history embeds majority-era behavior
  B1. member_pooled transfers near Bayes when behavior is personal
  B2. majority_question_rate is uninformative there (no better than
      the base-rate floor by a wide margin)

Run: python3 Code/tests/test_transfer_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import evaluate, metrics  # noqa: E402
from models_forecast import (MajorityQuestionRate, MemberPooledQuestionRate,  # noqa: E402
                             PartyQuestionRate)

PASS = []


def check(name, cond):
    PASS.append(cond)
    print(f"  {'ok' if cond else 'FAIL'}: {name}")


def make_world(n_per_party=60, n_rc=300, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    p_true_all = []
    # party 200 holds the majority in congresses 1-3; party 100 in congress 4
    for congress in (1, 2, 3, 4):
        majority = 200.0 if congress < 4 else 100.0
        # extra members give the majority its numerical edge
        parties = [100.0] * n_per_party + [200.0] * n_per_party
        extras = [majority] * 10
        icpsrs = list(range(2 * n_per_party)) + list(range(9000, 9010))
        pcs = parties + extras
        qtype = rng.integers(0, 2, n_rc)  # 0 procedural, 1 passage
        for rc in range(n_rc):
            q = "Previous Question" if qtype[rc] == 0 else "On Passage"
            for icpsr, pc in zip(icpsrs, pcs):
                in_maj = pc == majority
                if qtype[rc] == 0:
                    p = 0.9 if in_maj else 0.15
                else:
                    p = 0.75 if in_maj else 0.45
                rows.append((congress, "House", rc, float(icpsr), pc, q,
                             float(rng.random() < p)))
                p_true_all.append(p)
    df = pd.DataFrame(rows, columns=["congress", "chamber", "rollnumber", "icpsr",
                                     "party_code", "vote_question", "vote"])
    return df, np.array(p_true_all)


def make_personal_world(n_members=120, n_rc=300, seed=13):
    rng = np.random.default_rng(seed)
    personal = rng.normal(0.6, 1.2, n_members)  # stable across congresses
    rows, p_true = [], []
    for congress in (1, 2, 3, 4):
        for rc in range(n_rc):
            for i in range(n_members):
                p = 1 / (1 + np.exp(-personal[i]))
                rows.append((congress, "House", rc, float(i),
                             100.0 if i < n_members // 2 else 200.0,
                             "On Passage", float(rng.random() < p)))
                p_true.append(p)
    df = pd.DataFrame(rows, columns=["congress", "chamber", "rollnumber", "icpsr",
                                     "party_code", "vote_question", "vote"])
    return df, np.array(p_true)


def main():
    print("world A: majority-driven behavior, majority flips in test congress")
    df, p_true = make_world()
    train = df[df["congress"] < 4]
    test = df[df["congress"] == 4]
    bayes = metrics(test["vote"].to_numpy(), p_true[df["congress"].to_numpy() == 4])["log_loss"]

    res = {}
    for cls in (MajorityQuestionRate, MemberPooledQuestionRate, PartyQuestionRate):
        model = cls().fit(train)
        res[model.name] = evaluate(model, train, test, "syn_out", "test").overall["log_loss"]
        print(f"  {model.name}: test log loss {res[model.name]:.4f}")
    print(f"  bayes: {bayes:.4f}")

    check("A1 majority_question_rate transfers (near Bayes)",
          res["majority_question_rate"] < bayes + 0.05)
    check("A2 party_question_rate collapses across the flip",
          res["party_question_rate"] > res["majority_question_rate"] + 0.15)
    check("A3 pooled member history ANTI-transfers under the flip",
          res["member_pooled_question_rate"] > res["majority_question_rate"] + 0.3)

    print("world B: stable personal propensities, no majority effect")
    df, p_true = make_personal_world()
    train = df[df["congress"] < 4]
    test = df[df["congress"] == 4]
    bayes = metrics(test["vote"].to_numpy(), p_true[df["congress"].to_numpy() == 4])["log_loss"]
    res = {}
    for cls in (MajorityQuestionRate, MemberPooledQuestionRate):
        model = cls().fit(train)
        res[model.name] = evaluate(model, train, test, "syn_out", "test").overall["log_loss"]
        print(f"  {model.name}: test log loss {res[model.name]:.4f}")
    print(f"  bayes: {bayes:.4f}")

    check("B1 member_pooled transfers near Bayes on personal behavior",
          res["member_pooled_question_rate"] < bayes + 0.03)
    check("B2 majority baseline uninformative on personal behavior",
          res["majority_question_rate"] > bayes + 0.10)

    n = len(PASS)
    if all(PASS):
        print(f"\nAll {n} transfer-baseline checks passed.")
    else:
        print(f"\n{n - sum(PASS)} of {n} checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
