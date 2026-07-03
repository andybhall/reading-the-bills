"""Tests for metrics, splits, and baselines on synthetic data.

Run: python3 Code/tests/test_harness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import ConstantRate, MemberRate, PartyRollcallRate  # noqa: E402
from harness import apre, contested_rollcalls, evaluate, metrics  # noqa: E402

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


def make_synthetic(n_members=50, n_rollcalls=40, seed=0):
    """Two-party world: party 100 votes yea w.p. .9 on even rollcalls and .1
    on odd; party 200 the reverse. Member 0 is a maverick (always opposite)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_members):
        party = 100 if m < n_members // 2 else 200
        for r in range(n_rollcalls):
            p_yea = 0.9 if ((r % 2 == 0) == (party == 100)) else 0.1
            if m == 0:
                p_yea = 1 - p_yea
            rows.append({"congress": 118, "chamber": "House", "rollnumber": r,
                         "icpsr": m, "party_code": party,
                         "vote": float(rng.random() < p_yea)})
    df = pd.DataFrame(rows)
    df["split"] = np.where(rng.random(len(df)) < 0.8, "train", "test")
    return df


def test_metrics():
    print("metrics:")
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    m = metrics(y, p)
    expected_ll = -np.mean(np.log([0.9, 0.8, 0.8, 0.9]))
    check("log loss exact", abs(m["log_loss"] - expected_ll) < 1e-9)
    check("accuracy exact", m["accuracy"] == 1.0)
    check("auc perfect separation", m["auc"] == 1.0)
    m2 = metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))
    check("coin flip log loss = ln2", abs(m2["log_loss"] - np.log(2)) < 1e-9)
    # perfect predictor APRE = 1; rollcall-majority predictor APRE = 0
    y3 = np.array([1, 1, 1, 0, 0] * 2)
    rc3 = np.array([0] * 5 + [1] * 5)
    check("apre perfect = 1", apre(y3, y3.astype(float), rc3) == 1.0)
    maj = np.array([1.0] * 5 + [1.0] * 5)  # rollcall 0 majority yea; rc 1 majority yea
    check("apre majority = 0", apre(np.array([1,1,1,0,0,1,1,1,0,0]), maj,
                                    rc3) == 0.0)


def test_contested():
    print("contested stratum:")
    df = pd.DataFrame({
        "congress": 118, "chamber": "House",
        "rollnumber": [1] * 10 + [2] * 10,
        "vote": [1] * 5 + [0] * 5 + [1] * 10,
    })
    c = contested_rollcalls(df.iloc[:12], df.iloc[12:])  # pooled across train+eval
    check("50-50 rollcall contested", bool(c.loc[(118, "House", 1)]))
    check("unanimous rollcall not contested", not bool(c.loc[(118, "House", 2)]))
    # forecast-style: rollcall present only in eval still gets a stratum
    c2 = contested_rollcalls(df.iloc[:10], df.iloc[10:])
    check("eval-only rollcall gets stratum", not bool(c2.loc[(118, "House", 2)]))


def test_baselines_and_leakage():
    print("baselines on synthetic two-party data:")
    df = make_synthetic()
    train = df[df.split == "train"].drop(columns="split")
    test = df[df.split == "test"].drop(columns="split")

    results = {}
    for cls in (ConstantRate, MemberRate, PartyRollcallRate):
        model = cls().fit(train)
        res = evaluate(model, train, test, "synthetic", "test")
        results[cls.name] = res.overall["log_loss"]
    check("party >> member baseline on party-line data",
          results["party_rollcall_rate"] < results["member_rate"] - 0.2)
    check("member <= constant baseline",
          results["member_rate"] <= results["constant_rate"] + 0.02)
    check("party baseline near bayes log loss (~.33)",
          results["party_rollcall_rate"] < 0.45)

    # leakage guard: flipping test labels must not change predictions
    model = PartyRollcallRate().fit(train)
    feats = test.drop(columns=["vote"])
    p1 = model.predict_proba(feats)
    test_flipped = test.copy()
    test_flipped["vote"] = 1 - test_flipped["vote"]
    p2 = model.predict_proba(test_flipped.drop(columns=["vote"]))
    check("test labels cannot influence predictions", np.array_equal(p1, p2))

    # unseen member/rollcall falls back gracefully
    novel = pd.DataFrame([{"congress": 119, "chamber": "Senate", "rollnumber": 999,
                           "icpsr": 99999, "party_code": 328}])
    for cls in (ConstantRate, MemberRate, PartyRollcallRate):
        p = cls().fit(train).predict_proba(novel)
        check(f"{cls.name} handles unseen keys", np.isfinite(p).all() and 0 <= p[0] <= 1)


def test_split_determinism():
    print("split determinism:")
    from importlib.machinery import SourceFileLoader
    splits_mod = SourceFileLoader(
        "make_splits", str(Path(__file__).resolve().parent.parent / "02_make_splits.py")
    ).load_module()
    df = make_synthetic()[["congress", "chamber", "rollnumber", "icpsr"]]
    u1 = splits_mod.hash_unit(df, 42)
    u2 = splits_mod.hash_unit(df.sample(frac=1, random_state=1).sort_index(), 42)
    check("hash split independent of row order", np.array_equal(u1, u2))
    u3 = splits_mod.hash_unit(df, 43)
    check("different seed -> different split", not np.array_equal(u1, u3))
    check("hash units uniform-ish", 0.4 < u1.mean() < 0.6)


if __name__ == "__main__":
    test_metrics()
    test_contested()
    test_baselines_and_leakage()
    test_split_determinism()
    print(f"\nAll {PASS} checks passed.")
