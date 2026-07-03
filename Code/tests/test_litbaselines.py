"""Tests for the literature baselines on synthetic forecast data.

World 1 (from test_texttower): two topics, text reveals topic AND direction,
held-out rollcalls are new bills. Checks:
  1. gb_spatial (tfidf) beats constant_rate clearly — text-predicted bill
     parameters against fitted ideal points carry real signal
  2. gb_spatial with SHUFFLED text collapses toward constant — the gain
     comes from the text, not from a leak
  3. kraft_bilinear (bilinear on fixed synthetic embeddings) beats constant
     and approaches the Bayes floor

World 2: one dimension; the question TYPE flips the vote's direction
("expand" questions load +, "restrict" questions load -). NOMINATE dim1 is
a noisy copy of the true position. Checks:
  4. nominate_context_logit approaches the Bayes floor (it sees exactly the
     needed interaction) and beats constant by a wide margin.

Run: python3 Code/tests/test_litbaselines.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import ConstantRate  # noqa: E402
from harness import evaluate, metrics  # noqa: E402
from models_litbaselines import GBSpatial, NominateContextLogit, _kraft  # noqa: E402

PASS = []


def check(name, cond):
    PASS.append(cond)
    print(f"  {'ok' if cond else 'FAIL'}: {name}")


TOPIC_WORDS = {0: "firearms safety background checks weapons",
               1: "agriculture subsidies crop insurance farms"}


def make_world1(n_members=120, n_bills=500, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.5, (n_members, 2))
    topic = rng.integers(0, 2, n_bills)
    sign = rng.choice([-1.0, 1.0], n_bills)
    b0 = rng.normal(0, 0.2, n_bills)
    m = np.repeat(np.arange(n_members), n_bills)
    r = np.tile(np.arange(n_bills), n_members)
    logit = sign[r] * pos[m, topic[r]] + b0[r]
    y = (rng.random(len(m)) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({
        "congress": 118, "chamber": "House", "rollnumber": r, "icpsr": m,
        "party_code": 100.0, "vote": y,
        "vote_question": "On Passage", "bill_category": "legislation",
        "bill_type": "hr", "bill_no": r,
    })
    direction = {1.0: "expanding and increasing", -1.0: "restricting and reducing"}
    bills = pd.DataFrame({
        "congress": 118, "bill_type": "hr", "bill_no": np.arange(n_bills),
        "text": [f"A bill {direction[s]} {TOPIC_WORDS[t]} number {i}"
                 for i, (t, s) in enumerate(zip(topic, sign))],
        "sponsor_party": "D",
    })
    p_true = 1 / (1 + np.exp(-logit))
    return df, bills, p_true, topic, sign


def test_gb_spatial():
    print("gb_spatial on direction-revealing synthetic text:")
    df, bills, p_true, _, _ = make_world1()
    holdout = df["rollnumber"] >= 400
    train, test = df[~holdout], df[holdout]
    bayes = metrics(test["vote"].to_numpy(), p_true[holdout.to_numpy()])["log_loss"]

    const = evaluate(ConstantRate().fit(train), train, test, "syn", "test").overall["log_loss"]
    # GB stage-1 needs k=2 here: the world is 2-topic by construction
    gb = GBSpatial(text_mode="tfidf", k=2, svd_dim=16, bills_df=bills).fit(train)
    ll = evaluate(gb, train, test, "syn", "test").overall["log_loss"]

    shuf = bills.copy()
    shuf["text"] = shuf["text"].sample(frac=1, random_state=0).to_numpy()
    gb_s = GBSpatial(text_mode="tfidf", k=2, svd_dim=16, bills_df=shuf).fit(train)
    ll_s = evaluate(gb_s, train, test, "syn", "test").overall["log_loss"]

    print(f"  constant {const:.4f} | gb {ll:.4f} (alpha={gb.alpha}) "
          f"| gb-shuffled {ll_s:.4f} | bayes {bayes:.4f}")
    check("gb_spatial beats constant by wide margin", ll < const - 0.10)
    check("gb_spatial within reach of Bayes", ll < bayes + 0.15)
    check("shuffled text collapses the gain", ll_s > ll + 0.10)

    # k=1 is the PRODUCTION config. sklearn ravels single-column ridge
    # targets; unguarded, (x * A) broadcast (n,1)x(n,) into an (n,n) matrix
    # — 2.5TB at real scale, SIGKILLed by jetsam, and the k=2-only test
    # masked it (diagnosed 2026-07-03). Never test only the convenient k.
    gb1 = GBSpatial(text_mode="tfidf", k=1, svd_dim=16, bills_df=bills).fit(train)
    r1 = evaluate(gb1, train, test, "syn", "test")
    ll1 = r1.overall["log_loss"]
    print(f"  gb k=1 {ll1:.4f}")
    check("gb_spatial k=1 (production config) runs and beats constant", ll1 < const)

    gbc = GBSpatial(text_mode="tfidf", k=2, svd_dim=16, bills_df=bills,
                    calibrate=True).fit(train)
    llc = evaluate(gbc, train, test, "syn", "test").overall["log_loss"]
    print(f"  gb k=2 tcal {llc:.4f} (T={gbc.temperature:.2f}, b={gbc.bias:.2f})")
    check("calibrated gb not materially worse than raw", llc < ll + 0.02)


def test_kraft():
    print("kraft_bilinear on fixed synthetic embeddings:")
    df, bills, p_true, topic, sign = make_world1()
    holdout = df["rollnumber"] >= 400
    train, test = df[~holdout], df[holdout]
    bayes = metrics(test["vote"].to_numpy(), p_true[holdout.to_numpy()])["log_loss"]

    # fixed "sentence embedding": topic one-hots signed by direction + noise,
    # mimicking a pretrained encoder that captures both, plus dead dims
    rng = np.random.default_rng(0)
    E = np.zeros((len(bills), 8), dtype=np.float32)
    E[np.arange(len(bills)), topic] = sign
    E += rng.normal(0, 0.05, E.shape)
    emb = pd.DataFrame(E, columns=[f"e{j}" for j in range(E.shape[1])])
    emb.insert(0, "congress", 118)
    emb.insert(1, "chamber", "House")
    emb.insert(2, "rollnumber", bills["bill_no"].to_numpy())
    with tempfile.TemporaryDirectory() as td:
        emb_path = Path(td) / "syn_emb.parquet"
        emb.to_parquet(emb_path, index=False)
        m = _kraft()
        m.k, m.emb_file, m._bills_override = 4, str(emb_path), bills
        m.batch_size, m.max_epochs, m.min_epochs, m.lr = 4096, 150, 30, 0.05
        m.es_mode = "random"  # synthetic world has no dates
        m.fit(train)
        ll = evaluate(m, train, test, "syn", "test").overall["log_loss"]
    const = evaluate(ConstantRate().fit(train), train, test, "syn", "test").overall["log_loss"]
    print(f"  constant {const:.4f} | kraft {ll:.4f} | bayes {bayes:.4f}")
    check("kraft beats constant by wide margin", ll < const - 0.10)
    check("kraft within reach of Bayes", ll < bayes + 0.10)


def make_world2(n_members=150, n_rollcalls=600, seed=7):
    # position scale 2.5 keeps the Bayes floor well below the constant
    # model, so the beats-constant margin check is a real test
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 2.5, n_members)
    nom = pos + rng.normal(0, 0.3, n_members)  # noisy NOMINATE copy
    qtype = rng.integers(0, 2, n_rollcalls)    # 0: expand (+), 1: restrict (-)
    sign = np.where(qtype == 0, 1.0, -1.0)
    b0 = rng.normal(0, 0.2, n_rollcalls)
    m = np.repeat(np.arange(n_members), n_rollcalls)
    r = np.tile(np.arange(n_rollcalls), n_members)
    logit = sign[r] * pos[m] + b0[r]
    y = (rng.random(len(m)) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({
        "congress": 118, "chamber": "House", "rollnumber": r, "icpsr": m,
        "party_code": 100.0, "vote": y,
        "vote_question": np.where(qtype[r] == 0, "On Passage", "On Motion to Recommit"),
        "bill_category": "legislation", "bill_type": "hr", "bill_no": r,
        "nominate_dim1": nom[m], "nominate_dim2": 0.0,
    })
    bills = pd.DataFrame({"congress": 118, "bill_type": "hr",
                          "bill_no": np.arange(n_rollcalls),
                          "text": "x", "sponsor_party": "D"})
    p_true = 1 / (1 + np.exp(-logit))
    return df, bills, p_true


def test_nominate_context():
    print("nominate_context_logit on question-flips-direction world:")
    df, bills, p_true = make_world2()
    holdout = df["rollnumber"] >= 480
    train, test = df[~holdout], df[holdout]
    bayes = metrics(test["vote"].to_numpy(), p_true[holdout.to_numpy()])["log_loss"]
    const = evaluate(ConstantRate().fit(train), train, test, "syn", "test").overall["log_loss"]
    m = NominateContextLogit(bills_df=bills, batch_size=8192).fit(train)
    ll = evaluate(m, train, test, "syn", "test").overall["log_loss"]
    print(f"  constant {const:.4f} | nominate_context {ll:.4f} | bayes {bayes:.4f}")
    check("nominate_context beats constant by wide margin", ll < const - 0.10)
    check("nominate_context near Bayes (noise-limited)", ll < bayes + 0.10)


def main():
    test_gb_spatial()
    test_kraft()
    test_nominate_context()
    n = len(PASS)
    if all(PASS):
        print(f"\nAll {n} literature-baseline checks passed.")
    else:
        print(f"\n{n - sum(PASS)} of {n} checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
