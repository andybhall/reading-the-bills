"""Test the two-tower text model on synthetic forecast data.

World: two latent topics ("guns", "farms"). Members have independent
positions on each topic; a bill's text reveals its topic AND its direction
("expanding" vs "restricting" — the side of the cutpoint), so the bill's
discrimination vector is fully determined by text. Votes follow the
member's position on that topic. Held-out rollcalls are NEW bills — the
meta-only model cannot know what's at stake, the text tower can.
(First version of this test revealed direction only through an unobservable
sign, making the task information-theoretically impossible from features —
kept in mind as a design lesson for real bills, where direction must also
be inferable from text.) Checks:
  1. text tower beats the meta tower by a wide margin
  2. text tower approaches the Bayes log loss
Run: python3 Code/tests/test_texttower.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import evaluate, metrics  # noqa: E402
from models_texttower import TextTower  # noqa: E402

TOPIC_WORDS = {0: "firearms safety background checks weapons",
               1: "agriculture subsidies crop insurance farms"}


def make_world(n_members=120, n_bills=500, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.5, (n_members, 2))  # member x topic position
    topic = rng.integers(0, 2, n_bills)
    sign = rng.choice([-1.0, 1.0], n_bills)   # which side; revealed by text
    b0 = rng.normal(0, 0.2, n_bills)          # unobservable noise, kept small

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
    return df, bills, p_true


def main():
    df, bills, p_true = make_world()
    # forecast split: last 20% of bills are unseen
    holdout = df["rollnumber"] >= 400
    train, test = df[~holdout], df[holdout]
    bayes = metrics(test["vote"].to_numpy(), p_true[holdout.to_numpy()])["log_loss"]

    results = {}
    for use_text in (False, True):
        model = TextTower(k=4, use_text=use_text, svd_dim=16, batch_size=4096,
                          lr=0.05, max_epochs=150, min_epochs=30,
                          bills_df=bills).fit(train)
        res = evaluate(model, train, test, "synthetic", "test")
        results[model.name] = res.overall["log_loss"]
        print(f"{model.name}: test log loss {res.overall['log_loss']:.4f} "
              f"(epochs {model.epochs_run})")
    print(f"bayes: {bayes:.4f}")

    assert results["text_tower_4d"] < results["meta_tower_4d"] - 0.05, \
        "text tower should clearly beat meta tower on topic-structured data"
    assert results["text_tower_4d"] < bayes + 0.06, \
        f"text tower {results['text_tower_4d']:.4f} too far from Bayes {bayes:.4f}"
    print("All text-tower checks passed.")


if __name__ == "__main__":
    main()
