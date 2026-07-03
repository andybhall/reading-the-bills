"""Recovery test for the ideal-point model on synthetic spatial data.

Generates votes from the true model P(yea)=sigmoid(a_r*x_m + b_r), fits a
1D model, and checks (a) held-out log loss approaches the Bayes loss and
(b) learned positions recover true positions up to sign (|r| > 0.9).

Run: python3 Code/tests/test_idealpoint.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import evaluate, metrics  # noqa: E402
from models_idealpoint import IdealPoint  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    n_m, n_r = 200, 400
    x_true = rng.uniform(-1.5, 1.5, n_m)
    a_true = rng.normal(0, 1.5, n_r)
    b_true = rng.normal(0, 0.5, n_r)

    m = np.repeat(np.arange(n_m), n_r)
    r = np.tile(np.arange(n_r), n_m)
    p_true = 1 / (1 + np.exp(-(a_true[r] * x_true[m] + b_true[r])))
    df = pd.DataFrame({
        "congress": 118, "chamber": "House", "rollnumber": r, "icpsr": m,
        "vote": (rng.random(len(m)) < p_true).astype(float),
    })
    test_mask = rng.random(len(df)) < 0.15
    train, test = df[~test_mask], df[test_mask]
    bayes = metrics(test["vote"].to_numpy(), p_true[test_mask.nonzero()[0]])["log_loss"]

    model = IdealPoint(k=1, max_epochs=80).fit(train)
    res = evaluate(model, train, test, "synthetic", "test")
    ll = res.overall["log_loss"]
    print(f"bayes log loss {bayes:.4f} | model log loss {ll:.4f} "
          f"| epochs {model.epochs_run}")
    assert ll < bayes + 0.03, f"model log loss {ll:.4f} far from Bayes {bayes:.4f}"

    pos = model.member_positions().sort_values("icpsr")
    r_corr = np.corrcoef(pos["dim1"].to_numpy(), x_true)[0, 1]
    print(f"corr(learned, true) = {r_corr:+.4f}")
    assert abs(r_corr) > 0.9, f"poor recovery: |r| = {abs(r_corr):.3f}"
    print("All ideal-point recovery checks passed.")


if __name__ == "__main__":
    main()
