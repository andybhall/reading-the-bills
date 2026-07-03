"""Test within-bill context features on synthetic amendment-series data.

World: each bill gets a series of rollcalls over consecutive days; each
member holds a persistent stance on each BILL (yea-side w.p. 0.9 across all
of that bill's rollcalls). Text and metadata carry nothing — the ONLY
usable signal for a held-out rollcall is behavior on the same bill's
earlier rollcalls. Temporal holdout: last 20% of days.

The right performance floor is NOT full-information Bayes: the feature
carries ONE noisy observation of the member's bill stance, so the best
achievable on covered rows is the one-observation posterior predictive
(computed analytically below). First draft of this test wrongly demanded
near-Bayes and a permutation penalty larger than the total gain —
expectations fixed, not the feature (2026-07-03).

Checks:
  1. billctx tower beats the no-context tower (diluted by partial coverage)
  2. on covered rows, billctx reaches the one-observation posterior floor
  3. falsification: with each rollcall reassigned to a random bill the
     context gain vanishes — the model learns to ignore the noise features
     and lands back at the no-context loss

Run: python3 Code/tests/test_billctx.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import evaluate, metrics  # noqa: E402
from models_texttower import CtxStack, TextTower  # noqa: E402

PASS = []


def check(name, cond):
    PASS.append(cond)
    print(f"  {'ok' if cond else 'FAIL'}: {name}")


HI, LO = 0.97, 0.03  # strong persistence -> one prior vote is informative


def make_world(n_members=100, n_bills=60, rc_per_bill=8, seed=5):
    rng = np.random.default_rng(seed)
    stance = rng.choice([LO, HI], (n_members, n_bills))  # persistent per bill
    starts = rng.integers(0, 92, n_bills)  # bill's rollcalls on consecutive days
    rcs = [(b, j, starts[b] + j) for b in range(n_bills) for j in range(rc_per_bill)]
    rcs.sort(key=lambda t: (t[2], t[0]))  # rollnumber increases with time
    rows, p_true = [], []
    for rollnumber, (b, j, day) in enumerate(rcs):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(day))
        for m in range(n_members):
            p = stance[m, b]
            rows.append((118, "House", rollnumber, float(m),
                         100.0 if m < n_members // 2 else 200.0,
                         "On Passage", "hr", b, "legislation", date,
                         float(rng.random() < p)))
            p_true.append(p)
    df = pd.DataFrame(rows, columns=["congress", "chamber", "rollnumber", "icpsr",
                                     "party_code", "vote_question", "bill_type",
                                     "bill_no", "bill_category", "date", "vote"])
    bills = pd.DataFrame({"congress": 118, "bill_type": "hr",
                          "bill_no": np.arange(n_bills),
                          "text": "x", "sponsor_party": "D"})
    return df, bills, np.array(p_true)


def fit_eval(df, bills, use_billctx):
    cutoff = df["date"].quantile(0.8)
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    m = TextTower(k=4, use_text=False, batch_size=8192, lr=0.05,
                  max_epochs=120, min_epochs=25, bills_df=bills)
    m.use_billctx = use_billctx
    m.name = f"ctx={use_billctx}"
    m.fit(train)
    res = evaluate(m, train, test, "syn_ctx", "test")
    return m, train, test, res


def main():
    df, bills, p_true = make_world()

    _, train, test, res_no = fit_eval(df, bills, use_billctx=False)
    m_ctx, _, _, res_ctx = fit_eval(df, bills, use_billctx=True)
    ll_no, ll_ctx = res_no.overall["log_loss"], res_ctx.overall["log_loss"]
    bayes = metrics(test["vote"].to_numpy(),
                    p_true[df.index.isin(test.index)])["log_loss"]

    # one-observation posterior floor on covered rows: with equiprobable
    # stances, P(hi | prior=yea) = HI, so the best feature-based prediction
    # is q = HI^2 + LO^2 after a yea prior (symmetric after a nay)
    feats = m_ctx._bill_context(test)
    covered = feats[:, 0] == 1.0
    own = feats[:, 1]
    q = HI * HI + LO * LO
    p_floor = np.where(own > 0, q, np.where(own < 0, 1 - q, 0.5))
    y_test = test["vote"].to_numpy()
    floor_cov = metrics(y_test[covered], p_floor[covered])["log_loss"]
    p_test = np.asarray(m_ctx.predict_proba(test.drop(columns=["vote"])))
    ll_ctx_cov = metrics(y_test[covered], p_test[covered])["log_loss"]

    print(f"  no-ctx {ll_no:.4f} | ctx {ll_ctx:.4f} | bayes {bayes:.4f}")
    print(f"  covered rows ({covered.mean():.0%}): ctx {ll_ctx_cov:.4f} "
          f"| one-obs floor {floor_cov:.4f}")
    check("billctx beats no-context", ll_ctx < ll_no - 0.05)
    check("billctx reaches the one-observation floor on covered rows",
          ll_ctx_cov < floor_cov + 0.05)

    # falsification: reassign each ROLLCALL to a random bill (a per-bill
    # bijection would just rename bills; per-rollcall reassignment breaks
    # the series structure) — context features become noise and the model
    # must land back at the no-context loss
    rng = np.random.default_rng(0)
    n_bills = df["bill_no"].nunique()
    rc_bills = df[["rollnumber"]].drop_duplicates()
    rc_bills["bill_no"] = rng.integers(0, n_bills, len(rc_bills))
    df_perm = df.drop(columns=["bill_no"]).merge(rc_bills, on="rollnumber")
    _, _, _, res_perm = fit_eval(df_perm, bills, use_billctx=True)
    ll_perm = res_perm.overall["log_loss"]
    print(f"  permuted-bill ctx {ll_perm:.4f} (no-ctx {ll_no:.4f})")
    check("permutation kills the gain (back to no-context level)",
          ll_perm > ll_ctx + 0.03 and abs(ll_perm - ll_no) < 0.03)

    # E1b: residual stacking must capture the same signal without joint
    # training, and learn ~zero weights on the permuted world
    def base_factory():
        return TextTower(k=4, use_text=False, batch_size=8192, lr=0.05,
                         max_epochs=120, min_epochs=25, bills_df=bills)

    cutoff = df["date"].quantile(0.8)
    stack = CtxStack(base_factory=base_factory, name="ctx_stack_syn")
    stack.fit(df[df["date"] < cutoff])
    ll_stack = evaluate(stack, df[df["date"] < cutoff], df[df["date"] >= cutoff],
                        "syn_ctx", "test").overall["log_loss"]
    stack_p = CtxStack(base_factory=base_factory, name="ctx_stack_syn_perm")
    stack_p.fit(df_perm[df_perm["date"] < cutoff])
    ll_stack_p = evaluate(stack_p, df_perm[df_perm["date"] < cutoff],
                          df_perm[df_perm["date"] >= cutoff],
                          "syn_ctx", "test").overall["log_loss"]
    print(f"  stack {ll_stack:.4f} | stack-permuted {ll_stack_p:.4f}")
    check("stacked corrector captures the context gain", ll_stack < ll_no - 0.05)
    check("stacked corrector matches or beats joint training",
          ll_stack < ll_ctx + 0.02)
    check("stack learns ~nothing on permuted bills (stays at base)",
          abs(ll_stack_p - ll_no) < 0.03)

    n = len(PASS)
    if all(PASS):
        print(f"\nAll {n} bill-context checks passed.")
    else:
        print(f"\n{n - sum(PASS)} of {n} checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
