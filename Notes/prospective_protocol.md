# Prospective Validation Protocol

*Established 2026-06-11. This is the project's untouchable yardstick.*

## Rules

1. The frozen model (`Modified Data/results/frozen/prospective_model.pkl`,
   sha256 in the adjacent meta file) was fit on all yea/nay votes through
   the data snapshot date recorded in `prospective_model_meta.json`.
   **It is never refit, recalibrated, or replaced retroactively.**
2. The prospective evaluation set is every rollcall dated strictly AFTER
   the snapshot. These votes did not exist when the model was frozen, so
   no development decision can have adapted to them.
3. `Code/14_score_prospective.py` may be run any number of times as new
   votes occur (it is append-only measurement, not development). Metrics
   on the ledger are reported as-is — no filtering, no model selection.
4. If we later freeze an improved model, it gets a NEW artifact + meta
   file and its prospective clock starts at its own snapshot. Old frozen
   models keep being scored; we report both. No silent replacement.
5. Development work (new features, architectures) continues against the
   val sets of the development splits. The prospective number is quoted in
   any external write-up as the headline forecasting result.

## Why

The development test sets (forecast108_119, congressout118) are held out
but have been observed repeatedly across model iterations. The prospective
set is immune to that adaptive risk by construction: the future hadn't
happened yet.

## Current frozen artifacts

| artifact | snapshot | architecture |
|---|---|---|
| prospective_model.pkl | see meta json | emb2_mlp_mq_16d_tcal (k=16, MLP head, mq offset, temporal calibration) |
| prospective_model_v2.pkl | see v2 meta json | blend3_mlp_tfidf_emb3_tcal (S5 winner, 2026-07-03; logit blend of emb2-MLP + TF-IDF + Qwen-emb3 towers; weights in meta) |

**v2 window caveat (recorded 2026-07-03):** v2's data snapshot equals
v1's (2026-06-09) because both fit on the same frozen panel, so v2 is
scoreable on the same 2026-06-10+ window and directly comparable to v1.
However, that window's AGGREGATE v1 metrics were observed once (P1,
2026-07-03 morning) before v2 was frozen; the sprint's model selection
used only development val sets, but strict readers should treat votes
occurring after 2026-07-03 as v2's pristine prospective window. Both
numbers are reported; scored by 21_score_prospective_v2.py.
