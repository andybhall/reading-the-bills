# rollcalls

An auto-researcher benchmark for predicting congressional roll-call votes.
The goal: build progressively better predictors of how members of Congress
vote, then extract interpretable member attributes (ideology, party loyalty,
issue positions, ...) from what the predictors learn.
Full project definition: `Notes/vision.md`. Decisions: `Notes/decisions.md`.

## Reproduce

```bash
python3 Code/00_download_data.py      # fetch Voteview raw data (~320MB)
python3 Code/01_build_votes.py        # clean panel -> Modified Data/
python3 Code/02_make_splits.py        # canonical 80/10/10 split (seed 42)
python3 Code/tests/test_harness.py    # 18 checks on synthetic data
python3 Code/run_benchmark.py         # evaluate models -> leaderboard
```

Requires Python 3.13+ with pandas, numpy, pyarrow, requests.

## Benchmark (Regime A: held-out cell completion, congresses 101–119)

Primary metric: test log loss. Contested = rollcalls with train minority
share ≥ 35%.

| model | test log loss | test acc | contested acc | APRE |
|---|---|---|---|---|
| constant_rate | 0.655 | 63.7% | 51.0% | -0.32 |
| member_rate | 0.629 | 64.9% | 55.3% | -0.28 |
| party_rollcall_rate | 0.206 | 91.7% | 92.8% | 0.70 |
| ideal_point_1d | 0.193 | 91.6% | 91.2% | 0.69 |
| nominate_logit (frozen DW-NOMINATE) | 0.164 | 93.3% | 93.9% | 0.76 |
| ideal_point_2d | 0.161 | 93.3% | 93.8% | 0.76 |
| **ideal_point_8d** | **0.153** | **93.8%** | **94.3%** | **0.77** |

Learned 1D positions (fit per chamber) correlate with DW-NOMINATE dim1 at
r = 0.98 in both chambers, and 0.84-0.91 *within* party; see
`Modified Data/results/idealpoint_validation.json`.

## Benchmark (Regime B: forecast, congresses 108–119)

Entire rollcalls held out by date within congress-chamber; nothing
contemporaneous with the vote is observable. Bill features from GovInfo
BILLSTATUS (title, policy area, subjects, sponsor, question type).

| model | test log loss | test acc | contested acc | APRE |
|---|---|---|---|---|
| constant_rate | 0.617 | 70.9% | 52.7% | -0.11 |
| party_congress_rate | 0.589 | 68.6% | 58.8% | -0.19 |
| meta_tower_8d (no text) | 0.511 | 72.3% | 60.7% | -0.05 |
| party_question_rate | 0.466 | 76.6% | 69.8% | 0.11 |
| text_tower_8d | 0.464 | 77.9% | 69.4% | 0.16 |
| member_question_rate | 0.459 | 76.0% | 69.2% | 0.09 |
| emb_tower_mq_8d (leakage-clean text) | 0.435 | 79.0% | 75.9% | 0.20 |
| emb2_tower_mq_8d (v2 rollcall-level text) | 0.423 | 80.5% | 76.8% | 0.26 |
| text_tower_mq_8d | 0.411 | 80.1% | 75.0% | 0.24 |
| emb2tfidf_mq_16d_tcal | 0.393 | 81.8% | 77.6% | 0.31 |
| **emb2_mlp_mq_16d_tcal** | **0.349** | **85.2%** | **84.1%** | **0.44** |
| emb3_mlp_mq_16d_tcal (Qwen3 encoder) | 0.360 | 84.2% | 82.3% | 0.40 |

(emb3 is marginally better on val, worse on test — encoder scale is not
the current bottleneck; see decisions.md.)

Key text findings (full history in leaderboard.csv):
- Rollcall-level text construction (v2: vote_question + vote_desc +
  pre-vote bill summary) beats bill-level text — gives content to
  nominations (13% of votes) and amendments.
- An MLP projection head adds large discrimination gains but needs a
  TEMPORAL internal dev slice (last 5% of train rollcalls) for early
  stopping + calibration; random in-period cells miscalibrate forecast
  models catastrophically (0.64 -> 0.35 log loss from this fix alone).
- The leakage-clean v2 leader (emb2_mlp_mq_16d_tcal) uses only text
  knowable at vote time.

## Validation protocol

- **Prospective holdout (headline):** a frozen, sha256-pinned model
  (`Modified Data/results/frozen/`) fit through 2026-06-09 is scored only
  on rollcalls occurring after that date (`Code/14_score_prospective.py`;
  rules in `Notes/prospective_protocol.md`). Immune to adaptive overfitting
  by construction.
- **Within-congress forecast** (the table above): held-out future rollcalls
  within each congress. Test viewed across model iterations — trust the
  rankings; quote the prospective number externally.
- **Congress-out** (train 108-116, test 118): the champion drops to 61.6%
  acc / 0.670 log loss — cross-congress transfer (majority flips, no
  same-congress member history) is an open problem, documented in
  `Notes/decisions.md`.
- Error analysis (`Code/15_error_analysis.py`): remaining loss concentrates
  in amendments (29% of loss; most lack amendment-specific text) and long
  bills (33%; summary truncation) — these set the text roadmap.

## Issue-specific positions

`Notes/memo_issue_positions.md`: per-topic ideal points (12 policy areas +
nominations, validated — recovers libertarian non-interventionists and
hawkish Democrats from votes alone) and embedding probes (exploratory;
failed party-stereotype validation, diagnosis + path to validation in the
memo).

## Member signals

First extracted attribute: **party-loyalty residual** (loyalty on
party-unity votes beyond what the member's ideal point implies) —
`Notes/memo_party_loyalty.md`. Recovers canonical mavericks (Murkowski,
Collins, Manchin, Fitzpatrick) and flags multidimensional members whose
1D position misleads (C. Smith, Massie).

Numbers regenerate from `Modified Data/results/leaderboard.csv`; do not edit
by hand.

## Repo layout

- `Original Data/voteview/` — raw Voteview CSVs + download manifest (never modified)
- `Modified Data/` — panel, splits, results (all produced by `Code/`)
- `Code/` — pipeline, harness, models, tests
- `Notes/` — vision, decision log
- `logs/` — session logs of AI-assisted development (see `CLAUDE.md`)

Data files are gitignored (regenerate via the pipeline); the manifest, code,
and results are tracked.
