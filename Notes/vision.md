# Roll Calls: An Auto-Researcher for Predicting Congressional Votes

*Drafted 2026-06-11. Status: project definition, Module 1.*

## The idea

Build a Karpathy-style "auto-researcher" around one crisp prediction problem:
**predict how each member of Congress votes on each roll call.** A single
benchmark with fixed splits and cheap evaluation lets us (and eventually,
autonomous agent loops) iterate rapidly on models, with a leaderboard keeping
score.

The accuracy number is not the scientific product. The product is what a
strong predictor must learn in order to predict well. A model that beats the
classical spatial (NOMINATE-style) model has, by construction, found
systematic structure in congressional behavior beyond one or two ideology
dimensions. Interrogating that model yields new measurable attributes of
members of Congress:

- ideology (recovering and refining NOMINATE as a validation step)
- party loyalty / leadership alignment, separate from ideology
- issue-specific positions (trade, immigration, defense, ...) once roll calls
  are linked to topics and bill text
- procedural vs. substantive voting behavior
- parochial/district interests
- electoral-cycle responsiveness (does behavior shift as re-election nears?)
- within-career drift (freshman socialization, retirement effects)

Each candidate signal becomes a short research memo: how it was extracted,
how it is validated against external data, and what it shows. The best memos
seed standalone papers.

## Why this is a good auto-researcher substrate

1. **Huge, clean labels.** Millions of member×rollcall observations from
   Voteview, 1789-present, updated continuously.
2. **Cheap, unambiguous eval.** Held-out log loss / accuracy. No human
   judging, no noisy reward.
3. **Strong, well-understood baselines.** Party majority and NOMINATE set a
   high bar (~88-93% accuracy historically), so progress is meaningful, not
   trivial.
4. **Headroom from modern methods.** Embeddings, sequence models over vote
   histories, and bill-text features are barely exploited in the literature.
5. **Interpretability has direct social-science value** — the impact equation
   is satisfied on both factors: big question (what drives legislative
   behavior, and how should we measure legislators?) and real progress
   (better prediction is an objective yardstick that the field lacks).

## Task definition

Two evaluation regimes, in order of construction:

**Regime A — completion (Module 1).** Hold out a random 10% of
(member, rollcall) cells within each congress-chamber; predict the held-out
votes. Other members' votes on the same roll call are observable at
prediction time. This is the standard psychometric evaluation for
ideal-point models; it isolates representation quality and requires no
features beyond the vote matrix.

**Regime B — forecast (later).** Predict votes on roll calls occurring after
a time cutoff, given only prior votes and bill metadata/text. Nothing
contemporaneous with the vote is observable. Harder and more "real"
(useful for prospective prediction); requires bill-level features.

Target: binary yea (1) vs. nay (0). Present/abstain codes are excluded from
the prediction target (standard in the literature) but retained in the data;
modeling abstention is a candidate future signal.

## Metrics

- **Primary: log loss** (proper scoring rule; rewards calibration).
- Secondary: accuracy, AUC, Brier; **accuracy on contested votes** (roll
  calls with minority share ≥ 35% in training data) — lopsided votes make
  overall accuracy easy and flatter weak models.
- Comparability: APRE (aggregate proportional reduction in error), the
  NOMINATE literature's standard.
- All metrics reported overall and by chamber, congress, party, and
  contested/lopsided stratum.

## Architecture

```
Original Data/voteview/     raw CSVs + manifest.json (never modified)
Modified Data/              parquet panel, splits, results, leaderboard
Code/
  00_download_data.py       acquire raw data, verify, manifest
  01_build_votes.py         clean panel: one row per member x rollcall
  02_make_splits.py         canonical deterministic splits (seeded)
  harness.py                Model protocol, metrics, evaluation loop
  baselines.py              registered models
  run_benchmark.py          evaluate model(s) -> leaderboard
  tests/                    tests for metrics, splits, leakage
```

Models implement `fit(train) / predict_proba(test)`. The runner records
model name, code version, split hash, metrics, and timestamp to
`Modified Data/results/` and a cumulative leaderboard. Fixed seeds
throughout; splits are content-hashed so any drift is detected.

**Guardrails for (eventual) autonomous iteration:** agents never touch
`Original Data/`; splits are frozen; validation set for model selection,
test set only via the runner; every experiment logs config + code diff +
result. No test-set peeking, no metric shopping — the primary metric is
fixed in advance (log loss, Regime A test set).

## Roadmap

- **M1 (this session):** data acquisition + verification, clean panel,
  canonical splits, harness, trivial baselines (always-yea, member rate,
  party-rollcall majority). Pipeline proven end-to-end.
- **M2:** spatial models — logistic ideal points (1D, 2D, kD) fit by
  alternating/SGD; NOMINATE-score logit as reference; matrix factorization.
  Target: beat party-majority baseline on log loss.
- **M3:** Regime B (temporal forecast split) + bill metadata/text features
  (Congress.gov API, bill summaries, sponsor, topic codes).
- **M4:** richer models — member/bill embeddings, sequence models over vote
  history, possibly LLM-derived bill representations.
- **M5:** signal extraction — interpret learned representations, validate
  candidate member attributes against external benchmarks (NOMINATE, CFscores,
  district presidential vote, interest-group ratings), write memos.
- **M6:** the autonomous loop — agent proposes experiment, runs, logs,
  reports; human reviews at milestones.

## Data sources

- **Voteview** (Lewis et al.): members, roll calls, individual votes,
  NOMINATE scores. Primary source, downloaded with manifest.
- Later: Congress.gov / GovInfo (bill text), Comparative Agendas Project
  (topic codes), DIME/CFscores (validation), district returns (validation).
