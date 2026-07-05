# Memo: evaluation of coarse review round 2

*2026-07-05. 11 overall issues, 25 detailed comments. The review engages
the revised draft seriously; most comments are precision demands we
accept. One alleged bug (loyalty scale) was checked against code and is
a NOTATION error only — the pipeline computes the party-support recoding
the reviewer prescribes; the paper's equation is corrected to match.*

## Overall issues

**O1 feature provenance + conservative ablation.** PARTIAL. Provenance
table strengthened per-feature. The requested "conservative
leakage-clean rerun" already exists in our tables and gains a sentence:
the MiniLM tower alone uses only strictly pre-vote text (question,
description, dated summaries — no CRS-revisable fields) and scores
0.349; the blend's 0.324 includes a TF-IDF component with revisable
subject metadata. The paper now reports 0.349 explicitly as the
leakage-conservative bound.

**O2 measurement provenance map.** ACCEPT. New table: each quantity ->
estimator, training data, text used?, in-sample vs forecast. Abstract's
"its internals recover" corrected: positions/issues/cutpoints come from
benchmark spatial fits; ensemble supplies forecasts, surprises, and the
fit standard.

**O3 external construct validation.** PARTIAL (as r1): deferred to the
companion's data program, now stated in one candid sentence; measures
labeled "validated for convergence and face validity; external
validation is the companion's program."

**O4 uncertainty for measures.** PARTIAL. Added: rollcall-clustered
bootstrap SEs for the labeled loyalty extremes (caption states all
exceed 3x clustered SE, if they do — computed, not asserted);
identification-threshold sensitivity for the agenda shares (0.25/0.50);
binomial wording fixed to "standard error" with dependence caveat.
Full posterior uncertainty for all member/bill scores: revision-scale
work, flagged.

**O5 flip framing.** ACCEPT. Abstract's "no model survives" scoped to
the procedural-role component; accuracy-vs-log-loss divergence noted
honestly; Senate replication flagged as future work.

**O6 amendment hypotheses.** ACCEPT wording: the three-hypothesis
distinction (content uninformative / purpose fields thin / actor-context
dominant) stated, with our evidence assigned to the third and full
amendment text explicitly listed as untested data.

**O7 surprise estimands.** ACCEPT. Remaining "saw coming" phrasing
purged from Sections 1 and 6 (grep-verified); the temporal-test-window
surprises (true holdout) noted as the forecast-based variant; release
carries both, labeled.

**O8 cutpoint sensitivity.** ACCEPT cheap version: agenda-control shares
recomputed passage-only and at thresholds 0.25/0.50 (macros); 2D/chamber
variants flagged.

**O9 paired uncertainty for horse race.** ACCEPT: paired
rollcall-cluster bootstrap for the key forecast-test margins
(blend vs two-tower vs single tower vs member table), from saved
predictions; reported in the audit section.

**O10 model-spec table.** ACCEPT: appendix specification table + a
replication-target paragraph (commands, compute budget).

**O11 informational-bound proposition + demonstration.** PARTIAL: the
identifiability proposition added in two sentences; for the
demonstration we point to the repository's synthetic recovery suite
(planted direction-revealing text, planted majority-role worlds,
planted within-bill persistence — each recovered by exactly the models
with better held-out scores), summarized in the appendix rather than
building a new semi-synthetic panel.

## Detailed comments: all 25 ACCEPTED

(1) scale convention stated + threshold sensitivity; (2) notation fixed
to the code's party-support recoding — code verified correct; (3)
post-snapshot vs post-freeze separated, with the v1 post-June-12 subset
scored from the ledger; (4) "average-member cutpoint" named; (5) intro
panel/text-era distinction; (6) congress-out labeled skipped-congress
design (117th = validation); (7) "rules out" -> "provides no evidence
of"; (8) worked-example sentence separates forecast quantities from
realized-vote diagnostics; (9) both metrics spelled out; (10) amendment
deltas quoted with numbers (macros from leaderboard); (11) dashboard
labeled a single-tower approximation up front; (12) associational
language; (13) "enriched for recognizable episodes"; (14) NOMINATE
info-set stated (career scores include the test congress — retrospective
measurement comparison); (15) "identified rollcalls" + excluded count;
(16) estimand conditional on recorded yea/nay; (17) "9,451 individual
member-vote decisions"; (18) "held-out vote completion, not temporal
forecasting"; (19) SE wording + dependence; (20) nominations coding
rule; (21) caption formula -log p(realized); (22) "majority-direction
baseline," not chance; (23) table column labels; (24) dashboard logistic
on probabilities with formula; (25) representation language for the
amendment mechanism.
