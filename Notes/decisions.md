# Decision Log

Consequential choices, why they were made, and what would change them.
(Newest at bottom. Every entry flagged to Andy in the session log.)

**2026-06-11 — Python, not Stata.** This is an ML benchmark/harness with
custom model classes, hashing, parquet IO — the SOUL.md "affirmative case"
for a different language applies. Any downstream regression-style analyses
of extracted signals can still be done in Stata.

**2026-06-11 — Congress scope: 101–119 (1989–present).** Keeps downloads and
iteration fast (~11M votes) while covering the modern polarized era plus
enough history for temporal splits. Extending back is a one-line change in
`Code/00_download_data.py`; all code is congress-agnostic.

**2026-06-11 — Binary target: yea (1,2,3) vs nay (4,5,6); present/abstain
(7,8,9) excluded from the target.** Standard in the ideal-point literature.
Abstention rows are retained in the panel; modeling abstention is a future
extension. Code 0 (not a member at vote time) dropped. Presidents' announced
positions (icpsr ≥ 99000) dropped from the panel.

**2026-06-11 — Initial benchmark is Regime A (completion): random 10% of
cells held out within congress, other votes on the same rollcall observable.**
This isolates representation quality with no feature engineering and matches
how ideal-point models are evaluated. Regime B (temporal forecast with bill
features) is Module 3; it is the more decision-relevant task but needs bill
text infrastructure.

**2026-06-11 — Primary metric: log loss on the Regime A test set.** Proper
scoring rule, rewards calibration, fixed in advance to prevent metric
shopping. Accuracy on contested rollcalls (train minority share ≥ 35%) is
the key secondary because lopsided votes inflate overall accuracy. APRE
reported for comparability with the NOMINATE literature.

**2026-06-11 — Known data quirk, accepted.** Our per-rollcall yea tallies
match the official `yea_count` field on only ~80% of rollcalls, but
mismatches are tiny (median |diff| = 1 vote, max 6) — a known discrepancy
between clerk tallies and Voteview's individual records (Speaker non-votes,
vote changes). Individual cast codes are the labels; aggregate counts are
metadata only. No action needed.

**2026-06-11 — Smoothing: baseline rate estimates use add-k (k=5) shrinkage
toward the parent rate.** Prevents log-loss blowups on thin cells; k chosen
a priori, not tuned on test.

**2026-06-11 — Ideal points are fit pooled for prediction but per chamber
(and with party-sign initialization) for interpretation.** Two identification
failures diagnosed and fixed in session 02: (1) House and Senate share no
rollcalls, so the pooled fit flipped the Senate block's sign (only ~95
chamber-switchers tie the blocks); (2) within the Senate-only fit, the axis
orientation twisted smoothly across eras (corr with NOMINATE going -0.95 to
+0.95 from the 101st to 119th) — a local optimum with identical predictive
fit. Lesson recorded: predictive accuracy does not identify orientation;
interpretation requires explicit identification constraints. Party-sign init
(+0.5 R / -0.5 D) pins orientation only — within-party ordering, which is
what we validate against NOMINATE, remains free, so validation is not
circular. Benchmark numbers are unaffected (pooled models, prediction only).

**2026-06-11 — Ideal-point hyperparameters fixed a priori** (lr 0.05, AdamW
wd 1e-4, batch 131k, early stop on internal 2% train slice, patience 3).
Not tuned on val/test. k ∈ {1, 2, 8} compared on val as designed.

**2026-06-11 — Regime B forecast split: congresses 108–119, within
congress-chamber temporal cutoffs (80/10/10 by rollcall date), entire
rollcalls held out.** 108 because GovInfo BILLSTATUS begins there.
Within-congress cutoffs keep member composition stable; whole-congress-out
generalization is a separate harder eval for later. Split format matches
Regime A so the runner is unchanged.

**2026-06-11 — Contested stratum redefined to use all observed votes
(train + eval) on a rollcall** rather than train only, because forecast
splits hold out whole rollcalls (no train votes exist). Purely a reporting
stratum; labels never enter predictions.

**2026-06-11 — Bill-text features v1 = title + policy area + legislative
subjects (TF-IDF→SVD), question bucket, bill category, sponsor party.**
Known leakage caveat, accepted for v1 and flagged for Andy: subjects and
policy areas are assigned/revisable by CRS after introduction (possibly
after votes). A strict pre-vote variant — dated summary versions filtered to
action_date < vote date — is planned. Summary versions are already stored
with dates in bills.parquet to support this.

**2026-06-11 — Calibration is fit on the internal early-stop slice, never
val/test.** Temperature+bias transfer from in-period cells to future
rollcalls is imperfect but corrects global overconfidence; it cannot tune
to the eval distribution. The member x question offset gives the tower the
count-baseline's structure as a floor; its weight is learned.

**2026-06-11 — Both text variants reported: TF-IDF(title/subjects) and
leakage-clean sentence embeddings of pre-vote summary versions.** The clean
variant scores 0.435 vs 0.411 test log loss; the ~0.024 gap is an upper
bound on what post-vote subject/policy assignment leaks into TF-IDF. For
any externally-reported "true forecasting" claim, use the clean number.

**2026-06-11 — Forecast models use a TEMPORAL internal dev slice (last 5%
of train rollcalls by date) for early stopping and calibration.** Random
in-train cells are in-sample for rollcall parameters and miscalibrate
forecast models (MLP head: 82% accuracy but 0.64 log loss before the fix,
0.349 after). The slice remains strictly inside the train window — val and
test are untouched. Regime A models keep the random-cell slice (there the
eval cells share rollcalls with train by design).

**2026-06-11 — Embedding probe scores are NOT validated member attributes.**
Two iterations failed/ambiguously passed party-stereotype sanity checks
(documented in Notes/memo_issue_positions.md). Kept as exploratory output;
validation path = external interest-group ratings. The per-topic ideal
points ARE validated and are the issue-position instrument of record.

**2026-06-11 — Prospective validation protocol established
(Notes/prospective_protocol.md).** Frozen model artifact (sha256-pinned)
fit through the 2026-06-09 data snapshot; scored only on rollcalls after
that date. This is the project's headline forecasting number for any
external claim; development test sets have been observed repeatedly and
carry adaptive risk.

**2026-06-11 — Congress-out eval results carry two caveats.** (1) The
count baselines (member/party x question rate) key on congress and so
collapse to the constant model on an unseen congress — their congressout118
leaderboard rows are NOT real baselines; a fair cross-congress baseline
needs majority-status-aware features. (2) The champion model drops from
85.2% (within-congress) to 61.6% (congress-out): cross-congress transfer
is largely unsolved — the member x question offset collapses and the
text-to-side mapping does not survive a majority flip. Reported as a
limitation, not buried.

**2026-06-11 — Negative result: as-of-date cosponsor count/balance features
hurt the forecast model** (test 0.377 vs 0.349 without; val worse too).
Kept in the registry for reproducibility; champion remains
emb2_mlp_mq_16d_tcal. Possible causes (not yet investigated): early-life
cosponsor counts are noisy at vote time; optimization interference.

**2026-06-12 — Module A (encoder upgrade) verdict: representation is not
the bottleneck.** Qwen3-Embedding-0.6B (1024d, 6000-char budget) vs MiniLM
(384d, 1500 chars), all else identical: val marginally better (0.424 vs
0.427), test marginally worse (0.360 vs 0.349; rerun variance ~0.003).
Crucially the long-bill loss bucket got WORSE with more text (test 0.568
vs 0.518) — the long-bill hypothesis (truncation) is rejected; omnibus
votes hinge on deal context, not summarizable content. Selection-by-val
makes v3 nominally preferred, but the conflict is flagged and the frozen
prospective artifact (v2-based) is unchanged. Resource priority moves to
amendment text (M-B) and context features (M-D), not bigger encoders.

**2026-07-03 — Literature baselines added to the forecast regimes
(Paper A horse race).** GB-2011 two-stage analog (1D ideal points +
ridge from text to bill parameters; TF-IDF and modern-embedding
variants, each raw and temperature-calibrated), Kraft-2016 bilinear
analog, and a NOMINATE x metadata logit. Result: none beats the
member x question count baseline in either forecast regime; raw GB is
worse than the CONSTANT model on log loss (0.80-1.05) despite ~66-72%
accuracy — miscalibration the 2011-16 literature's accuracy metrics
could not see. Calibration recovers 0.24-0.37 log loss; discrimination
(AUC 0.69 vs champion 0.90) remains the binding constraint. Fairness
caveat: our GB analog is two-stage ridge, not joint generative
inference — quote the tcal variant as best-case classical and AUC for
architecture comparisons. Bug history: the k=1 production config
initially SIGKILLed via a silent (n,1)x(n,) -> (n,n) numpy broadcast
(2.5TB lazy alloc, jetsam kill, no traceback) because sklearn ravels
single-column ridge targets; fixed with shape-guarded prediction and a
k=1 test. Lesson: test the production configuration, not just the
convenient one.

**2026-07-03 — Fair congress-out baselines: majority-status keys
transfer, member history ANTI-transfers.** `majority_question_rate`
(party-is-majority x chamber x qbucket, pooled across congresses;
majority from member composition, knowable pre-vote) scores 0.617 on
congressout118 — beating the champion's 0.670 on log loss (champion
keeps the accuracy edge, 61.6% vs 55.1%). `member_pooled_question_rate`
scores 0.740: pooled member history embeds majority-era behavior and
is poisoned by the 117->118 House flip. Synthetic two-world test
(tests/test_transfer_baselines.py) isolates both mechanisms. The
congressout leaderboard rows for congress-keyed count baselines remain
non-baselines; majority_question_rate is the honest floor.

**2026-07-03 — M-D attempt: majority-status features do NOT fix
congress-out transfer. Negative result; champion unchanged.**
`emb2maj_mlp_mq_16d_tcal` (champion + in-majority indicator + pooled
majority x question logit as z features): congressout118 test 0.696 /
58.9% — WORSE than the plain champion (0.670 / 61.6%) and still above
the majority baseline's 0.617; forecast108_119 test 0.361 vs 0.349 —
a small within-congress cost too. Interpretation: with the mq offset
available in training, the model has no incentive to weight the
majority columns, and at eval time they cannot substitute for the
collapsed member offsets and flipped text-to-side mapping. Fixing
transfer likely requires majority-conditioned TRAINING (e.g., training
across multiple majority regimes with member offsets keyed to majority
status), not feature addition — deferred beyond Paper A; reported as
a limitation with an honest baseline floor.

**2026-06-11 — Synthetic-test design lesson recorded:** a bill's text must
reveal the *direction* of the proposal (which side of the cutpoint), not
just its topic, for text features to beat metadata baselines. First version
of the text-tower test accidentally made direction unobservable and the
task information-theoretically impossible. Real bill titles do encode
direction ("To repeal...", "To expand..."), but imperfectly — relevant when
interpreting real-data gains.
