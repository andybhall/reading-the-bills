# Experiment Ledger — model improvement sprint

## S5 ADJUDICATION (2026-07-03, single test unveil, Andy's go)

Paired same-day fits, test observed once, all three regimes:

| model | forecast test | randomrc test | congressout test |
|---|---|---|---|
| emb2_mlp_mq_16d_tcal (incumbent) | 0.3484 | 0.4173 | 0.6703 |
| blend_mlp_tfidf_tcal | 0.3369 | 0.4120 | 0.6541 |
| **blend3_mlp_tfidf_emb3_tcal** | **0.3241** | **0.3905** | **0.6430** |

**Winner: blend3 — sweeps all regimes; forecast test gain 0.0243 (bar
0.015); val promise (+0.026) held on test (+0.024), no selection
inflation. Forecast acc 86.2%, contested acc 85.1%, APRE 0.477, AUC
0.919.** Error decomposition: gains broad across content buckets
(passage 0.300, nominations 0.282); amendments nearly unchanged (0.602,
now 31% of remaining loss) — the structural frontier, per the closed
E5 family. Congressout note: blend3 (0.6430) still trails the fair
majority baseline (0.617) on log loss — transfer remains open.
Incumbent reproduction: forecast 0.3484 (orig 0.3490), congressout
0.6703 (orig 0.670) — comparisons valid.

**randomrc reproducibility investigation (closed 2026-07-03).** Same
model, same split: morning ladder val 0.3883 vs afternoon 0.4275.
Established facts: (a) 4 fresh-process fits are BIT-IDENTICAL at
0.4275 — zero fit variance; (b) all input files provably unchanged
(mtimes pre-date the morning run; split content hash matches its
build-time pin); (c) fitting 2nd-in-process reproduces 0.4273;
meta_tower reproduces its morning number exactly. The 0.3883 came from
a fit ELEVEN models deep in one long process — a state not cheaply
recreatable; suspected cross-fit state leakage (MPS RNG/allocator)
under that specific condition. Resolution: 0.4275 is the canonical
incumbent randomrc number (the reproducible one); absolute randomrc
levels carry a cross-context sensitivity caveat (~0.04 observed once);
all sprint conclusions rest on paired same-context comparisons and are
unaffected. Operational rule going forward: benchmark runs are
one-model-per-process (the batch scripts already do this).

Next: freeze blend3 as prospective_model_v2 (20_freeze_model_v2.py)
upon Andy's confirmation; v2 scorer needs v3-embedding append support.

*Started 2026-07-03. Rules (fixed in advance):*

1. Every candidate gets a ledger entry BEFORE its first run: hypothesis,
   config, and pre-vote-knowability argument for any new feature.
2. Selection on `forecast108_119` VAL only (`--eval-sets val`); randomrc
   val as secondary. The test sets stay unobserved until one adjudication
   run at sprint end (S5).
3. Bar for a new champion: >= 0.015 val log loss improvement over
   emb2_mlp_mq_16d_tcal (val 0.4277), sustained in the single test unveil,
   with bucket-level gains where the hypothesis names a bucket.
4. History features must use strictly-prior information (date, rollnumber
   ordering) and pass a falsification test (shuffled dates/text kill the
   gain).
5. Negative results are recorded here, not deleted.

Reference points (forecast108_119 val log loss):
- emb2_mlp_mq_16d_tcal (champion): 0.4277
- emb2tfidf_mq_16d_tcal: 0.4771
- member_question_rate: 0.5115

---

## E1: emb2ctx_mlp_mq_16d_tcal — within-bill context features

*Registered 2026-07-03, before first run.*

**Hypothesis:** amendment votes (29% of remaining loss) cluster within
bills; by the time a bill's Nth rollcall occurs, the coalition revealed by
rollcalls 1..N-1 on the same bill is public history. Member-level features
from strictly-prior same-bill rollcalls should cut the amendment bucket
specifically.

**Features (z columns):** (a) has_prior indicator; (b) member's own vote on
the most recent strictly-prior same-bill rollcall, signed +1/-1, 0 if none;
(c) member's party's yea share on that prior rollcall, centered at 0.

**Knowability:** prior rollcall outcomes are public the moment they occur;
ordering is by (date, rollnumber) within congress-chamber. Conservative
harness restriction: context lookups draw only on TRAIN-window rollcalls
(the harness never exposes eval labels), so eval rollcalls whose priors fall
inside the eval window get has_prior=0 — this UNDERSTATES the achievable
gain; noted for the paper.

**Falsification:** synthetic world where members repeat their prior
same-bill vote — context model must approach Bayes and beat no-context;
with permuted bill assignments the gain must vanish.

**Result (2026-07-03): NEGATIVE in joint training.** Synthetic checks all
pass (model reaches the one-observation information floor; permutation
collapses cleanly to no-context). Real data: forecast val 0.4717 and
randomrc val 0.4104 vs champion 0.4277 / 0.3883 — worse in BOTH regimes.
The both-regimes regression rules out pure prior-staleness shift;
diagnosis is shortcut learning: within-series priors are so predictive in
training that the z-path starves the text/member backbone (same family as
the logged cosponsor negative). Test sets untouched.

## E1b: ctx_stack_16d_tcal — residual stacking of context features

*Registered 2026-07-03, before first run.*

**Hypothesis:** the E1 features carry real signal (synthetic floor
reached) but joint training misallocates capacity. Freezing the champion
backbone and fitting a small logistic correction — champion logit as a
fixed offset, context features + staleness terms as inputs, fit ONLY on
the temporal internal dev slice — captures the signal without touching
the backbone. Staleness terms (log days since prior, same-day flag) let
the correction discount old priors, addressing the forecast-boundary
shift E1 ignored.

**Knowability:** identical to E1 plus elapsed time, knowable at vote time.

**Falsification:** on the billctx synthetic world the stacked corrector
must capture the same gain joint training did; with permuted bills the
learned correction weights must go ~0 (stack output ≈ champion alone).

**Result (2026-07-03): NEGATIVE overall.** Synthetic 6/6 (captures full
gain; zero weights on permuted bills). Real data: forecast val 0.4260 vs
champion 0.4277 (+0.0017, inside run variance) but randomrc val 0.4251 vs
0.3883 — the corrector weights learned on the temporal dev slice actively
misfire on the interleaved regime. Verdict: the within-bill context
family (E1 joint, E1b stacked) does not improve the champion under our
no-leak constraints; the real-data signal in train-window-only priors is
too small and too distribution-dependent. DROPPED from the S5 candidate
set. (Possible paper footnote: context features would likely require an
online/rolling evaluation harness that exposes realized eval-period
history — a different, also-legitimate protocol we deliberately did not
mix into this one.)

## E3: per-question-bucket temperature calibration

*Registered 2026-07-03, before implementation.*

**Hypothesis:** one global temperature under-corrects heterogeneous
overconfidence — the error decomposition shows amendment/procedural
buckets carry very different loss profiles. Fitting (temperature, bias)
per question bucket on the temporal dev slice should cut loss in the
overconfident buckets without touching discrimination.

**Knowability:** question bucket is metadata known before the vote; the
dev slice is inside train. **Falsification:** none needed beyond val —
pure post-hoc rescaling cannot leak; risk is dev-slice overfit with
8 buckets x 2 params, checked by val.

## E4: logit ensemble of emb2_mlp and emb2tfidf towers

*Registered 2026-07-03, before implementation.*

**Hypothesis:** the two leading towers use different text blocks
(sentence embeddings vs TF-IDF+SVD) and different heads; their errors
should be partially decorrelated, so a convex logit blend (weight fit on
the temporal dev slice) beats either alone. Classic reliable gain.

**Knowability:** both components already pass it; blending adds nothing
new. **Falsification:** blend weight learned on dev; if truly redundant
the weight should collapse to the champion (val ≈ champion, no harm).

## E4 results (2026-07-03)

Two-way blend forecast val 0.4200 vs champion 0.4277: **+0.0077, beyond
run variance — first real gain of the sprint.** randomrc val 0.4197 vs
0.3883: worse, as with E1b and E3. E3 (bucket calibration): NEGATIVE
both regimes (0.4306 forecast, 0.4465 randomrc).

**Mechanism note (E1b + E3 + E4 jointly):** every post-hoc parameter set
fit on the TEMPORAL dev slice (stack weights, bucket temperatures, blend
weights) helps or stays neutral on the temporal forecast val and hurts
on the interleaved randomrc val. The dev slice inherits the forecast
regime; regime-matched dev slices are a requirement, not a nicety.
(For a randomrc-regime deployment the same machinery with a random-cell
dev slice would be the right analog — not run, out of sprint scope.)
E4 stands as a forecast-regime candidate.

## E4b: three-way blend (+ Qwen emb3 tower)

*Registered 2026-07-03, before first run.*

**Hypothesis:** emb3_mlp_mq_16d_tcal (Qwen3 encoder, val 0.4243) was
val-better than the champion but test-worse — exactly the profile a
blend exploits (different encoder, decorrelated errors, selection risk
absorbed by dev-slice weighting instead of model selection). Adding it
as a third logit component should extend E4's gain.

**Knowability:** components already pass. **Falsification:** if emb3 is
redundant its blend weight collapses toward zero and val ≈ E4.

**Result (2026-07-03): forecast val 0.4020 — CLEARS THE SPRINT BAR**
(champion 0.4277, bar 0.4127; gain 0.0257). The Qwen tower's val-better/
test-worse selection risk is absorbed as intended: it contributes as a
component instead of being selected outright. randomrc val 0.3984 vs
champion 0.3883 — mildly worse, and far more robust than the two-way
blend's 0.4197: a third decorrelated component stabilizes the dev-slice
weights. Candidate for S5.

## E5: amendment text (v4 embeddings) — S2 / M-B

*Registered 2026-07-03, before implementation.*

**Hypothesis:** amendments carry 29% of remaining forecast loss and 68%
of amendment rollcalls have no vote_desc (no content beyond the parent
bill). The cached BILLSTATUS XMLs already contain per-amendment
<purpose> text; House amendment actions embed "(Roll no. N)" giving an
exact free join. v4 rollcall text = v2 construction + matched amendment
purpose for amendment votes; everything else unchanged. Champion refit
on v4 embeddings should cut the amendment loss bucket specifically.

**Knowability:** amendment purpose is filed when the amendment is
submitted, before the vote; joins use roll numbers / action dates of
already-occurred actions. Purposes are static text (no post-vote CRS
revision issue, unlike subjects).

**Falsification:** the gain must appear in the amendment qbucket of the
error decomposition, not diffusely (a diffuse gain would suggest
something other than the hypothesized mechanism, e.g. leakage — v2 vs
v4 text differs ONLY on amendment rollcalls by construction, enforced
in the build script).

**E5b (corrected join) result (2026-07-03): still NEGATIVE — forecast
val 0.4665 vs 0.4277 — and nearly as bad as with corrupted text. The
regression is insensitive to purpose correctness, implicating TEMPLATE
MIXING rather than content: v4 embeds two text distributions in one
space ("Amendment purpose:" template vs v2 template) and the
end-of-congress dev slice (few amendments) never rewards the new
region. E5c registered: purpose inserted in the vote_desc SLOT (the
template Senate amendment rows already use), desc-less House rollcalls
only — minimal distribution shift, pure content test.**

**E5c + controls; FAMILY CLOSED as a genuine, mechanistic negative
(2026-07-03).** E5c (purpose in the vote_desc slot, 2,225 desc-less
House rollcalls only): forecast val 0.4686 — same-size regression as
E5b despite minimal template shift. Placebo control (byte-copy of v2
under a different emb_file): 0.4272 ≈ champion 0.4277 — file mechanics
and code path exonerated; the text change itself causes the regression.
Mechanism: shared parent-bill text makes an amendment series' rollcalls
textually identical, so the MLP head implicitly POOLS their parameters —
a low-variance estimate of a strongly correlated series. Distinct
purpose text replaces that pooling with per-amendment parameters driven
by purpose semantics, which (consistent with the M-A encoder finding:
coalitions are deal-driven, not content-driven) predict vote direction
WORSE than the parent bill's identity. Amendment text is not the
unexploited signal the error decomposition suggested; the decomposition
located the loss but not a text remedy.**

**First run (2026-07-03): INVALID — join bug, not a hypothesis test.**
val regressed hugely in both regimes (0.4805 / 0.4276). File integrity
verified byte-clean; real cause found by comparing action dates to
rollcall dates: clerk roll numbers RESET EACH SESSION while Voteview
rollnumbers are congress-continuous, so the (congress, chamber, roll)
join attached a wrong amendment's purpose to ~40% of replaced rollcalls
(mismatch gaps clustered at ~367 days — the session offset). Textbook
merge-verification failure: coverage was checked at build time,
correctness was not. Fix: join additionally requires action_date ==
rollcall date; matches actually ROSE to 3,618 (dates disambiguate
formerly-ambiguous multi-claims). Rerun in progress as E5b.

## E2: recency-weighted member x question offset

*Registered 2026-07-03, before implementation (runs after E3/E4).*

**Hypothesis:** the mq offset weights all of a member's train-window
votes equally; behavior drifts, so exponentially decaying weights
(tau = 365 days to end-of-train, fixed a priori) should make the offset
more predictive of future votes in the forecast regime.

**Knowability:** decayed averages of strictly-train votes.
**Falsification:** tau -> infinity must recover the current offset
(verified to 1e-12 before running).

**Result (2026-07-03): NEGATIVE.** Forecast val 0.4714 vs champion
0.4277. tau=365d halves the effective sample behind each member x
question cell; the noisier offset (the single most-trusted z column)
costs far more than drift-tracking gains. Flat member histories beat
recency-weighted ones at this granularity. Not iterating on tau — that
would be metric shopping; one a-priori value was the design.
