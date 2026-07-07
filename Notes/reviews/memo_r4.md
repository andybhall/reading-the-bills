# Memo: Response Plan for coarse.ink Review Round 4

Reviewer: `openrouter/openai/gpt-5.5-pro`, with a referee brief requesting
(1) theoretical contribution for APSR, (2) writing quality/clarity at the
APSR bar, (3) methodological soundness. Review: `coarse_review_r4.md`.
Verdict: major revision. The 15 detailed comments contain 4 unique items
(the extractor duplicated them); the overall feedback carries the weight.

The central charge is fair and actionable: the paper is
"methods-without-theory" --- prediction, measurement, and theory are
blended, and the construct being measured is never pinned down. The
reviewer's own reading of our mechanism evidence (procedural posture and
title dominate; Senate null; transition failures) points to the honest
frame: much of what text adds is a measurement of the *political
situation the agenda creates* --- party agenda control and leadership
cueing --- not a purer measure of member ideology. That frame fits every
result we have, including the negatives.

## Major points

### T1. Add a theory/construct section — ADOPT (the round's centerpiece)
New Section 2 ("What Bill Text Can Measure") stating: the object each
section measures (held-out fit of a *vote-only* spatial representation in
P1; forecastable coalition behavior *conditional on the curated agenda*
in P2--P3); the hypothesized role of text (identifying the situation a
bill creates: must-pass deal, suspension consensus, program-building
spending); and what each section can and cannot support. The Senate null
and majority-transition results move from "boundaries" into the theory:
they are predictions of the agenda-control frame (Senate floor access is
negotiated, so text carries less marginal information; procedural
member-history is role, not preference). Intro and conclusion
restructured accordingly. This is an authorial-judgment change --- flagged
to Andy in the report, with the reframe stated plainly so it can be
vetoed.

### T2. Separate the vote-only spatial result from the text contribution — ADOPT
Restate point one as two explicitly separated results: (a) a vote-only
flexibility result (completion regime; background/benchmark), and (b) the
text forecasting result. Intro sentence structure changed so the
completion result is never presented as evidence about text. The
101st--119th vs 108th--119th split stated where the headline numbers
appear.

### T3. Timing rules (24h-before / at-question) — PARTIAL
Full reruns under two timing regimes are not possible for question/
description (Clerk-generated at the vote; no earlier timestamp exists)
and CRS summaries carry day-level action dates already enforced as
strictly-before. What we can and will state precisely in the audit
appendix: the timing rule for each field, and that the "vote question
removed" and "bill summary removed" ablation rows (Table 4) bound the
contribution of at-the-vote fields. The leakage-clean tower is promoted
to co-headline in the abstract as well as §3.2. REBUT the implication
that summaries can postdate votes: the join uses action_date strictly
before vote date by construction (Appendix C states this).

### T4. Validation hierarchy + clustered uncertainty everywhere — PARTIAL
All four regimes exist (completion, random-rollcall, temporal,
congress-out, prospective) plus the new seen-bill and chamber splits;
§2's evaluation paragraph will present them explicitly as a hierarchy
and state which claims attach to which level. Congress-out remains a
documented boundary (its numbers are in Table 2 and §6). Full clustered
CIs for every pairwise comparison: deferred except where already
computed (paired bootstrap for headline margins); noted in text.

### T5. Rename protest → majority defection throughout — ADOPT
Section 4 retitled "Text Anticipates Majority-Party Defections"; $protest
voting$ appears only where the construct is discussed (the FL/DMM
connection and the worked episodes). Abstract/intro/conclusion edited to
match. The Table 5 "5/4" cells the reviewer flags do not exist --- the
table reads n/5 in every row (OCR artifact on their side); noted, no fix.

### T6. Mechanism = agenda/procedure as much as content — ADOPT (via T1)
This is the reframe. The decomposition the reviewer requests is largely
Table 4 (redactions) + Table 7 (cutpoint components) + Appendix policy
table; §4.2's closing paragraph already states the division of labor. New
theory section makes it the paper's actual claim rather than a caveat.

### T7. Cutpoint target naming + reconciliation — ADOPT
Rename target "realized rollcall cutpoint" at first use; state explicitly
that targets come from the per-chamber-congress 1D fits of §2 (not the
8D completion model); reconcile N (3,549 identified test rollcalls; 3,527
after the |cut|<=4 display/evaluation trim --- state the trim where N
appears); cite the existing threshold-sensitivity macros for the |a|
screen. Peress/Richman overlap remains the named next test (r3 deferral
stands).

### T8. Document the 2011--2016 reimplementations — ADOPT (appendix)
Expand Appendix (litrace) with per-model implementation notes:
architecture, text inputs, optimizer, tuning budget, calibration, and
what "as published" vs "modernized" means for each row. Content exists in
the ledger/repo; this is exposition.

### T9. Known-truth synthetic section with protest-shock DGP — DEFER (flagged)
A full DGP with text-driven protest shocks and a text-independent placebo
is a real but sizeable addition. The existing synthetic suite (planted
structure recovery) is described in one sentence; we will expand the
description to a paragraph with its actual quantities. The protest-shock
DGP is queued as follow-up work; flagged to Andy rather than silently
dropped, since it is the reviewer's strongest unmet methodological ask.

### T10. Downstream application — ADOPT
New short subsection at the end of §5 (or §6): "What the measures are
for." Two exhibits from existing outputs, no retraining:
(a) a pre-vote revolt-risk series for the 118th House holdout (mean
predicted majority-defection share by vote, over time), showing spikes on
the CRs and discharge-petition/suspension deals --- a direct, usable
measure of when party government is under stress before the vote occurs;
(b) the agenda map already implicit in Figure 3a: where the 118th's
final-passage agenda cut relative to the majority median. One figure,
one paragraph of substantive inference about agenda control.

### T11. Benchmark card — ADOPT
New appendix table (script-generated): by congress block and chamber,
rollcalls, member-votes, share bill-linked, share with pre-vote summary,
share title-only, share amendment-linked; and holdout log loss by text
coverage cell. Serves the release (P6) as well.

### T12. Forecaster uncertainty/support — PARTIAL
Document the existing embedding-distance out-of-support warning in
Appendix F (it is already implemented in the dashboard); add a
rollcall-cluster residual interval to the displayed cutpoint; full
conformal coverage study deferred and named.

## Unique detailed comments

| # | Item | Verdict | Action |
|---|------|---------|--------|
| 1 | "every flip / none of the placebos" needs event-level support | ADOPT | add pointer that the all/none claim is the event-level content of Appendix Fig 5; averages summarize it (they do — the figure shows per-transition points) |
| 2 | "any vote-based measure" too broad | ADOPT | adopt reviewer's narrower sentence, tied to the tested component |
| 3 | amendment finding widened to "legislative language" | ADOPT | adopt reviewer's rewrite; keep "worse, not better" and "no incremental signal" as separate claims |
| 4 | forecaster inputs vs section feature set | ADOPT | add the sentence: interface defaults to final-passage; amendment/procedure forecasts conditional on supplied metadata |

## Sequencing

W1 theory section + intro/abstract/conclusion restructure (T1, T2, T5,
T6). W2 mechanical fixes (T3 audit statement, T7 renames/reconciliation,
4 detailed comments, abstract co-headline). W3 downstream application
(T10; script 36). W4 benchmark card (T11; script 37) + litrace appendix
expansion (T8). W5 forecaster documentation (T12). T9 flagged as queued
follow-up.
