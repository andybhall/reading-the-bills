# Reading the Bills: Text, Votes, and the Measurement of Congress

**Date**: 07/05/2026
**Domain**: social_sciences/political_science
**Taxonomy**: academic/working_paper
**Filter**: Active comments

---

## Overall Feedback

Here are some overall reactions to the document.

**Outline**

The paper has a compelling measurement agenda and an unusually rich forecasting exercise, but several central claims run ahead of the current research design. The main risks are feature timing, same-bill leakage in the temporal split, and the gap between predicting defections and validating protest voting as a construct.

The paper takes on a real problem in legislative measurement: vote-only models miss information in the agenda and in bill content. The draft is strong on scale, engineering, and comparison to simple baselines, and the case material around the December 2024 continuing resolution is promising. To meet a top-journal bar, the paper needs cleaner evidence that the gains come from pre-vote policy text rather than summaries, metadata, or prior votes on related measures.

**The headline forecast uses features the appendix treats as not leakage-clean**

Table 1 and Section 3.2 make the 0.324 log-loss ensemble the main forecasting result, while Appendix C says the ensemble's TF-IDF component includes CRS-revisable policy metadata and that the fully conservative headline is the MiniLM tower at 0.348. That is more than a footnote: the difference between 0.348 and 0.324 is a large share of the claimed gain over the strongest single component, and the introduction reports the more favorable number as if all inputs were pre-vote. The paper needs to either document, with archived BILLSTATUS snapshots or timestamps, that every subject and policy-metadata field used by the TF-IDF tower existed in that form before each roll call, or move 0.348 to the headline result. The same leakage-clean rerun should appear for the protest-vote analysis in Section 4 and the cutpoint analysis in Section 5, since both rely on the same representation family. Without that rerun, readers cannot tell whether text content or post-vote curation is driving the last increment of performance.

**Reading the bills is often reading summaries and floor metadata**

Section 2 says the model reads vote question, vote description, and the latest bill summary; Appendix D adds title-subject metadata and short input templates. The paper's title, Section 1, and Section 4 mechanism language say it is reading bills, but these inputs are often CRS summaries or floor metadata rather than the statutory text actually before members. That distinction matters for continuing resolutions, omnibus packages, rules, substitutes, and manager's amendments, where the latest summary may lag or omit the language that created the coalition split. The fix is not cosmetic: add an audit for a sample of high-stakes cases, including H.R. 10545 in Figure 2, the Social Security Fairness Act in Table 3, and at least one amendment vote from Section 6, showing the exact text version available before the vote and whether the model used it. A stronger revision would add a full-text-versus-summary comparison using versioned bill and amendment texts; if that is not feasible, the paper should frame the contribution as reading pre-vote summaries and floor metadata, not bills.

**The temporal split leaves same-bill and agenda leakage unresolved**

The temporal forecast in Section 2 holds out the final ten percent of roll calls within each congress-chamber, which is a useful design but not enough for the paper's broader forecasting language. Bills often generate multiple roll calls, and earlier amendments, rule votes, recommittals, or related passage votes can appear in the training window before a later held-out vote on the same measure. In that setting, the model may learn a coalition response to a bill family rather than infer the response from text alone. Table 1's congress-out panel also shows a large drop: the final ensemble moves from 0.324 log loss and 86.2 percent accuracy in the temporal split to 0.643 and 62.1 percent in congress-out transfer. The revision should add a bill-family-held-out temporal split that removes all prior roll calls on the same measure, plus a member-held-out or new-member test. At minimum, Table 1 should separate performance on bills with prior roll calls in the training window from performance on wholly unseen measures.

**The completion benchmark is not evidence for the text claim**

Section 3.1 compares an eight-dimensional spatial model to frozen DW-NOMINATE in a random held-out completion setting where same-rollcall votes remain observable. That result may be useful for ideal-point measurement, but it is not a test of reading bill text, and it includes pre-text congresses. A more flexible spatial model should often beat frozen two-dimensional scores under random vote completion, especially for the hard cases labeled in Figure 1. The paper currently folds this result into the three-part answer in Section 1, which makes the text contribution look broader than the evidence shows. The completion result should be presented as a separate vote-only measurement result, not as part of the bill-reading claim. For the forecasting claim, the main comparison should be against stronger non-text models in the same temporal and bill-held-out regimes: dynamic ideal points, issue-specific member histories, party-by-topic baselines, sponsor-party baselines, and recent same-bill history where that history is allowed.

**Protest voting is predicted, but not yet validated as protest voting**

Section 4 defines the detection task as ranking majority-party members by defection probability on roll calls where the party majority voted yea. That is a reasonable prediction task, but it is not the same as identifying majority members who voted against bills they actually preferred. Defections can reflect sincere ideological disagreement, constituency pressure, leadership conflict, procedural objections, or missing bill context, and the current evidence does not separate these mechanisms. Figure 2 is a strong case illustration, but Table 3 already shows a case, the Social Security Fairness Act, where the conflict was procedural rather than textual. The revision should build an external validation set: public statements, whip-count reports, caucus letters, discharge-petition histories, and the Fowler-Lewis or Duck-Mayr-Montgomery classifications where they overlap. Then report precision, recall, and false-positive examples for protest episodes, rather than only within-vote AUC for generic majority defections.

**The text-feature evidence does not yet rule out titles, procedure, or boilerplate**

Sections 4.2 and 5 offer nearest neighbors, a ridge probe with r = 0.25, and the sparse term model in Figure 4. These are useful first diagnostics, but they do not show that the model is using substantive policy language rather than titles, vote questions, procedural phrases, sponsor cues, or recurring omnibus language. Table 4 is suggestive in the same direction: metadata alone gives much of the cutpoint performance, and the separate text components are weak by themselves. The paper should add a ranked ablation table for the main vote-forecasting task, not only the cutpoint task: title-only, question-only, summary-only, summary with titles and proper nouns redacted, no vote question, no sponsor party, boilerplate stripped, and shuffled text within policy area. Counterfactual edits would also help: for H.R. 10545, the Social Security Fairness Act, and BIOSECURE, show how predicted majority defection changes when fiscal-package language, discharge-petition cues, or China/security language is removed or replaced. A policy-area table should report text gains separately for appropriations and continuing resolutions, suspensions, amendments, social policy, national security, and tax bills.

**The cutpoint result needs uncertainty and a clearer interpretation**

Section 5 predicts realized cutpoints taken from spatial fits, with the cutpoint reported only when |a_v| is at least 0.35. Those cutpoints are estimated from the same realized roll-call system the paper wants to move beyond, and the paper does not report uncertainty in the dependent variable. The incremental text gain over metadata in Table 4 is modest: MAE improves from 0.657 to 0.637 and direction accuracy from 70.8 to 74.5 percent. That may still be meaningful, but the paper should show when text changes the inference and whether those changes survive uncertainty in the fitted cutpoints. Add bootstrap or posterior intervals for cutpoints, then propagate them into the Section 5 evaluation. The main table should also separate final passage, amendments, procedural votes, contested votes, and lopsided votes, because the single spatial cutpoint is least credible for omnibus and procedural roll calls.

**The prospective audit is too small to verify the headline claims**

Table 2 is valuable, but it scores only 43 roll calls and 9,451 member-vote decisions, with a wide interval for the v1 log loss. The text in Section 3.2 says the frozen artifact verifies that the temporal results do not reflect researcher iteration, but the current evidence is too thin for that statement. The table also lacks matched post-freeze baselines, including the no-text counterpart, member-history model, and leakage-clean MiniLM tower. The revision should report the post-freeze audit as preliminary unless a larger set of roll-call clusters is available, and it should score all main baselines on the same frozen sample. Appendix D should also clarify whether the blend values 0.57, 0.12, and 0.42 are unconstrained coefficients or convex weights, since their sum exceeds one and exact model specification matters for the audit.

**Hard-case member measurement lacks worked validation**

Figure 1 says the new spatial model fixes the canonical hard cases, but the paper does not show what the revised measurement means for any one member. A fit gain for Ron Paul or Justin Amash is persuasive only if readers can see that the model is capturing known ends-against-the-middle behavior rather than giving flexible parameters to unusual voters. Add a worked table for the labeled members in Figure 1: DW-NOMINATE GMP, eight-dimensional GMP, the ten held-out roll calls with the largest log-loss improvement, and short labels for those bills. For Ron Paul, include votes such as TARP, Patriot Act or FISA reauthorizations, defense spending, and continuing resolutions; for Amash, use civil-liberties, leadership-procedure, and spending votes. The table should also compare the resulting classifications to Duck-Mayr and Montgomery-style ends-against-middle cases and Fowler-Lewis protest-vote episodes where they overlap.

**No Senate-facing validation of congressional claims**

The paper is framed as a measurement study of Congress, but the main narrative examples are House examples, and the majority-transition boundary is also House-centered. Readers will wonder whether the text gains survive in the Senate, where the agenda, cloture, amendment practice, and individual-member incentives differ sharply from the House. Add a chamber-disaggregated version of Table 1, plus chamber-specific versions of the Section 4 majority-defection AUC and the Section 5 cutpoint results. A strong revision would also include at least two Senate worked cases: for example, H.R. 1892, the Bipartisan Budget Act of 2018, as a spending-compromise/right-flank case, and H.R. 8404, the Respect for Marriage Act, as a cross-party coalition case. For each case, show the nearest-neighbor bills, the ranked defectors, and the predicted cutpoint.

**Forecast probabilities need calibration evidence**

The paper reports log loss and accuracy, but it does not show whether the vote probabilities are calibrated in the settings where users would rely on them. That matters because the protest-voting claim and the interactive forecaster are probability claims, not only ranking exercises. Add reliability plots and calibration tables for the temporal forecast and the frozen post-freeze sample, with bins by predicted probability and separate panels for party, chamber, contested votes, and lopsided votes. Report Brier score and expected calibration error for the final ensemble, the no-text counterpart, and the member-history baseline. For Section 4, include a roll-call-level calibration plot comparing predicted majority-defection shares to realized shares by decile.

**Bill-location results need an ex-post benchmark**

Section 5 positions the paper against the Peress and Richman bill-location literature, but it never checks whether the proposed prospective locations line up with those retrospective estimators on an overlapping sample. The current target is a cutpoint derived from the paper’s own spatial fits, so readers cannot see whether the new estimates recover the older object as a special case. Add an overlap exercise that estimates Peress-style proposal/status-quo locations, or a Richman-style status-quo test, after the vote and then correlates those ex-post quantities with the text-predicted pre-vote cutpoints. Natural cases to include are TARP, the Affordable Care Act, Dodd-Frank, the Tax Cuts and Jobs Act, and the Inflation Reduction Act. The table should report the text model, metadata-only model, and classical ex-post estimate side by side, so the paper can show what the prospective method adds.

**Recommendation**: major revision. The paper has a strong idea and enough promising evidence to merit revision, but the current design does not yet support the full strength of the claims about reading bill text, forecasting future votes, and detecting protest voting. The main results need to be rerun under leakage-clean inputs and stricter held-out designs, with external validation for the protest and cutpoint interpretations.

**Key revision targets**:

1. Make the leakage-clean MiniLM specification the headline or prove that the ensemble's CRS-derived metadata was available and unchanged before each vote; rerun Tables 1, 3, and 4 under the leakage-clean feature set.
2. Add stricter out-of-sample tests: bill-family-held-out temporal forecasts, no-prior-same-bill forecasts, congress-out transfer with recalibration rules fixed in advance, and a member-held-out or new-member evaluation.
3. Add a main-text ablation and counterfactual-edit section showing which text components drive vote forecasts, with redactions for titles, proper nouns, procedural phrases, sponsor cues, and boilerplate.
4. Validate protest-vote claims against external evidence from public statements, whip counts, caucus conflicts, discharge-petition histories, and existing protest-vote classifications; report false positives and false negatives.
5. Rework Section 5 to propagate uncertainty in realized cutpoints, separate vote types, and state the incremental value of text over metadata in practical terms.

**Status**: [Pending]

---

## Detailed Comments (18)

### 1. Rephrase the member-level improvement claim

**Status**: [Pending]

**Quote**:
> In the psychometric completion setting, an expressive spatial model fits held-out votes better than frozen DW-NOMINATE member by member, on the NOMINATE literature’s own statistic, and the improvements concentrate in precisely the cases that motivated a decade of bespoke scaling models.

**Feedback**:
The phrase ‘member by member’ reads like pointwise dominance. If Section 3.1 reports higher GMP for 67.1% of members, the text should describe this as a distributional member-level improvement concentrated in hard cases, not as uniform superiority for every member.

---

### 2. Separate protest-vote and ends-against-middle literatures

**Status**: [Pending]

**Quote**:
> Protest voting—majority members defecting on bills they arguably prefer—is the newest concern of the scaling literature *(Fowler and Lewis, 2026; Duck-Mayr and Montgomery, 2023)*, where it has been treated retrospectively, by reinterpreting votes already cast.

**Feedback**:
This sentence merges two related but distinct objects. Section 4 evaluates majority-party defections, while Duck-Mayr and Montgomery appear to be framed elsewhere as ends-against-the-middle scaling cases. Assign Fowler and Lewis to the majority-party protest-vote claim and describe Duck-Mayr and Montgomery as related mismeasurement literature unless their result directly covers the same majority-defection task.

---

### 3. Identify the normalization behind reported cutpoints

**Status**: [Pending]

**Quote**:
> The spatial fits estimate $P(\text{yea})=\sigma(a_{v}x_{i}+b_{v}+c_{i})$ per chamber-congress (party-sign initialized); a rollcall’s *average-member cutpoint* is $-(b_{v}+\bar{c})/a_{v}$, reported when $|a_{v}|\geq 0.35$ on the fitted scale, with *direction* $\text{sign}(a_{v})$.

**Feedback**:
The likelihood is invariant to affine rescalings such as x_i'=αx_i+β, a_v'=a_v/α, and b_v'=b_v−(a_v/α)β. Under that transformation, the reported cutpoint becomes αx*+β, the |a_v| screen changes to |a_v|/|α|, and α<0 flips the direction. Even after standardizing x, the transformation a_v^λ=a_v−λ and c_i^λ=c_i+λx_i preserves fitted probabilities while changing the cutpoint denominator. The paper should state the identifying normalization used for cutpoints, directions, and the |a_v| threshold, and apply the same alignment in any recovery exercise.

---

### 4. Clarify exclusion of scored cells from roll-call parameter estimates

**Status**: [Pending]

**Quote**:
> against 0.849 for frozen DW-NOMINATE positions with estimated rollcall parameters; the full ladder of intermediate models appears in Appendix Table 5.

**Feedback**:
‘Estimated rollcall parameters’ should specify whether the held-out member-vote being scored is excluded from the likelihood used to estimate that roll call’s parameters. If a held-out yea enters even an intercept-only roll-call fit, the fitted yes rate can move from 8/9 to 9/10 and mechanically raise the probability assigned to that same held-out yea. State that each scored y_iv is omitted when estimating the corresponding roll-call parameters.

---

### 5. Define the scalar coordinate used in position correlations

**Status**: [Pending]

**Quote**:
> Convergent validity is preserved: the model’s member positions correlate with DW-NOMINATE at $r=.98$ and with Nokken–Poole per-congress scores *(Nokken and Poole, 2004)* at $r=0.96$–0.98, so the improvement in fit does not come at the cost of the scale’s interpretation.

**Feedback**:
The model is described as having eight-dimensional member positions, so a Pearson correlation with DW-NOMINATE requires choosing a scalar coordinate or projection. Rotations and alternative projections of the fitted space can produce different correlations. Specify the oriented coordinate or projection used, and state how the Nokken–Poole comparison is aggregated within congress.

---

### 6. Distinguish post-snapshot and strictly post-freeze audit scores

**Status**: [Pending]

**Quote**:
> It recorded 0.331 log loss (85.7% accuracy) overall and 0.343 on the strictly post-freeze subset, matching its development-time performance (Table 2).

**Feedback**:
Table 2 is described as covering votes cast after the June 9 snapshot, while the prose separately gives a strictly post-freeze score of 0.343. Since the displayed row reports 0.331, these appear to be different audit sets. Label 0.331 as post-snapshot performance and add the strictly post-freeze subset to Table 2 if that is the no-iteration audit being claimed.

---

### 7. Narrow the attribution of the benchmark gap

**Status**: [Pending]

**Quote**:
> The distance between that literature and the present ensemble is the distance between the text tools of the two eras.

**Feedback**:
The comparison changes more than the vintage of the text tools: the older architectures are reimplemented in a modern pipeline, while the target is the current ensemble rather than a matched text-only model. The evidence supports the narrower claim that those reimplementations do not match the present forecasting pipeline; it does not isolate text-tool era as the sole source of the gap.

---

### 8. Limit the center-pull mechanism to ends-against-middle cases

**Status**: [Pending]

**Quote**:
> majority members voting against bills they arguably prefer, which one-dimensional models absorb by pulling the member's estimated position toward the center (Fowler and Lewis, 2026; Duck-Mayr and Montgomery, 2023).

**Feedback**:
A majority-party defection does not by itself imply center-pull in a one-dimensional fit. For example, if opposition members are at x=-1, mainstream majority members at x=0.5, and flank dissenters at x=2, a yea coalition for x<1.2 and nay votes from x>1.2 can be fit by keeping the dissenters on the extreme flank. The center-pull mechanism should be limited to ends-against-the-middle patterns where defectors appear to side with the opposition despite latent preferences closer to their party.

---

### 9. State the ranking universe in Figure 2

**Status**: [Pending]

**Quote**:
> The five members the text-reading model ranks likeliest to defect, labeled at right, all voted nay.

**Feedback**:
‘Defect’ normally means a majority-party member voting against the party majority, but this sentence only verifies that the labeled members voted nay. If the ranking is among majority-party members, say so explicitly and state that the majority party’s modal vote on H.R. 10545 was yea. Otherwise, the caption is describing vote prediction rather than defection detection.

---

### 10. Support the claim that the same members drive both results

**Status**: [Pending]

**Quote**:
> The members whose scaling improves most in Figure 1 are the same protest-prone members whose defections the text model prices here: what a one-dimensional model records as inexplicable moderation is, to a model that reads the agenda, systematic and forecastable behavior.

**Feedback**:
The reported AUC gain from 0.75 to 0.80 is a within-vote pairwise ranking statistic over defector/nondefector comparisons. It does not identify which members generated the improved pairwise orderings, so the high-gain Figure 1 members could be different from the members driving the AUC improvement. Add a member-level overlap table linking scaling-gain rank, defection opportunities, and text-minus-no-text prediction gain before claiming they are the same members.

---

### 11. Align the defection-task definition with the evaluated subset

**Status**: [Pending]

**Quote**:
> Define the detection task on every held-out rollcall in which a member’s party majority voted yea: rank that party’s members by predicted defection probability.

**Feedback**:
The following result is reported only for 517 roll calls with at least three defections, not every majority-yea roll call. AUC is defined with one or two defectors as long as at least one nondefector is present, so the at-least-three rule is a substantive filter rather than a definitional necessity. Define the task as the filtered subset or report a separate result for all positive-defection roll calls.

---

### 12. Define within-vote AUC aggregation and ties

**Status**: [Pending]

**Quote**:
> model attains a within-vote AUC of 0.80, against 0.75 for its no-text counterpart—text resolves roughly a fifth of the remaining ranking errors.

**Feedback**:
A within-vote AUC can be averaged equally over roll calls or weighted by the number of defector/nondefector pairs. With two votes having AUCs 0.50 and 1.00 and 300 versus 12 pairs, the unweighted mean is 0.75 but the pair-weighted mean is 0.519. State the aggregation rule, tie handling, and the remaining-error calculation, which appears to be (0.80−0.75)/(1−0.75)=0.20.

---

### 13. Use signed evidence before claiming inversion

**Status**: [Pending]

**Quote**:
> the procedural-vote component of member history inverts: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
Log loss is unsigned. A log loss of 1.105 corresponds to a geometric mean probability of exp(−1.105)=0.331 on realized votes, which shows poor procedural transfer across flips but not a direction reversal. Unless Appendix Figure 5 reports a signed coefficient or contribution reversal, describe this as degraded procedural transfer rather than inversion.

---

### 14. Clarify the information set in the identifiability claim

**Status**: [Pending]

**Quote**:
> if two member-rollcall states induce the same conditional vote distribution given available information, no measure computed from that information can distinguish them.

**Feedback**:
Equal conditional vote probabilities alone do not prevent a measure from distinguishing states using observed covariates. For instance, if I∈{A,B}, Pr(Y=1|I=A)=Pr(Y=1|I=B)=1/2, and M(I)=1{I=A}, then M is computed from available information and separates A from B. The claim is valid if the states are observationally equivalent in the full available information set, or if ‘measure’ is restricted to a function of the conditional vote distribution; state that restriction.

---

### 15. Specify how the no-text counterpart is refit

**Status**: [Pending]

**Quote**:
> The no-text counterpart deletes all text blocks and keeps everything else.

**Feedback**:
‘Keeps everything else’ could mean refitting the model after removing text, or reusing the champion’s blend weights and calibration while zeroing text inputs. Those choices can yield different probabilities: with component logits (0,1,0), the stated blend weights would give σ(0.12), while a no-text refit putting unit weight on the second tower gives σ(1). Because this baseline identifies the contribution of text, state whether early stopping, blend weights, and temperature calibration are re-estimated after removing text.

---

### 16. Make the ledger entries match the stated tests

**Status**: [Pending]

**Quote**:
> and regime inheritance (parameters fit on the temporal development slice transfer to temporal evaluation and break on interleaved evaluation).
> 
> Table 7: Ledger audit trail (full record with timestamps in the repository).
> 
> |  ID | Hypothesis | Falsification test | Outcome  |
> | --- | --- | --- | --- |
> |  M-A | bigger encoder helps long bills | long-bill loss bucket | negative  |
> |  E1 | within-bill context helps | shuffled-bill placebo | negative  |
> |  E1b | stacked context corrector | permuted-bill weights → 0 | null  |
> |  E2 | recency-weighted rates help | τ → ∞ recovers flat | negative  |
> |  E3 | per-bucket calibration helps | val only | negative  |
> |  E4/E4b | tower blends help | dev-weight collapse if redundant | positive  |
> |  E5a-c | amendment text helps | placebo file rebuild | negative  |
> |  S5 | blend beats champion | one-shot test unveil | positive  |

**Feedback**:
The prose lists a regime-inheritance result, but the printed ledger rows do not include that test. Several ‘Falsification test’ entries also describe the sample-opening protocol rather than the metric or inequality that could falsify the hypothesis; for example, ‘val only’ and ‘one-shot test unveil’ do not state a performance criterion. Add the regime-inheritance row and replace protocol-only cells with explicit criteria such as validation log-loss improvement over global calibration or one-shot log loss below the champion.

---

### 17. Separate fractional-logit fitted means from raw probabilities

**Status**: [Pending]

**Quote**:
> the cutpoint comes from a logistic fit of predicted probabilities (as fractional responses) on current-member positions,  $\mathrm{logit}(\hat{p}_i) = \alpha +\beta x_i$ ,  $x^{*} = -\alpha /\beta$

**Feedback**:
A fractional-logit fit uses targets p̂_i and fitted means q_i=logit^{-1}(α+βx_i); it does not imply logit(p̂_i)=α+βx_i for every member. With x=(−1,0,1) and p̂=(0.2,0.8,0.8), exact equality would require α=logit(0.8) and β=0 from the last two points, which cannot also fit p̂=0.2 at x=−1. Present x* as the 0.5 crossing of the fitted curve q_i, not of the raw predicted probabilities.

---

### 18. Label sponsor-coalition mismatch cautiously

**Status**: [Pending]

**Quote**:
> Forecasts whose predicted coalition contradicts the bill's own sponsorship are flagged as off-agenda extrapolations—a live demonstration that floor agendas are curated and the model has never seen off-agenda bills.

**Feedback**:
The stated trigger is a mismatch between predicted coalition and sponsor party, but sponsor party is not the same as agenda control or expected floor support. A cross-party bill can be sponsored by one party while drawing more support from the other, and an off-agenda proposal could still be predicted to attract its sponsor party. Label this diagnostic as a sponsor-coalition mismatch rather than proof of off-agenda extrapolation.

---
