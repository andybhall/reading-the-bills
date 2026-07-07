# Reading the Bills: Text, Votes, and the Measurement of Congress

**Date**: 07/05/2026
**Domain**: social_sciences/political_science
**Taxonomy**: academic/working_paper
**Filter**: Active comments

---

## Overall Feedback

Here are some overall reactions to the document.

**Outline**

The paper has a strong empirical core, but the present draft makes larger claims about measuring Congress than the design can yet support.

The dataset, forecasting pipeline, frozen-artifact audit, and failure ledger are valuable. The paper is also unusually clear about some limits, especially the House-Senate split and majority-transition failures. The main weakness is that prediction, measurement, and theory are still blended together in ways that leave the contribution less settled than the results deserve.

**The central construct is still not pinned down**

Section 1 asks what language-model reading contributes to the measurement of Congress, and Section 7 says reading bills makes Congress more measurable in three ways. The evidence, though, is organized around predictive tasks: held-out vote fit in Section 3, majority-defection ranking in Section 4, and cutpoint prediction in Section 5. The paper never settles whether the measured object is member ideology, agenda-induced coalition shape, party discipline, procedural role, or forecastable roll-call behavior conditional on the realized agenda. This matters because these constructs imply different theories and different validity tests; a predictor that uses titles and vote questions well may be evidence about party agenda control rather than a new measure of representation. A revision should add a short theory section before Section 3 that states the object being measured, the role bill text is hypothesized to play, and the claims each empirical section can and cannot support. The Senate null results in Sections 3.2 and 4.3 and the majority-transition failures in Section 6 should be folded into that theory rather than left as boundaries at the end.

**The no-text spatial result is conflated with the text contribution**

Section 3.1’s completion exercise involves no bill text; it shows that an eight-dimensional spatial model fits held-out votes slightly better than a logit using frozen DW-NOMINATE positions. That is a useful benchmark, especially for Ron Paul and Justin Amash, but it is not evidence that reading bills improves measurement. The introduction and conclusion present this as the first of the paper’s three answers to what modern language models add, which blurs two contributions: a more flexible roll-call model and a text-based forecasting model. Because bill-side information begins only with the 108th Congress, the 101st-119th headline also mixes a no-text completion era with a text-era forecast exercise. The convergent-validity claim in Section 3.1, based on high correlations with DW-NOMINATE and Nokken-Poole scores, does not show that the tail improvements are better ideology rather than better fit to strategic or procedural voting. The fix is to split the contributions: either make the completion exercise a background diagnostic, or add external validation for the changed member placements, such as comparisons to Duck-Mayr and Montgomery or Fowler and Lewis estimates, issue-specific voting records, and named-case evidence coded before seeing the model.

**Feature timing is not yet strong enough for the forecasting claims**

Section 3.1 says the model uses vote question, vote description, and the latest bill summary dated before the vote; Appendix C says vote question and description are fixed at the vote. That is weaker than the claim in Sections 3.2 and 5 that the model forecasts before votes are cast, because same-day floor text, vote descriptions, CRS summaries, and subject metadata can be created or revised close to, or after, the action. The paper recognizes one version of this problem by reporting the clean MiniLM tower at 0.348 log loss, while still using the ensemble with CRS-curated metadata as the headline result at 0.324 in Section 3.2 and the introduction. This choice invites a reader to attribute leakage-prone gains to bill reading. A revision should make the leakage-clean tower the main headline unless every input in the ensemble can be tied to an archived pre-vote timestamp. It should also rerun Sections 3-5 under at least two timing rules: features available 24 hours before the vote, and features available at the time the question was called, with raw source hashes or snapshots for summaries, titles, vote descriptions, and amendment records.

**The validation design overstates generalization beyond a within-Congress setting**

Section 3.2’s primary temporal holdout is the final ten percent of rollcalls within each congress-chamber. That is a sensible development test, but it keeps the same legislators, party configuration, and much of the same floor agenda in the training set; this is a favorable setting for member-history and procedural cues. The paper’s own congress-out panel in Table 2 is much less favorable: the final ensemble’s log loss rises to 0.643 and accuracy falls to 62.1%, while the text-plus-metadata rows without member history do not clearly dominate simple party or member-history baselines on log loss. The prospective audit is valuable, but Section 3.2 reports only 43 rollcalls and uses a predecessor artifact rather than the final model, so it cannot carry the headline on its own. The paper should reorganize validation as a hierarchy: within-Congress temporal, leave-one-Congress-out, leave-bill-family-out, and chamber-specific tests, with uncertainty clustered by rollcall and Congress rather than only paired rollcall bootstraps for one margin. The claims about future forecasting and the Appendix F interactive forecaster should then be tied to the setting where performance is actually shown, which appears to be especially the House within a stable majority period.

**Protest voting is not validated as distinct from majority-party defection**

Section 4.2 is careful for one sentence, saying the observable is majority-party defection and that this includes sincere disagreement and constituency pressure as well as protest. The surrounding text then repeatedly calls the result advance detection of protest voting, including in Section 1 and Section 7. The detection task in Section 4.3 ranks majority-party members who vote nay when their party majority votes yea; it does not identify whether those members arguably prefer the bill, nor does it separate protest from policy disagreement or procedural conflict. This matters because the link to Fowler and Lewis and to scaling bias depends on the narrower protest construct, not on any majority defection. The paper should either rename the empirical result as majority-defection forecasting throughout, or build a labeled validation set of protest episodes using Fowler-Lewis classifications, floor statements, whip reports, discharge-petition cases, or pre-specified case criteria. Table 5 also needs correction: Top-5 correct is defined as a count among the five likeliest defectors, yet the cells show values such as 5/4 and 4/4.

**The mechanism evidence points to procedural cues at least as much as bill content**

The phrase reading the bill is doing more work than the diagnostics support. In Section 4.2 the featured American Relief Act example had no summary, so the model read only the vote question and title; Table 4 shows that deleting the vote question hurts log loss more than deleting the summary; Table 5 shows the suspension-versus-passage edit moves predicted defection more than removing spending-program sentences. Section 6 says amendment purpose text makes prediction worse and that amender party, not amendment language, drives direction accuracy. These are interesting findings, but they suggest the model is reading the floor agenda and procedural posture as much as substantive legislative content. The paper should decompose all main claims into title, vote question, description, summary, sponsor or amender party, and other metadata components, using the same held-out tasks rather than only selected diagnostics. If procedural language remains the main signal, the theoretical frame should shift toward party agenda control and leadership cueing; if substantive summaries add independent value, the paper should show the examples where summary content changes the predicted coalition after metadata and titles are held fixed.

**The cutpoint section needs an independent target and uncertainty**

Section 5 says a bill’s text can locate it before anyone votes, but the target is the realized cutpoint from the paper’s own spatial fits, reported only when |a_v| >= 0.35. That target is a roll-call outcome, not the proposal or status-quo location studied by Peress or Richman, and it mixes policy content with agenda control, party pressure, and strategic voting. The draft should also state whether these targets come from the one-dimensional fit in the definitions or the eight-dimensional completion model in Section 3.1. The main improvement over metadata alone is modest in Table 7: MAE moves from 0.657 to 0.637 and direction accuracy from 70.8% to 74.5%, while the text-only rows are far weaker than metadata. A revision should relabel the current target as realized roll-call cutpoint, propagate uncertainty from the spatial fit into the prediction evaluation, justify the |a_v| threshold with sensitivity checks, and add at least one independent benchmark such as proposal/status-quo estimates, expert-coded bill ideology, or issue-specific labeled cases. The sample-size mismatch between the 3,549 rollcalls stated in Section 5 and the 3,527 All rollcalls in Appendix Table 11 also needs reconciliation.

**Comparisons to existing text and vote models need fuller documentation**

Section 3.2 and Appendix Table 9 make a strong claim that 2011-2016 text-based architectures fail to improve on a member-by-question table under proper scoring. The paper gives too little detail on those reimplementations to know whether the comparison is a test of the published ideas, the authors’ specific implementations, or the tuning budget used here. This matters because the paper’s main contribution is partly framed as modern language models reviving an earlier research program; a weak or opaque reimplementation could make the gain look larger than it is. The main forecast table also separates feature sets, but readers need to see the strongest classical and near-classical baselines in the same panel: DW-NOMINATE with context, recent-vote models, sponsor or cosponsor cues where available, bill-family history, and issue-area baselines. A revision should move the main comparator details out of the appendix, report hyperparameter ranges and validation rules for each older model, and include confidence intervals or clustered paired differences for every headline comparison. If some baselines are infeasible because of data coverage, state that and show the common-sample comparisons that are possible.

**Known-truth validation is too compressed**

The paper's empirical tests are extensive, but the only known-truth evidence appears as a one-sentence reference to a synthetic suite in Appendix B. For a measurement paper, readers need to see when higher vote-forecast scores also recover true member positions, cutpoints, and protest events, rather than only observed roll-call regularities. Add a full simulation section or appendix with a DGP: 435 members drawn from two party-centered ideology distributions, bills generated with text vectors for policy area, must-pass status, and procedure, votes drawn from P(y=1)=sigma(a_v x_i+b_v+c_i) plus a right-flank protest shock triggered by the bill text. Score DW-NOMINATE, the eight-dimensional spatial fit, the no-text tower, and the text tower on recovery of x_i, cutpoints, coalition direction, and protest flags, not just log loss. Include a placebo DGP where text is independent of the coalition; the text model should then lose its advantage if the interpretation is doing the work.

**No downstream use of the new measures**

The draft says reading bills makes Congress more measurable, but it never uses the new measures to answer a congressional question that existing measures answer differently. That leaves the APSR-level payoff less developed than the prediction evidence. A compact application would do a lot: use Section 5 cutpoint forecasts for all final-passage House bills in the 118th Congress to map which parts of the agenda cut inside the Republican conference, near the chamber median, or on the Democratic side, then compare that map to DW-NOMINATE-based party-unity and roll-rate summaries. Another natural application is to take the Section 4 defection scores and construct a pre-vote revolt-risk series for the McCarthy/Johnson House, checking whether it spikes on continuing resolutions, discharge-petition bills, and bipartisan fiscal packages. The point is to show one concrete inference about party government or agenda control that the text-based measurement changes.

**Benchmark composition and missingness are underdocumented**

The benchmark is advertised as a public resource, but the draft does not yet give a benchmark card that lets readers see what is in it and what is missing. The current text reports the 10.6 million decisions, the 108th-Congress start of BILLSTATUS text, and a few coverage percentages, but selection into legislation-linked, summary-rich, amendment-linked, and unlinked votes is not shown. Add a table by Congress, chamber, and vote type with counts of roll calls and member-votes, share linked to a bill, share with pre-vote title only, share with pre-vote summary, share with amendment purpose text, and share excluded because the member was present/absent rather than yea/nay. Then report the main forecast log loss separately for linked bills, title-only bills, summary-bearing bills, amendments, and non-legislative or unlinked votes. This matters because the strongest claims about reading bills are based on cells where readable text exists; users of the released benchmark need to know how much of Congress those cells cover.

**Released forecaster lacks uncertainty and support checks**

The interactive forecaster is a claimed output, yet the paper gives no user-facing uncertainty or out-of-support diagnostics. Average calibration curves do not tell a user how much to trust a pasted hypothetical bill's cutpoint, member vote forecast, or top-defector list. Add prediction intervals for the forecaster using a rollcall-cluster bootstrap, conformal residuals, or the frozen ensemble's between-tower spread, and validate their coverage separately for House/Senate, passage/amendment/procedure, and title-only versus summary-rich inputs. The interface should also report a support measure, such as distance to nearest training bills in embedding space within the same question type and sponsor-party cell, because Appendix F already admits sponsor-coalition mismatches are unvalidated. A short worked example using a pre-vote text snapshot for one later-realized bill would show how the interval and support flags change the interpretation.

**Recommendation**: major revision. The paper has a promising empirical contribution and several unusually good research-practice features, but the main claims currently outrun the construct validation and timing evidence. A top political science venue would need a clearer theory of what is being measured, leakage-clean headline results, and tighter validation of the protest and cutpoint claims before treating this as more than a strong forecasting paper.

**Key revision targets**:

1. Add a theory and construct section that distinguishes ideology, realized roll-call behavior, party agenda control, procedural role, and representation, then map each empirical section to one of those claims.
2. Make the leakage-clean feature set the headline unless all ensemble inputs have archived pre-vote provenance; rerun the main results under 24-hour-before-vote and same-day-floor-question timing rules.
3. Expand validation beyond the within-Congress final-ten-percent holdout, including leave-one-Congress-out, leave-bill-family-out, chamber-specific estimates, and clustered uncertainty for all headline comparisons.
4. Either rename the Section 4 outcome as majority-party defection or validate a labeled protest-vote construct; correct Table 5 and show how much of the defection gain comes from text rather than member history.
5. Rework Section 5 around realized roll-call cutpoints, add uncertainty and threshold sensitivity, and benchmark against an independent bill-location or bill-ideology measure where possible.

**Status**: [Pending]

---

## Detailed Comments (15)

### 1. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 2. Vote-based-measure claim is too broad

**Status**: [Pending]

**Quote**:
> Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference, and scores estimated in one majority configuration should be carried across configurations only with care (Bateman, Clinton and Lapinski, 2017). The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ .

**Feedback**:
I expected the interpretation to stay tied to the tested object: the procedural-vote component of member history. The reported result shows that this component loses transfer performance at House majority flips. It does not establish what “any vote-based measure” captures, since another vote-based measure could include time-varying ideal points, party-control interactions, or a separate procedural dimension. The caution about carrying scores across majority configurations follows from the test; the first half of the sentence reaches beyond the experiment described here. Rewrite “Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference” as “This member-history component appears to capture procedural role that changes with majority control, not only stable preference” because the experiment tests one model component rather than all vote-based measures.

---

### 3. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 4. Vote-based-measure claim is too broad

**Status**: [Pending]

**Quote**:
> Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference, and scores estimated in one majority configuration should be carried across configurations only with care (Bateman, Clinton and Lapinski, 2017). The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ .

**Feedback**:
I expected the interpretation to stay tied to the tested object: the procedural-vote component of member history. The reported result shows that this component loses transfer performance at House majority flips. It does not establish what “any vote-based measure” captures, since another vote-based measure could include time-varying ideal points, party-control interactions, or a separate procedural dimension. The caution about carrying scores across majority configurations follows from the test; the first half of the sentence reaches beyond the experiment described here. Rewrite “Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference” as “This member-history component appears to capture procedural role that changes with majority control, not only stable preference” because the experiment tests one model component rather than all vote-based measures.

---

### 5. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 6. Vote-based-measure claim is too broad

**Status**: [Pending]

**Quote**:
> Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference, and scores estimated in one majority configuration should be carried across configurations only with care (Bateman, Clinton and Lapinski, 2017). The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ .

**Feedback**:
I expected the interpretation to stay tied to the tested object: the procedural-vote component of member history. The reported result shows that this component loses transfer performance at House majority flips. It does not establish what “any vote-based measure” captures, since another vote-based measure could include time-varying ideal points, party-control interactions, or a separate procedural dimension. The caution about carrying scores across majority configurations follows from the test; the first half of the sentence reaches beyond the experiment described here. Rewrite “Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference” as “This member-history component appears to capture procedural role that changes with majority control, not only stable preference” because the experiment tests one model component rather than all vote-based measures.

---

### 7. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 8. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 9. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 10. Vote-based-measure claim is too broad

**Status**: [Pending]

**Quote**:
> Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference, and scores estimated in one majority configuration should be carried across configurations only with care (Bateman, Clinton and Lapinski, 2017). The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ .

**Feedback**:
I expected the interpretation to stay tied to the tested object: the procedural-vote component of member history. The reported result shows that this component loses transfer performance at House majority flips. It does not establish what “any vote-based measure” captures, since another vote-based measure could include time-varying ideal points, party-control interactions, or a separate procedural dimension. The caution about carrying scores across majority configurations follows from the test; the first half of the sentence reaches beyond the experiment described here. Rewrite “Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference” as “This member-history component appears to capture procedural role that changes with majority control, not only stable preference” because the experiment tests one model component rather than all vote-based measures.

---

### 11. Amendment finding broadens beyond its evidence

**Status**: [Pending]

**Quote**:
> The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ . Short purpose text adds nothing once bill identity and procedural context are modeled, a measured limit on what legislative language reveals, at least in the representations tested here.

**Feedback**:
I first read the paragraph as a boundary for amendment purpose text. The closing phrase widens it to “legislative language,” which is broader than the stated experiment; bill summaries, titles, and vote questions are different text fields in the paper. The paragraph also shifts from “makes prediction worse” to “adds nothing.” Those claims imply different measured effects: negative incremental performance versus zero incremental signal. Rewrite “Short purpose text adds nothing once bill identity and procedural context are modeled, a measured limit on what legislative language reveals, at least in the representations tested here” as “In these amendment models, purpose text provides no useful incremental signal and can degrade forecasts once bill identity and procedural context are modeled, a measured limit on what amendment-purpose language reveals in the representations tested here” because the evidence stated here concerns amendment-purpose fields, not legislative text as a whole.

---

### 12. Forecaster inputs omit reported metadata features

**Status**: [Pending]

**Quote**:
> Two caveats bound the exercise. The prediction target is the cutpoint of our own spatial fit—a realized-vote quantity with estimation error of its own—and we have not yet benchmarked the predictions against the proposal/status-quo estimates of Peress (2013) or Richman (2011) on an overlapping sample; that comparison, which requires reimplementing those estimators, is the natural next test. An interactive forecaster operationalizes the section's result: a user pastes a hypothetical bill's text, selects a sponsoring party and chamber, and receives a predicted cutpoint and a member-by-member vote forecast, with a sponsor-coalition mismatch flag for inputs off the curated agenda (Appendix F).

**Feedback**:
At this point I expected the user inputs to match the feature set used for the section’s headline numbers. They do not as described: the model above uses institutional metadata including question type and, for amendments, amender party, while the forecaster sentence lists only text, sponsoring party, and chamber. If the tool fixes the question type to final passage, it is not operating the full cutpoint model for amendments or procedure votes; if it accepts those fields elsewhere, the section should say so here. This matters because the section attributes much of the gain to metadata. Add “The public interface defaults to final-passage bill forecasts; amendment or procedure forecasts require the question type and, where applicable, amender party, so forecasts are conditional on those supplied or defaulted metadata.” after “An interactive forecaster operationalizes the section's result:” to address the mismatch between the reported model inputs and the stated user inputs.

---

### 13. All/none transition claim needs event-level support

**Status**: [Pending]

**Quote**:
> Users of these methods should know where they stop, and both boundaries are measured rather than conjectured. The first concerns majority transitions. At every House majority flip in the window—and at none of the placebo transitions—the procedural-vote component of member history stops transferring: the model's procedural log loss averages 1.105 across flips against 0.613 across placebos, while final-passage transfer is flip-invariant
> 
> <!-- PAGE BREAK -->
> 
> ![img-6.jpeg](img-6.jpeg)
> Figure 4: Bill-summary terms most associated with coalition direction, from a sparse linear model on final-passage votes. Positive coefficients (red) indicate language associated with conservative-yea coalitions; negative coefficients (blue), liberal-yea coalitions.
> 
> (0.447 against 0.427; Appendix Figure 5).

**Feedback**:
On first read, the phrase “At every House majority flip” sounds like an event-level statement. The numbers supplied immediately after it are only averages. Averages of 1.105 and 0.613 do not imply all flips fail and no placebos fail; for instance, a placebo set with losses 1.226 and 0.000 has average 0.613 while containing a placebo worse than the flip mean. Appendix Figure 5 may be the intended support, so the section should not leave the all/none claim to be inferred from the averages. Add “The all/none statement refers to the individual transition losses in Appendix Figure 5; the averages reported here summarize that event-level comparison” after “(0.447 against 0.427; Appendix Figure 5).” to address the gap between the event-level claim and the averaged statistics.

---

### 14. Vote-based-measure claim is too broad

**Status**: [Pending]

**Quote**:
> Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference, and scores estimated in one majority configuration should be carried across configurations only with care (Bateman, Clinton and Lapinski, 2017). The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ .

**Feedback**:
I expected the interpretation to stay tied to the tested object: the procedural-vote component of member history. The reported result shows that this component loses transfer performance at House majority flips. It does not establish what “any vote-based measure” captures, since another vote-based measure could include time-varying ideal points, party-control interactions, or a separate procedural dimension. The caution about carrying scores across majority configurations follows from the test; the first half of the sentence reaches beyond the experiment described here. Rewrite “Much of what any vote-based measure captures on the procedural agenda is institutional role rather than stable preference” as “This member-history component appears to capture procedural role that changes with majority control, not only stable preference” because the experiment tests one model component rather than all vote-based measures.

---

### 15. Amendment finding broadens beyond its evidence

**Status**: [Pending]

**Quote**:
> The second concerns amendments. Supplying amendments' purpose text makes prediction worse, not better, in pre-registered, placebo-controlled experiments (Appendix E); what moves amendment prediction is actor identity—the amender's party takes direction accuracy from approximately a coin flip to  $77\%$ . Short purpose text adds nothing once bill identity and procedural context are modeled, a measured limit on what legislative language reveals, at least in the representations tested here.

**Feedback**:
I first read the paragraph as a boundary for amendment purpose text. The closing phrase widens it to “legislative language,” which is broader than the stated experiment; bill summaries, titles, and vote questions are different text fields in the paper. The paragraph also shifts from “makes prediction worse” to “adds nothing.” Those claims imply different measured effects: negative incremental performance versus zero incremental signal. Rewrite “Short purpose text adds nothing once bill identity and procedural context are modeled, a measured limit on what legislative language reveals, at least in the representations tested here” as “In these amendment models, purpose text provides no useful incremental signal and can degrade forecasts once bill identity and procedural context are modeled, a measured limit on what amendment-purpose language reveals in the representations tested here” because the evidence stated here concerns amendment-purpose fields, not legislative text as a whole.

---
