# Measuring Congress in the AI Era: What Modern Vote Prediction Reveals About Members, Bills, and Institutions

**Date**: 07/05/2026
**Domain**: social_sciences/political_science
**Taxonomy**: academic/working_paper
**Filter**: Active comments

---

## Overall Feedback

Here are some overall reactions to the document.

**Outline**

The paper is ambitious and has several promising pieces: the temporal validation, the prospective freeze idea, and the attempt to turn prediction failures into institutional evidence are all valuable. The main problem is that the paper often moves too quickly from high predictive performance to measurement claims about ideology, loyalty, agenda control, amendments, and institutional role. Several of those claims need tighter validation, clearer separation between forecasting and in-sample measurement, and more careful tests against simpler institutional explanations.

This is a strong and timely project with a potentially important benchmark for congressional vote prediction. The paper does well in treating evaluation regimes as part of the measurement problem rather than as a technical afterthought, and the frozen prospective scoring protocol is a real strength. At the current stage, though, the framing and several substantive conclusions are stronger than the evidence shown in the main text supports.

**The prediction-to-measurement claim is overstated**

The Introduction and Abstract argue that prediction quality bounds measurement quality and that a better vote predictor has, by construction, captured more behavioral structure. Section 5 then shows that member histories can encode institutional role rather than stable preference, especially across the 2022 House flip. That is an important finding, but it cuts against the earlier framing: a model can predict better by using signals that are useful for forecasting yet poor measures of the construct the paper wants to name. This matters because the paper uses the model’s internals to produce ideology, loyalty, issue-position, cutpoint, and out-of-character measures in Sections 3 and 4. A revision should separate predictive information from construct-valid measurement more explicitly, perhaps by decomposing the predictor into stable member preference, party/role, agenda, and rollcall-context components and stating which derived measures are based on which component. The opening claim should also be softened: prediction is evidence that some structure has been captured, but not evidence by itself that the captured structure is the intended political quantity.

**Pre-vote information and leakage controls need a fuller audit**

Section 2.1 says the task uses everything observable before a rollcall, and Section 2.3 emphasizes calibration and hash-pinned prospective evaluation. The main text does not yet give enough detail to verify that every feature obeys that rule, especially dated bill summaries, amendment purpose text, member histories, calibration slices, and empirical member-by-question rates. The concern becomes sharper in Section 3.4, where out-of-character votes are described as decisions the frozen champion, "fit on the member's entire record," got confidently wrong; that is not the same estimand as surprise given information available before the vote. If the model uses post-vote behavior to define what was surprising, the resulting measure is retrospective fit residuals, not a forecasting-based measure of unexpected defection. The paper should add a feature-time audit table for Section 2.1 listing each input, timestamp, construction rule, and whether it is recomputed in an expanding-window fashion. For Section 3.4, the authors should either recompute surprises using strictly pre-vote training histories or rename the measure as an in-sample residual and avoid claims about what the model "could not see coming."

**Prospective certification is thinner and less aligned with the headline than the paper suggests**

The Abstract says the model is verified prospectively on 9,451 votes after it was cryptographically frozen, while Section 2.3 states that a predecessor model frozen on June 12, 2026 was scored on June 10–30 votes and that the blend "has begun its own prospective clock." Table 2 then reports both v1 and v2 with the same June 9 snapshot, which leaves unclear which exact artifact supports the headline claim. This matters because the paper uses the prospective freeze to certify not just accuracy but also the downstream measurement instrument. A short 43-rollcall window can be useful as a contamination check, but it is not enough by itself to validate claims about contested votes, amendments, majority-control transfer, or derived member measures. The revision should align the Abstract, Section 2.3, and Table 2: identify the exact frozen artifact, freeze date, training cutoff, vote window, and whether the champion or a predecessor was evaluated. It should also report uncertainty intervals for Table 2 and, if possible, add a longer prospective window covering enough rollcalls to break out contested votes, amendments, party-split votes, and chamber-specific performance.

**The benchmark comparison may not yet isolate the value of the modern text model**

Table 1 and Table 7 show large gains for the three-tower blend in the within-congress temporal forecast regime, but the comparison to classical and simpler institutional baselines needs tightening. The NOMINATE × context logit in Table 1 performs weakly relative to the member-history table, while Appendix Table 6 shows frozen DW-NOMINATE is very strong in the completion regime; readers will want to know whether the forecast baseline is a strong implementation of modern dynamic or context-rich ideal-point prediction. Similarly, Section 4.1 shows metadata alone nearly matches text plus metadata for cutpoint location, and Section 5.2 shows a majority-status count table beats the champion on cross-congress log loss. These results raise a natural question: how much of the headline performance comes from member histories and institutional metadata rather than language-model reading of legislative text? The paper should add a single decomposition table across the main forecasting regimes with party-only, member-history-only, metadata-only, text-only, text plus metadata, and full-history models, all calibrated in the same way. A stronger classical baseline should also be included, such as a dynamic ideal-point or hierarchical member-party-question model fit only on prior votes and scored under the same temporal splits.

**Measurement validation is too dependent on face validity and correlations with existing scores**

Section 3.1’s correlation of learned positions with DW-NOMINATE is a useful sanity check, but it does not validate the newer measures in Sections 3.2–3.4. The loyalty residual in Figure 5 is said to identify mavericks and to have "seen" Jeff Van Drew’s party switch coming, yet the text does not show an out-of-sample test, uncertainty bands, or comparison to known party-unity and leadership-defection measures. Issue-specific positions in Section 3.3 are mainly validated by named examples and correlations with the overall dimension; that leaves open whether the deviations are stable member traits, sparse-vote artifacts, or products of agenda selection within policy areas. The out-of-character measure in Section 3.4 is described as "not model noise," but the evidence shown is inspection of recognizable cases, not a systematic validation. The revision should add uncertainty estimates for member positions, loyalty residuals, issue positions, cutpoints, and surprise scores, preferably via bootstrap or repeated splits. It should also validate the new measures against external data: interest-group ratings by issue, caucus membership, public party-switch events using pre-switch data only, district characteristics for crossovers, and known leadership-pressure votes.

**The amendment conclusion is stronger than the design establishes**

Section 5.1 concludes that amendment coalitions form around deals, not language, because adding amendment purpose text makes prediction worse. The mechanism offered is plausible, but the reported design also suggests a modeling explanation: replacing shared parent-bill text with amendment-specific purpose language removes useful pooling over amendment series. If the model is forced to trade a stable bill-level signal for short and noisy amendment text, worse performance does not by itself show that amendment content is politically irrelevant. Section 4.1 already notes that amender party is highly informative for amendment direction, which means the missing ingredients may be sponsor/amender identity, amendment sequence, rule structure, or bill-series context rather than deals in a general sense. A revision should run a cleaner set of amendment ablations: parent bill text only, purpose text only, parent plus purpose text, parent plus purpose plus amender-party metadata, and a hierarchical model that keeps amendment-series pooling while adding purpose text. The paper can then state the boundary more carefully, for example that short purpose text adds little once bill identity and procedural context are modeled.

**The majority-control result is based on too narrow a stress test**

Section 5.2 is one of the paper’s most interesting parts, but the claim that "no model survives a change in majority control" rests mainly on the 2022 House flip and the 118th Congress column of Table 1. A single recent flip is informative, especially because procedural voting is central to the mechanism, but it is not enough to support the general claim across Congresses, chambers, and types of votes. The design should distinguish a majority-status shock from other simultaneous changes: agenda composition, new members, post-January-6 party dynamics, divided government, speaker politics, and changes in rollcall mix. The authors should repeat the transfer exercise over all available majority-control changes in the 101st–119th Congresses, with no-flip congresses as placebo transitions. The analysis should also break performance out by procedural, passage, amendment, nomination, and party-unity votes, since the role-based anti-transfer mechanism should be strongest for some categories and much weaker for others. If the result remains concentrated in House procedural votes after flips, that would still be valuable, but the conclusion should be narrowed accordingly.

**Rollcall and agenda selection are acknowledged but not built into the measurement claims**

The paper repeatedly says it measures Congress, but the data are recorded rollcall votes, whose content is selected by leaders and shaped by agenda control. Section 4.2 acknowledges the off-agenda problem in the dashboard, and Section 5.2 notes that cross-time comparisons are entangled with what the chamber votes on, yet Sections 3 and 4 often treat the derived quantities as member and bill measures without carrying that selection caveat into the estimands. This is especially relevant for issue-specific positions in Section 3.3 and cutpoint distributions in Section 4, because policy areas and vote types are not sampled in the same way across chambers, parties, or majority configurations. The paper should define each measure as conditional on the observed rollcall agenda unless it supplies a correction for agenda selection. A concrete fix would be to report how issue-position estimates and cutpoint distributions change after stratifying or reweighting by question type, chamber, Congress, and majority party, and to add a sensitivity table showing whether the named substantive findings survive those restrictions. The interactive forecaster in Appendix B should also avoid implying support predictions for bills drawn from an unobserved agenda without clear extrapolation warnings.

**Derived measures need explicit estimands and algorithms**

The paper introduces several new measurement objects, but the main text does not define them precisely enough for readers to know what is being estimated. The loyalty residual, issue-specific positions, bill cutpoints, and out-of-character votes are described in words, yet the paper does not give the equations, normalizations, inclusion rules, or estimation sequence that turn model output into these quantities. This matters because the paper’s contribution is not only a better predictor; it is the claim that the predictor can be used as a measurement instrument. A companion paper can develop member measures at greater length, but this paper still needs the core definitions. Add a short methods subsection with notation: define the vote probability \(\hat p_{iv}\), the party-unity vote set, the expected party-support model used for the loyalty residual, the per-policy-area scaling procedure, the cutpoint computation such as \(-\alpha_v/\beta_v\) with the discrimination threshold, and the surprise score such as \(-\log \hat p_{iv}(y_{iv})\). A compact pseudocode block showing how one rollcall and one member move through the pipeline would make the measures auditable.

**Text coverage across Congresses is not documented**

The benchmark spans the 101st through 119th Congresses, but the bill-side BILLSTATUS text is described as available from the 108th Congress forward. The paper does not show how much rollcall-level text exists in each Congress, how missing summaries or amendment purpose fields are coded, or whether the model’s text advantage is concentrated in the post-108th period. That gap matters for a paper whose headline model reads legislative text and whose benchmark covers 10.6 million voting decisions. Readers need to know when the text tower is actually using substantive text rather than titles, question strings, metadata, or missing-value patterns. Add a coverage table by Congress, chamber, and question type reporting the fraction of rollcalls with vote question, vote description, bill summary, sponsor, and amendment purpose text. Then report the main temporal-forecast ablation separately for the 101st–107th and 108th–119th Congresses, using the same calibrated party-only, member-history, metadata, text-only, and full-model variants.

**A worked landmark rollcall is missing**

The paper gives many aggregate figures, but it never walks through a single rollcall from inputs to predictions to derived measurements. For a measurement paper built around an opaque ensemble, that worked case would do real work: it would show readers what the instrument sees, where the cutpoint comes from, why particular votes are surprising, and how member-level measures enter the prediction. A natural case is the 2024 House Ukraine security supplemental final-passage vote, since it connects directly to the paper’s claim about libertarian non-interventionism and cross-party foreign-policy factions. Another good choice is the 2023 Fiscal Responsibility Act, which has visible party splits and institutional pressure. For one such vote, show the pre-vote text fields, the predicted probabilities for a small set of named members, the realized votes, the estimated cutpoint and discrimination, the relevant issue-position deviations, the loyalty residuals, and the resulting surprise scores. This would make the measurement claims much easier to calibrate than another aggregate plot.

**Bill cutpoints lack comparison to existing agenda measures**

Section 4 claims that the model makes standard agenda-control quantities immediate, but the bill-level cutpoints are not benchmarked against prior proposal, status-quo, or pivotal-politics estimates. The DW-NOMINATE comparison validates member positions, not the new rollcall cutpoints or the interpretation of their distribution as negative agenda control. This matters because the bill-level measurement is one of the paper’s main extensions beyond classical member scaling. Add a comparison using a standard agenda-control setting, such as the rollcalls studied in Richman (2011) or Peress (2013), and report whether the new cutpoints recover similar locations or classifications. If exact overlap is limited, compute the paper’s cutpoints for a small set of well-known majority-management votes—ACA repeal attempts, the 2017 tax bill, debt-ceiling votes, and Ukraine aid—and compare the inferred pivotal region with the documented coalition. A table breaking these comparisons out by final passage, amendments, and procedural votes would also clarify where the cutpoint interpretation is strongest.

**Recommendation**: major revision. The project has a strong empirical base and could make a valuable contribution, but the main claims about measurement, institutional limits, and text require more validation than the current version provides. The paper should narrow some headline statements and add targeted tests that separate prediction gains from construct-valid political measurement.

**Key revision targets**:

1. Revise the prediction-to-measurement framing by decomposing model signal into stable preference, institutional role, member history, metadata, and text components, and state which downstream measures use each component.
2. Add a full pre-vote feature audit and recompute the out-of-character measure using strictly prior information, or relabel it as an in-sample residual.
3. Clarify the prospective freeze evidence by aligning the Abstract, Section 2.3, and Table 2; identify the evaluated artifact and add uncertainty plus a longer prospective evaluation if available.
4. Strengthen the benchmark horse race with calibrated text-only, metadata-only, member-history-only, party-only, and dynamic ideal-point baselines under the same temporal and transfer regimes.
5. Add systematic validation and uncertainty for loyalty residuals, issue-specific positions, cutpoints, and surprise scores, including external benchmarks and historical majority-flip replications.

**Status**: [Pending]

---

## Detailed Comments (17)

### 1. Abstract should tie the “strongest predictor” claim to a comparison set

**Status**: [Pending]

**Quote**:
> We build the strongest vote predictor to date—an ensemble reading rollcall-level text with modern language-model encoders, anchored on member histories, with calibrated probabilities (86.2% accuracy on strictly future votes; 85.1% on contested votes; verified prospectively on 9,451 votes cast after the model was cryptographically frozen)—and then use it as a measurement instrument.

**Feedback**:
The phrase “strongest vote predictor to date” reads as a literature-wide ranking, but the abstract does not state the benchmark, comparison set, or metric on which that ranking rests. Since vote-prediction performance depends on the rollcall mix, temporal split, scoring rule, and definition of contested votes, the claim should be anchored to the evidence the paper directly reports. A safer version would be: “We build the strongest vote predictor on our benchmark...” That preserves the paper’s point without asking the abstract to carry more than the reported comparisons can show.

---

### 2. “Every history-based measure” is broader than the NOMINATE comparison allows

**Status**: [Pending]

**Quote**:
> Second, when majority control of the House changed in 2022, every history-based measure partially inverted: pooled member records *anti-transfer* across the flip (a member who reliably supported procedural motions in the majority reliably opposes them in the minority), our best model’s probabilities on the unseen congress are worse-calibrated than a three-way count table that merely knows who holds the chamber, and a simple logit on frozen DW-NOMINATE scores nearly matches the deep model’s transfer accuracy (62.0% versus 62.1%). The lesson is constitutional in the small: a large share of what any vote-based measure captures is institutional role, not stable preference, and measures built within one majority configuration should be carried across configurations only with care. That career-spanning NOMINATE scores—estimated from behavior in both roles—are the most durable object in our entire horse race is a quiet vindication of the traditional design.

**Feedback**:
The first sentence says “every history-based measure” partially inverted, but the paragraph then says career-spanning NOMINATE scores, which are also estimated from voting behavior, are the most durable object in the comparison. The intended distinction seems to be between pooled member-history features learned within one majority configuration and career-spanning spatial scores estimated across both majority and minority roles. Say that directly. For example: “the pooled member-history features learned within the prior majority configuration partially inverted.” That wording fits the procedural-role mechanism and avoids sweeping in the NOMINATE result that the paragraph wants to praise.

---

### 3. Signed ideal point is not the same as ideological extremity

**Status**: [Pending]

**Quote**:
> And the residual is essentially uncorrelated with ideological extremity ( $\rho = -.05$ ): the measure captures a dimension of behavior that position does not, which is precisely its purpose. The labeled extremes—who emerge from the full distribution rather than from selection—are the members journalists and whips would name: Walter Jones as the era's outstanding maverick, then Justin Amash, Jeff Van Drew (measured before his party switch—the residual saw it coming), Collin Peterson. The residual is discipline net of preference—the quantity party-unity scores are often mistaken for.

**Feedback**:
The reported correlation appears to be against the member’s signed ideal point, not against ideological extremity. Those are different quantities. Extremity usually means distance from the center, such as |x|. A variable can be uncorrelated with signed position while being a direct function of extremity: if X is symmetric around zero and R = X^2, then Cov(X,R)=0, even though R is entirely determined by |X|. If the calculation uses signed ideal point, revise the sentence to say so. If the intended claim is about extremity, report the correlation with absolute ideal point or another stated extremity measure.

---

### 4. The loyalty-residual tail is partly bounded by construction

**Status**: [Pending]

**Quote**:
> Using the instrument, we compute each member's expected behavior on party-unity votes given their position, and define the loyalty residual as actual minus expected unity-vote support. Figure 5 shows the complete recent-era distribution—every member, not a curated list—and its two systematic properties. The distribution is tight and asymmetric: most members are within a point or two of their position-predicted loyalty, with a long defector tail and almost no excess-loyalty tail, indicating that party pressure binds nearly everyone to a common ceiling while defection is where the individual variation lives.

**Feedback**:
The substantive interpretation is plausible, but the asymmetry also follows partly from the residual’s feasible range. If actual unity support is A in [0,1] and expected support is E in [0,1], then R=A−E must lie between −E and 1−E. When E is high, the positive tail is mechanically short. For instance, at E=0.95, the largest possible positive residual is 0.05, while the negative side can be much larger in magnitude. The text should acknowledge this bound before reading the lack of an excess-loyalty tail as evidence of party pressure. A modest fix is to say the pattern is “consistent with a loyalty ceiling,” while noting that the bounded residual scale also limits positive residuals when expected unity support is high.

---

### 5. Conditioning on one position does not remove all preference variation

**Status**: [Pending]

**Quote**:
> And the residual is essentially uncorrelated with ideological extremity ( $\rho = -.05$ ): the measure captures a dimension of behavior that position does not, which is precisely its purpose. The labeled extremes—who emerge from the full distribution rather than from selection—are the members journalists and whips would name: Walter Jones as the era's outstanding maverick, then Justin Amash, Jeff Van Drew (measured before his party switch—the residual saw it coming), Collin Peterson. The residual is discipline net of preference—the quantity party-unity scores are often mistaken for.

**Feedback**:
The final sentence overstates what the adjustment can identify. If actual party-unity support depends on the position variable P, an unmodeled issue-specific preference Q, and discipline D, then a residual from an expected-support model given P still contains the part of Q not captured by P. In symbols, if A=f(P)+g(Q)+D, subtracting E[A|P] leaves g(Q)−E[g(Q)|P] as well as the discipline component. This matters because the paper itself argues that issue-specific positions contain meaningful structure beyond the overall dimension. A better phrasing would be: “The residual is discipline net of the position variable used in the expected-support model, though it may also contain preference variation not captured by that position.”

---

### 6. Issue-domain counts and coverage ranges do not match Table 3

**Status**: [Pending]

**Quote**:
> |  Policy area | N | Party gap | SD (D) | SD (R) | ρ(overall) | Largest liberal dev.  |
> | --- | --- | --- | --- | --- | --- | --- |
> |  Economics & Public Finance | 1,562 | 1.87 | 0.37 | 0.42 | 0.99 | Mannion (NY)  |
> |  Armed Forces & National Security | 1,495 | 1.73 | 0.49 | 0.37 | 0.97 | Roy (TX)  |
> |  Congress | 1,399 | 2.05 | 0.27 | 0.21 | 0.95 | Brecheen (OK)  |
> |  International Affairs | 1,297 | 1.73 | 0.43 | 0.46 | 0.97 | Paul (KY)  |
> |  Health | 1,189 | 2.06 | 0.31 | 0.29 | 0.97 | Boebert (CO)  |
> |  Energy | 1,178 | 2.12 | 0.41 | 0.32 | 0.96 | Good (VA)  |
> |  Crime & Law Enforcement | 1,086 | 1.97 | 0.45 | 0.32 | 0.96 | Burlison (MO)  |
> |  Finance & Financial Sector | 1,072 | 1.99 | 0.38 | 0.26 | 0.97 | Good (VA)  |
> |  Transportation & Public Works | 1,040 | 1.93 | 0.28 | 0.41 | 0.97 | Mrvan (IN)  |
> |  Government Operations & Politics | 1,026 | 2.05 | 0.33 | 0.28 | 0.97 | Scott (SC)  |
> |  Public Lands & Natural Resources | 934 | 2.02 | 0.37 | 0.34 | 0.97 | Donalds (FL)  |
> |  Taxation | 917 | 2.11 | 0.30 | 0.20 | 0.97 | Flake (AZ)  |
> |  Nominations | 328 | 2.06 | 0.34 | 0.29 | 0.98 | Kyl (AZ)  |

**Feedback**:
The prose and table are not aligned. The phrase “economics, defense, crime, health, and ten other domains” implies fourteen domains, but Table 3 reports thirteen rows. The coverage range also does not match the table: including all rows, N runs from 328 to 1,562; excluding Nominations as a non-major area, it runs from 917 to 1,562, not roughly 800 to 1,400. This is easy to fix. Refer to “the thirteen reported domains,” and state the coverage range from the table, for example: “Excluding Nominations, Table 3 reports 917 to 1,562 members per area under the fifty-vote rule; Nominations has 328.”

---

### 7. High correlations do not by themselves show that issue deviations are structured

**Status**: [Pending]

**Quote**:
> Three systematic facts emerge. First, one dimension is almost sufficient—within-area positions correlate with overall positions at  $\rho = .95 - .99$  (Figure 6)—which is precisely why the deviations that remain are informative rather than noise: they are the structured residue that survives a near-perfect one-dimensional fit.

**Feedback**:
The high correlations show that most between-member variation is one-dimensional. They do not, by themselves, show that the residual deviations are structured rather than small random errors. A simple counterexample makes the point: let an overall score X have Var(X)=1, and let a within-area score be Y=X+ε, where ε is independent noise with Var(ε)=1/0.98^2−1≈0.041. Then Corr(X,Y)=0.98, while the deviation Y−X is entirely noise. The stronger interpretation may still be supported by the factional patterns discussed later, but the correlation is not the evidence for that step. Rephrase this sentence so the correlation establishes the near-one-dimensional baseline, and let the named factional deviations carry the claim about structure.

---

### 8. The “widest within-party spreads” claim is too broad

**Status**: [Pending]

**Quote**:
> |  Policy area | N | Party gap | SD (D) | SD (R) | ρ(overall) | Largest liberal dev.  |
> | --- | --- | --- | --- | --- | --- | --- |
> |  Economics & Public Finance | 1,562 | 1.87 | 0.37 | 0.42 | 0.99 | Mannion (NY)  |
> |  Armed Forces & National Security | 1,495 | 1.73 | 0.49 | 0.37 | 0.97 | Roy (TX)  |
> |  Congress | 1,399 | 2.05 | 0.27 | 0.21 | 0.95 | Brecheen (OK)  |
> |  International Affairs | 1,297 | 1.73 | 0.43 | 0.46 | 0.97 | Paul (KY)  |
> |  Health | 1,189 | 2.06 | 0.31 | 0.29 | 0.97 | Boebert (CO)  |
> |  Energy | 1,178 | 2.12 | 0.41 | 0.32 | 0.96 | Good (VA)  |
> |  Crime & Law Enforcement | 1,086 | 1.97 | 0.45 | 0.32 | 0.96 | Burlison (MO)  |
> |  Finance & Financial Sector | 1,072 | 1.99 | 0.38 | 0.26 | 0.97 | Good (VA)  |
> |  Transportation & Public Works | 1,040 | 1.93 | 0.28 | 0.41 | 0.97 | Mrvan (IN)  |
> |  Government Operations & Politics | 1,026 | 2.05 | 0.33 | 0.28 | 0.97 | Scott (SC)  |
> |  Public Lands & Natural Resources | 934 | 2.02 | 0.37 | 0.34 | 0.97 | Donalds (FL)  |
> |  Taxation | 917 | 2.11 | 0.30 | 0.20 | 0.97 | Flake (AZ)  |
> |  Nominations | 328 | 2.06 | 0.34 | 0.29 | 0.98 | Kyl (AZ)  |

**Feedback**:
The party-gap statement is supported by the table: Energy and Taxation have the largest gaps, while Armed Forces and International Affairs have the smallest. The within-party-spread claim needs narrowing. For Democrats, the largest SD is Armed Forces at 0.49, followed by Crime at 0.45 and International Affairs at 0.43. For Republicans, the largest SD is International Affairs at 0.46, followed by Economics at 0.42 and Transportation at 0.41; Armed Forces is 0.37. So the table supports a claim that foreign-policy areas have high within-party dispersion, not that both closest-gap areas are the widest in both parties. A more accurate sentence would be: “within-party dispersion is also high: Armed Forces has the largest Democratic spread and International Affairs has the largest Republican spread.”

---

### 9. Left of the floor median does not alone determine the party coalition

**Status**: [Pending]

**Quote**:
> Cutpoint locations relative to the chamber’s pivotal actors are the raw material of the agenda-control literature *(Richman, 2011; Peress, 2013; Aldrich, Montgomery and Sparks, 2014)*, and the full-agenda measurement makes the standard quantities immediate. In the 118th House, only 15.0% of the 984 identified cutpoints fall between the floor median (0.48) and the majority-party median (1.77)—the region where a vote splits the majority party against its own median—while 69.3% fall to the *left* of the floor median, where the Republican conference votes together against most Democrats. That asymmetry—an agenda engineered to divide the opposition rather than the majority—is negative agenda control rendered as a histogram, computed from every recorded vote of the congress rather than a curated subset.

**Feedback**:
A cutpoint left of the floor median tells readers that the floor median and members farther right are on the same side of the threshold, assuming a fixed discrimination orientation. It does not, by itself, show that most Democrats are on the other side. If the Democratic median were −1 and a cutpoint were −2, the cutpoint would still be left of the floor median, but most Democrats and Republicans would lie on the same side. The sentence also needs to account for the sign of the discrimination parameter, since that determines which side corresponds to yea support. Revise the claim to condition on cutpoints that fall between the Democratic median and the floor median, and note that the coalition interpretation depends on discrimination orientation.

---

### 10. The ledger claim needs an auditable summary in the paper

**Status**: [Pending]

**Quote**:
> Every development experiment was pre-registered in the repository's ledger before its first run, with a hypothesis, a pre-vote-knowability argument for any new feature, and a falsification test. Selection used validation sets only; the three-model final comparison observed the test sets once. The ledger records, with diagnosed mechanisms: the encoder-scale negative (a  $16 \times$  larger text budget does not improve forecasts and worsens long-bill votes); the amendment-text negatives and their placebo control (Section 5); the recency-weighting negative (decayed member rates add estimator noise exceeding any drift signal); the within-bill context negative (features that pass synthetic tests at the information-theoretic optimum destabilize joint training on real data); the per-bucket calibration negative; and the regime-inheritance result (post-hoc parameters fit on a temporal development slice transfer to temporal evaluation and break on interleaved evaluation). Full reproduction instructions accompany the repository.

**Feedback**:
This paragraph makes a strong audit claim, but the appendix gives only a narrative summary. Readers cannot see run IDs, timestamps, hashes, split names, or the preregistered falsification criteria. Since the paper uses the ledger as evidence against specification search, the manuscript should contain enough of the ledger to make the claim checkable without leaving the paper. Add a compact table with experiment ID, preregistration timestamp, first-run timestamp, hypothesis, new feature, pre-vote-knowability argument, falsification test, validation split, whether test data were touched, and outcome. The repository can carry the full record, but the paper should show the audit trail for the experiments it relies on most.

---

### 11. Amendment-text coverage needs a denominator

**Status**: [Pending]

**Quote**:
> Amendment votes carry roughly thirty percent of the instrument's remaining error (Table 5), and two-thirds of amendment rollcalls lack amendment-specific text—so supplying the amendments' actual purpose language, parsed from BILLSTATUS records and joined through recorded-vote references, looked like the obvious improvement. Pre-registered, it failed three ways, and the failure is the finding.

**Feedback**:
The experiment’s coverage is hard to read from this sentence. Did the BILLSTATUS join fill most of the two-thirds of amendment rollcalls that lacked amendment-specific text, or only a smaller subset with usable recorded-vote references? That denominator matters because a near-complete join and a selected partial join support different interpretations of the negative result. Add a sentence such as: “The BILLSTATUS join added nonempty purpose language for N of M amendment rollcalls, including N_missing of the M_missing rollcalls that previously lacked amendment-specific text.”

---

### 12. The evaluation regimes vary along several axes, not one difficulty scale

**Status**: [Pending]

**Quote**:
> A model's fitness as a measurement instrument depends on what kind of prediction it is good at, so we evaluate under regimes of increasing difficulty: completion (random individual votes held out, same-rollcall votes observable—the psychometric setting in which ideal points are traditionally assessed); random rollcall holdout (whole rollcalls held out, interleaved in time—the 2011–2016 literature's setting); temporal forecast (the final ten percent of each congress-chamber's rollcalls—a true forecaster's setting); congress-out transfer (train through the 116th Congress, predict the 118th, across the majority flip); and prospective (hash-pinned frozen models scored only on votes that occur after freezing, following the logic of the reusable holdout problem (Dwork et al., 2015)). Our primary metric is log loss—a proper scoring rule that rewards honest probabilities—because measurement consumers use the probabilities, not just the classifications; accuracy, AUC, contested-vote accuracy, and APRE are reported throughout for comparability.

**Feedback**:
The regimes are well chosen, but “increasing difficulty” makes them sound like a single ordered ladder. They are not. A hash-pinned prospective test guards against post-freeze adaptation, but it may be a short same-Congress horizon; congress-out transfer across a majority flip is a different distributional stress test. Random rollcall holdout changes the held-out unit while remaining interleaved in time. A clearer phrase would be “regimes with different and progressively stricter information constraints.” That captures the design without implying a monotone difficulty ordering that the definitions do not establish.

---

### 13. Figure 5’s “every member” claim needs the caption’s eligibility rule

**Status**: [Pending]

**Quote**:
> esidual as actual minus expected unity-vote support. Figure 5 shows the complete recent-era distribution—every member, not a curated list—and its two systematic properties. The distribution is tight and asymmetric: most members are within a point or two of their position-predicted loyalty, with a long defector tail and almost no excess-loyalty tail, indicating that party pressure binds nearly everyone to a common ceiling while defection is where the individual variation lives. And the residual is essentially uncorrelated with ideological extremity ( $\rho = -.05$ ): the measure captures a dimension of behavior that position does not, which is precisely its purpose. The labeled extremes—who emerge from the full distribution rather than from selection—are the members journalists and whips would name: Walter Jones as the era's outstanding maverick, then Justin Amash, Jeff Van Drew (measured before his party switch—the residual saw it coming), Collin Peterson. The residual is discipline net of preference—the quantity party-unity scores are often mistaken for.
> 
> <!-- PAGE BREAK -->
> 
> Party loyalty beyond ideology, 115th-119th Congresses
> 
> ![img-5.jpeg](img-5.jpeg)
> Figure 5: The loyalty residual for every member serving in the 115th-119th Congresses (at least 100 party-unity votes; territorial delegates excluded). Left: the full distribution by party. Right: the residual against the member's ideal point, with the most extreme residuals labeled. Blue: Democr

**Feedback**:
The main text says the figure shows “every member,” while the caption restricts the plotted set to members with at least 100 party-unity votes and excludes territorial delegates. Those are reasonable restrictions, but they should be stated in the prose. Replace the unqualified phrase with something like: “Figure 5 shows the recent-era distribution for every eligible member under the caption’s inclusion rule.”

---

### 14. Log loss should not be described as calibration alone

**Status**: [Pending]

**Quote**:
> The 118th Congress column of Table 1 is the deepest lesson. Trained through 2021 and asked about 2023–24—across the House majority flip—the champion’s probabilities are worse-calibrated than a count table keyed only on majority status, chamber, and question type (0.643 versus 0.617); pooled member histories are outright poisonous (log loss 0.740), because the member who reliably supported procedural motions as a majority member reliably opposes them in the minority.

**Feedback**:
The parenthetical values are log losses, not calibration statistics. Log loss combines calibration, discrimination, and sharpness: a model can have worse log loss because it ranks cases poorly even if its probability bins are reasonably calibrated. Unless the paper also reports a reliability curve, expected calibration error, or a calibration decomposition for this comparison, the sentence should say “higher log loss,” not “worse-calibrated.” The substantive transfer point still stands with that wording.

---

### 15. The TF-IDF omission rationale needs the missing implementation condition

**Status**: [Pending]

**Quote**:
> The dashboard wraps the frozen champion's MiniLM component tower (the blend's TF-IDF component requires corpus-level context a single pasted bill lacks; the MiniLM tower alone carries the largest blend weight). Pasted text is assembled into the training text template and embedded with the training encoder; a synthetic bill record carries the user-specified sponsoring party through the same features used in training; member vote probabilities are computed for the full current roster; and the displayed cutpoint is the position at which predicted support crosses one half, from a logistic fit of predicted votes on the current congress's estimated positions—mirroring the paper's cutpoint estimation

**Feedback**:
The implementation choice is understandable, but the stated reason does not follow for standard frozen TF-IDF. If the training corpus fixes the idf weights, a new pasted document can be transformed using its term frequencies and those fixed idf values; no inference-time corpus is needed for that ordinary construction. If this tower requires some additional corpus-relative state, retrieval step, normalization, or deployment resource, name it. Otherwise the sentence overstates why the dashboard cannot include the TF-IDF component.

---

### 16. The dashboard cutpoint rule needs an explicit range and slope condition

**Status**: [Pending]

**Quote**:
> the displayed cutpoint is the position at which predicted support crosses one half, from a logistic fit of predicted votes on the current congress's estimated positions—mirroring the paper's cutpoint estimation

**Feedback**:
A logistic curve can cross one half far outside the observed member range. If q(x)=1/(1+exp(−(10+x))) and current-member positions lie in [−1,1], every member has fitted support above 0.999, but the fitted crossing is still x=−10. So “near-unanimous predicted support” does not itself imply “no identified cutpoint.” Add the display rule: report a cutpoint only when the fitted crossing lies within the observed range of current-member positions and the fitted slope exceeds a stated minimum; otherwise suppress it as unidentified or extrapolated.

---

### 17. “Post-hoc parameters” is ambiguous in a preregistration paragraph

**Status**: [Pending]

**Quote**:
> Selection used validation sets only; the three-model final comparison observed the test sets once. The ledger records, with diagnosed mechanisms: the encoder-scale negative (a  $16 \times$  larger text budget does not improve forecasts and worsens long-bill votes); the amendment-text negatives and their placebo control (Section 5); the recency-weighting negative (decayed member rates add estimator noise exceeding any drift signal); the within-bill context negative (features that pass synthetic tests at the information-theoretic optimum destabilize joint training on real data); the per-bucket calibration negative; and the regime-inheritance result (post-hoc parameters fit on a temporal development slice transfer to temporal evaluation and break on interleaved evaluation). Full reproduction instructions accompany the repository.

**Feedback**:
The phrase “post-hoc parameters” sits awkwardly in a paragraph meant to reassure readers about preregistration and validation-only selection. If the parameters were fitted after the base model but only on the temporal development slice, say that. If they were chosen after looking at evaluation behavior, they should be separated from the validation-only claim. A clean revision would be: “parameters fitted only on the temporal development slice transfer to temporal evaluation and break on interleaved evaluation.”

---
