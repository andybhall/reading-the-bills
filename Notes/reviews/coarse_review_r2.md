# Measuring Congress in the AI Era: What Modern Vote Prediction Reveals About Members, Bills, and Institutions

**Date**: 07/05/2026
**Domain**: social_sciences/political_science
**Taxonomy**: academic/working_paper
**Filter**: Active comments

---

## Overall Feedback

Here are some overall reactions to the document.

**Outline**

The paper is ambitious and unusually transparent about prediction regimes, prospective scoring, and negative results. Its strongest contribution is the disciplined benchmark for congressional vote prediction and the evidence that modern text/member-history models improve within-congress forecasting while failing across some institutional transitions. The main weakness is that several measurement claims run ahead of what the design establishes, especially where quantities are derived from spatial refits or in-sample residuals rather than the advertised forecasting instrument.

The paper has the ingredients of a strong contribution: a large benchmark, careful temporal splits, a model registry, and a serious attempt to connect prediction to congressional measurement. The distinction between within-congress prediction, cross-congress transfer, and prospective scoring is valuable. Still, the current draft needs a tighter alignment between the prediction exercise and the substantive measures before the claims about ideology, loyalty, issue positions, amendments, and institutional role are fully convincing.

**Feature-timing claims need a stronger audit than Appendix B provides**

Section 2.1 states that every forecast uses only information observable before the roll call, and Section 2.4 makes this timing discipline central to the model’s credibility. Appendix B, however, gives only a compact list of timestamp rules, and several high-value inputs deserve more proof. Voteview question and description are described as “fixed at the vote,” but roll-call descriptions in public datasets can be cleaned, standardized, or linked after the fact; if they encode procedural context or bill identifiers not available before the vote, the temporal forecast is less strict than claimed. CRS policy areas and bill summaries are also tricky: the appendix says policy metadata is revisable and excluded from leakage-clean text variants, but Section 3.3 uses policy areas to estimate issue-specific positions, and Section 4.2 uses bill category and metadata in cutpoint prediction. The amendment join in Sections 4.2 and 5.1 also relies on recorded-vote references and a session-offset correction, which may be perfectly valid, but the paper does not show a row-level timing audit or examples proving that these joins would have been possible before each vote. This matters because the paper’s main contribution is not only accuracy but measurement from a supposedly pre-vote instrument. A revision should add a feature provenance table with, for each feature used in Tables 1, 2, 6, and 10, the source timestamp, the exact exclusion rule when the timestamp is missing or after the vote, and an ablation that reruns the headline temporal and cutpoint results using only the most conservative leakage-clean feature set.

**The measurement objects are not always the champion model’s internals**

The abstract and introduction say the ensemble is used as a measurement instrument and that its internals recover positions, loyalty, issue positions, bill cutpoints, and out-of-character votes. In the body, several of these measures appear to come from separate spatial fits rather than from the ensemble. Section 2.2 defines spatial fits as estimating P(yea)=sigma(a_v x_i+b_v+c_i), Section 3.1 validates positions from the “benchmark’s one-dimensional spatial model,” Section 3.3 refits that spatial model within policy areas, and Section 4 estimates cutpoints from realized roll calls. Only the out-of-character score in Section 3.4 clearly uses the frozen ensemble probabilities. This creates a mismatch between the prediction claims in Section 2 and the measurement claims in Sections 3 and 4: readers cannot tell which quantities are learned by the deep text model, which are conventional spatial estimates run on the same data, and which are post-vote summaries of realized coalitions. The distinction matters because a spatial refit can recover DW-NOMINATE-like positions even if the language model contributes little to construct-valid measurement. The paper should add a single measurement map showing, for each reported quantity, the estimator, training data, whether the vote being measured is included, and whether text enters the estimate; the abstract and Section 1 should then be revised so that “model internals” is reserved for quantities actually produced by the ensemble.

**Construct validity is asserted more strongly than the evidence supports**

Section 3.1 treats the r=.98 correlation with DW-NOMINATE as evidence that “two instruments” discover the same object, but DW-NOMINATE is another behavior-based roll-call scale, not an external ground truth for ideology. Section 3.2 labels the unity-vote residual as loyalty beyond ideology, yet the residual can also contain issue-specific preference variation, district pressure, agenda composition, vote-type mix, chamber effects, or model misspecification; the paragraph acknowledges some of this but still uses the maverick interpretation quite strongly. Section 3.3 says issue-specific positions recover libertarian non-interventionism and hawkish Democrats “from votes alone,” but the evidence shown is mostly high correlations with the overall dimension plus named outliers in Table 4 and Figure 6. These are useful face-validity checks, not enough validation for new measures intended for broad use. The paper should add external validation exercises tailored to each construct: for loyalty, compare the residuals with CQ party unity scores, whip-vote lists, or known party-switch/defection episodes; for issue positions, compare domain scores with interest-group ratings or hand-coded issue scales; for out-of-character votes, evaluate whether top residuals predict contemporaneous news coverage, primary threats, retirements, or district-level cross-pressure better than simpler DW-NOMINATE residuals. Without those checks, the measures are promising diagnostics rather than validated constructs.

**Uncertainty and dependence are underdeveloped for member- and bill-level measures**

The paper reports millions of individual voting decisions, but the effective sample size is far smaller because votes are clustered by roll call, member, party, chamber, Congress, bill, and vote type. Section 2.4 gives a rollcall-cluster bootstrap interval for the small prospective ledger, which is a good start, but the same discipline is not carried through to the measurement sections. Section 3.2 says binomial sampling error is at most about five points for the loyalty residual with 100 unity votes, but unity votes are not independent Bernoulli trials: members face party strategies, repeated bills, procedural series, and common shocks. Section 3.3 reports issue-specific deviations with a fifty-vote threshold, yet no uncertainty intervals or stability checks indicate whether labeled outliers in Table 4 are distinguishable from noise after conditioning on party, chamber, and Congress. Section 4 reports 984 identified cutpoints in the 118th House and percentages in agenda-control regions, but gives no uncertainty over member positions, discriminations, cutpoints, or the |a_v| identification threshold. This matters because the paper asks readers to use the released scores as measures, not just predictions. A revision should add uncertainty estimates for loyalty residuals, issue deviations, and cutpoints using a rollcall- or bill-clustered bootstrap, posterior/parametric simulation from the spatial fits, or split-half stability tests, and the tables should identify which named cases remain extreme under those intervals.

**The majority-flip result is important but framed too broadly**

Section 5.2 contains one of the paper’s most interesting findings, but the wording sometimes exceeds the evidence. The abstract says “no model survives a change in majority control,” while the section later narrows the claim to the procedural-role component of voting behavior. The tables also point in different directions depending on the metric: in Table 1 the majority-status count table beats the champion on new-Congress log loss, but the champion has higher accuracy; in Table 2 the text-plus-metadata model without member history has the best congress-out accuracy among the listed ablations, while its log loss is poor. That pattern suggests a calibration and class-imbalance story, not simply a failure of predictive signal. The evidence across House transitions is useful, but the paper does not show the analogous Senate analysis, chamber-specific base rates, or whether the finding depends on including procedural votes that dominate the role mechanism. The conclusion that “a large share of what any vote-based measure captures is institutional role, not stable preference” should be tied more tightly to vote types and chambers where the paper actually demonstrates it. A revision should decompose the majority-flip result by chamber, party, question type, contested status, and calibration, and should rewrite the headline claim as a claim about procedural and majority-role behavior unless broader evidence is added.

**The amendment interpretation needs evidence beyond a failed text ablation**

Section 5.1 argues that amendment coalitions are organized by deals rather than content because adding amendment purpose text makes prediction worse. The negative result is interesting, but the inference from “purpose text does not improve this model” to “deals, not content” is too strong. Amendment purpose fields are short, formulaic, and missing for many roll calls; they may also be bad proxies for the policy content of amendments, especially if the relevant meaning is in legislative language, floor debate, structured-rule context, or the parent bill’s bargaining history. The mechanism offered—that replacing parent-bill text with amendment purpose text breaks useful pooling within amendment series—is plausible, but that points to model design and information representation as much as to the political nature of amendments. The paper should distinguish three hypotheses: amendment text is substantively uninformative, amendment purpose fields are too thin, and amendment coalitions depend on actor/procedural context. A stronger test would compare parent-bill text, amendment purpose, full amendment text where available, amender party, amendment sequence, rule type, and within-bill random effects under the same temporal split; if actor/procedure features dominate full amendment language, the substantive conclusion would be much better supported.

**The out-of-character measure mixes retrospective fit with forecast surprise**

Section 3.4 is careful to call the historical scores in-sample residuals, but the surrounding framing still risks conflating two different objects. The frozen champion is fit on each member’s entire record through mid-2026 and then asked about historical votes, so the score measures incompatibility with a full-record behavioral summary, not what was surprising before the vote. That is a valid descriptive residual, but it is not the same as “votes no model saw coming,” a phrase used in Sections 1 and 6. It also means late-career behavior can inform the model’s assessment of early-career votes, and repeated patterns can be learned away after the fact, as the discussion of Squad protest voting notes. For studying persuasion, pressure, and idiosyncratic representation, the forecast-based version is usually the more relevant estimand. The paper should present the retrospective residual and the strictly pre-vote residual as separate measures, report their correlation, and show examples where they diverge; Figure 8 and the released ranking should label clearly which version is being used.

**The agenda-control and cutpoint claims need stronger validation against institutional baselines**

Section 4 uses cutpoints from spatial fits to describe the 118th House agenda and interprets the distribution as negative agenda control rendered as a histogram. That is a promising use of the model, but the current evidence does not yet separate agenda control from the realized mix of vote types, party-line procedural votes, consensus votes, and the orientation conventions of the fitted discriminations. The threshold for reporting cutpoints, |a_v| >= 0.35 in Section 2.2, may also shape the 984 identified cutpoints used for the 15.0% and 69.1% figures, yet the paper does not show sensitivity to that cutoff. Table 6 shows that metadata alone predicts cutpoints nearly as well as text plus metadata, which means the institutional baseline is doing much of the work; the interpretation should reflect that more directly. Readers will want to know whether the histogram looks the same for final passage only, for contested votes only, for non-procedural votes, and under two-dimensional or chamber-congress-specific spatial specifications. A revision should add those sensitivity checks and compare the cutpoint-based agenda-control quantities with established hand-coded or theory-driven subsets from the agenda-control literature.

**Horse-race gains lack paired uncertainty checks**

The model comparison tables are central to the paper’s claim that the ensemble is the strongest predictor on this benchmark, but most entries are reported as point estimates only. The sole interval appears for the small prospective ledger; Tables 1, 2, 9, and 10 do not show whether differences such as 0.337 versus 0.324 log loss, or 85.6% versus 86.2% accuracy, are stable under the rollcall-level dependence in the data. This matters because the paper uses these margins to justify which model becomes the measurement instrument. A revision should add paired rollcall-cluster bootstrap intervals, or a block bootstrap by bill within chamber-congress, for the log-loss and accuracy differences between the champion, the two-tower blend, the MiniLM tower, the member-history table, and the NOMINATE-context logit. The same table should break the paired differences out by House/Senate, Congress, question type, and contested versus lopsided votes, so readers can see where the ensemble’s advantage is doing real work.

**Champion architecture is not specified enough**

The paper describes the final model as a logistic blend of three two-tower networks, but the published text does not give a full recipe for the instrument whose outputs drive the later measures. A benchmark paper can rely on a repository for code, but the article still needs enough detail for readers to know what was estimated: encoder names, whether encoders were frozen, embedding dimensions, member-history construction, text templates, missing-text handling, optimizer, loss, training windows, calibration method, and blend weights. Without that, the model is hard to compare with future work, and the negative results in Appendix D are harder to interpret. Add a model-specification table for each tower and for the blend, including the exact pre-vote inputs used in the temporal and congress-out regimes. A short “replication target” paragraph should also state the compute budget and the command or pipeline stage that regenerates Table 1 from raw Voteview and BILLSTATUS files.

**The informational-bound claim needs a demonstration**

The abstract and introduction build the paper around the claim that prediction quality bounds measurement quality, but the paper does not yet make that claim precise. The empirical sections show that better predictors recover useful patterns, yet they do not define the class of measures for which the bound holds or connect log loss to recoverability of ideology, loyalty, issue positions, or cutpoints. This matters because the statement is the paper’s framing contribution, not a side remark. The proof would benefit from a short proposition: if two member-bill states induce the same conditional distribution of votes given the available information, then any measure derived from that information and prediction rule cannot distinguish them; a simple Bayes-error or mutual-information version would be enough. Pair that with a worked semi-synthetic example on the rollcall panel: inject an “ends-against-the-middle” faction or an issue-specific defense bloc, fit a member-rate table, a one-dimensional spatial model, and the champion, and show that only the models with improved heldout scoring recover the planted member positions or cutpoints.

**Recommendation**: major revision. The paper has a strong benchmark and several promising measurement ideas, but the current version overstates the link between prediction accuracy and validated congressional measures. The main results may well survive, but the paper needs clearer estimator provenance, stronger leakage documentation, and more direct construct validation before it would meet the standard for a top political methodology or legislative studies venue.

**Key revision targets**:

1. Add a row-level feature-timing audit and rerun the headline temporal and cutpoint results using a conservative leakage-clean feature set.
2. Create a measurement provenance table identifying which quantities come from the ensemble, which from spatial refits, which are in-sample, and which are true forecasts; revise the abstract and Section 1 accordingly.
3. Validate loyalty residuals, issue-specific positions, and out-of-character scores against external benchmarks or clearly defined historical outcomes, including comparisons to simpler DW-NOMINATE and party-history residuals.
4. Add uncertainty and stability estimates for member residuals, issue deviations, and bill cutpoints using clustering or split-sample methods that respect roll-call dependence.
5. Narrow and decompose the majority-flip and amendment claims by vote type, chamber, calibration, and information source, with additional tests separating text limitations from procedural or actor-driven explanations.

**Status**: [Pending]

---

## Detailed Comments (25)

### 1. Spatial scale is not identified before applying the cutpoint threshold

**Status**: [Pending]

**Quote**:
> The *spatial fits* estimate $P(\text{yea})=\sigma(a_{v}x_{i}+b_{v}+c_{i})$ per chamber-congress with party-sign initialization; a rollcall’s *cutpoint* is $-(b_{v}+\bar{c})/a_{v}$, reported only when $|a_{v}|\geq 0.35$ (the location is unidentified for near-valence votes), and its *direction* is $\text{sign}(a_{v})$.

**Feedback**:
The spatial likelihood fixes probabilities through the product $a_v x_i$, not through the separate scales of $a_v$ and $x_i$. For any $s>0$ and $t$, setting $x'_i=sx_i+t$, $a'_v=a_v/s$, $b'_v=b_v-a_vt/s$, and $c'_i=c_i$ leaves $a'_v x'_i+b'_v+c'_i=a_vx_i+b_v+c_i$. Party-sign initialization can choose orientation, but it does not by itself fix location or scale. That matters here because the reporting rule changes under rescaling: $|a'_v|=|a_v|/s$, so the $0.35$ identification threshold has no stable meaning until the scale is fixed. The definition should state the normalization used within each chamber-congress, such as centering and scaling $x_i$, centering $c_i$, and applying the $|a_v|\geq0.35$ threshold only on that standardized scale.

---

### 2. The loyalty-residual formula is on the yea scale, not the party-support scale

**Status**: [Pending]

**Quote**:
> The *loyalty residual* is $\bar{y}_{i}^{U}-\overline{\hat{p}}_{i}^{U}$ over member $i$’s party-unity votes $U$ (majorities of the two parties opposed), where the expectation comes from the spatial fit; members need at least 100 unity votes.

**Feedback**:
As written, $\hat p_{iv}$ is a yea probability, so $\bar y_i^U-\overline{\hat p}_i^U$ measures excess yeas relative to the model. That is not the same as excess support for the member’s party on unity votes. A two-vote example shows the problem. Suppose the member’s party majority is yea on vote 1 and nay on vote 2; a loyal pattern $(1,0)$ and a disloyal pattern $(0,1)$ both have mean yea rate $1/2$. If the expected yea probabilities are $(0.9,0.1)$, both patterns also give residual $1/2-(0.9+0.1)/2=0$. The definition should recode the outcome to party support: let $m_{iv}=1$ when the member’s party majority votes yea and $0$ otherwise, define $z_{iv}=1\{y_{iv}=m_{iv}\}$ and $q_{iv}=m_{iv}\hat p_{iv}+(1-m_{iv})(1-\hat p_{iv})$, then compute the residual as $\bar z_i^U-\bar q_i^U$.

---

### 3. The June scoring window is post-snapshot, not fully post-freeze

**Status**: [Pending]

**Quote**:
> The single-tower predecessor (v1) was frozen on June 12, 2026 (weights hash-pinned, fit through the June 9 data snapshot) and scored 0.331 log loss and  $85.7\%$  accuracy on the 43 rollcalls and 9,451 votes Congress cast June 10–30 (rollcall-cluster bootstrap  $95\%$  interval for the log loss: 0.265–0.402). That interval is wide—43 rollcalls is a contamination check, not a precision instrument—and its value is what it rules out: the development-test numbers are not inflated by adaptive iteration. The ensemble champion (v2) was frozen on July 3, 2026 on the same data snapshot; scored on the same window it ties the predecessor on log loss with one point higher accuracy, and its pristine window (votes after July 3) accumulates going forward.

**Feedback**:
The dates do not support the full post-freeze language. For v1, the freeze date is June 12, but the scoring window begins June 10. For v2, the freeze date is July 3, while the entire June 10–30 scoring window precedes the freeze. The June 10–30 results are therefore a post-snapshot check relative to the June 9 data snapshot, not a fully post-freeze prospective certificate for the entire window, and they cannot certify v2 against adaptive choices made before July 3. The text should separate three objects: v1’s post-snapshot June 10–30 score, the subset of v1 votes after June 12 if that is meant as a post-freeze test, and v2’s genuinely post-freeze ledger after July 3.

---

### 4. The stated cutpoint is an average-member threshold

**Status**: [Pending]

**Quote**:
> The *spatial fits* estimate $P(\text{yea})=\sigma(a_{v}x_{i}+b_{v}+c_{i})$ per chamber-congress with party-sign initialization; a rollcall’s *cutpoint* is $-(b_{v}+\bar{c})/a_{v}$, reported only when $|a_{v}|\geq 0.35$ (the location is unidentified for near-valence votes), and its *direction* is $\text{sign}(a_{v})$.

**Feedback**:
Under the fitted model, the probability-one-half point is member-specific when $c_i$ varies. Since $\sigma(\eta)=1/2$ exactly when $\eta=0$, the halfway point for member $i$ satisfies $a_vx_i+b_v+c_i=0$, so $x_i=-(b_v+c_i)/a_v$ when $a_v\neq0$. The reported $-(b_v+\bar c)/a_v$ is therefore the threshold for a member with $c_i=\bar c$, not a universal rollcall threshold. That may be the right rollcall-level summary, but the convention should be named explicitly, for example as an “average-member cutpoint.”

---

### 5. The text-era benchmark is conflated with the full 101st–119th vote panel

**Status**: [Pending]

**Quote**:
> On an open benchmark of 10.6 million voting decisions (101st–119th Congresses), we develop an ensemble model that reads rollcall-level text with sentence encoders, maps it to vote-choice parameters through deep heads, anchors on each member’s behavioral history, and reports calibrated probabilities.

**Feedback**:
This sentence reads as though the text-reading ensemble is trained and evaluated on Congresses 101–119. Section 2.1 later says the forecasting results use the text-era Congresses only, 108th–119th, while the earlier Congresses enter only non-text completion-regime spatial fits. That scope distinction matters because the sentence ties the 101st–119th benchmark directly to sentence encoders and rollcall-level text. Revise the introduction to distinguish the full Voteview panel from the text-era forecasting benchmark, so readers do not infer that the language-model ensemble uses comparable bill text for the 101st–107th Congresses.

---

### 6. The congress-out design skips the 117th Congress

**Status**: [Pending]

**Quote**:
> *congress-out transfer* (train through the 116th Congress, predict the 118th, across the majority flip); and *prospective* (hash-pinned frozen models

**Feedback**:
The 118th House majority flip follows the 117th Congress. Training only through the 116th and testing on the 118th therefore skips an intervening Congress, rather than simply crossing from the immediately prior majority configuration into the next one. That design may be deliberate, but the label should say so because the test also includes omitted 117th-Congress voting histories, member turnover, and a longer time gap. If the 117th Congress was held out for validation, say that directly; if it was excluded, describe the regime as a skipped-Congress transfer that includes a majority flip.

---

### 7. The prospective interval is too wide to rule out adaptive inflation

**Status**: [Pending]

**Quote**:
> The single-tower predecessor (v1) was frozen on June 12, 2026 (weights hash-pinned, fit through the June 9 data snapshot) and scored 0.331 log loss and  $85.7\%$  accuracy on the 43 rollcalls and 9,451 votes Congress cast June 10–30 (rollcall-cluster bootstrap  $95\%$  interval for the log loss: 0.265–0.402). That interval is wide—43 rollcalls is a contamination check, not a precision instrument—and its value is what it rules out: the development-test numbers are not inflated by adaptive iteration.

**Feedback**:
Even setting aside the freeze-date issue, the stated interval does not justify the strong “rules out” language. The temporal forecast log loss for the blend is 0.324, while the reported 95% interval for the prospective v1 log loss is 0.265–0.402. The upper endpoint is 0.078 log-loss units above 0.324, so the interval is still compatible with materially worse prospective performance. Since v1 is also not the final ensemble, this result is better described as a sanity check against gross contamination than as a formal exclusion of adaptive-iteration effects.

---

### 8. The worked example mixes pre-vote forecasts with retrospective spatial quantities

**Status**: [Pending]

**Quote**:
> Before the aggregate exhibits, one rollcall end to end—chosen from the genuine holdout window, so every model quantity below was computed without access to this vote or any later one. On December 20, 2024, the House passed H.R. 10545, the American Relief Act (roll no. 1235), the continuing resolution that averted a Christmas shutdown after the collapse of the leadership’s first package—a storied majority-splitting vote.

**Feedback**:
“Every model quantity below” is too broad. The frozen ensemble probabilities can be true pre-vote forecasts, but the rollcall-specific spatial discrimination and cutpoint reported in the example are estimated from the realized yea/nay pattern for that roll call. The later reference to the “recorded 366-34 coalition” confirms that this part is retrospective. The example would be clearer if it said that the ensemble probabilities were computed without access to the vote, while the surprise scores and spatial discrimination/cutpoint are diagnostics of the realized roll call.

---

### 9. The cutpoint-prediction parenthetical mixes two metrics

**Status**: [Pending]

**Quote**:
> On 3,549 held-out future rollcalls, text plus metadata predicts cutpoint locations with a mean error of 0.64 member-standard-deviations ( $r = 0.55$ , against 0.86 with no features) and calls the coalition's direction correctly  $74.5\%$  of the time (chance:  $55.8\%$ ).

**Feedback**:
The parenthetical reads as though it compares correlations: $r=0.55$ for text plus metadata against 0.86 with no features. Table 6 shows that 0.86 is the no-feature mean absolute error, not a correlation; the no-feature cutpoint correlation is 0.00. Spell out both metrics in the sentence, for example: “MAE = 0.637 and $r=0.55$, versus MAE = 0.856 and $r=0.00$ with no features.” Otherwise the baseline appears to have a stronger location correlation than the model being defended.

---

### 10. The amendment-text negative result needs the actual deltas

**Status**: [Pending]

**Quote**:
> Pre-registered, it failed three ways, and the failure is the finding. Corrupted joins, corrected joins, and minimal-footprint text insertions all made prediction *worse* by similar amounts; a placebo rebuild that changed the data file without changing content reproduced the champion exactly, exonerating the pipeline.

**Feedback**:
This is the evidentiary center of the amendment section, but the paper does not report the effect sizes for the three failed variants or the placebo in the text or Table 7. Readers cannot tell whether “worse by similar amounts” means a rounding-level loss, a large degradation, or a validation-only result. Add a small table or sentence giving the matched evaluation rows, metric, and deltas for the corrupted join, corrected join, minimal-footprint insertion, and placebo rebuild. The negative finding is interesting, but it needs numbers next to it.

---

### 11. The dashboard is not the full frozen champion

**Status**: [Pending]

**Quote**:
> The dashboard wraps the frozen champion's MiniLM component tower. (An earlier draft claimed the blend's TF-IDF component could not run on a single pasted bill; that was wrong—frozenidf weights transform new documents fine. The component is omitted as an engineering simplification, justified by its small blend weight of 0.12.)

**Feedback**:
The first sentence makes the dashboard sound like an interface to the frozen champion. The parenthetical then says a blend component is omitted, so the dashboard is a MiniLM-only deployment approximation. That distinction matters near probability and cutpoint thresholds: even a 0.12 blend weight can change a displayed side when the MiniLM logit is close to zero. State up front that the dashboard does not run the exact three-tower champion and explain whether the displayed probabilities have been checked against the full blend on held-out or historical inputs.

---

### 12. The sparse text model supports association, not causal recruitment

**Status**: [Pending]

**Quote**:
> Bills whose summaries establish programs, authorize assistance, and provide grants and services to workers and employees recruit liberal-yea coalitions; bills about rules, regulations, permits, budgets, statutory revision (sec, code, amends), and injury (the tort-reform lexicon) recruit conservative-yea ones.

**Feedback**:
The TF-IDF exercise estimates predictive associations in observed final-passage votes. It does not identify the language as recruiting members in a causal or generative sense. The same coefficient pattern could arise because sponsors choose different policy substance, agenda control filters what reaches the floor, or summary terms proxy for issue area. Use descriptive language here: summaries with those terms are associated with liberal-yea or conservative-yea coalitions in observed final-passage roll calls.

---

### 13. Confident residuals are not enough to rule out model noise

**Status**: [Pending]

**Quote**:
> Figure 8 shows the decisions the model still gets confidently wrong even with full hindsight. These are not model noise; inspected, they are recognizable episodes—high-pressure defections, district-driven crossovers, protest votes—now identifiable systematically rather than anecdotally.

**Feedback**:
The conclusion is too categorical. Even a well-calibrated model that assigns probability 0.99 to yea will produce about 10 nay outcomes among 1,000 such predictions, and those errors are compatible with stochastic residual noise. Qualitative inspection may show that many top residuals are substantively rich, but that does not establish that the whole tail excludes calibration error, misspecification, or random realizations. A more defensible claim is that the ranking is enriched for recognizable episodes and is a systematic sampling frame, while some cases may still be model error or noise.

---

### 14. The NOMINATE transfer comparison needs an information-set clarification

**Status**: [Pending]

**Quote**:
> Career-spanning NOMINATE scores, by contrast, nearly match the deep model’s cross-congress accuracy—a genuine vindication of traditional measurement design.

**Feedback**:
This sentence invites a forecast-style reading, but “career-spanning NOMINATE” can mean scores estimated using a member’s full voting record, including votes from the held-out Congress being predicted. If so, the comparison is a retrospective measurement comparison, not a same-information forecasting comparison. If the NOMINATE scores are frozen before the held-out Congress, the paper should say exactly which release or estimation window is used. The result can still vindicate the durability of the design, but readers need to know whether the inputs obey the same temporal constraint as the deep model.

---

### 15. The cutpoint-prediction target set should be identified rollcalls

**Status**: [Pending]

**Quote**:
> For every rollcall in the benchmark we take the realized cutpoint and direction (which party's side the yeas fall on) from the spatial fits, hold out the final twenty percent of each congress-chamber's rollcalls, and ask how well features available before the vote predict them: nothing; metadata (question type, sponsor party, bill category); text alone (in classical and modern representations); or text and metadata together.

**Feedback**:
The table caption later says cutpoint error and direction accuracy are computed on identified rollcalls. That is a narrower target set than “every rollcall in the benchmark,” since the paper reports cutpoints only when the discrimination clears the identification threshold. This is not just wording; performance on identified votes may differ from performance on all held-out votes. Revise the paragraph to say “identified rollcalls” and give the number excluded by the identification rule, or explain why the direction metric is also restricted to the same subset.

---

### 16. The task is conditional on a recorded yea/nay decision

**Status**: [Pending]

**Quote**:
> We predict individual voting decisions: given everything observable before a rollcall, output each member’s probability of voting yea. The panel covers every yea/nay decision in the 101st through 119th Congresses from Voteview *(Lewis et al., 2026)*—10.6 million decisions—with bill-side information (titles, policy areas, dated summary versions, spon

**Feedback**:
The first sentence sounds like an unconditional probability of voting yea for each eligible member-rollcall pair. The next sentence narrows the panel to recorded yea/nay decisions, so the estimand appears to be $P(\text{yea}\mid\text{yea or nay recorded})$, not a probability that also models abstentions, absences, or present votes. Say that directly in the task definition. Otherwise readers may think the 10.6 million rows are a full eligibility panel rather than the binary subset of recorded yea/nay choices.

---

### 17. The prospective denominator should say member-vote decisions

**Status**: [Pending]

**Quote**:
> its frozen, hash-pinned predecessor scored 85.7% on the 9,451 votes Congress cast in the three weeks after that artifact was finalized

**Feedback**:
Here “votes” can be read as roll-call events, especially because the paper often uses “rollcall votes” in surrounding prose. The denominator is individual member-vote decisions: Section 2.4 says the window contains 43 rollcalls and 9,451 votes. Use that wording in the abstract as well, for example “9,451 individual member-vote decisions from 43 roll calls.” The evidentiary claim is otherwise easy to misread.

---

### 18. Completion is prediction, but not temporal forecasting

**Status**: [Pending]

**Quote**:
> Still, completion-regime performance should never be quoted as "prediction."

**Feedback**:
The warning is right in spirit, but the word choice is too strong. Completion holds out individual votes and predicts them using observed same-rollcall information, so it is still a prediction or imputation task. What it is not is a temporal forecast. A cleaner version would be: “completion-regime performance should be quoted as held-out vote completion, not as temporal forecasting.”

---

### 19. The five-point sampling statement is a one-standard-error calculation

**Status**: [Pending]

**Quote**:
> with at least 100 unity votes per member, binomial sampling error is at most about five points, several times smaller than the labeled extremes.

**Feedback**:
The five-point number is the worst-case binomial standard error, not a bound on realized sampling error. For a binomial share, $SE(\hat p)=\sqrt{p(1-p)/n}$, and $p(1-p)\leq1/4$, so with $n=100$ the maximum standard error is $\sqrt{1/(4\cdot100)}=0.05$. A rough two-standard-error interval is about ten points before accounting for dependence across votes. Replace “sampling error is at most” with “the binomial standard error is at most,” and note that clustered or repeated procedural votes can make the binomial calculation optimistic.

---

### 20. Nominations need a separate policy-domain coding rule

**Status**: [Pending]

**Quote**:
> Because every rollcall here carries a policy area from its bill's text metadata, the spatial model can be re-estimated within policy areas, giving each member a position in each of the thirteen policy domains of Table 4—the structure a single dimension is forced to average away. Coverage is broad: under the fifty-vote rule, between 917 and 1,562 members receive a position in each of the twelve legislative areas (nominations, a Senate-only domain, covers 328), and Table 4 reports the full per-topic accounting.

**Feedback**:
The first sentence says every rollcall gets its policy area from bill text metadata, but the next sentence identifies nominations as a separate Senate-only domain. Nominations are not bill rollcalls, so the bill-metadata source cannot apply uniformly to all thirteen domains. Adjust the wording to say that legislative rollcalls use bill policy metadata and nominations are coded under a separate rule.

---

### 21. The surprise caption should use the realized-vote probability

**Status**: [Pending]

**Quote**:
> Table 5: A worked rollcall from the holdout window: H.R. 10545, American Relief Act (December 20, 2024). Positions and loyalty residuals from the paper's measures;  $\hat{p}$  is the frozen ensemble's pre-vote forecast; surprise is  $-\log \hat{p}$  (realized vote).

**Feedback**:
The caption can be read as applying $-\log \hat p$ directly to the reported yea probability for every row. The nay rows show that the table instead uses the negative log probability of the realized vote. For Massie, $-\log(0.43)=0.84$, while $-\log(1-0.43)=0.56$, matching the reported 0.55 after rounding. Write the formula as $-\log \Pr(\text{realized vote})$, i.e. $-\log\hat p$ for yeas and $-\log(1-\hat p)$ for nays.

---

### 22. The 55.8% direction benchmark is not chance

**Status**: [Pending]

**Quote**:
> On 3,549 held-out future rollcalls, text plus metadata predicts cutpoint locations with a mean error of 0.64 member-standard-deviations ( $r = 0.55$ , against 0.86 with no features) and calls the coalition's direction correctly  $74.5\%$  of the time (chance:  $55.8\%$ ).

**Feedback**:
The 55.8% number is the constant no-feature baseline in Table 6, not chance accuracy for a two-class task. If the majority direction share is 0.558, a fair coin gives 50%, and random draws from the marginal distribution give $0.558^2+0.442^2=0.507$. Call it a “constant baseline” or “majority-direction baseline” rather than chance.

---

### 23. Table 9’s random-rollcall AUC column is unlabeled

**Status**: [Pending]

**Quote**:
> |  Model | Forecast: LL / AUC / contested acc. |   |   | Random rollcall: LL  |   |
> | --- | --- | --- | --- | --- | --- |
> |  Constant rate | 0.617 | 0.500 | 52.7 | 0.659 | 0.500  |

**Feedback**:
The table contains five numeric metrics, but the random-rollcall side names only LL even though the last column is plainly an AUC column: the constant-rate row is 0.500, and the values range like AUCs. Label the columns separately as Forecast LL, Forecast AUC, Forecast contested accuracy, Random-rollcall LL, and Random-rollcall AUC. This is a small fix, but it prevents readers from misreading the last column as an unnamed loss metric.

---

### 24. The dashboard’s cutpoint fit should specify probabilities versus thresholded votes

**Status**: [Pending]

**Quote**:
> member vote probabilities are computed for the full current roster; and the displayed cutpoint is the position at which predicted support crosses one half, from a logistic fit of predicted votes on the current congress's estimated positions—mirroring the paper's cutpoint estimation on realized votes.

**Feedback**:
After the previous clause says the dashboard computes probabilities, “predicted votes” is ambiguous. If the logistic fit uses hard 0/1 classifications from thresholded probabilities, it can discard near-threshold information and run into separation. If it uses the predicted probabilities as fractional responses, say so and give the formula, for example $\operatorname{logit}(p_i)=\alpha+\beta x_i$ with cutpoint $x^*=-\alpha/\beta$. The dashboard description should be reproducible without guessing this step.

---

### 25. The amendment mechanism should not imply unshown per-amendment parameters

**Status**: [Pending]

**Quote**:
> Mechanism: when a bill’s amendments share the parent bill’s text, the model assigns the series nearly identical parameters—an implicit pooled estimate of a correlated block. Purpose language replaces that pooling with per-amendment, content-driven parameters that predict worse than the bill’s identity does.

**Feedback**:
“Parameters” sounds like rollcall-specific quantities estimated for each amendment, but the experiment described here appears to change the text representation fed into the prediction model. Unless the model really estimates separate per-amendment parameters in this ablation, use representation language instead: shared parent-bill text gives amendment series nearly identical text-side representations, while purpose language creates per-amendment representations that performed worse in the pre-registered variants. That phrasing matches the experiment without implying an additional parameterization.

---
