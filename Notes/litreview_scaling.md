# Roll-call scaling in APSR / AJPS / JOP / PA, 2010–2026

*Compiled 2026-07-05 for the reorganization of Paper A. Each entry: what
it measures, what fit/validation it reports, and the substantive payoff
we should learn from. Sources: web sweep + full-text where accessible;
entries marked (canon) predate the window but define standard practice.*

## The measurement canon and its fit statistics

- **Poole & Rosenthal, NOMINATE (canon).** The field's fit vocabulary:
  correct classification (CC%), aggregate proportional reduction in
  error (APRE), and geometric mean probability (GMP). Voteview publishes
  per-member GMP and error counts — meaning any new measure can be
  compared member-by-member on NOMINATE's own yardstick. GMP =
  exp(−log loss): our primary metric already is the field's, in
  different units. **Action: report GMP everywhere; add the member-level
  GMP comparison.**
- **Clinton, Jackman & Rivers 2004 APSR (canon).** Bayesian IRT framing;
  the license for treating scaling as statistical modeling with fit
  checks and posterior uncertainty.
- **Carroll, Lewis, Lo, Poole & Rosenthal 2013 AJPS.** Utility-function
  comparison (Gaussian vs quadratic) adjudicated by... CC/GMP-style fit.
  Template: architecture claims settled by predictive fit on votes.

## Estimation advances 2010–2026

- **Lauderdale 2010 PA, "Unpredictable Voters in Ideal Point
  Estimation."** Heteroskedastic IRT: a legislator-specific variance
  measuring how much behavior is NOT conditioned on the main dimension.
  Substantive use: who votes off-dimension (mavericks), diagnosing
  unmodeled dimensions. **Our per-member mean predictive loss from the
  champion is the direct modern analog — richer, since our model first
  absorbs text/issue/loyalty structure before calling the rest
  unpredictable. Deliver as a measure; cite as lineage.**
- **Shor & McCarty 2011 APSR.** Common-space scores across legislatures;
  validation = bridging + convergent correlations. Template for
  "validate against everything external you can find."
- **Imai, Lo & Olmsted 2016 APSR.** Fast (EM) ideal points for massive
  data; validation = correlation with MCMC estimates + held-out
  prediction. Precedent for computational scaling claims in APSR.
- **Duck-Mayr & Montgomery 2023 PA, GGUM ("Ends Against the Middle").**
  Non-monotone response: extremists vote with the opposite pole against
  the middle. Fit improvements in the 116th; fixes AOC-type mis-scaling.
- **Fowler & Lewis 2026 PA, "Accounting for Protest Voting."** Majority
  members voting nay on bills they prefer (Squad). Adjusted scores make
  AOC et al. the most liberal members; validation = qualitative cases +
  stronger correlation with non-roll-call measures; implications for
  polarization trends and responsiveness estimates. **The newest thread,
  and our surprise/loyalty machinery detects exactly these votes without
  a bespoke scaling model. Connect explicitly (do we recover the Squad?).**

## Text + votes

- **Lauderdale & Clark 2014 AJPS.** LDA-weighted issue-specific ideal
  points. Validation: issue-position face validity, known cases.
  Direct ancestor of our per-topic positions (ours: policy-area splits
  + modern text; theirs: topic shares as weights).
- **Kim, Londregan & Ratkovic 2018 PA (SFA).** Joint sparse factor model
  of votes + speech; finds ideology + a leadership dimension.
  Precedent: text and votes share latent structure; leadership as a
  second dimension rhymes with our loyalty residual.
- **Gerrish & Blei 2011/12; Kraft et al. 2016 (ML venues, canon for
  us).** Covered in our lit-baseline horse race.

## Bills, cutpoints, status quos, agendas

- **Richman 2011 APSR, "Parties, Pivots, and Policy: The Status Quo
  Test."** Uses estimated status-quo locations to adjudicate cartel vs
  pivot theories. Bill-side locations as theory tests.
- **Peress 2013, "Estimating Proposal and Status Quo Locations"** (votes
  + cosponsorship). The standing method for bill locations; needs
  realized votes. **Our text model locates bills BEFORE any vote — the
  prospective capability this literature lacks. That is claim (c).**
- **Crespin & Rohde 2010 JOP.** Appropriations votes scale differently —
  issue-specific dimensionality within bill types.
- **Aldrich, Montgomery & Sparks 2014 PA.** Low dimensionality is partly
  an agenda artifact (parties structure what reaches the floor).
  Rhymes with our agenda-selection boundary and cutpoint-mass exhibit.
- **Bateman, Clinton & Lapinski 2017 AJPS.** Cross-time comparisons of
  ideal points confounded by agenda change; issue-coded analysis.
  Supports our role-vs-preference transfer finding and per-congress
  humility.

## Bridging / external validity

- **Bonica 2013/2014 AJPS (CFscores); Bonica 2018 AJPS.** 2018:
  supervised ML mapping contributions -> roll-call scores; validation =
  out-of-sample prediction of scores. Precedent for ML-as-measurement
  in AJPS, and for out-of-sample fit as the validation currency.
- **Jessee 2016 AJPS.** Limits of bridging assumptions — caution for
  cross-population claims (we stay within Congress).
- **Nokken & Poole 2004 (canon).** Per-congress scores; the standard
  for "positions move" robustness. Voteview ships them; our panel has
  them. **Action: report convergent r with Nokken-Poole per congress.**

## The standard package a scaling paper is expected to deliver

1. Fit vs incumbents on THEIR statistics: CC%, APRE, GMP (overall, by
   congress/chamber, and per member where possible).
2. Convergent validity: r with DW-NOMINATE (overall and within party),
   Nokken-Poole; face-validity rosters for known groups (Squad, Freedom
   Caucus, canonical mavericks).
3. Out-of-sample vote prediction (post-Bonica/Imai this is expected;
   our regime audit exceeds the standard).
4. At least one NEW quantity with substantive payoff, validated:
   issue positions (L&C), variance/unpredictability (Lauderdale),
   protest-adjusted scores (F&L), bill/status-quo locations (Peress).
5. A named-cases exhibit connecting the new quantity to episodes the
   reader knows (AOC/Squad in F&L; mavericks in Lauderdale).

## Gaps this paper can own

- (a) A measure that beats NOMINATE on NOMINATE's own member-level GMP,
  in-sample AND under genuine forecasting, with a frozen prospective
  certificate no scaling paper has offered.
- (b) Legislators: loyalty residual (discipline net of ideology; nests
  the F&L protest phenomenon without a bespoke likelihood);
  model-based unpredictability (Lauderdale's tau, modernized);
  issue-specific positions at full coverage. Bills: cutpoints for the
  whole agenda with policy-area structure; cutpoint mass vs chamber/
  party medians as an agenda-control exhibit (Richman/Aldrich).
- (c) PROSPECTIVE bill locations from text alone (Peress needs votes +
  cosponsors after the fact) — plus the interactive forecaster.
