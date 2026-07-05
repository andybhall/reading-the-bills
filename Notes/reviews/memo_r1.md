# Memo: evaluation of coarse review round 1

*2026-07-05. Verdict per comment: ACCEPT / PARTIAL (with what we will and
won't do, and why) / REJECT (with argument). Changes implemented after
this memo; round 2 review follows.*

## Overall issues

**O1. Prediction→measurement framing overstated.** ACCEPT. The reviewer
is right that Section 5 undercuts the strong form of the intro's claim:
a model can predict well using role/agenda signals that are not the
construct. Fix: reframe the bound as *informational* (prediction bounds
how much structure exists to measure) and make the paper's arc explicit
— Sections 3–4 extract measures, Section 5 shows which parts of the
predictive signal are construct-valid and which are role. Abstract
softened accordingly. No new analysis needed; the paper already contains
the decomposition evidence, the framing just has to own it.

**O2. Feature-time audit + surprises estimand.** ACCEPT both. (a) New
appendix table: every feature, its timestamp rule, and its knowability
argument. (b) The reviewer's catch is correct and important: the frozen
champion is fit through June 2026, so scoring historical votes with it
yields IN-SAMPLE residuals, not forecast surprises. We relabel the
measure as "out-of-character votes (in-sample residuals)", strike
"could not see coming", and state that the prospective ledger produces
the true forecast version going forward. Recomputing expanding-window
surprises for 20 years is deferred (compute) and flagged as such.

**O3. Prospective alignment + uncertainty.** ACCEPT. Abstract now
attributes the prospective certificate to the frozen June-12 predecessor
explicitly; Table gains rollcall-cluster bootstrap CIs (computed in P1);
snapshot/artifact identities spelled out. Longer window: cannot be
manufactured — the ledger accumulates; stated.

**O4. Decomposition table + stronger classical forecast baseline.**
PARTIAL. We assemble the requested single decomposition table from
already-benchmarked models (party-only, member-history-only,
metadata-only, text-only, text+member+meta, blend) under identical
regimes — these runs exist in the leaderboard. A *dynamic* ideal-point
forecaster is a genuine additional build; our NOMINATE×context logit is
the classical context-rich representative, and we say plainly that a
dynamic-IRT forecast baseline is future work. (Reject-with-argument on
building it now: the within-congress temporal split leaves little room
for dynamics to matter — positions are near-constant within a congress —
so its expected contribution over nominate_context is small.)

**O5. Uncertainty + external validation for new measures.** PARTIAL.
Adding: binomial SE statement for loyalty residuals (with n≥100 unity
votes, SE ≤ 0.05; extremes shown are 2–4× that), explicit verification
of the Van Drew pre-switch claim (checked below; if his residual window
mixes post-switch service we weaken the sentence), and uncertainty
language for cutpoints (identification threshold already enforced).
External benchmarks (interest-group ratings, caucus rosters) are the
companion paper's validation program — data acquisition, honestly
deferred and stated.

**O6. Amendment conclusion stronger than design.** ACCEPT the narrowing;
REJECT additional runs, with argument. The reviewer's alternative
mechanism (losing pooling) is not an alternative — it IS our stated
mechanism. But they are right that "deals, not language" overreaches.
The claim is narrowed to their formulation: short purpose text adds
nothing once bill identity and procedural context are modeled — and we
now cross-reference the amender-party result (direction accuracy 51%→77%
from actor identity alone), which affirmatively supports "actor and
context over content." Purpose-only/hierarchical variants would test
claims we no longer make.

**O7. Majority-flip generality.** ACCEPT — the one big new computation.
We add a multi-transition study: congress-out evaluation at every
transition in the window with bill-text coverage (110, 112, 116, 118 =
flips; 113, 115, 117 = placebo non-flips), champion architecture vs the
majority-status table, House, with by-question-type breakdowns. If the
degradation concentrates in flip transitions and procedural votes, the
claim survives narrowed to exactly that; either way the paper reports it.

**O8. Agenda-conditional estimands.** ACCEPT (prose): every derived
measure is now defined as conditional on the realized rollcall agenda,
stated once prominently and echoed at each measure. Reweighting
sensitivity: deferred with flag (the issue-position pipeline predates
this session; a stratified re-run is queued as robustness for the
revision after Andy's read).

**O9. Explicit estimands.** ACCEPT: new "Measurement definitions"
subsection with the notation the reviewer requests (p̂, loyalty residual,
per-topic scaling, cutpoint −(b+c̄)/a with |a| threshold, surprise
−log p̂(y)).

**O10. Text coverage by congress.** ACCEPT the table; the pre/post-108
ablation is mooted by a scope clarification the paper should have made
anyway: all forecast-regime results use congresses 108–119 only
(text-era); 101–107 enters only the completion regime, where no text is
used. Stated explicitly + coverage table added.

**O11. Worked rollcall.** ACCEPT — good idea, cheap, high value. We walk
the 2024 Ukraine supplemental final passage (118th House, inside the
temporal test window): text fields, predicted vs realized votes for
named members, cutpoint, issue deviations, surprises.

**O12. Cutpoint benchmarking on known votes.** PARTIAL, merged with O11:
a landmark-votes table (FRA 2023, Ukraine 2024, IRA passage, etc.) with
our cutpoints against the documented coalitions. Full Peress/Richman
replication is beyond scope and stated as the natural next test.

## Detailed comments

1 abstract scope — ACCEPT ("on our benchmark").
2 "every history-based measure" — ACCEPT reviewer's wording.
3 extremity vs signed — ACCEPT-CLARIFY: the computation already uses
  |x| (code: corr(|x|, r)); the text now says "absolute ideal point."
4 bounded residual tail — ACCEPT: mechanical-bound caveat added.
5 residual contains unmodeled preference — ACCEPT their sentence.
6 thirteen-vs-fourteen + coverage range — ACCEPT: prose tied to table.
7 ρ high ≠ structured residue — ACCEPT: logic reordered; factional
  patterns carry the structure claim.
8 widest-spread overbreadth — ACCEPT their corrected sentence.
9 left-of-floor coalition inference — ACCEPT: new quantity (share
  between D median and floor median) + orientation note; macro added.
10 ledger audit table — ACCEPT: compact appendix table (IDs, dates,
  hypothesis, falsification, outcome).
11 amendment-join denominator — ACCEPT: exact counts as macros.
12 "increasing difficulty" — ACCEPT their phrase.
13 "every member" eligibility — ACCEPT.
14 log loss ≠ calibration — ACCEPT ("higher log loss").
15 TF-IDF dashboard rationale — ACCEPT: the reviewer is technically
  right (frozen idf transforms fine); the true reason is an engineering
  simplification given the component's small blend weight — now stated
  honestly.
16 dashboard cutpoint display rule — ACCEPT: in-range condition added
  to the code alongside the existing slope threshold; both documented.
17 "post-hoc parameters" — ACCEPT their sentence.

## Not doing, with reasons (summary)

- Dynamic-IRT forecast baseline (O4): expected marginal within-congress;
  future work.
- Expanding-window surprise recomputation (O2): compute-heavy; measure
  relabeled honestly instead; prospective ledger supplies the true
  forecast version going forward.
- Purpose-only / hierarchical amendment models (O6): test claims the
  narrowed text no longer makes.
- External-ratings validation (O5): data acquisition; companion paper.
- Issue-position agenda reweighting (O8): queued as revision robustness.
