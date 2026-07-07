# Memo: Response Plan for coarse.ink Review Round 3

Reviewer model: `openrouter/openai/gpt-5.5-pro` (upgraded from `gpt-5.5` in
rounds 1–2). The review copy carried a prepended "Note to the Referee"
directing attention to (1) depth of the text-feature analysis, (2) concrete
examples/members/bills, (3) missing analyses. Review:
`coarse_review_r3.md`. Verdict: major revision, 11 major points, 18
detailed comments.

Legend: **ADOPT** (this round), **PARTIAL** (adopt a feasible core, defer
the rest with argument), **DEFER** (argued in text/repo, not done this
round), **REBUT** (reviewer premise wrong; fix exposition if our text
invited the misreading).

---

## Major points

### R1. Headline uses features the appendix calls not leakage-clean — PARTIAL
Correct as stated and we flagged it ourselves in Appendix C. We cannot
prove CRS subject-term vintages (BILLSTATUS does not version them), so the
honest response is prominence, not proof:
- Promote the dual headline into §3.2 main text: ensemble \champForecastLL
  with the leakage-clean single MiniLM tower \incForecastLL beside it, and
  state explicitly which claim each supports.
- Add a leakage-clean protest comparison: the MiniLM-tower preds
  (`emb2_mlp_mq_16d_tcal`) already exist; recompute defection AUC so the
  §4 text-vs-no-text contrast is clean-tower vs no-text as well as
  ensemble vs no-text. (No retraining needed.)
- §5 is already clean: verified that the headline `embeddings_meta`
  feature set contains no CRS-revisable TF-IDF; only the non-headline
  "Text: TF-IDF" row uses it. State this in the §5 text.

### R2. Same-bill / bill-family leakage in the temporal split — ADOPT
Fair and testable. New analysis: split held-out rollcalls by whether the
same bill (congress × bill id) had any rollcall in the training window;
report champion and no-text log loss on both strata (macros + a sentence
in §2/§3). If the text gain concentrates on seen-bill votes, that is a
finding to report, not hide. Bill-family (related-measure) linkage beyond
same-bill is DEFERred (no reliable cross-bill family key in BILLSTATUS).
Member-held-out: DEFER with argument — new-member cold start is a
different task; congress-out already bounds transfer.

### R3. Completion benchmark is not text evidence — ADOPT (framing)
Correct; §3.1 involves no text and we should say so plainly. One sentence
in §3.1 ("a vote-only result about model flexibility, not text") and a
matching clause in the intro. The three-part structure survives: point one
is "outperforms existing approaches" (both settings), not "text wins
everywhere."

### R4. Protest predicted ≠ protest validated — PARTIAL
The full external-validation program (whip counts, statements, caucus
letters) is a paper of its own. This round:
- Tighten language: we detect/forecast *majority defections*, including
  but not limited to protest votes; the construct claim runs through the
  worked episodes, not the AUC.
- Where Fowler–Lewis name specific episodes we can check overlap
  qualitatively, cite them in the episode discussion.
- The SSFA paragraph already reports a non-protest defection case honestly.
DEFER the systematic external panel; note it as the companion-paper
validation path (consistent with r1/r2 deferral of interest-group
ratings).

### R5. Text-feature evidence doesn't rule out titles/procedure/boilerplate — ADOPT (the round's centerpiece)
Exactly the user's directive. Two new analyses, both inference-only on the
frozen champion's MiniLM tower (no retraining, no refitting):
1. **Redaction ablations** on holdout rollcalls: original text vs
   (a) title/proper-noun redaction, (b) summary removed (question+
   description only), (c) question removed, (d) numbers/years stripped,
   (e) within-policy-area text shuffle placebo. Report Δ log loss and
   Δ defection-AUC per variant. The shuffle placebo is the killer control:
   if performance survives text from a *different* bill in the same policy
   area, the "reading" claim dies.
2. **Counterfactual edits** on the worked exemplars (H.R. 10545, SSFA,
   BIOSECURE): remove/replace the fiscal-deal language, the
   benefits-expansion language, the China/security language; show the
   predicted majority-defection share and the five likeliest defectors
   under each edit. This is the "how is the model figuring it out"
   exhibit.
Plus **policy-area-conditional text gains** from existing preds
(appropriations/CR vs suspension vs amendment vs tax etc.): cheap, from
stored predictions.

### R6. Cutpoint uncertainty + vote-type split — PARTIAL
Vote-type split of Table 4 (passage / amendment / procedural): ADOPT,
cheap. Delta-method SEs on realized cutpoints and a robustness line
(results on the subset with tight cutpoint SEs): ADOPT. Full posterior
propagation: DEFER (the target is admittedly a fitted quantity; we say so).

### R7. Prospective audit too small — PARTIAL
Soften §3.2 to "consistent with development performance" and label the
audit preliminary until more clusters accumulate. Score the no-text and
member-rate baselines on the same post-snapshot sample (cheap; artifacts
exist). Clarify blend weights are unconstrained logistic-stacking
coefficients (they are; sum > 1 is expected). All ADOPT.

### R8. Worked hard-case member table — ADOPT
Directly serves the user's "specific members" directive. New table: for
Ron Paul and Justin Amash (and the other Fig 1 labels in the repo
version), the held-out votes with the largest per-vote fit improvement
over the DW-NOMINATE model, with bill short-titles. Generated from
member_fit cell-level output; no new fitting.

### R9. Senate-facing validation — PARTIAL
Chamber-disaggregated forecast LL and defection AUC: ADOPT (macros from
existing preds). Senate worked cases (RFMA etc.): DEFER to keep §4 to one
deep episode; noted as extension.

### R10. Calibration evidence — ADOPT
We have the machinery (calibration.pdf existed pre-simplification).
Refresh: reliability plot for champion + no-text on the forecast holdout,
Brier and ECE macros, and a rollcall-level predicted-vs-realized
defection-share panel for §4. Appendix figure + two sentences.

### R11. Ex-post bill-location benchmark (Peress/Richman overlap) — DEFER
A faithful Peress reimplementation is out of scope this round; the paper
already states the target is our own spatial fit's cutpoint. Add an
explicit limitation sentence naming the missing comparison and what it
would take.

---

## Detailed comments (18)

| # | Item | Verdict | Action |
|---|------|---------|--------|
| 1 | "member by member" overclaims | ADOPT | rephrase: distributional improvement, higher for \memberGMPshare% |
| 2 | Fowler–Lewis vs Duck-Mayr–Montgomery conflated | ADOPT | split the citation sentence by construct |
| 3 | cutpoint normalization unstated | ADOPT | state normalization (x standardized per chamber-congress, party-sign orientation) in Definitions |
| 4 | held-out cells in rollcall-param fits? | REBUT (exposition) | verified in 24_member_fit.py: both models fit on identical training cells, scored on cells neither saw; say so in §3.1 |
| 5 | which coordinate correlates .98 | ADOPT | specify first-dimension projection + per-congress aggregation |
| 6 | 0.331 post-snapshot vs 0.343 post-freeze labels | ADOPT | relabel prose + add post-freeze row/label to Table 2 |
| 7 | era-gap attribution too broad | ADOPT | narrow the sentence |
| 8 | center-pull not implied by any defection | ADOPT | limit mechanism claim to ends-against-middle patterns |
| 9 | Fig 2 ranking universe | ADOPT | caption: rank over all members chamber-wide; majority modal vote yea; the five are majority members |
| 10 | "same members" claim unsupported | ADOPT | compute the overlap (scaling-gain rank vs text-minus-no-text defection gain) and either report or soften to what it shows |
| 11 | ≥3-defection filter vs task definition | ADOPT | define task on filtered subset; also report all-positive-defection AUC (rerun 27 with both) |
| 12 | AUC aggregation/ties unstated | ADOPT | state unweighted mean over rollcalls, rank-based ties (verified in 27) |
| 13 | "inverts" vs degrades | ADOPT | reword to degraded transfer (signed analysis not in hand) |
| 14 | identifiability claim too strong | ADOPT | restrict to observational equivalence given the full information set |
| 15 | no-text refit ambiguity | REBUT (exposition) | notext_mq_16d_tcal is independently trained/stopped/calibrated; state it |
| 16 | ledger rows vs stated tests | ADOPT | add regime-inheritance row; replace protocol cells with criteria |
| 17 | fractional-logit x* wording | ADOPT | x* = 0.5 crossing of the fitted curve |
| 18 | off-agenda flag naming | ADOPT | rename to sponsor–coalition mismatch flag |

---

## Implementation plan (modules)

1. **M1 — precision + framing fixes** (detailed 1–18 adopts; R3, R4
   language, R7 language, R11 limitation). Text-only; recompile.
2. **M2 — leakage-clean + stratified evidence** (R1 protest AUC from
   emb2 preds; R2 seen-bill/fresh-bill split; R9 chamber split; policy-area
   text gains). One new script, macros, no retraining.
3. **M3 — text-feature depth** (R5 redaction ablations + counterfactual
   edits on exemplars). One new script using the dashboard's single-tower
   inference path; new table + possibly a small figure. The centerpiece.
4. **M4 — members and calibration** (R8 Paul/Amash worked table; R10
   calibration figure + macros; comment 10 overlap check).
5. **M5 — cutpoint robustness** (R6 vote-type split + SE screen; §5 text).
6. Recompile, re-review (round 4) if the user wants a convergence check.

Compute: everything is inference-only or refits of small linear models;
no tower retraining, no touching frozen artifacts.
