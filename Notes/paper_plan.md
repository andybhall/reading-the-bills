# Research Plan: Roll-Call Prediction in the AI Era

*Drafted 2026-06-12 after Andy set the focus: update the Gerrish-Blei /
Kraft-et-al. literature for modern AI, and deliver substantive takeaways
from text embeddings (deviations, better prediction).*

## Framing

Two deliverables, likely two papers (or one paper + companion):

### Paper A (methods/benchmark): "Forecasting Congressional Votes in the AI Era"

The 2011-2016 text-based vote prediction literature (Gerrish-Blei ICML'11,
NeurIPS'12; Kraft-Jain-Rush EMNLP'16; Lauderdale-Clark AJPS'14) used LDA
and static word vectors, evaluated almost exclusively on random holdouts.
We update every layer and show what changes:

1. **Modern text stack**: long-context sentence embeddings (and LLM-distilled
   bill digests) replacing LDA/word2vec; rollcall-level text covering
   amendments and nominations.
2. **Honest evaluation**: random-holdout (their setting) vs within-congress
   temporal forecast vs whole-congress-out vs frozen prospective scoring.
   We already know these differ enormously (94% / 85% / 62%); the
   literature's headline numbers live in the easiest cell. This evaluation
   audit is itself a contribution.
3. **Calibration as a first-class metric** (log loss, temperature transfer
   across time) — absent from the prior thread.
4. **Open benchmark + leaderboard** so the field can iterate (the
   auto-researcher infrastructure).

Headline tables: regime x model leaderboard; ablation stack (text
construction -> encoder -> head -> calibration); error decomposition;
prospective ledger result; congress-out transfer failure + partial fixes.

### Paper B (substantive): "What Vote-Prediction Models Know About Members of Congress"

Signal extraction from the trained models, each signal validated externally:

1. **Party-loyalty residual** (done, v1): loyalty beyond ideology;
   mavericks/soldiers; validate vs known cases + leadership rosters.
2. **Issue-specific positions & deviations** (done, v1): per-topic ideal
   points; libertarian non-interventionists etc.; extend to per-congress
   fits to track realignment (e.g., GOP trade position over time).
3. **Surprise votes / out-of-character detection**: largest per-vote
   residuals from the champion model — a dataset of "votes the model
   couldn't see coming"; case-study validation (do they correspond to
   known defection episodes, vote-buying, district pressure?).
4. **Member drift**: per-congress member embeddings; who moves, when
   (primary threats? leadership entry? retirement?).
5. **Agenda vs preference** (from the probe post-mortem): revealed issue
   positions vs stereotype expectations diverge (border/defense approps);
   model the divergence rather than assume it away.
External validation sources to acquire: interest-group ratings (LCV, NRA,
AFL-CIO, Club for Growth), district presidential vote, leadership rosters.

## Near-term modules (in order)

- **M-A (now): encoder upgrade** — long-context modern embedding model +
  expanded text budget (error analysis: 33% of loss is long-bill
  truncation). Cheap, isolates "is representation the bottleneck."
- **M-B: amendment text** — Congress.gov API (needs free API key from Andy)
  for amendment purpose/description; 29% of loss is amendments.
- **M-C: LLM-distilled digests** for long bills (needs Anthropic API key);
  distill-then-embed; also fixes probe grounding.
- **M-D: cross-congress transfer** — majority-status features, party-of-
  agenda-setter interactions; attack the 62% congress-out result.
- **M-E: deviation analytics** (Paper B items 3-4): surprise-vote dataset,
  per-congress positions/drift.
- **M-F: external validation data** for Paper B.

## Evaluation discipline (unchanged)

Frozen primary metrics per split; prospective protocol is the external
headline; all new text features must pass the pre-vote-knowability test.
