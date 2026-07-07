# Deep dive: should the model read full bill text, not just summaries?

*2026-07-07. Evaluation memo with new measurements; no models changed,
no frozen artifacts touched.*

## 1. What the models read today (verified in code and data)

Every registered model's text input comes from `08_embed_bills.py`:
`vote_question. vote_desc. bill_text`, where `bill_text` is the **latest
CRS summary dated strictly before the vote**, else the **title**, else
empty. MiniLM tower: 1,500 chars (~256 tokens). Qwen tower: 6,000 chars
— *more of the summary*, never statutory text. TF-IDF tower: title +
policy area + subjects. The pipeline has never ingested the GovInfo
BILLS collection (grep-verified: no full-text reference in Code/).

## 2. What our own experiments already tell us

- **M-A (ledger, negative):** a 16× text budget on *summary* text did
  not help and hurt long bills. Caution against "longer input is
  better," but it never tested *different* text.
- **E5a–c (negative):** amendment purpose text hurt; actor identity
  carried the signal.
- **Redaction ablations (P5v8):** wrong-bill text actively misleads
  (+.064 LL) — the encoder uses bill-specific content; titles alone
  carry large signal (generic-title ablation +.012; title-only cell
  below).
- **Benchmark card:** title-only rollcalls show the champion at .286 LL
  vs .439 no-text — the biggest relative text gain of any cell, from
  titles alone. Where there is *no* text (unlinked), text adds ~nothing.

Together: content works, but the binding constraint has been
**coverage/timeliness of the summary**, not summary richness.

## 3. New measurements (today)

**a. The ingredients are already on disk.** Our 7,461 BILLSTATUS XML
files contain `<textVersions>`: dated version entries with direct
govinfo URLs (no API key; fetch probe returned HTTP 200). 99% of linked
bills have ≥1 dated version. Trap: enrolled versions carry an *empty*
date — must be excluded under the timing rule.

**b. Where full text adds coverage.** Of 19,537 legislation-linked
rollcalls: the 17,277 summary-bearing ones nearly all (97%) also have
pre-vote full text — there full text is a *richness* bet (weak prior
per M-A). But of the **2,260 title-only rollcalls** (the cell where the
model currently reads ~nothing), **58% in the 114th–119th have a dated
full text strictly before the vote**: **1,144 rollcalls / 607 distinct
bills, concentrated in the 118th (431) and 119th (167)** — exactly our
holdout and prospective windows, where CRS lag is worst.

**c. Marquee cases cut both ways.**
- *H.R. 82 (Social Security Fairness Act):* CRS never wrote a summary —
  the champion read only the title — while full text had been posted
  for **22 months** before the vote. Clear win case.
- *H.R. 10545 (Dec 2024 CR):* **no dated text version at all**. The most
  extreme same-day deals are not rescued. Lead-time distribution for
  the modern title-only cell: 702 rollcalls >30d lead, 229 at 8–30d,
  ~350 same-day (unusable under the strict rule, as dates lack times).

**d. Encoding reality.** Fetch probes: a small bill's full text is
~2 KB → **256 tokens (fits the existing MiniLM window)**; the
Infrastructure Act is 2.7 MB → **~680k tokens** (no encoder holds it;
Qwen's 32k context covers ~5%). Naive truncation reads the enacting-
clause/short-title boilerplate — demonstrated to shift predictions in
the wrong direction on the SSFA example. Corpus: 7,443 distinct linked
bills; fetching all pre-vote versions ≈ 1 GB, hours not days.

## 4. Mechanisms, ranked

**A. Coverage/timeliness (strong, measurable, the real prize).** Give
text to the 1,144 modern rollcalls whose bills have pre-vote full text
but no pre-vote summary. This attacks the documented failure cell and
the congresses where the prospective ledger keeps scoring.

**B. Richness beyond summaries (weak prior).** Where both exist,
replace/augment the summary with full text. M-A and the 6k-char Qwen
tower's modest incremental weight (0.12 for TF-IDF, 0.42 Qwen vs 0.57
MiniLM) suggest limited headroom. Test only after A, separately.

**C. Product/UX ("cooler").** The web app could accept full text pasted
by users and preprocess it server-side into the model's input form —
independent of whether training changes.

## 5. How to encode full text (options)

1. **Truncate into existing towers** — rejected: reads boilerplate
   (demonstrated).
2. **Chunk-and-pool** (embed 256-token chunks, mean/attention-pool) —
   simple, leakage-clean, but pooling over 680k tokens of statutory
   language dilutes; untested distribution shift vs summary-trained
   towers.
3. **Long-context Qwen (32k)** — covers most ordinary bills whole;
   megabills still truncate; compute ~10× v3 embedding run.
4. **Summarize-then-embed (recommended for A):** generate a summary
   *from the pre-vote full text* with a local LLM, then feed the
   existing summary-shaped pipeline. Stays in-distribution (towers were
   trained on summaries), handles any length, and directly fills the
   missing-summary cell. **Critical leakage risk:** a modern LLM
   "knows" outcomes of famous bills (that H.R. 82 became law); the
   summarizer must be strictly extractive/grounded (temperature 0,
   instruction to describe only what the provided text says, no
   names-in-the-news framing), with an audited sample and an outcome-
   token screen. This risk must be pre-registered as the experiment's
   main knowability concern.

## 6. Pre-registered experiment plan (proposed ledger entry M-B)

*Hypothesis:* supplying pre-vote full-text-derived summaries to
rollcalls that lack CRS summaries improves holdout forecast log loss,
concentrated in the title-only cell.
*Design:* build `rollcall_text_v5`: bill_text := CRS pre-vote summary
if it exists, **else LLM summary of the latest dated pre-vote text
version** (strictly-before rule; undated/enrolled excluded), else
title. Train the standard MiniLM-MLP tower on v5; evaluate on
forecast108_119; report overall LL and the title-only-cell LL.
*Falsification:* no improvement in the title-only cell → negative,
recorded. *Placebo:* same pipeline with full texts randomly reassigned
among the 607 target bills (content vs coverage-mechanics). *Controls:*
same-day = unavailable; one-model-per-process; frozen v2 untouched (a
positive result feeds a future v3 freeze, not the current artifacts).
*Cost:* ~1 GB fetch + ~600 LLM summaries (local, hours) + one tower
train (~30 min) — 1–2 days total.
*Secondary (M-B2, only if M-B positive):* richness test on
summary-bearing rollcalls (chunked-Qwen or replace-summary variants).

## 7. Recommendation

**Yes — but targeted at coverage, not wholesale replacement.** The
evidence says the model's text problem is that summaries *arrive late
or never* for exactly the bills that matter most in recent congresses;
full text fixes that for ~1,100 modern rollcalls, including cases like
SSFA where the model read six words while 22 months of statutory text
sat unused. Wholesale "read the whole bill instead of the summary" has
a weak prior (M-A) and hard encoding costs (680k-token megabills), and
same-day deals (the Dec 2024 CR) are out of reach for any text source.
Run M-B as pre-registered; if positive, it strengthens the paper
(the title becomes more literal, the benchmark card gains a row, and
the prospective ledger's 119th-Congress scoring benefits directly) and
the web app inherits a principled way to accept full text.

## 8. Paper and app implications if M-B succeeds

- Paper: new coverage row + an honest note that "reading the bills"
  runs through summaries when available and statutory text when not;
  ARA footnote unchanged (nothing rescues same-day).
- App: accept pasted full text; server-side summarize-then-embed
  preprocessor replicates the training transform.
