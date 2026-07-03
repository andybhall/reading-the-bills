# Memo: Issue-Specific Member Positions

*Drafted 2026-06-11. Code: `Code/10_issue_positions.py` (validated
instrument), `Code/11_issue_probes.py` (exploratory, NOT yet validated).
Data: `Modified Data/results/issue_positions.parquet`,
`issue_probe_scores.parquet`.*

## Instrument 1: per-topic ideal points (VALIDATED)

1D spatial models fit separately within each of 12 policy areas (+ Senate
nominations), congresses 108-119, Regime A training votes; positions
z-scored within chamber-topic and compared to a pooled "overall" fit on
the same sample. 23 chamber-topic fits, 14,870 member-topic positions.

**Headline structure.** Topic positions correlate r = 0.95-0.99 with the
overall dimension — one dimension dominates congressional voting in every
policy area, consistent with the polarization literature. The signal is in
the deviations (topic z − overall z):

- **House, Armed Forces/National Security — leftward deviators:** Chip Roy,
  Ron Paul, Eli Crane, Burlison, Amash, Massie, Brecheen, Flake. This is
  precisely the libertarian / Freedom-Caucus non-interventionist wing,
  recovered from votes alone.
- **Senate, International Affairs — leftward deviators:** Rand Paul (-1.1
  SD), Mike Lee (-0.96), then Daines, Young, Moran, Braun, Hawley — the
  Republican non-interventionists, old and new right.
- **Senate, International Affairs — rightward deviators:** the famously
  hawkish Democrats (Menendez, Schumer, Blumenthal, Rosen) and
  establishment-internationalist Republicans (Romney, Blunt, Portman,
  Rounds).

These are exactly the patterns a knowledgeable observer would predict, and
none of them are visible in a single NOMINATE dimension.

**Uses.** The member x topic matrix supports: issue-based candidate/member
profiles, within-party faction detection (non-interventionists, trade
skeptics), and tracking issue realignment over time (v2: per-congress
topic fits).

## Instrument 2: embedding probes (EXPLORATORY — failed validation, kept
for the record)

The two-tower model maps any text to a discrimination vector, so we scored
members against hypothetical bill texts ("expand background checks...").
Two iterations:

- v1 (phi with metadata blocks): party-separation sanity checks failed
  badly — most probes inverted. Diagnosis: with passage/sponsor features
  zeroed, the text map leans on agenda/sponsorship cues entangled during
  training.
- v2 (text-only phi + within-issue contrasts): mixed. Guns now separates
  correctly (D-R = +0.45 on the restrict-expand contrast); environment is
  still inverted; military spending and border-security probes show
  *Democrats* higher — which may be partially REAL revealed preference in
  the 118th-119th (defense and border appropriations pass with
  Democrat + establishment-Republican coalitions against Freedom-Caucus
  nays; Democrats supported the 2024 border package Republicans killed),
  but we cannot distinguish "real revealed preference" from "instrument
  artifact" with party stereotypes as the only benchmark.

**Verdict:** do not use probe scores as member attributes yet. Path to
validation: (a) benchmark against external interest-group ratings (LCV,
NRA, AFL-CIO, Cato trade scores) rather than party stereotypes; (b)
restrict probe scoring to high-tenure members (freshmen positions are
noisy); (c) stronger text encoder; (d) train the probe tower on
passage-of-sponsored-legislation votes only, where text direction and vote
direction are most tightly linked.

## The substantive lesson

"Issue positions" from roll calls measure *revealed voting on the bills
that reached the floor*, not survey-style issue attitudes. Agenda control
means these can diverge sharply (the border-package episode). That
divergence is itself a research finding — but it must be modeled, not
assumed away.
