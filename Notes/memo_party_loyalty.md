# Memo: Party Loyalty Beyond Ideology

*Drafted 2026-06-11. Status: v1, first signal-extraction memo. Code:
`Code/09_member_signals.py`. Data: `Modified Data/results/member_signals.parquet`.*

## Question

How much of each member's party-line voting is explained by their own
ideological position, and who deviates — in which direction — from that
prediction? The residual is a member attribute distinct from ideology:
"party soldiers" vote with the party *more* than their ideal point implies;
"mavericks" less.

## Method

1. Fit the validated 1D ideal-point model (per chamber, party-sign
   initialization; validation against DW-NOMINATE: r = 0.98) on Regime A
   training votes, congresses 101-119.
2. Restrict to **party-unity rollcalls** in training data: a majority of
   voting Democrats opposes a majority of voting Republicans, both parties
   with >= 10 voters on the rollcall.
3. For each D/R member's vote on those rollcalls, compare:
   - actual: did they vote with their party's majority side?
   - ideology-implied: the 1D model's probability of voting with that side.
4. **Loyalty residual** per member-congress = actual rate − implied rate.

## Why the residual is meaningful

The 1D model has two parameters per member (position + intercept) and
cannot memorize individual votes; what it captures is the spatial/ideological
component of party-line voting. Systematic positive residuals mean a member
votes with the party even when their position implies they shouldn't
(discipline, agenda alignment, career incentives); negative residuals mean
crossing the party despite an ideal point that implies loyalty.

## Results

- 21,874 of 35,442 rollcalls (101st-119th) are party-unity votes.
- 10,207 member-congress observations with >= 100 unity votes.
- Overall calibration is exact: mean actual with-party rate 91.26% vs.
  ideology-implied 91.29%. Residual SD = 3.2pp, so the measure separates
  members by several percentage points of unexplained (dis)loyalty.
- Residuals are nearly orthogonal to within-party extremity (r = -0.16) —
  this is by construction mostly, and confirms we measure something
  distinct from ideology.

**Senate, 118th — least loyal vs. ideology-implied:** Graham (-12.5pp),
Murkowski (-12.0), Manchin (-7.3), Tillis (-5.5), Kennedy, McConnell,
Collins, Young, Moran, Rounds. The canonical mavericks (Murkowski, Collins,
Manchin) fall out immediately, plus late-career McConnell and the 118th-
Congress version of Graham — both real phenomena, not artifacts.

**House, 118th — least loyal:** Fitzpatrick (-7.5pp, the House's most
famous centrist Republican), Bacon, Buck, Gallego (D-AZ, mid-Senate-run
repositioning), Wagner, Calvert.

**Most loyal vs. ideology-implied (Senate 118):** Hoeven, Wicker, Fischer,
Thune, Fetterman — reliable party-line voters whose moderate-ish scaled
positions understate their loyalty.

**The most interesting cells are the positive-residual House members:**
Chris Smith (+10.1pp) and Massie (+8.9pp). These are members whose 1D
position is *misleading* — Smith's social conservatism + economic
moderation averages to "moderate" in 1D, so the model expects defection
that never comes; Massie's libertarian extremity predicts anti-leadership
defection, but in the 118th (post-Rules-Committee deal) he voted with the
party far more than his position implies. The positive tail thus mixes
genuine party discipline with 1D model misfit for multidimensional members
— exactly the members the 8D model's predictive gains come from.
Cross-referencing this residual against higher-k representations is v2 of
this memo.

## Caveats

- In-sample predictions (descriptive measurement, not benchmark evaluation;
  the frozen benchmark metrics are untouched).
- The residual conflates party pressure with dimensions a 1D model omits
  (e.g., libertarian or parochial dimensions); the 8D model's gains show
  such dimensions exist. Cross-referencing residuals against higher-k
  models is the natural v2.
- Unity-vote selection is itself endogenous to agenda setting (we only see
  votes leadership allows to occur) — interpretation is conditional on the
  observed agenda.
