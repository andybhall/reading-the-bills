"""Counterfactual text edits on worked exemplars (review r3, R5).

Two exemplars chosen to separate the two channels the redaction
ablations identify (30_text_ablation.py):

1. American Relief Act CR (118 House 1235). Its pre-vote text is
   title-only --- the bill appeared the day before the vote, so no
   summary existed; the model read exactly what a contemporaneous
   observer could: "On Motion to Suspend the Rules and Pass. American
   Relief Act, 2025". Edits therefore target the title's semantics and
   the procedural posture:
     original / generic title ("A bill to provide for certain
     matters") / question changed to "On Passage" / BIOSECURE title
     substituted.

2. Appalachian Regional Development Act Amendments (109 House 1138):
   among summary-rich majority-yea holdout passage votes, the one
   where the champion's predicted defection share is largest --- a
   revolt the model detected from a real pre-vote summary (realized
   25%, predicted 21%): a regional spending authorization drawing a
   fiscal-conservative revolt, the direction-terms pattern in one
   bill. Edits target policy content:
     original / spending-program sentences removed / summary replaced
     by another same-policy-area holdout bill's summary.

Reported per variant: mean predicted defection probability among
majority members, expected defectors, and how many of the model's
five likeliest defectors actually voted nay. The frozen tower and all
non-text features are identical across variants.

Outputs: Draft/tables/counterfactual_edits.tex, counterfactual_macros.tex
Run: python3 Code/34_counterfactual_edits.py
"""

import importlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
OUT = ROOT / "Draft" / "tables"
sys.path.insert(0, str(ROOT / "Code"))

abl = importlib.import_module("30_text_ablation")

SPEND = re.compile(r"grant|assistance|authoriz|appropriat|develop|program"
                   r"|funds|funding", re.I)


def cut_sentences(text, pattern):
    kept = [s for s in re.split(r"(?<=[.;]) ", text) if not pattern.search(s)]
    return " ".join(kept)


def get_row(d, cong, ch, roll):
    m = d[(d.congress == cong) & (d.chamber == ch) & (d.rollnumber == roll)]
    assert len(m) == 1, f"exemplar {cong}-{ch}-{roll} not found"
    return m.iloc[0]


def main():
    d = abl.holdout_frame()
    ara = get_row(d, 118, "House", 1235)
    bio = get_row(d, 118, "House", 1120)
    arda = get_row(d, 109, "House", 1138)

    # deterministic same-area donor: the first same-policy-area holdout
    # bill (by congress, rollnumber) with a real summary
    donors = d[(d.policy_area == arda.policy_area)
               & (d.bill_no != arda.bill_no)
               & (d.bill_text.str.len() > 300)].sort_values(
        ["congress", "rollnumber"])
    donor = donors.iloc[0]
    print("donor:", donor.desc[:60])

    panels = [
        ("American Relief Act CR, Dec.\\ 2024 (pre-vote text is "
         "title-only)", ara, [
            ("Original (title + suspension question)",
             abl.compose(ara.question, ara.desc, ara.bill_text)),
            ("Title replaced by generic language",
             abl.compose(ara.question, "A bill to provide for certain "
                         "matters", "A bill to provide for certain "
                         "matters")),
            ("Question changed to On Passage",
             abl.compose("On Passage", ara.desc, ara.bill_text)),
            ("BIOSECURE Act title substituted",
             abl.compose(ara.question, bio.desc, bio.bill_text)),
        ]),
        ("Appalachian Regional Development Act Amendments, 2006 (full "
         "pre-vote summary)", arda, [
            ("Original (title + summary)",
             abl.compose(arda.question, arda.desc, arda.bill_text)),
            ("Spending-program sentences removed",
             abl.compose(arda.question, arda.desc,
                         cut_sentences(arda.bill_text, SPEND))),
            ("Generic title, summary kept",
             abl.compose(arda.question, "A bill to provide for certain "
                         "matters", arda.bill_text)),
            ("Same-area bill's summary substituted",
             abl.compose(arda.question, arda.desc, donor.bill_text)),
        ]),
    ]

    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code", "bioname"]]
    tower = abl.load_tower()

    tex, macros = [], []
    mtags = [["cfAraOrig", "cfAraGeneric", "cfAraPassage", "cfAraBio"],
             ["cfArdaOrig", "cfArdaCut", "cfArdaGeneric", "cfArdaSwap"]]
    for (label, r, variants), tags in zip(panels, mtags):
        cells = votes[(votes.congress == r.congress)
                      & (votes.chamber == r.chamber)
                      & (votes.rollnumber == r.rollnumber)].merge(
            mem, on=["congress", "chamber", "icpsr"], how="left",
            suffixes=("_v", ""))
        cells["vote_question"] = r.vote_question
        cells["bill_category"] = "legislation"
        cells["bill_type"], cells["bill_no"] = r.bill_type, r.bill_no
        cells["qb"] = r.qb
        maj = cells[cells.party_code == 200.0]  # GOP majority, 118th-119th
        single = d[(d.congress == r.congress) & (d.chamber == r.chamber)
                   & (d.rollnumber == r.rollnumber)]
        tex.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        tex.append(f"\\multicolumn{{4}}{{l}}{{\\quad\\emph{{realized "
                   f"outcome: {int((maj.vote == 0).sum())} of {len(maj)} "
                   f"majority members defected "
                   f"({100*(maj.vote == 0).mean():.0f}\\%)}}}} \\\\")
        for (vlabel, text), tag in zip(variants, tags):
            p = abl.score(tower, single, [text], cells)
            c = cells.assign(p_yea=p)
            mj = c[c.party_code == 200.0]
            exp_def = float((1 - mj.p_yea).sum())
            mean_def = float((1 - mj.p_yea).mean())
            top5 = mj.nsmallest(5, "p_yea")
            hits = int((top5.vote == 0).sum())
            tex.append(f"\\quad {vlabel} & {100*mean_def:.1f}\\% & "
                       f"{exp_def:.0f} & {hits}/5 \\\\")
            macros.append(f"\\newcommand{{\\{tag}}}{{{100*mean_def:.1f}}}")
            print(f"{vlabel:44s} mean def {100*mean_def:5.1f}%  "
                  f"expected {exp_def:5.1f}  top5 hits {hits}/5")

    (OUT / "counterfactual_edits.tex").write_text(
        "\\begin{tabular}{p{7.6cm}ccc}\n\\toprule\n"
        "Text supplied to the frozen model & "
        "\\shortstack{Predicted\\\\defection share} & "
        "\\shortstack{Implied\\\\defectors} & "
        "\\shortstack{Top-5 picks\\\\who defected} \\\\\n\\midrule\n"
        + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "counterfactual_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote counterfactual_edits.tex, counterfactual_macros.tex")


if __name__ == "__main__":
    main()
