"""Worked validation for the hard-case members (review r3, R8).

For Ron Paul and Justin Amash — the canonical members that
one-dimensional scaling mis-measures — list the held-out completion
cells where the eight-dimensional spatial model most improves on the
frozen-DW-NOMINATE model, with the bill behind each vote. The point is
face validity: the improvements should sit on recognizable
ends-against-the-middle votes (TARP-type packages, surveillance
reauthorizations, spending bills), not on arbitrary procedure.

Inputs: cell-level held-out predictions saved by the regimeA benchmark
(both models scored on identical cells neither trained on).
Outputs: Draft/tables/hardcase_votes.tex, hardcase_macros.tex

Run: python3 Code/31_hardcase_table.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"

KEY = ["congress", "chamber", "rollnumber"]

MEMBERS = {  # icpsr(s) per member; Amash has two Voteview icpsr records
    "Ron Paul (R-TX)": [14290],
    "Justin Amash (MI)": [21143, 91143],
}
TOP_N = 6


def main():
    ours = pd.read_parquet(
        RES / "preds" / "regimeA_seed42_ideal_point_8d.parquet")
    nom = pd.read_parquet(
        RES / "preds" / "regimeA_seed42_nominate_logit.parquet")
    m = ours.merge(nom, on=KEY + ["icpsr", "vote"],
                   suffixes=("_ours", "_nom"))
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        KEY + ["date", "vote_question", "vote_desc"]]
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        KEY + ["bill_type", "bill_no"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "title"]]

    # probability assigned to the vote actually cast, under each model
    m["p_cast_ours"] = m.p_yea_ours.where(m.vote == 1, 1 - m.p_yea_ours)
    m["p_cast_nom"] = m.p_yea_nom.where(m.vote == 1, 1 - m.p_yea_nom)
    m["gain"] = m.p_cast_ours - m.p_cast_nom

    rows, macros = [], []
    for label, icpsrs in MEMBERS.items():
        d = m[m.icpsr.isin(icpsrs)]
        top = (d.nlargest(TOP_N, "gain")
               .merge(rc, on=KEY, how="left")
               .merge(links, on=KEY, how="left")
               .merge(bills, on=["congress", "bill_type", "bill_no"],
                      how="left"))
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{label}}}: "
                    f"{len(d):,} held-out votes}} \\\\")
        for r in top.itertuples():
            desc = r.title if isinstance(r.title, str) else r.vote_desc
            desc = (str(desc) or str(r.vote_question)).replace("&", "\\&")
            desc = desc[:44] + ("\\dots" if len(desc) > 44 else "")
            cast = "Yea" if r.vote == 1 else "Nay"
            rows.append(
                f"\\quad {desc} & {r.congress} & {cast} & "
                f"{r.p_cast_nom:.2f} & {r.p_cast_ours:.2f} \\\\")
        print(label, "top gains:")
        print(top[["congress", "vote_desc", "title", "vote",
                   "p_cast_nom", "p_cast_ours"]].to_string(index=False))

    (OUT / "hardcase_votes.tex").write_text(
        "\\begin{tabular}{p{7.2cm}cccc}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{$P(\\text{vote cast})$} \\\\\n"
        "\\cmidrule(lr){4-5}\n"
        "Held-out vote & Congress & Cast & DW-NOM & This paper "
        "\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    for label, icpsrs in MEMBERS.items():
        d = m[m.icpsr.isin(icpsrs)]
        tag = "paul" if "Paul" in label else "amash"
        macros.append(f"\\newcommand{{\\{tag}HeldoutN}}{{{len(d):,}}}")
    (OUT / "hardcase_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote hardcase_votes.tex, hardcase_macros.tex")


if __name__ == "__main__":
    main()
