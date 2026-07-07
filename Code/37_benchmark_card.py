"""Benchmark card: composition and text coverage (review r4, T11).

Documents what the released benchmark contains and where readable text
exists: by congress block and chamber, rollcall and member-vote counts,
the share of rollcalls linked to a bill, the share whose leakage-clean
text template draws on a pre-vote CRS summary versus a title only, and
the share linked through the amendment-records join. Then reports
holdout forecast log loss by text-coverage cell (summary-bearing /
title-only / description-only) for the champion and its no-text
counterpart, so users can see which cells carry the reading-the-bills
claims.

Outputs: Draft/tables/benchmark_card.tex, benchmark_card_macros.tex
Run: python3 Code/37_benchmark_card.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"
KEY = ["congress", "chamber", "rollnumber"]
EPS = 1e-6

BLOCKS = [(101, 107, "101--107 (pre-text era)"),
          (108, 113, "108--113"),
          (114, 119, "114--119")]


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[KEY + ["date"]]
    votes = pd.read_parquet(MOD / "votes.parquet")
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        KEY + ["bill_category"]]
    src = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet")[
        KEY + ["text_source"]]
    nv = votes.dropna(subset=["vote"]).groupby(KEY).size().rename("n_votes")
    d = (rc.merge(links, on=KEY, how="left")
           .merge(src, on=KEY, how="left")
           .merge(nv, on=KEY, how="left"))
    d["linked"] = d.bill_category == "legislation"
    d["summary"] = d.text_source == "summary"
    d["title_only"] = d.text_source == "title"

    rows = []
    for lo, hi, label in BLOCKS:
        for ch in ("House", "Senate"):
            g = d[(d.congress.between(lo, hi)) & (d.chamber == ch)]
            rows.append(
                f"{label} & {ch} & {len(g):,} & "
                f"{int(g.n_votes.sum()):,} & "
                f"{100*g.linked.mean():.0f}\\% & "
                f"{100*g.summary.mean():.0f}\\% & "
                f"{100*g.title_only.mean():.0f}\\% \\\\")
            print(label, ch, len(g), int(g.n_votes.sum()),
                  round(100*g.linked.mean()), round(100*g.summary.mean()),
                  round(100*g.title_only.mean()))

    # holdout log loss by text-coverage cell
    cov = d[KEY + ["text_source"]]
    ll_rows, macros = [], []
    for tag, model in (("champ", "blend3_mlp_tfidf_emb3_tcal"),
                       ("notext", "notext_mq_16d_tcal")):
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        p = p.dropna(subset=["vote"]).merge(cov, on=KEY, how="left")
        cells = {}
        for lab, mask in (("summary", p.text_source == "summary"),
                          ("titleonly", p.text_source == "title"),
                          ("desconly", p.text_source.isin(
                              ["desc_only", "none"]) | p.text_source.isna())):
            q = np.clip(p[mask].p_yea, EPS, 1 - EPS)
            ll = float(-np.mean(np.where(p[mask].vote == 1,
                                         np.log(q), np.log(1 - q))))
            cells[lab] = (ll, int(mask.sum()))
            macros.append(f"\\newcommand{{\\cov{lab}{tag.capitalize()}}}"
                          f"{{{ll:.3f}}}")
        ll_rows.append((tag, cells))
        print(tag, {k: (round(v[0], 3), v[1]) for k, v in cells.items()})

    head = ("\\begin{tabular}{llccccc}\n\\toprule\n"
            "Congresses & Chamber & Rollcalls & "
            "\\shortstack{Member\\\\votes} & "
            "\\shortstack{Bill-\\\\linked} & "
            "\\shortstack{Pre-vote\\\\summary} & "
            "\\shortstack{Title\\\\only} \\\\\n\\midrule\n")
    mid = "\n".join(rows)
    tail = ("\n\\midrule\n\\multicolumn{7}{l}{\\emph{Temporal-holdout "
            "log loss by text coverage:}} \\\\\n"
            "\\multicolumn{3}{l}{} & Member votes & "
            "\\multicolumn{2}{c}{Champion} & No-text \\\\\n")
    lab = {"summary": "Summary-bearing rollcalls",
           "titleonly": "Title-only rollcalls",
           "desconly": "Description-only or unlinked"}
    ch_cells, nt_cells = ll_rows[0][1], ll_rows[1][1]
    for k in ("summary", "titleonly", "desconly"):
        tail += (f"\\multicolumn{{3}}{{l}}{{\\quad {lab[k]}}} & "
                 f"{ch_cells[k][1]:,} & "
                 f"\\multicolumn{{2}}{{c}}{{{ch_cells[k][0]:.3f}}} & "
                 f"{nt_cells[k][0]:.3f} \\\\\n")
    (OUT / "benchmark_card.tex").write_text(
        head + mid + tail + "\\bottomrule\n\\end{tabular}\n")
    (OUT / "benchmark_card_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote benchmark_card.tex, benchmark_card_macros.tex")


if __name__ == "__main__":
    main()
