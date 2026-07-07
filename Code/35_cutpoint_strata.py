"""Cutpoint prediction by vote type and target precision (review r3, R6).

Stratifies the Section 5 held-out cutpoint predictions (text+metadata
feature set) two ways:
1. By question bucket: passage, amendment, procedural, cloture,
   resolution --- the single spatial cutpoint is least credible for
   omnibus/procedural votes, so per-type results bound the claim.
2. By target precision proxy: |a_v| above/below the test-set median.
   Realized cutpoints are estimated quantities; strongly discriminating
   rollcalls pin their cutpoints most precisely, so results on the
   high-|a| half indicate how much of the headline error is target
   noise rather than prediction failure.

Outputs: Draft/tables/cutpoint_strata.tex, cutpoint_strata_macros.tex
Run: python3 Code/35_cutpoint_strata.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEAS = ROOT / "Modified Data" / "results" / "measures"
OUT = ROOT / "Draft" / "tables"
KEY = ["congress", "chamber", "rollnumber"]


def stats(d):
    mae = float((d.pred_embeddings_meta - d.cut_std).abs().mean())
    r = float(np.corrcoef(d.pred_embeddings_meta, d.cut_std)[0, 1])
    return mae, r, len(d)


def main():
    rc = pd.read_parquet(MEAS / "cutpoint_rollcalls.parquet")
    a = pd.read_parquet(MEAS / "cutpoint_targets_cache.parquet")[KEY + ["a"]]
    d = rc[rc.test & rc.identified & rc.pred_embeddings_meta.notna()]
    d = d[d.cut_std.between(-4, 4)].merge(a, on=KEY, how="left")
    med = d.a.abs().median()

    rows, macros = [], []
    for label, sub in [("All rollcalls", d)] + [
            (b.capitalize(), d[d.qbucket == b])
            for b in ("passage", "amendment", "procedural", "cloture",
                      "resolution") if (d.qbucket == b).sum() >= 50] + [
            (f"High discrimination ($|a_v|$ above median)",
             d[d.a.abs() > med]),
            (f"Low discrimination ($|a_v|$ below median)",
             d[d.a.abs() <= med])]:
        mae, r, n = stats(sub)
        rows.append(f"{label} & {n:,} & {mae:.3f} & {r:.2f} \\\\")
        print(f"{label:42s} n={n:5,}  MAE={mae:.3f}  r={r:.2f}")

    (OUT / "cutpoint_strata.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Subset & Rollcalls & MAE & $r$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    hi = stats(d[d.a.abs() > med])
    pa = stats(d[d.qbucket == "passage"])
    macros = [f"\\newcommand{{\\cutHiDiscR}}{{{hi[1]:.2f}}}",
              f"\\newcommand{{\\cutHiDiscMAE}}{{{hi[0]:.2f}}}",
              f"\\newcommand{{\\cutPassageR}}{{{pa[1]:.2f}}}",
              f"\\newcommand{{\\cutPassageMAE}}{{{pa[0]:.2f}}}"]
    (OUT / "cutpoint_strata_macros.tex").write_text("\n".join(macros) + "\n")
    print("wrote cutpoint_strata.tex, cutpoint_strata_macros.tex")


if __name__ == "__main__":
    main()
