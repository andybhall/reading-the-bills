"""The holdout's habitual defectors, both flanks (P5v10).

For every majority-yea passage-family rollcall in the temporal holdout,
a defection is a majority member voting nay. For each member with at
least eight defections, this reports how well the models price THEIR
defections in advance: the mean within-vote percentile of the member's
predicted defection probability among their party's members, computed
only on the votes where the member actually defected (100 = the model
ranked them the likeliest defector in the party every time).

The panel is the honest home for the left flank: Omar, Tlaib,
Ocasio-Cortez, and Bush are not completion-regime hard cases (frozen
DW-NOMINATE already classifies them at 97-99%), but they are among the
chamber's most frequent majority defectors when their party holds the
floor, and the text model prices their defections.

Outputs: Draft/tables/defector_panel.tex, defector_macros.tex
Run: python3 Code/38_defector_panel.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"

sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket

KEY = ["congress", "chamber", "rollnumber"]
MODELS = {"champ": "blend3_mlp_tfidf_emb3_tcal",
          "notext": "notext_mq_16d_tcal"}
MIN_DEFECTIONS = 8


def defection_percentiles(model):
    p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[KEY + ["vote_question"]]
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code", "bioname",
         "state_abbrev"]]
    d = (p.merge(rc, on=KEY, how="left")
          .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                 suffixes=("_v", "")))
    d["qb"] = question_bucket(d["vote_question"])
    d = d[d.qb.isin(["passage", "resolution", "cloture"])
          & d.party_code.isin([100.0, 200.0])]
    maj = (d.groupby(KEY + ["party_code"])["vote"].mean()
           .rename("party_rate").reset_index())
    d = d.merge(maj, on=KEY + ["party_code"])
    d = d[d.party_rate >= 0.5]
    # percentile of predicted defection probability within party-rollcall
    d["pct"] = (d.groupby(KEY + ["party_code"])["p_yea"]
                .rank(pct=True, ascending=True)) * 100
    # low p_yea -> high defection risk -> rank ascending puts them low,
    # so invert: percentile of defection risk
    d["pct"] = 100 - d["pct"]
    return d


def main():
    frames = {t: defection_percentiles(m) for t, m in MODELS.items()}
    base = frames["champ"]
    opp = (base.groupby("icpsr").size().rename("opportunities"))
    dd = base[base.vote == 0]  # realized defections
    g = (dd.groupby(["icpsr", "bioname", "state_abbrev", "party_code"])
         .agg(defections=("vote", "size"), pct_champ=("pct", "mean"))
         .reset_index())
    nt = (frames["notext"][frames["notext"].vote == 0]
          .groupby("icpsr")["pct"].mean().rename("pct_notext"))
    g = (g.merge(nt, on="icpsr").merge(opp, on="icpsr")
         .query(f"defections >= {MIN_DEFECTIONS}"))
    # rank by defection RATE within party: Republican majorities dominate
    # the holdout windows, so raw counts would show one flank only and
    # bury the modern left flank, whose windows are short
    g["rate"] = g.defections / g.opportunities
    g = (g.sort_values("rate", ascending=False)
         .groupby("party_code").head(8))
    g = g.sort_values(["party_code", "rate"], ascending=[False, False])

    # named-case block: the modern left flank, whose Democratic-majority
    # windows (116th-117th holdouts) are too short for its members to
    # reach the top-8 rate cut; selection here is by name, and the table
    # says so
    squad = ["OMAR, Ilhan", "TLAIB, Rashida",
             "OCASIO-CORTEZ, Alexandria", "PRESSLEY, Ayanna", "BUSH, Cori"]
    sq = (dd[dd.bioname.isin(squad)]
          .groupby(["icpsr", "bioname", "state_abbrev", "party_code"])
          .agg(defections=("vote", "size"), pct_champ=("pct", "mean"))
          .reset_index().merge(nt, on="icpsr").merge(opp, on="icpsr"))
    sq["rate"] = sq.defections / sq.opportunities
    sq = sq.sort_values("rate", ascending=False)

    rows = []
    panels = [("Republican majorities (top 8 by defection rate)",
               g[g.party_code == 200.0]),
              ("Democratic majorities (top 8 by defection rate)",
               g[g.party_code == 100.0]),
              ("The modern left flank (named cases; 116th--117th "
               "windows)", sq)]
    for plabel, panel in panels:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{plabel}}}}} \\\\")
        for r in panel.itertuples():
            last = r.bioname.split(",")[0].title()
            pty = "D" if r.party_code == 100.0 else "R"
            rows.append(f"\\quad {last} ({pty}-{r.state_abbrev}) & "
                        f"{r.defections} & {100*r.rate:.0f}\\% & "
                        f"{r.pct_champ:.0f} & {r.pct_notext:.0f} \\\\")
    (OUT / "defector_panel.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{Mean pre-vote risk percentile} \\\\\n"
        "\\cmidrule(lr){4-5}\n"
        "Member & Defections & Rate & Text model & No-text \\\\\n"
        "\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")

    macros = []
    named = pd.concat([g, sq]).drop_duplicates(subset=["icpsr"])
    for tag, needle in (("omar", "OMAR"), ("tlaib", "TLAIB"),
                        ("bush", "BUSH, Cori"), ("aoc", "OCASIO"),
                        ("massie", "MASSIE"), ("royC", "ROY, ")):
        m = named[named.bioname.str.startswith(needle)]
        if len(m):
            macros += [
                f"\\newcommand{{\\{tag}Defections}}"
                f"{{{int(m.defections.iloc[0])}}}",
                f"\\newcommand{{\\{tag}PctChamp}}"
                f"{{{m.pct_champ.iloc[0]:.0f}}}",
                f"\\newcommand{{\\{tag}PctNotext}}"
                f"{{{m.pct_notext.iloc[0]:.0f}}}"]
    (OUT / "defector_macros.tex").write_text("\n".join(macros) + "\n")
    print(g[["bioname", "party_code", "defections", "pct_champ",
             "pct_notext"]].round(0).to_string(index=False))
    print("wrote defector_panel.tex, defector_macros.tex")


if __name__ == "__main__":
    main()
