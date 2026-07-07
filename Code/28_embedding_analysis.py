"""How does the model read a bill? (P5v7: embedding-space evidence)

Two analyses of the text representation underlying the forecasting
results, both restricted to pre-vote information:

1. Nearest neighbors. For exemplar holdout rollcalls, the closest
   training rollcalls in sentence-embedding space, with each neighbor's
   realized majority-party defection share and coalition direction. If
   the encoder organizes bills by the political situation they create,
   a shutdown-averting CR's neighbors should be earlier must-pass
   spending deals with the same right-flank revolt signature --- and
   they are. Output: Draft/tables/neighbors.tex.

2. A revolt probe. Ridge regression of the majority-party defection
   share on the text embedding alone, fit on training-window
   majority-side passage votes, evaluated on the holdout window.
   Output: correlation + top-predicted examples, as macros in
   Draft/tables/embedding_macros.tex.

Run: python3 Code/28_embedding_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RES = MOD / "results"
OUT = ROOT / "Draft" / "tables"

sys.path.insert(0, str(ROOT / "Code"))
from models_forecast import question_bucket

EXEMPLARS = [  # (congress, chamber, rollnumber, short label)
    (118, "House", 1235, "American Relief Act (CR), Dec.\\ 2024"),
    (118, "House", 1174, "Social Security Fairness Act, Nov.\\ 2024"),
    (118, "House", 1120, "BIOSECURE Act, Sept.\\ 2024"),
]


def build_frame():
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes.congress >= 108]
    mem = pd.read_parquet(MOD / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    votes = votes.merge(mem, on=["congress", "chamber", "icpsr"],
                        how="left", suffixes=("", "_m"))
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "date", "vote_question",
         "vote_desc", "bill_number"]]
    key = ["congress", "chamber", "rollnumber"]
    # majority party per congress-chamber = larger caucus
    caucus = (votes.drop_duplicates(key[:2] + ["icpsr"])
              .groupby(["congress", "chamber", "party_code"])["icpsr"].count()
              .rename("n").reset_index())
    majparty = (caucus[caucus.party_code.isin([100.0, 200.0])]
                .sort_values("n").groupby(["congress", "chamber"]).last()
                .rename(columns={"party_code": "maj_party"})[["maj_party"]]
                .reset_index())
    votes = votes.merge(majparty, on=["congress", "chamber"])
    mv = votes[votes.party_code == votes.maj_party]
    g = (mv.groupby(key)["vote"].agg(["mean", "size"])
         .rename(columns={"mean": "maj_yea", "size": "maj_n"}).reset_index())
    g["defect_share"] = 1 - g.maj_yea
    g = g.merge(rc, on=key, how="left")
    g["qb"] = question_bucket(g["vote_question"])
    emb = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet")
    ecols = [c for c in emb.columns if c.startswith("e")]
    g = g.merge(emb[key + ecols], on=key, how="inner")
    # temporal flag consistent with the forecast split
    g = g.sort_values(["congress", "chamber", "date", "rollnumber"])
    rank = g.groupby(["congress", "chamber"]).cumcount()
    size = g.groupby(["congress", "chamber"])["rollnumber"].transform("size")
    g["test"] = (rank + 0.5) / size >= 0.90
    return g, ecols


def neighbors(g, ecols):
    E = g[ecols].to_numpy()
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    train = ~g.test.to_numpy()
    rows = []
    for cong, ch, roll, label in EXEMPLARS:
        i = g.index[(g.congress == cong) & (g.chamber == ch)
                    & (g.rollnumber == roll)]
        if not len(i):
            print(f"exemplar missing: {label}")
            continue
        i = g.index.get_loc(i[0])
        sims = E @ E[i]
        # neighbors restricted to rollcalls the model trained on AND that
        # precede the exemplar (the Dec 19 failed twin of the Dec 20 CR
        # sits at cosine 0.99 but is itself in the holdout window)
        before = (pd.to_datetime(g["date"]) <
                  pd.to_datetime(g.iloc[i]["date"])).to_numpy()
        sims[~(before & train)] = -np.inf
        order = np.argsort(sims)[::-1][:3]
        tgt = g.iloc[i]
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}"
                    f": realized majority defection "
                    f"{100*tgt.defect_share:.0f}\\%}} \\\\")
        for j in order:
            n = g.iloc[j]
            desc = (str(n.vote_desc) if isinstance(n.vote_desc, str)
                    else str(n.bill_number))
            desc = desc.replace("&", "\\&")[:44]
            rows.append(" & ".join([
                f"\\quad {desc}", f"{n.congress}",
                f"{sims[j]:.2f}", f"{100*n.defect_share:.0f}\\%"]) + " \\\\")
    (OUT / "neighbors.tex").write_text(
        "\\begin{tabular}{p{7.8cm}ccc}\n\\toprule\n"
        "Nearest earlier bills in the encoder's text space & Congress & "
        "\\shortstack{Text\\\\similarity} & "
        "\\shortstack{Its majority\\\\defection} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote neighbors.tex ({len(rows)} rows)")


def revolt_probe(g, ecols):
    d = g[g.qb.isin(["passage", "resolution", "cloture"]) & (g.maj_yea >= 0.5)
          & (g.maj_n >= 50)]
    tr, te = d[~d.test], d[d.test]
    X_tr, X_te = tr[ecols].to_numpy(), te[ecols].to_numpy()
    ridge = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(X_tr, tr.defect_share)
    pred = ridge.predict(X_te)
    r = float(np.corrcoef(pred, te.defect_share)[0, 1])
    te = te.assign(pred=pred)
    top = te.nlargest(5, "pred")[["vote_desc", "bill_number", "congress",
                                  "pred", "defect_share"]]
    print(f"revolt probe: test r = {r:.3f} (n={len(te)})")
    print(top.to_string(index=False))
    macros = [f"\\newcommand{{\\revoltProbeR}}{{{r:.2f}}}",
              f"\\newcommand{{\\revoltProbeN}}{{{len(te):,}}}"]
    macros += ssfa_auc_macros()
    (OUT / "embedding_macros.tex").write_text("\n".join(macros) + "\n")
    top.to_csv(RES / "measures" / "revolt_probe_top.csv", index=False)


def ssfa_auc_macros():
    """Within-vote defection AUC on the Social Security Fairness Act
    (118 House roll 1174) for the champion vs. its no-text twin: text
    similarity alone does not flag this revolt (Table: neighbors are
    consensual benefit bills), so the paper reports how much of the
    detection survives, attributing the remainder to institutional
    and member-history features."""
    from sklearn.metrics import roc_auc_score
    mem = pd.read_parquet(RES / "measures" / "members_house118.parquet")[
        ["icpsr", "party_code"]]
    out = []
    for model, tag in [("blend3_mlp_tfidf_emb3_tcal", "text"),
                       ("notext_mq_16d_tcal", "notext")]:
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        d = p[(p.congress == 118) & (p.chamber == "House")
              & (p.rollnumber == 1174)].merge(mem, on="icpsr")
        g = d[d.party_code == 200.0].dropna(subset=["vote"])
        auc = roc_auc_score((g.vote == 0).astype(int), 1 - g.p_yea)
        out.append(f"\\newcommand{{\\ssfaAUC{tag}}}{{{auc:.2f}}}")
        if tag == "text":
            out.append(f"\\newcommand{{\\ssfaDefectors}}"
                       f"{{{int((g.vote == 0).sum())}}}")
    return out


def main():
    g, ecols = build_frame()
    print(f"{len(g)} rollcalls with embeddings; "
          f"{int(g.test.sum())} in holdout windows")
    neighbors(g, ecols)
    revolt_probe(g, ecols)


if __name__ == "__main__":
    main()
