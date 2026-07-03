"""Generate every Paper A table as a TeX fragment from results files.

Reads ONLY machine-produced outputs (leaderboard.csv, run JSONs, saved
predictions, prospective report JSONs) — no hand-entered numbers, per
SOUL.md. Each table is a booktabs tabular written to Draft/tables/ for
\\input into the draft; numbers.tex defines \\newcommand macros for every
inline statistic the text quotes.

Canonical row rule: for each (model, split, eval_set) the LATEST
leaderboard row wins — superseded reruns (e.g. the pre-unveil randomrc
rows; see the 2026-07-03 reproducibility note in the experiment ledger)
are thereby excluded automatically.

Run: python3 Code/make_tables.py   (idempotent; overwrites Draft/tables/)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "Modified Data" / "results"
OUT = ROOT / "Draft" / "tables"

PRETTY = {
    "constant_rate": "Constant rate",
    "member_rate": "Member rate",
    "party_rollcall_rate": "Party $\\times$ rollcall majority",
    "party_congress_rate": "Party $\\times$ congress rate",
    "party_question_rate": "Party $\\times$ question rate",
    "member_question_rate": "Member $\\times$ question rate",
    "majority_question_rate": "Majority $\\times$ question rate",
    "member_pooled_question_rate": "Member history (pooled)",
    "nominate_logit": "DW-NOMINATE logit (frozen scores)",
    "ideal_point_1d": "Ideal points (1D)",
    "ideal_point_2d": "Ideal points (2D)",
    "ideal_point_8d": "Ideal points (8D)",
    "nominate_context_logit": "NOMINATE $\\times$ context logit",
    "gb_spatial_tfidf": "GB-style spatial (TF-IDF)",
    "gb_spatial_tfidf_tcal": "GB-style spatial (TF-IDF, calibrated)",
    "gb_spatial_emb2": "GB-style spatial (mod.\\ embeddings)",
    "gb_spatial_emb2_tcal": "GB-style spatial (mod.\\ emb., calibrated)",
    "kraft_bilinear_16d": "Kraft-style bilinear",
    "meta_tower_8d": "Metadata tower",
    "text_tower_mq_8d": "TF-IDF tower + member offset",
    "emb_tower_mq_8d": "Embedding tower (v1 bill text)",
    "emb2_tower_mq_8d": "Embedding tower (v2 rollcall text)",
    "emb2_tower_mq_16d": "Embedding tower (v2, $k{=}16$)",
    "emb2_mlp_mq_16d": "MiniLM-MLP tower (uncalibrated)",
    "emb2_mlp_mq_16d_tcal": "MiniLM-MLP tower (prior champion)",
    "emb2tfidf_mq_16d_tcal": "Embeddings + TF-IDF tower",
    "emb3_mlp_mq_16d_tcal": "Qwen-encoder tower",
    "blend_mlp_tfidf_tcal": "Two-tower blend",
    "blend3_mlp_tfidf_emb3_tcal": "Three-tower blend (champion)",
    "emb2maj_mlp_mq_16d_tcal": "+ majority features (M-D attempt)",
}


def canonical(lb: pd.DataFrame) -> pd.DataFrame:
    lb = lb.sort_values("run_utc")
    return lb.drop_duplicates(["model", "split", "eval_set"], keep="last")


def fmt(x, nd=3):
    return "---" if pd.isna(x) else f"{x:.{nd}f}"


def pct(x):
    return "---" if pd.isna(x) else f"{100 * x:.1f}"


def get(lb, model, split, col, eval_set="test"):
    r = lb[(lb.model == model) & (lb.split == split) & (lb.eval_set == eval_set)]
    return r[col].iloc[0] if len(r) else np.nan


def tabular(path: Path, header: str, rows: list[str], align: str):
    body = " \\\\\n".join(rows) + " \\\\"
    path.write_text(
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    print(f"wrote {path.relative_to(ROOT)}")


def regime_audit(lb):
    models = ["constant_rate", "member_question_rate", "majority_question_rate",
              "nominate_context_logit", "gb_spatial_tfidf_tcal", "kraft_bilinear_16d",
              "emb2_mlp_mq_16d_tcal", "blend_mlp_tfidf_tcal", "blend3_mlp_tfidf_emb3_tcal"]
    splits = ["randomrc108_119", "forecast108_119", "congressout118"]
    rows = []
    for m in models:
        cells = [PRETTY[m]]
        for s in splits:
            cells.append(fmt(get(lb, m, s, "log_loss")))
        for s in splits:
            cells.append(pct(get(lb, m, s, "accuracy")))
        rows.append(" & ".join(cells))
    tabular(OUT / "regime_audit.tex",
            "Model & \\multicolumn{3}{c}{Log loss} & \\multicolumn{3}{c}{Accuracy (\\%)}",
            rows, "l" + "c" * 6)


def regimeA(lb):
    models = ["constant_rate", "member_rate", "party_rollcall_rate",
              "ideal_point_1d", "nominate_logit", "ideal_point_2d", "ideal_point_8d"]
    rows = []
    for m in models:
        rows.append(" & ".join([
            PRETTY[m],
            fmt(get(lb, m, "regimeA_seed42", "log_loss")),
            pct(get(lb, m, "regimeA_seed42", "accuracy")),
            pct(get(lb, m, "regimeA_seed42", "contested_accuracy")),
            fmt(get(lb, m, "regimeA_seed42", "apre"), 2)]))
    tabular(OUT / "regimeA.tex",
            "Model & Log loss & Acc.\\ (\\%) & Contested acc.\\ (\\%) & APRE",
            rows, "lcccc")


def litrace(lb):
    models = ["constant_rate", "party_question_rate", "member_question_rate",
              "nominate_context_logit", "gb_spatial_tfidf", "gb_spatial_tfidf_tcal",
              "gb_spatial_emb2_tcal", "kraft_bilinear_16d", "text_tower_mq_8d",
              "emb2_mlp_mq_16d_tcal", "blend3_mlp_tfidf_emb3_tcal"]
    rows = []
    for m in models:
        rows.append(" & ".join([
            PRETTY[m],
            fmt(get(lb, m, "forecast108_119", "log_loss")),
            fmt(get(lb, m, "forecast108_119", "auc")),
            pct(get(lb, m, "forecast108_119", "contested_accuracy")),
            fmt(get(lb, m, "randomrc108_119", "log_loss")),
            fmt(get(lb, m, "randomrc108_119", "auc"))]))
    tabular(OUT / "litrace.tex",
            "Model & \\multicolumn{3}{c}{Forecast: LL / AUC / contested acc.} & "
            "\\multicolumn{2}{c}{Random rollcall: LL / AUC}",
            rows, "lccccc")


def ablation(lb):
    steps = ["meta_tower_8d", "text_tower_mq_8d", "emb_tower_mq_8d",
             "emb2_tower_mq_8d", "emb2_tower_mq_16d", "emb2_mlp_mq_16d",
             "emb2_mlp_mq_16d_tcal", "emb3_mlp_mq_16d_tcal",
             "blend_mlp_tfidf_tcal", "blend3_mlp_tfidf_emb3_tcal"]
    # stepwise labels: each row names the change from the row above
    step_label = {"emb2_mlp_mq_16d": "\\quad + MLP head",
                  "emb2_mlp_mq_16d_tcal": "\\quad + temporal calibration"}
    rows = []
    for m in steps:
        rows.append(" & ".join([
            step_label.get(m, PRETTY[m]),
            fmt(get(lb, m, "forecast108_119", "log_loss")),
            pct(get(lb, m, "forecast108_119", "accuracy")),
            pct(get(lb, m, "forecast108_119", "contested_accuracy")),
            fmt(get(lb, m, "forecast108_119", "apre"), 2)]))
    tabular(OUT / "ablation.tex",
            "Configuration & Log loss & Acc.\\ (\\%) & Contested (\\%) & APRE",
            rows, "lcccc")


def error_decomp():
    """Champion-vs-blend loss by question bucket, recomputed from saved
    test predictions (not parsed from report text)."""
    import sys
    sys.path.insert(0, str(ROOT / "Code"))
    from models_forecast import question_bucket
    rows_out = []
    preds = {}
    for m in ("emb2_mlp_mq_16d_tcal", "blend3_mlp_tfidf_emb3_tcal"):
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{m}.parquet")
        links = pd.read_parquet(ROOT / "Modified Data" / "rollcall_bills.parquet")[
            ["congress", "chamber", "rollnumber", "vote_question"]]
        p = p.merge(links, on=["congress", "chamber", "rollnumber"], how="left")
        p["qbucket"] = question_bucket(p["vote_question"])
        eps = 1e-7
        q = np.clip(p["p_yea"], eps, 1 - eps)
        p["ll"] = -(p["vote"] * np.log(q) + (1 - p["vote"]) * np.log(1 - q))
        preds[m] = p
    a, b = preds["emb2_mlp_mq_16d_tcal"], preds["blend3_mlp_tfidf_emb3_tcal"]
    ga = a.groupby("qbucket")["ll"].agg(["mean", "size"])
    gb = b.groupby("qbucket")["ll"].mean()
    share = (gb * ga["size"]) / (gb * ga["size"]).sum()
    for qb in ga.sort_values("mean", ascending=False).index:
        rows_out.append(" & ".join([
            qb, f"{ga.loc[qb, 'size']:,.0f}", fmt(ga.loc[qb, "mean"]),
            fmt(gb.loc[qb]), pct(share.loc[qb])]))
    tabular(OUT / "error_decomp.tex",
            "Question type & $N$ & Prior champion LL & Blend LL & Share of loss (\\%)",
            rows_out, "lrccc")


def prospective():
    rows = []
    for tag, f in (("v1 (emb2-MLP tower)", "prospective_report.json"),
                   ("v2 (three-tower blend)", "prospective_report_v2.json")):
        r = json.loads((RES / "frozen" / f).read_text())
        m = r["metrics"]
        rows.append(" & ".join([
            tag, r["snapshot"], f"{r['rollcalls_scored']}",
            f"{r['votes_scored']:,}", fmt(m["log_loss"]), pct(m["accuracy"]),
            fmt(m["auc"])]))
    tabular(OUT / "prospective.tex",
            "Artifact & Snapshot & Rollcalls & Votes & Log loss & Acc.\\ (\\%) & AUC",
            rows, "llrrccc")


def numbers(lb):
    """Inline macros: every number the draft text quotes."""
    def macro(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    v1 = json.loads((RES / "frozen" / "prospective_report.json").read_text())
    v2 = json.loads((RES / "frozen" / "prospective_report_v2.json").read_text())
    champ, inc = "blend3_mlp_tfidf_emb3_tcal", "emb2_mlp_mq_16d_tcal"
    lines = [
        macro("champForecastLL", fmt(get(lb, champ, "forecast108_119", "log_loss"))),
        macro("champForecastAcc", pct(get(lb, champ, "forecast108_119", "accuracy"))),
        macro("champContestedAcc", pct(get(lb, champ, "forecast108_119", "contested_accuracy"))),
        macro("champAPRE", fmt(get(lb, champ, "forecast108_119", "apre"), 2)),
        macro("incForecastLL", fmt(get(lb, inc, "forecast108_119", "log_loss"))),
        macro("champGain", fmt(get(lb, inc, "forecast108_119", "log_loss")
                               - get(lb, champ, "forecast108_119", "log_loss"))),
        macro("champCongressoutLL", fmt(get(lb, champ, "congressout118", "log_loss"))),
        macro("majorityCongressoutLL", fmt(get(lb, "majority_question_rate",
                                               "congressout118", "log_loss"))),
        macro("gbRawForecastLL", fmt(get(lb, "gb_spatial_tfidf", "forecast108_119", "log_loss"))),
        macro("prospectiveVoneLL", fmt(v1["metrics"]["log_loss"])),
        macro("prospectiveVoneAcc", pct(v1["metrics"]["accuracy"])),
        macro("prospectiveVtwoLL", fmt(v2["metrics"]["log_loss"])),
        macro("prospectiveVtwoAcc", pct(v2["metrics"]["accuracy"])),
        macro("prospectiveN", f"{v1['votes_scored']:,}"),
        macro("prospectiveRC", f"{v1['rollcalls_scored']}"),
        macro("regimeAbestLL", fmt(get(lb, "ideal_point_8d", "regimeA_seed42", "log_loss"))),
        macro("regimeAbestAcc", pct(get(lb, "ideal_point_8d", "regimeA_seed42", "accuracy"))),
        macro("regimeAnominateLL", fmt(get(lb, "nominate_logit", "regimeA_seed42", "log_loss"))),
        macro("champForecastAUC", fmt(get(lb, champ, "forecast108_119", "auc"), 2)),
        macro("champRandomrcLL", fmt(get(lb, champ, "randomrc108_119", "log_loss"))),
        macro("incRandomrcLL", fmt(get(lb, inc, "randomrc108_119", "log_loss"))),
        macro("champCongressoutAcc", pct(get(lb, champ, "congressout118", "accuracy"))),
        macro("nomctxCongressoutAcc", pct(get(lb, "nominate_context_logit",
                                              "congressout118", "accuracy"))),
        macro("nomctxCongressoutLL", fmt(get(lb, "nominate_context_logit",
                                             "congressout118", "log_loss"))),
        macro("kraftForecastLL", fmt(get(lb, "kraft_bilinear_16d",
                                         "forecast108_119", "log_loss"))),
        macro("gbTcalForecastLL", fmt(get(lb, "gb_spatial_tfidf_tcal",
                                          "forecast108_119", "log_loss"))),
        macro("gbForecastAUC", fmt(get(lb, "gb_spatial_tfidf", "forecast108_119",
                                       "auc"), 2)),
        macro("memberqForecastLL", fmt(get(lb, "member_question_rate",
                                           "forecast108_119", "log_loss"))),
        macro("constantForecastLL", fmt(get(lb, "constant_rate",
                                            "forecast108_119", "log_loss"))),
        macro("forecastTestN", f"{int(get(lb, champ, 'forecast108_119', 'n')):,}"),
    ]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {(OUT / 'numbers.tex').relative_to(ROOT)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lb = canonical(pd.read_csv(RES / "leaderboard.csv"))
    regime_audit(lb)
    regimeA(lb)
    litrace(lb)
    ablation(lb)
    error_decomp()
    prospective()
    numbers(lb)
    # completeness check: no placeholder cells in any table the draft inputs
    missing = []
    for f in OUT.glob("*.tex"):
        if "---" in f.read_text():
            missing.append(f.name)
    if missing:
        print(f"WARNING: placeholder cells in {missing}")
    else:
        print("all tables complete — no placeholder cells")


if __name__ == "__main__":
    main()
