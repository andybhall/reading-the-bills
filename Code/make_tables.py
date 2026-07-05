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
        ll = get(lb, m, "regimeA_seed42", "log_loss")
        rows.append(" & ".join([
            PRETTY[m],
            pct(get(lb, m, "regimeA_seed42", "accuracy")),
            fmt(get(lb, m, "regimeA_seed42", "apre"), 2),
            fmt(np.exp(-ll), 3),  # GMP, the NOMINATE literature's statistic
            fmt(ll)]))
    tabular(OUT / "regimeA.tex",
            "Model & CC (\\%) & APRE & GMP & Log loss",
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
            "Model & Fcst.\\ LL & Fcst.\\ AUC & Fcst.\\ cont.\\ acc.\\ (\\%) & "
            "Rand.\\ LL & Rand.\\ AUC",
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


def issue_topics():
    """Systematic per-topic summary: coverage, party gap, within-party
    dispersion, correlation with the overall dimension, and the largest
    recent-era deviators in each direction."""
    iss = pd.read_parquet(RES / "issue_positions.parquet")
    iss = iss[(iss.topic != "OVERALL") & (iss.n_votes >= 50)].copy()
    pos = pd.read_parquet(RES / "member_positions_1d.parquet")
    iss = iss.merge(pos[["icpsr", "last_congress"]], on="icpsr", how="left")

    def nm(r):
        last = str(r.bioname).split(",")[0].title()
        return f"{last} ({r.state_abbrev})"

    rows = []
    order = (iss.groupby("topic")["icpsr"].count()
             .sort_values(ascending=False).index)
    for t in order:
        d = iss[iss.topic == t]
        dd, rr = d[d.party_code == 100.0], d[d.party_code == 200.0]
        gap = rr.z.median() - dd.z.median()
        rho = np.corrcoef(d.overall_z, d.z)[0, 1]
        rec = d[d.last_congress >= 114].sort_values("deviation")
        lib = nm(rec.iloc[0]) if len(rec) else "---"
        con = nm(rec.iloc[-1]) if len(rec) else "---"
        rows.append(" & ".join([
            t.replace(" and ", " \\& "), f"{len(d):,}", fmt(gap, 2),
            fmt(dd.z.std(), 2), fmt(rr.z.std(), 2), fmt(rho, 2),
            lib, con]))
    tabular(OUT / "issue_topics.tex",
            "Policy area & $N$ & Party gap & SD (D) & SD (R) & "
            "$\\rho$(overall) & Largest liberal dev. & Largest conservative dev.",
            rows, "lrccccll")


def cutpoint_pred():
    """The text-to-cutpoint horse race + text-component ablation."""
    r = json.loads((RES / "measures" / "cutpoint_pred.json").read_text())
    label = {"constant": "Constant", "metadata": "Metadata only",
             "tfidf_svd": "Text: TF--IDF", "embeddings": "Text: embeddings",
             "embeddings_meta": "Text + metadata"}
    rows = []
    for k in ("constant", "metadata", "tfidf_svd", "embeddings", "embeddings_meta"):
        m = r["sets"][k]
        rows.append(" & ".join([label[k], fmt(m["cut_mae"]), fmt(m["cut_r"], 2),
                                pct(m["dir_acc_identified"])]))
    ab_label = {"t_question": "\\quad question only",
                "t_desc": "\\quad description only",
                "t_summary": "\\quad bill summary only"}
    for k, m in r["component_ablation"].items():
        rows.append(" & ".join([ab_label[k], fmt(m["cut_mae"]),
                                fmt(m["cut_r"], 2),
                                pct(m["dir_acc_identified"])]))
    tabular(OUT / "cutpoint_pred.tex",
            "Features & Cutpoint MAE & Cutpoint $r$ & Direction acc.\\ (\\%)",
            rows, "lccc")


def decomposition(lb):
    """Signal decomposition (review r1, O4): one table, identically
    evaluated, isolating party, member history, metadata, text, and
    their combinations across the two forecasting regimes."""
    rows_spec = [
        ("party_question_rate", "Party history only"),
        ("member_question_rate", "Member history only"),
        ("meta_tower_8d", "Rollcall metadata only"),
        ("text_tower_8d", "Text + metadata (no member history)"),
        ("text_tower_mq_8d", "Text + metadata + member history"),
        ("emb2_mlp_mq_16d_tcal", "+ modern encoder, deep head, calibration"),
        ("blend3_mlp_tfidf_emb3_tcal", "+ ensemble (final)"),
    ]
    rows = []
    for m, lab in rows_spec:
        rows.append(" & ".join([
            lab,
            fmt(get(lb, m, "forecast108_119", "log_loss")),
            pct(get(lb, m, "forecast108_119", "accuracy")),
            fmt(get(lb, m, "congressout118", "log_loss")),
            pct(get(lb, m, "congressout118", "accuracy"))]))
    tabular(OUT / "decomposition.tex",
            "Information used & \\multicolumn{2}{c}{Temporal forecast} & "
            "\\multicolumn{2}{c}{Congress-out}",
            rows, "lcccc")


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
    # GMP versions of headline fits (exp(-log loss)); the field's statistic
    lines += [
        macro("champForecastGMP", fmt(np.exp(-get(lb, champ, "forecast108_119",
                                                  "log_loss")), 3)),
        macro("regimeAbestGMP", fmt(np.exp(-get(lb, "ideal_point_8d",
                                                "regimeA_seed42", "log_loss")), 3)),
        macro("regimeAnominateGMP", fmt(np.exp(-get(lb, "nominate_logit",
                                                    "regimeA_seed42", "log_loss")), 3)),
    ]
    # agenda-control quantities: cutpoint mass between floor and majority
    # medians in the 118th House (majority: R), per cartel-theory exhibits
    cut = pd.read_parquet(RES / "measures" / "cutpoints_house118.parquet")
    memb = pd.read_parquet(RES / "measures" / "members_house118.parquet")
    cut = cut[cut.identified & cut.cutpoint.between(-4, 4)]
    floor_med = memb.x.median()
    rmed = memb.loc[memb.party_code == 200.0, "x"].median()
    dmed = memb.loc[memb.party_code == 100.0, "x"].median()
    lo, hi = sorted([floor_med, rmed])
    lines += [
        macro("blockoutShare", pct(((cut.cutpoint >= lo) & (cut.cutpoint <= hi)).mean())),
        macro("cutLeftOfFloor", pct((cut.cutpoint < floor_med).mean())),
        # the party-line region: cutpoints between the two party medians,
        # where (given orientation) the parties are on opposite sides
        macro("cutBetweenParties", pct(cut.cutpoint.between(
            *sorted([dmed, floor_med])).mean())),
        macro("houseFloorMed", fmt(floor_med, 2)),
        macro("houseRMed", fmt(rmed, 2)),
        macro("houseDMed", fmt(dmed, 2)),
        macro("nCutHouse", f"{len(cut):,}"),
    ]
    # amendment-join coverage denominators (review r1 #11)
    am = pd.read_parquet(ROOT / "Modified Data" / "amendments.parquet")
    rcs = pd.read_parquet(ROOT / "Modified Data" / "rollcalls.parquet")
    rcs = rcs[rcs.congress >= 108].copy()
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "Code"))
    from models_forecast import question_bucket as _qb
    rcs["qb"] = _qb(rcs["vote_question"])
    amrc = rcs[rcs.qb == "amendment"].copy()
    nodesc = amrc["vote_desc"].isna()
    # session-offset amender join, recomputed (same code path as analysis)
    from importlib import import_module
    tcp = import_module("23_text_cutpoints")
    amrc["amdt_sponsor"] = tcp.amendment_sponsors(
        amrc.rename(columns={})).to_numpy()
    cov = amrc["amdt_sponsor"].notna().mean()
    filled_nodesc = int((nodesc & amrc["amdt_sponsor"].notna()).sum())
    lines += [
        macro("amdtTotal", f"{len(amrc):,}"),
        macro("amdtNoDesc", f"{int(nodesc.sum()):,}"),
        macro("amdtFilledNoDesc", f"{filled_nodesc:,}"),
        macro("amdtSponsorCov", pct(cov)),
    ]
    # prospective bootstrap CI (v1 ledger, rollcall-cluster, 2000 reps)
    led = pd.read_parquet(RES / "frozen" / "prospective_ledger.parquet")
    rng = np.random.default_rng(42)
    keys = list(led.groupby(["congress", "chamber", "rollnumber"]).groups.values())
    lls = []
    for _ in range(2000):
        idx = np.concatenate([keys[i] for i in rng.integers(0, len(keys), len(keys))])
        s = led.loc[idx]
        q = np.clip(s.p_yea, 1e-12, 1 - 1e-12)
        lls.append(float(-(s.vote * np.log(q) + (1 - s.vote) * np.log(1 - q)).mean()))
    lines += [macro("prospLLlo", fmt(np.percentile(lls, 2.5))),
              macro("prospLLhi", fmt(np.percentile(lls, 97.5)))]
    # post-FREEZE subset for v1 (freeze June 12; snapshot June 9): r2 #3
    pf = led[pd.to_datetime(led.date) > "2026-06-12"]
    qpf = np.clip(pf.p_yea, 1e-12, 1 - 1e-12)
    llpf = float(-(pf.vote * np.log(qpf) + (1 - pf.vote) * np.log(1 - qpf)).mean())
    lines += [
        macro("postFreezeLL", fmt(llpf)),
        macro("postFreezeN", f"{len(pf):,}"),
        macro("postFreezeRC", f"{pf.groupby(['congress','chamber','rollnumber']).ngroups}"),
    ]
    # identification-threshold sensitivity for agenda shares (r2 O8/D1)
    cutraw = pd.read_parquet(RES / "measures" / "cutpoints_house118.parquet")
    cutraw = cutraw[cutraw.cutpoint.between(-4.2, 4.2)]
    for thr, tag in ((0.25, "Lo"), (0.50, "Hi")):
        cc = cutraw[cutraw.a.abs() >= thr]
        lines += [macro(f"cutBetweenParties{tag}", pct(cc.cutpoint.between(
            *sorted([dmed, floor_med])).mean()))]
    cpass = cutraw[cutraw.identified & (cutraw.qbucket == "passage")]
    lines += [macro("cutBetweenPartiesPassage", pct(cpass.cutpoint.between(
        *sorted([dmed, floor_med])).mean()))]
    # amendment-experiment deltas (r2 #10): forecast val rows from ledgered runs
    for name, mac in (("emb4_mlp_mq_16d_tcal", "amdtSwapLL"),):
        pass  # values pulled below from full history, not canonical dedup
    hist = pd.read_csv(RES / "leaderboard.csv")
    hv = hist[(hist.split == "forecast108_119") & (hist.eval_set == "val")]
    e5 = hv[hv.model == "emb4_mlp_mq_16d_tcal"].log_loss.tolist()
    if len(e5) >= 3:
        lines += [macro("eFiveA", fmt(e5[0])), macro("eFiveB", fmt(e5[1])),
                  macro("eFiveC", fmt(e5[2]))]
    champ_ctrl = hv[hv.model == "emb2_mlp_mq_16d_tcal"].log_loss
    placebo = hv[hv.model == "emb_placebo_mlp_mq_16d_tcal"].log_loss
    lines += [macro("eFivePlacebo", fmt(placebo.iloc[-1])),
              macro("champValCtrl", fmt(champ_ctrl.iloc[-1]))]
    # paired rollcall-cluster bootstrap for headline forecast margins (r2 O9)
    pr = {m: pd.read_parquet(RES / "preds" / f"forecast108_119_{m}.parquet")
          for m in ("blend3_mlp_tfidf_emb3_tcal", "blend_mlp_tfidf_tcal",
                    "emb2_mlp_mq_16d_tcal")}
    base = pr["blend3_mlp_tfidf_emb3_tcal"].sort_values(
        ["congress", "chamber", "rollnumber", "icpsr"]).reset_index(drop=True)
    rckey = base.groupby(["congress", "chamber", "rollnumber"]).ngroup().to_numpy()
    def _ll(p, y):
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    y = base.vote.to_numpy()
    ll3 = _ll(base.p_yea.to_numpy(), y)
    diffs = {}
    for m, tag in (("blend_mlp_tfidf_tcal", "Two"), ("emb2_mlp_mq_16d_tcal", "One")):
        o = pr[m].sort_values(["congress", "chamber", "rollnumber",
                               "icpsr"]).reset_index(drop=True)
        diffs[tag] = _ll(o.p_yea.to_numpy(), y) - ll3
    rng2 = np.random.default_rng(7)
    ngroups = rckey.max() + 1
    idx_by_g = pd.Series(np.arange(len(y))).groupby(rckey).apply(np.array)
    for tag, dvec in diffs.items():
        gsum = pd.Series(dvec).groupby(rckey).sum().to_numpy()
        gn = pd.Series(dvec).groupby(rckey).size().to_numpy()
        boots = []
        for _ in range(2000):
            pick = rng2.integers(0, ngroups, ngroups)
            boots.append(gsum[pick].sum() / gn[pick].sum())
        lines += [macro(f"pairedDelta{tag}Lo", fmt(np.percentile(boots, 2.5))),
                  macro(f"pairedDelta{tag}Hi", fmt(np.percentile(boots, 97.5)))]
    # Nokken-Poole convergent validity for the per-congress fits
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")
    np_rs = []
    for cong, ch in [(118, "House"), (118, "Senate"), (119, "House"), (119, "Senate")]:
        pos = pd.read_parquet(RES / "measures" / f"members_{ch.lower()}{cong}.parquet")
        mm = mem[(mem.congress == cong) & (mem.chamber == ch)][
            ["icpsr", "nokken_poole_dim1"]]
        j = pos.merge(mm, on="icpsr").dropna(subset=["nokken_poole_dim1"])
        np_rs.append(np.corrcoef(j.x, j.nokken_poole_dim1)[0, 1])
    lines += [macro("npRmin", fmt(min(np_rs), 2)),
              macro("npRmax", fmt(max(np_rs), 2))]
    # member-level held-out fit summary
    mf = pd.read_parquet(RES / "measures" / "member_fit.parquet")
    mf = mf[mf.n >= 100]
    lines += [
        macro("memberGMPshare", pct((mf.gmp_ours > mf.gmp_nom).mean())),
        macro("memberGMPmedOurs", fmt(mf.gmp_ours.median(), 3)),
        macro("memberGMPmedNom", fmt(mf.gmp_nom.median(), 3)),
        macro("paulGMPnom", fmt(mf.loc[mf.bioname.str.contains(
            "PAUL, Ronald", na=False), "gmp_nom"].iloc[0], 2)),
        macro("paulGMPours", fmt(mf.loc[mf.bioname.str.contains(
            "PAUL, Ronald", na=False), "gmp_ours"].iloc[0], 2)),
    ]
    # transition-study summary macros
    tr = json.loads((RES / "measures" / "transitions.json").read_text())
    def _mean(flip, bucket=None):
        vals = []
        for e in tr.values():
            if e["flip"] == flip:
                m = e["models"]["champion"]
                vals.append(m["by_qbucket"].get(bucket) if bucket else m["log_loss"])
        return float(np.mean([v for v in vals if v is not None]))
    lines += [
        macro("flipLL", fmt(_mean(True))),
        macro("placeboLL", fmt(_mean(False))),
        macro("flipProcLL", fmt(_mean(True, "procedural"))),
        macro("placeboProcLL", fmt(_mean(False, "procedural"))),
        macro("flipPassLL", fmt(_mean(True, "passage"))),
        macro("placeboPassLL", fmt(_mean(False, "passage"))),
    ]
    # protest-detection comparison (P5v6 point 2)
    pdj = json.loads((RES / "measures" / "protest_detection.json").read_text())
    lines += [
        macro("protAUCtext", fmt(pdj["blend3_mlp_tfidf_emb3_tcal"]["auc_within_rollcall"], 2)),
        macro("protAUCnotext", fmt(pdj["notext_mq_16d_tcal"]["auc_within_rollcall"], 2)),
        macro("protN", f"{pdj['blend3_mlp_tfidf_emb3_tcal']['n_defections']:,}"),
        macro("protRC", f"{pdj['blend3_mlp_tfidf_emb3_tcal']['n_rollcalls_3plus']}"),
        macro("notextForecastLL", fmt(get(lb, "notext_mq_16d_tcal",
                                          "forecast108_119", "log_loss"))),
    ]
    cp = json.loads((RES / "measures" / "cutpoint_pred.json").read_text())["sets"]
    lines += [
        macro("cutComboMAE", fmt(cp["embeddings_meta"]["cut_mae"], 2)),
        macro("cutComboR", fmt(cp["embeddings_meta"]["cut_r"], 2)),
        macro("cutMetaMAE", fmt(cp["metadata"]["cut_mae"], 2)),
        macro("cutMetaR", fmt(cp["metadata"]["cut_r"], 2)),
        macro("cutConstMAE", fmt(cp["constant"]["cut_mae"], 2)),
        macro("dirCombo", pct(cp["embeddings_meta"]["dir_acc_identified"])),
        macro("dirMeta", pct(cp["metadata"]["dir_acc_identified"])),
        macro("dirTfidf", pct(cp["tfidf_svd"]["dir_acc_identified"])),
        macro("dirConst", pct(cp["constant"]["dir_acc_identified"])),
        macro("cutTestN", f"{cp['embeddings_meta']['n_test_cut']:,}"),
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
    issue_topics()
    cutpoint_pred()
    decomposition(lb)
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
