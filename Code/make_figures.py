"""Generate Paper A's figures from results files. No hand-entered data.

Outputs PDF figures to Draft/figures/. Inputs: the measurement layer
(22_extract_measures.py), the validated member instruments
(03/09/10 outputs), saved test predictions, and the leaderboard.

Run: python3 Code/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "Modified Data" / "results"
MEAS = RES / "measures"
OUT = ROOT / "Draft" / "figures"

DEM, REP, OTHER = "#2166ac", "#b2182b", "#888888"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "figure.dpi": 150,
})


def party_color(pc):
    return np.where(pc == 100.0, DEM, np.where(pc == 200.0, REP, OTHER))


def shortname(bioname, state):
    last = str(bioname).split(",")[0].title()
    return f"{last} ({state})"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Draft/figures/{name}")


# ---------------------------------------------------------------- F1
def f1_validation():
    pos = pd.read_parquet(RES / "member_positions_1d.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, ch in zip(axes, ("House", "Senate")):
        d = pos[(pos.chamber == ch) & pos.nominate_dim1.notna()]
        ax.scatter(d.nominate_dim1, d.dim1, s=7, alpha=0.45,
                   c=party_color(d.party_code), linewidths=0)
        r = np.corrcoef(d.nominate_dim1, d.dim1)[0, 1]
        ax.set_title(f"{ch}  (r = {r:.2f})")
        ax.set_xlabel("DW-NOMINATE (first dimension)")
        ax.axhline(0, lw=0.4, color="#cccccc", zorder=0)
        ax.axvline(0, lw=0.4, color="#cccccc", zorder=0)
    axes[0].set_ylabel("Learned position (this paper)")
    save(fig, "validation_scatter.pdf")


# ---------------------------------------------------------------- F2
def f2_loyalty():
    """The full loyalty-residual distribution for every member (recent
    era), not a curated tail: (a) its shape by party; (b) its
    independence from ideology, with the extremes labeled where they sit
    in the complete cloud."""
    sig = pd.read_parquet(RES / "member_signals.parquet")
    recent = sig[sig.congress >= 115]
    recent = recent.assign(w=recent.n_unity,
                           rw=recent.loyalty_residual * recent.n_unity,
                           xw=recent.ideal_1d * recent.n_unity)
    g = (recent.groupby(["icpsr", "bioname", "party_code", "state_abbrev"])
         .agg(rw=("rw", "sum"), xw=("xw", "sum"), n=("w", "sum"))
         .reset_index())
    # vote-weighted means: unweighted congress-means overweight short
    # stints (e.g. a 140-vote party-switch transition period)
    g["resid"], g["x"] = g.rw / g.n, g.xw / g.n
    g = g[(g.n >= 100) & g.party_code.isin([100.0, 200.0])]
    g = g[~g.state_abbrev.isin(["GU", "PR", "VI", "DC", "AS", "MP"])]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1, 1.5]})

    bins = np.linspace(-0.22, 0.10, 55)
    for pc, color, lab in ((100.0, DEM, "Democrats"), (200.0, REP, "Republicans")):
        ax1.hist(g.loc[g.party_code == pc, "resid"], bins=bins, color=color,
                 alpha=0.55, label=lab)
    ax1.axvline(0, color="#444444", lw=0.8)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_xlabel("Loyalty residual")
    ax1.set_ylabel("Members")
    ax1.set_title(f"All {len(g):,} members")

    ax2.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax2.scatter(g.x, g.resid, s=10, alpha=0.5, c=party_color(g.party_code),
                linewidths=0)
    lab = pd.concat([g.nsmallest(8, "resid"), g.nlargest(2, "resid")])
    lab = lab.sort_values("resid")
    for i, r in enumerate(lab.itertuples()):  # alternate label sides
        dx = 5 if i % 2 else -5
        ax2.annotate(shortname(r.bioname, r.state_abbrev), (r.x, r.resid),
                     fontsize=7, xytext=(dx, -2), textcoords="offset points",
                     ha="left" if dx > 0 else "right")
    rho = np.corrcoef(g.x.abs(), g.resid)[0, 1]
    ax2.set_xlabel("Ideal point (liberal $\\leftarrow$ $\\rightarrow$ conservative)")
    ax2.set_ylabel("Loyalty residual")
    ax2.set_title(f"Loyalty is not ideology "
                  f"($\\rho$ with extremity = {rho:.2f})")
    fig.suptitle("Party loyalty beyond ideology, 115th–119th Congresses",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "loyalty_distribution.pdf")


# ---------------------------------------------------------------- F3
TOPIC_SHORT = {
    "Economics and Public Finance": "Economics & Public Finance",
    "Armed Forces and National Security": "Armed Forces & Nat. Security",
    "International Affairs": "International Affairs",
    "Government Operations and Politics": "Government Operations",
    "Crime and Law Enforcement": "Crime & Law Enforcement",
    "Finance and Financial Sector": "Finance & Financial Sector",
    "Public Lands and Natural Resources": "Public Lands & Nat. Resources",
    "Transportation and Public Works": "Transportation & Public Works",
}


def load_issues(min_votes=50):
    iss = pd.read_parquet(RES / "issue_positions.parquet")
    iss = iss[(iss.topic != "OVERALL") & (iss.n_votes >= min_votes)].copy()
    pos = pd.read_parquet(RES / "member_positions_1d.parquet")
    iss = iss.merge(pos[["icpsr", "last_congress"]], on="icpsr", how="left")
    iss["short"] = iss.topic.map(TOPIC_SHORT).fillna(iss.topic)
    return iss


def f3_issue_grid():
    """Every measured legislator, every major topic: topic position
    against overall position. Deviations are vertical distances from the
    45-degree line; the two largest recent-era deviations per panel are
    labeled."""
    iss = load_issues()
    topics = (iss.groupby("topic")["icpsr"].count()
              .sort_values(ascending=False).head(9).index.tolist())
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 8.6), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), topics):
        d = iss[iss.topic == t]
        ax.plot([-3, 3], [-3, 3], color="#bbbbbb", lw=0.7, zorder=0)
        ax.scatter(d.overall_z, d.z, s=5, alpha=0.35,
                   c=party_color(d.party_code), linewidths=0)
        lab = d[d.last_congress >= 114].reindex(
            d.deviation.abs().sort_values(ascending=False).index)
        lab = lab[lab.last_congress >= 114].head(2)
        for r in lab.itertuples():
            ax.annotate(shortname(r.bioname, r.state_abbrev),
                        (r.overall_z, r.z), fontsize=6.5,
                        xytext=(3, -6), textcoords="offset points")
        rho = np.corrcoef(d.overall_z, d.z)[0, 1]
        ax.set_title(f"{iss.loc[iss.topic == t, 'short'].iloc[0]}"
                     f"  ($\\rho$ = {rho:.2f})", fontsize=8.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    for ax in axes[-1]:
        ax.set_xlabel("Overall position", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Position on this issue", fontsize=8.5)
    fig.suptitle("Issue-specific positions of every measured legislator, "
                 "101st–119th Congresses", fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "issue_grid.pdf")


def f3b_topic_polarization():
    """Party positions by topic: medians, interquartile ranges, and the
    implied party gap, for all thirteen topics."""
    iss = load_issues()
    stats = []
    for t, d in iss.groupby("topic"):
        dd, rr = d[d.party_code == 100.0].z, d[d.party_code == 200.0].z
        stats.append((d["short"].iloc[0], dd.median(), rr.median(),
                      dd.quantile(0.25), dd.quantile(0.75),
                      rr.quantile(0.25), rr.quantile(0.75)))
    st = pd.DataFrame(stats, columns=["topic", "dmed", "rmed",
                                      "d25", "d75", "r25", "r75"])
    st["gap"] = st.rmed - st.dmed
    st = st.sort_values("gap")
    y = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hlines(y, st.dmed, st.rmed, color="#cccccc", lw=1.2, zorder=1)
    ax.hlines(y - 0.14, st.d25, st.d75, color=DEM, lw=3, alpha=0.35)
    ax.hlines(y + 0.14, st.r25, st.r75, color=REP, lw=3, alpha=0.35)
    ax.scatter(st.dmed, y - 0.14, color=DEM, s=34, zorder=3, label="Democratic median")
    ax.scatter(st.rmed, y + 0.14, color=REP, s=34, zorder=3, label="Republican median")
    ax.set_yticks(y)
    ax.set_yticklabels(st.topic, fontsize=8.5)
    ax.set_xlabel("Issue-specific position (bars: interquartile ranges)")
    ax.set_title("How far apart the parties are, issue by issue")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "topic_polarization.pdf")


# ---------------------------------------------------------------- F4
def f4_cutpoints():
    rc = pd.read_parquet(MEAS / "cutpoints_house118.parquet")
    mem = pd.read_parquet(MEAS / "members_house118.parquet")
    rc = rc[rc.identified & rc.cutpoint.between(-4.2, 4.2)]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})

    for pc, color, lab in ((100.0, DEM, "Democrats"), (200.0, REP, "Republicans")):
        x = mem.loc[mem.party_code == pc, "x"]
        ax1.hist(x, bins=55, range=(-4.2, 4.2), color=color, alpha=0.55,
                 label=lab, density=True)
    ax1.hist(rc.cutpoint, bins=70, range=(-4.2, 4.2), histtype="step",
             color="#333333", lw=1.1, density=True, label="Rollcall cutpoints")
    # medians for the agenda-control reading (cartel-theory exhibits):
    # cutpoints between the floor and majority medians are votes that
    # could roll the majority party
    for x, lab, c in ((mem.x.median(), "floor\nmedian", "#333333"),
                      (mem.loc[mem.party_code == 100.0, "x"].median(),
                       "D median", DEM),
                      (mem.loc[mem.party_code == 200.0, "x"].median(),
                       "R median", REP)):
        ax1.axvline(x, color=c, lw=0.9, ls="--", alpha=0.8)
        ax1.annotate(lab, (x, ax1.get_ylim()[1] * 0.86), fontsize=6.5,
                     ha="center", color=c)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_ylabel("Density")
    ax1.set_title("Members and the bills that divide them: 118th House")

    areas = (rc.groupby("policy_area")["cutpoint"].count()
             .sort_values(ascending=False).head(8).index.tolist())
    d = rc[rc.policy_area.isin(areas)]
    order = (d.groupby("policy_area")["cutpoint"].median()
             .sort_values().index.tolist())
    rng = np.random.default_rng(0)
    for i, area in enumerate(order):
        c = d.loc[d.policy_area == area, "cutpoint"]
        ax2.scatter(c, i + rng.uniform(-0.16, 0.16, len(c)), s=6, alpha=0.4,
                    color="#555555", linewidths=0)
        ax2.scatter(c.median(), i, s=70, color="#111111", marker="|", zorder=3)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([a[:34] for a in order], fontsize=8)
    ax2.set_xlabel("Position on the liberal–conservative scale "
                   "(cutpoint = where a member is indifferent)")
    save(fig, "cutpoints_house118.pdf")


# ---------------------------------------------------------------- F5
def f5_surprises():
    s = pd.read_parquet(MEAS / "surprises.parquet")
    s = s[s.congress >= 113].copy()
    # separate INDIVIDUAL out-of-character votes from bloc events (mass
    # defections on one rollcall, e.g. party-strategy nays on suspension
    # bills — interesting, but a different measure): drop rollcalls that
    # appear repeatedly at the top of the surprise ranking
    rc_key = ["congress", "chamber", "rollnumber"]
    counts = s.groupby(rc_key)["icpsr"].transform("count")
    s = s[counts <= 3]
    # suspension-bill nays are a distinct strategic category (discussed in
    # the text); excluded here so the figure shows the varied individual
    # defections, capped per question type for diversity
    s = s[~s.vote_question.fillna("").str.contains("Suspend", case=False)]
    s = s.drop_duplicates(rc_key).drop_duplicates("icpsr")
    from models_forecast import question_bucket
    s["qb"] = question_bucket(s["vote_question"])
    s = s.groupby("qb", group_keys=False).head(4)
    s = s.sort_values("vote_ll", ascending=False).head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(s))
    actual = s.vote.to_numpy()
    ax.hlines(y, s.p_yea, actual, color="#bbbbbb", lw=1.0, zorder=1)
    ax.scatter(s.p_yea, y, s=30, c=party_color(s.party_code), zorder=3,
               label="Model probability of yea")
    ax.scatter(actual, y, s=48, marker="s", facecolors="none",
               edgecolors="#111111", zorder=3, label="Actual vote")
    qb = s.vote_question.fillna("").str.replace("On ", "", regex=False)
    labs = [f"{shortname(b, st)} — {q[:26]}, {int(c)}th"
            for b, st, q, c in zip(s.bioname, s.state_abbrev, qb, s.congress)]
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("P(yea): model (dot) versus outcome (square)")
    ax.set_title("Out-of-character votes: the model's largest in-sample surprises")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "surprises.pdf")


# ---------------------------------------------------------------- F6
def f6_calibration():
    """Left: member-vote reliability curves on the forecast holdout.
    Right: rollcall-level predicted vs realized majority-defection
    share (review r3, R10 — the Section 4 quantities are probability
    claims, so their calibration is shown directly)."""
    from models_forecast import question_bucket
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    specs = [("notext_mq_16d_tcal", "No-text counterpart", "#b2182b", "--"),
             ("emb2_mlp_mq_16d_tcal", "Leakage-clean tower", "#4393c3", "-"),
             ("blend3_mlp_tfidf_emb3_tcal", "Ensemble", "#111111", "-")]
    for model, label, color, ls in specs:
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        bins = np.quantile(p.p_yea, np.linspace(0, 1, 16))
        idx = np.clip(np.searchsorted(bins, p.p_yea) - 1, 0, 14)
        g = p.groupby(idx).agg(conf=("p_yea", "mean"), acc=("vote", "mean"))
        ax.plot(g.conf, g.acc, ls, color=color, lw=1.6, label=label,
                marker="o", ms=3)
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, zorder=0)
    ax.set_xlabel("Predicted probability of yea")
    ax.set_ylabel("Observed yea rate")
    ax.set_title("(a) Member-vote reliability", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    rc = pd.read_parquet(ROOT / "Modified Data" / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question"]]
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    key = ["congress", "chamber", "rollnumber"]
    for model, label, color, ls in specs[::2]:  # no-text and ensemble
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        d = (p.merge(rc, on=key, how="left")
              .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                     suffixes=("_v", "")))
        d["qb"] = question_bucket(d["vote_question"])
        d = d[d.qb.isin(["passage", "resolution", "cloture"])
              & d.party_code.isin([100.0, 200.0])]
        maj = (d.groupby(key + ["party_code"])["vote"].mean()
               .rename("party_rate").reset_index())
        d = d.merge(maj, on=key + ["party_code"])
        d = d[d.party_rate >= 0.5]
        g = d.groupby(key).agg(
            pred=("p_yea", lambda s: 1 - s.mean()),
            real=("vote", lambda s: 1 - s.mean()),
            n=("vote", "size"))
        g = g[g.n >= 50]
        q = pd.qcut(g.pred, 10, duplicates="drop")
        b = g.groupby(q, observed=True).agg(pred=("pred", "mean"),
                                            real=("real", "mean"))
        ax2.plot(b.pred, b.real, ls, color=color, lw=1.6, label=label,
                 marker="o", ms=3)
    lim = 0.20
    ax2.plot([0, lim], [0, lim], color="#aaaaaa", lw=0.8, zorder=0)
    ax2.set_xlabel("Predicted majority-defection share (rollcall deciles)")
    ax2.set_ylabel("Realized majority-defection share")
    ax2.set_title("(b) Defection-share calibration", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "calibration.pdf")


# ---------------------------------------------------------------- F7
def f7_regimes():
    lb = pd.read_csv(RES / "leaderboard.csv").sort_values("run_utc")
    lb = lb.drop_duplicates(["model", "split", "eval_set"], keep="last")
    lb = lb[lb.eval_set == "test"]
    models = [("member_question_rate", "Member history table"),
              ("gb_spatial_tfidf_tcal", "2011-style (calibrated)"),
              ("kraft_bilinear_16d", "2016-style bilinear"),
              ("emb2_mlp_mq_16d_tcal", "Modern tower"),
              ("blend3_mlp_tfidf_emb3_tcal", "Final blend")]
    splits = [("randomrc108_119", "Random rollcall\nholdout"),
              ("forecast108_119", "Temporal\nforecast"),
              ("congressout118", "New congress\n(majority flip)")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.arange(len(splits))
    colors = plt.cm.viridis(np.linspace(0.85, 0.05, len(models)))
    for (m, lab), color in zip(models, colors):
        ys = [lb[(lb.model == m) & (lb.split == s)].log_loss.iloc[0]
              for s, _ in splits]
        ax.plot(xs, ys, "-o", color=color, lw=1.5, ms=5, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in splits])
    ax.set_ylabel("Test log loss (lower = better)")
    ax.set_title("The same models under three forecasting regimes")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "regime_lines.pdf")


# ---------------------------------------------------------------- F8
def f8_cutpoint_prediction():
    rc = pd.read_parquet(MEAS / "cutpoint_rollcalls.parquet")
    d = rc[rc.test & rc.identified & rc.pred_embeddings_meta.notna()]
    d = d[d.cut_std.between(-4, 4)]
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.3, 0.85, 0.7]})
    ax1.plot([-4, 4], [-4, 4], color="#bbbbbb", lw=0.8, zorder=0)
    ax1.scatter(d.pred_embeddings_meta, d.cut_std, s=6, alpha=0.25,
                color="#777777", linewidths=0)
    qb = pd.qcut(d.pred_embeddings_meta, 12, duplicates="drop")
    bm = d.groupby(qb, observed=True).agg(p=("pred_embeddings_meta", "mean"),
                                          c=("cut_std", "mean"))
    ax1.plot(bm.p, bm.c, "-o", color="#111111", lw=1.8, ms=3.5, zorder=3)
    r = np.corrcoef(d.pred_embeddings_meta, d.cut_std)[0, 1]
    # landmark annotations: the most discriminating passage votes with
    # recognizable short titles among the held-out rollcalls
    links = pd.read_parquet(ROOT / "Modified Data" / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "bill_type", "bill_no"]]
    bills = pd.read_parquet(ROOT / "Modified Data" / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "title"]]
    dd = (d.merge(links, on=["congress", "chamber", "rollnumber"], how="left")
           .merge(bills, on=["congress", "bill_type", "bill_no"], how="left"))
    dd = dd[(dd.qbucket == "passage") & dd.title.notna()]
    dd["short"] = dd.title.str.extract(r"([A-Z][A-Za-z ]{3,28}? Act)")[0]
    lab = dd[dd.short.notna()].nlargest(2, "cut_std")  # right-cutting acts
    lab = pd.concat([lab, dd[dd.short.notna()].nsmallest(2, "cut_std")])
    offsets = [(5, 2), (-2, -10), (5, 2), (-2, -10)]
    for (dx, dy), row in zip(offsets, lab.itertuples()):
        ax1.annotate(row.short, (row.pred_embeddings_meta, row.cut_std),
                     fontsize=6, xytext=(dx, dy),
                     textcoords="offset points", color="#111111")
    ax1.set_xlabel("Predicted cutpoint (text + metadata)")
    ax1.set_ylabel("Realized cutpoint")
    ax1.annotate(f"r = {r:.2f}", (0.05, 0.92), xycoords="axes fraction", fontsize=9)

    j = json.loads((MEAS / "cutpoint_pred.json").read_text())
    names = [("metadata", "Metadata only"), ("tfidf_svd", "Text: TF-IDF"),
             ("embeddings", "Text: embeddings"),
             ("embeddings_meta", "Text + metadata")]
    y = np.arange(len(names))
    mae = [j["sets"][k]["cut_mae"] for k, _ in names]
    acc = [100 * j["sets"][k]["dir_acc_identified"] for k, _ in names]
    bar_colors = ["#aaaaaa", "#aaaaaa", "#aaaaaa", "#111111"]
    ax2.barh(y, mae, height=0.6, color=bar_colors)
    ax2.axvline(j["sets"]["constant"]["cut_mae"], color="#333333", lw=0.9,
                ls=":")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n for _, n in names], fontsize=8.5)
    ax2.set_xlabel("Cutpoint mean absolute error\n(member SDs)", fontsize=8.5)
    ax2.set_title("(b) Location error", fontsize=9.5)
    ax2.set_xlim(0, 0.95)
    ax3.barh(y, acc, height=0.6, color=bar_colors)
    ax3.axvline(100 * j["sets"]["constant"]["dir_acc_identified"],
                color="#333333", lw=0.9, ls=":")
    ax3.set_yticks(y)
    ax3.set_yticklabels([])
    ax3.set_xlabel("Direction accuracy (%)", fontsize=8.5)
    ax3.set_title("(c) Direction", fontsize=9.5)
    ax3.set_xlim(50, 80)
    ax1.set_title("(a) Predicted vs. realized cutpoints", fontsize=9.5)
    fig.tight_layout()
    save(fig, "cutpoint_prediction.pdf")


# ---------------------------------------------------------------- F9
def f9_direction_terms():
    t = pd.read_parquet(MEAS / "cutpoint_terms.parquet")
    top = pd.concat([t.nsmallest(12, "dir_coef"), t.nlargest(12, "dir_coef")])
    top = top.sort_values("dir_coef")
    y = np.arange(len(top))
    colors = np.where(top.dir_coef > 0, REP, DEM)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.barh(y, top.dir_coef, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top.term, fontsize=8.5)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Coefficient: bill-summary language $\\rightarrow$ "
                  "coalition direction")
    save(fig, "direction_terms.pdf")


# ---------------------------------------------------------------- F10
def f10_member_gmp():
    """Member-by-member fit on the field's canonical statistic: each
    member's percent of held-out votes correctly classified under our
    spatial model versus under the classical model (frozen DW-NOMINATE
    positions, estimated rollcall parameters), on identical votes.
    The GMP version tells the same story and its macros remain in the
    text as the proper-scoring variant."""
    mf = pd.read_parquet(MEAS / "member_fit.parquet")
    mf = mf[mf.n >= 100]
    cc_nom, cc_ours = 100 * (1 - mf.err_nom), 100 * (1 - mf.err_ours)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([55, 100], [55, 100], color="#bbbbbb", lw=0.8, zorder=0)
    ax.scatter(cc_nom, cc_ours, s=8, alpha=0.4,
               c=party_color(mf.party_code), linewidths=0)
    named = mf.bioname.str.contains("PAUL, Ronald|AMASH", na=False)
    lab = pd.concat([
        mf.assign(g=mf.err_nom - mf.err_ours,
                  ccn=cc_nom, cco=cc_ours).nlargest(6, "g"),
        mf[named].assign(g=0.0, ccn=cc_nom[named], cco=cc_ours[named])])
    for i, r in enumerate(lab.itertuples()):
        ax.annotate(shortname(r.bioname, r.state_abbrev),
                    (r.ccn, r.cco), fontsize=6.5,
                    xytext=(5, -2 if i % 2 else 4),
                    textcoords="offset points")
    share = (mf.err_ours < mf.err_nom).mean()
    ax.set_xlabel("Held-out votes correctly classified (%), "
                  "DW-NOMINATE-based model")
    ax.set_ylabel("Held-out votes correctly classified (%), "
                  "this paper's spatial model")
    ax.annotate(f"higher under this model for {share:.0%} of members",
                (0.05, 0.95), xycoords="axes fraction", fontsize=9)
    ax.set_xlim(55, 100.5)
    ax.set_ylim(55, 100.5)
    save(fig, "member_cc.pdf")


# ---------------------------------------------------------------- F11
def f11_transitions():
    """Majority-transition study: champion transfer loss for every House
    test congress, flips vs placebo transitions, with the by-vote-type
    breakdown that isolates the procedural-role mechanism."""
    r = json.loads((MEAS / "transitions.json").read_text())
    rows = sorted(((int(c), e) for c, e in r.items()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    xs = np.arange(len(rows))
    colors = [REP if e["flip"] else "#999999" for _, e in rows]
    ax1.bar(xs, [e["models"]["champion"]["log_loss"] for _, e in rows],
            color=colors)
    ax1.scatter(xs, [e["models"]["majority_question_rate"]["log_loss"]
                     for _, e in rows], marker="_", s=220, color="#111111",
                label="Majority-status table")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(c) for c, _ in rows])
    ax1.set_xlabel("Test congress (red: majority flipped)")
    ax1.set_ylabel("Test log loss")
    ax1.set_title("(a) All votes", fontsize=9.5)
    ax1.legend(frameon=False, fontsize=8)

    for qb, marker, lab in (("procedural", "o", "Procedural"),
                            ("passage", "s", "Final passage"),
                            ("amendment", "^", "Amendment")):
        f = [e["models"]["champion"]["by_qbucket"].get(qb) for _, e in rows]
        ax2.scatter([x for x, v in zip(xs, f) if v is not None],
                    [v for v in f if v is not None], marker=marker, s=34,
                    color=[c for c, v in zip(colors, f) if v is not None],
                    label=lab, alpha=0.9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(c) for c, _ in rows])
    ax2.set_xlabel("Test congress")
    ax2.set_title("(b) By vote type", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "transitions.pdf")


# ---------------------------------------------------------------- F12
def f12_protest():
    """One majority-splitting vote (H.R. 10545, the shutdown-averting
    American Relief Act, Dec 20 2024, holdout window), predicted twice
    by the same architecture — once without reading the bill, once
    reading it. The no-text model cannot see the flank revolt; the
    text model prices it member by member."""
    mem = pd.read_parquet(MEAS / "members_house118.parquet")
    names = mem[["icpsr", "x", "bioname", "state_abbrev", "party_code"]]
    panels = [("notext_mq_16d_tcal", "Without reading the bill"),
              ("blend3_mlp_tfidf_emb3_tcal", "Reading the bill")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for ax, (model, title) in zip(axes, panels):
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        d = p[(p.congress == 118) & (p.chamber == "House")
              & (p.rollnumber == 1235)].merge(names, on="icpsr")
        yea, nay = d[d.vote == 1], d[d.vote == 0]
        ax.scatter(yea.x, yea.p_yea, s=10, alpha=0.45,
                   c=party_color(yea.party_code), linewidths=0,
                   label="voted yea")
        ax.scatter(nay.x, nay.p_yea, s=42, marker="x", lw=1.4,
                   color="#111111", label="voted nay")
        ax.axhline(0.5, color="#bbbbbb", lw=0.7)
        ax.set_xlabel("Member ideal point (liberal $\\to$ conservative)")
        ax.set_title(title)
        if model.startswith("blend3"):
            lab = d[d.vote == 0].nsmallest(5, "p_yea")
            for i, r in enumerate(lab.itertuples()):
                ax.annotate(shortname(r.bioname, r.state_abbrev),
                            (r.x, r.p_yea), fontsize=6.5,
                            xytext=(4, -9 - 4 * (i % 2)),
                            textcoords="offset points")
    axes[0].set_ylabel("Predicted P(yea), pre-vote")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "protest_detection.pdf")


def f14_defector_capture():
    """Pooled companion to the single-vote exhibit (P5v10): across all
    majority-yea holdout rollcalls with >= 3 defections, the share of
    actual defectors found among the top-k majority members ranked by
    predicted defection probability, averaged over rollcalls. Chance =
    ranking members at random (k / party size)."""
    from models_forecast import question_bucket
    key = ["congress", "chamber", "rollnumber"]
    rc = pd.read_parquet(ROOT / "Modified Data" / "rollcalls.parquet")[
        key + ["vote_question"]]
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    ks = np.arange(1, 41)
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    specs = [("blend3_mlp_tfidf_emb3_tcal", "Reading the bill",
              "#111111", "-"),
             ("notext_mq_16d_tcal", "Without reading the bill",
              "#b2182b", "--")]
    chance = None
    per_vote = {}
    for model, label, color, ls in specs:
        p = pd.read_parquet(RES / "preds" / f"forecast108_119_{model}.parquet")
        d = (p.merge(rc, on=key, how="left")
              .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                     suffixes=("_v", "")))
        d["qb"] = question_bucket(d["vote_question"])
        d = d[d.qb.isin(["passage", "resolution", "cloture"])
              & d.party_code.isin([100.0, 200.0])]
        maj = (d.groupby(key + ["party_code"])["vote"].mean()
               .rename("party_rate").reset_index())
        d = d.merge(maj, on=key + ["party_code"])
        d = d[d.party_rate >= 0.5]
        d["defect"] = (d.vote == 0).astype(int)
        recalls, sizes, aucs = [], [], {}
        for gk, x in d.groupby(key + ["party_code"]):
            nd = x.defect.sum()
            if nd < 3 or nd == len(x):
                continue
            x = x.sort_values("p_yea")           # likeliest defectors first
            cum = x.defect.to_numpy().cumsum()
            recalls.append([cum[min(k, len(x)) - 1] / nd for k in ks])
            sizes.append(len(x))
            rr = (1 - x.p_yea).rank().to_numpy()
            n1, n0 = int(nd), int(len(x) - nd)
            aucs[gk] = float((rr[x.defect == 1].sum()
                              - n1 * (n1 + 1) / 2) / (n1 * n0))
        per_vote[model] = aucs
        r = np.mean(recalls, axis=0)
        ax.plot(ks, 100 * r, ls, color=color, lw=1.8, label=label)
        if chance is None:
            chance = np.mean([[min(k / n, 1) for k in ks] for n in sizes],
                             axis=0)
    # panel (a): the same detection statistic vote by vote — the text
    # gain is a distribution, concentrated on the votes member history
    # gets most wrong
    a_t = per_vote["blend3_mlp_tfidf_emb3_tcal"]
    a_n = per_vote["notext_mq_16d_tcal"]
    common = sorted(set(a_t) & set(a_n))
    xs = np.array([a_n[k] for k in common])
    ys = np.array([a_t[k] for k in common])
    ax0.plot([0.2, 1], [0.2, 1], color="#bbbbbb", lw=0.8, zorder=0)
    ax0.scatter(xs, ys, s=9, alpha=0.4, color="#333333", linewidths=0)
    share = (ys > xs).mean()
    ax0.annotate(f"text model higher on {share:.0%} of votes",
                 (0.05, 0.95), xycoords="axes fraction", fontsize=9)
    ax0.set_xlabel("Defection AUC without reading the bill")
    ax0.set_ylabel("Defection AUC reading the bill")
    ax0.set_title("(a) Vote by vote", fontsize=10)
    ax.set_xlabel("Majority members screened, in order of predicted risk")
    ax.set_ylabel("Share of actual defectors found (%)")
    ax.set_title("(b) Pooled screening curve", fontsize=10)
    ax.plot(ks, 100 * chance, ":", color="#999999", lw=1.2,
            label="Chance (random ranking)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "defector_capture.pdf")


def f13_revolt_risk():
    """A pre-vote measure of party stress (review r4, downstream use):
    the champion's predicted majority-defection share for every
    majority-yea passage-family vote in the 118th House holdout window,
    plotted by vote date against the realized share. The point is that
    the measure exists BEFORE each vote: leadership stress on the
    curated agenda is forecastable, and it spikes on the bipartisan
    deals and discharge-petition bills, not routine business."""
    from models_forecast import question_bucket
    key = ["congress", "chamber", "rollnumber"]
    p = pd.read_parquet(
        RES / "preds" / "forecast108_119_blend3_mlp_tfidf_emb3_tcal.parquet")
    rc = pd.read_parquet(ROOT / "Modified Data" / "rollcalls.parquet")[
        key + ["date", "vote_question", "vote_desc"]]
    mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")[
        ["congress", "chamber", "icpsr", "party_code"]]
    d = (p.merge(rc, on=key, how="left")
          .merge(mem, on=["congress", "chamber", "icpsr"], how="left",
                 suffixes=("_v", "")))
    d = d[(d.congress == 118) & (d.chamber == "House")]
    d["qb"] = question_bucket(d["vote_question"])
    d = d[d.qb.isin(["passage", "resolution", "cloture"])
          & (d.party_code == 200.0)]                    # GOP majority
    g = d.groupby(key + ["date", "vote_desc"], dropna=False).agg(
        pred=("p_yea", lambda s: 1 - s.mean()),
        real=("vote", lambda s: 1 - s.mean()),
        n=("vote", "size")).reset_index()
    g = g[(g.n >= 150) & (g.real < 0.5)]                # majority-yea votes

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    lim = max(g.real.max(), g.pred.max()) * 1.06
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=0.8, zorder=0)
    ax.scatter(g.pred, g.real, s=26, alpha=0.75, color="#333333",
               linewidths=0, zorder=2)
    # binned means make the monotone relationship legible through the
    # vertical scatter
    q = pd.qcut(g.pred, 6, duplicates="drop")
    b = g.groupby(q, observed=True).agg(p=("pred", "mean"),
                                        r=("real", "mean"))
    ax.plot(b.p, b.r, "-o", color=REP, lw=1.8, ms=4, zorder=3,
            label="binned mean")
    lab = pd.concat([g.nlargest(3, "real"), g.nlargest(2, "pred")])
    lab = lab.drop_duplicates(subset=["rollnumber"]).sort_values(
        "real", ascending=False)
    offs = [(6, 2), (6, -10), (6, 2), (6, 2), (-6, -10)]
    for (dx, dy), (_, r) in zip(offs, lab.iterrows()):
        ax.annotate(str(r.vote_desc)[:26], (r.pred, r.real), fontsize=6.5,
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel("Predicted majority-defection share (before the vote)")
    ax.set_ylabel("Realized majority-defection share")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "revolt_risk.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f1_validation()
    f2_loyalty()
    f3_issue_grid()
    f3b_topic_polarization()
    f5_surprises() if (MEAS / "surprises.parquet").exists() else print("skip F5 (no measures yet)")
    f4_cutpoints() if (MEAS / "cutpoints_house118.parquet").exists() else print("skip F4 (no measures yet)")
    f6_calibration()
    f7_regimes()
    f8_cutpoint_prediction()
    f9_direction_terms()
    if (MEAS / "member_fit.parquet").exists():
        f10_member_gmp()
    else:
        print("skip F10 (no member_fit yet)")
    if (MEAS / "transitions.json").exists():
        f11_transitions()
    if (RES / "preds" / "forecast108_119_notext_mq_16d_tcal.parquet").exists():
        f12_protest()
        f13_revolt_risk()
        f14_defector_capture()


if __name__ == "__main__":
    main()
