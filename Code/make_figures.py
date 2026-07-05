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
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    specs = [("gb_spatial_tfidf", "2011-style spatial + text", "#b2182b", "--"),
             ("emb2_mlp_mq_16d_tcal", "Modern tower (calibrated)", "#4393c3", "-"),
             ("blend3_mlp_tfidf_emb3_tcal", "Final blend", "#111111", "-")]
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
    ax.set_title("Calibration on future rollcalls")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    ax1.plot([-4, 4], [-4, 4], color="#bbbbbb", lw=0.8, zorder=0)
    ax1.scatter(d.pred_embeddings_meta, d.cut_std, s=6, alpha=0.3,
                color="#333333", linewidths=0)
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
    lab = dd[dd.short.notna()].nlargest(4, "cut_std")  # right-cutting acts
    lab = pd.concat([lab, dd[dd.short.notna()].nsmallest(2, "cut_std")])
    for i, row in enumerate(lab.itertuples()):
        ax1.annotate(row.short, (row.pred_embeddings_meta, row.cut_std),
                     fontsize=6, xytext=(4, 3 if i % 2 else -8),
                     textcoords="offset points", color="#111111")
    ax1.set_xlabel("Predicted cutpoint (text + metadata)")
    ax1.set_ylabel("Realized cutpoint")
    ax1.set_title(f"Held-out future rollcalls (r = {r:.2f})")

    j = json.loads((MEAS / "cutpoint_pred.json").read_text())
    names = [("metadata", "Metadata only"), ("tfidf_svd", "Text: TF-IDF"),
             ("embeddings", "Text: embeddings"),
             ("embeddings_meta", "Text + metadata")]
    y = np.arange(len(names))
    mae = [j["sets"][k]["cut_mae"] for k, _ in names]
    acc = [100 * j["sets"][k]["dir_acc_identified"] for k, _ in names]
    ax2b = ax2.twiny()
    ax2.barh(y + 0.18, mae, height=0.34, color="#4393c3", label="Cutpoint MAE")
    ax2b.barh(y - 0.18, acc, height=0.34, color="#b2182b", label="Direction acc. (%)")
    ax2.axvline(j["sets"]["constant"]["cut_mae"], color="#4393c3", lw=0.9, ls=":")
    ax2b.axvline(100 * j["sets"]["constant"]["dir_acc_identified"],
                 color="#b2182b", lw=0.9, ls=":")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n for _, n in names], fontsize=8.5)
    ax2.set_xlabel("Cutpoint MAE (member-SD units; dotted: no-feature baseline)",
                   fontsize=8, color="#4393c3")
    ax2b.set_xlabel("Direction accuracy, % (dotted: baseline)", fontsize=8,
                    color="#b2182b")
    ax2.set_xlim(0, 1.0)
    ax2b.set_xlim(50, 70)
    fig.suptitle("Predicting where a bill cuts the chamber, before the vote",
                 fontweight="bold", y=1.04)
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
    ax.set_title("The language of left- and right-recruiting bills\n"
                 "(final-passage votes; blue: liberal-yea, red: conservative-yea)")
    save(fig, "direction_terms.pdf")


# ---------------------------------------------------------------- F10
def f10_member_gmp():
    """Member-by-member fit on the NOMINATE literature's own statistic:
    each member's geometric mean probability under our champion versus
    under the classical model (frozen DW-NOMINATE positions, estimated
    rollcall parameters), computed on identical votes."""
    mf = pd.read_parquet(MEAS / "member_fit.parquet")
    mf = mf[mf.n >= 100]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0.2, 1], [0.2, 1], color="#bbbbbb", lw=0.8, zorder=0)
    ax.scatter(mf.gmp_nom, mf.gmp_ours, s=8, alpha=0.4,
               c=party_color(mf.party_code), linewidths=0)
    lab = mf.assign(g=mf.gmp_ours - mf.gmp_nom).nlargest(8, "g")
    for i, r in enumerate(lab.itertuples()):
        ax.annotate(shortname(r.bioname, r.state_abbrev),
                    (r.gmp_nom, r.gmp_ours), fontsize=6.5,
                    xytext=(5, -2 if i % 2 else 4),
                    textcoords="offset points")
    share = (mf.gmp_ours > mf.gmp_nom).mean()
    ax.set_xlabel("Held-out GMP, DW-NOMINATE-based model")
    ax.set_ylabel("Held-out GMP, this paper's spatial model")
    ax.set_title("Member-by-member fit on identical held-out votes\n"
                 f"(higher for {share:.0%} of members; largest gains labeled)")
    save(fig, "member_gmp.pdf")


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
    ax1.set_title("Cross-congress transfer, all transitions")
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
    ax2.set_title("The damage is procedural, and flips cause it")
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "transitions.pdf")


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


if __name__ == "__main__":
    main()
