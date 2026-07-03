"""Generate Paper A's figures from results files. No hand-entered data.

Outputs PDF figures to Draft/figures/. Inputs: the measurement layer
(22_extract_measures.py), the validated member instruments
(03/09/10 outputs), saved test predictions, and the leaderboard.

Run: python3 Code/make_figures.py
"""

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
    sig = pd.read_parquet(RES / "member_signals.parquet")
    recent = sig[sig.congress >= 115]
    g = (recent.groupby(["icpsr", "bioname", "party_code", "state_abbrev"])
         .agg(resid=("loyalty_residual", "mean"), n=("n_unity", "sum"))
         .reset_index())
    g = g[g.n >= 300]
    rows = []
    for pc in (100.0, 200.0):
        p = g[g.party_code == pc].sort_values("resid")
        rows.append(p.head(9))   # mavericks
        rows.append(p.tail(4))   # loyalists
    d = pd.concat(rows).sort_values("resid")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    y = np.arange(len(d))
    ax.hlines(y, 0, d.resid, color=party_color(d.party_code), lw=1.4, alpha=0.75)
    ax.scatter(d.resid, y, c=party_color(d.party_code), s=26, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([shortname(b, s) for b, s in zip(d.bioname, d.state_abbrev)],
                       fontsize=8)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Party-unity loyalty beyond ideology "
                  "(share of unity votes, 115th–119th Congresses)")
    ax.set_title("Mavericks and soldiers: loyalty residuals")
    save(fig, "loyalty_caterpillar.pdf")


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
    rc = rc[rc.identified & rc.cutpoint.between(-3.2, 3.2)]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})

    for pc, color, lab in ((100.0, DEM, "Democrats"), (200.0, REP, "Republicans")):
        x = mem.loc[mem.party_code == pc, "x"]
        ax1.hist(x, bins=45, range=(-3.2, 3.2), color=color, alpha=0.55,
                 label=lab, density=True)
    ax1.hist(rc.cutpoint, bins=60, range=(-3.2, 3.2), histtype="step",
             color="#333333", lw=1.1, density=True, label="Rollcall cutpoints")
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


if __name__ == "__main__":
    main()
