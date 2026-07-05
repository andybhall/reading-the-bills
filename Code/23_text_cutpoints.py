"""Can a bill's text predict where it will cut the chamber?

The paper's vote-level results show text improves member-vote forecasts;
this script tests the sharper measurement claim behind the dashboard:
that text predicts a ROLLCALL'S location --- its cutpoint on the
ideological scale and which party's side the yeas fall on --- better
than non-text alternatives.

Design:
1. Fit the 1D spatial model per congress-chamber (108th-119th, party-sign
   initialized) on all votes; realized cutpoints (-(b+c̄)/a, standardized
   by the chamber-congress member distribution) and directions (sign of
   a: + = yeas attract conservatives) are the measurement targets.
   Positions use all votes; the PREDICTORS below never see any held-out
   rollcall's parameters --- this evaluates measurement-target
   prediction, complementing (not replacing) the vote-level forecasts.
2. Temporal holdout within each congress-chamber: first 80% of rollcalls
   train the predictors, last 20% (by date) are the test set. Identified
   rollcalls only (|a| >= 0.35) for the cutpoint task.
3. Predictor sets: constant; metadata (question bucket, sponsor party,
   bill category, chamber); TF-IDF+SVD of pre-vote rollcall text; MiniLM
   sentence embeddings of the same text; embeddings + metadata.
   Ridge for cutpoints, logistic for direction; regularization chosen by
   cross-validation WITHIN the training window only.
4. Interpretability: (a) direction and location coefficients of a sparse
   TF-IDF logistic/ridge (the terms that make a bill cut left or right);
   (b) text-component ablation (question / description / bill summary
   separately, TF-IDF pipeline).

Outputs to Modified Data/results/measures/:
  cutpoint_rollcalls.parquet   targets + features + test predictions
  cutpoint_pred.json           metrics per predictor set + ablation
  cutpoint_terms.parquet       term coefficients (direction + location)

Run: python3 Code/23_text_cutpoints.py   (~12 min: 24 spatial fits)
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
embed_mod = importlib.import_module("08_embed_bills")
from models_forecast import question_bucket  # noqa: E402
from models_idealpoint import IdealPoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
OUT = MOD / "results" / "measures"

MIN_DISCRIM = 0.35
CONGRESSES = range(108, 120)


def fit_targets() -> pd.DataFrame:
    votes = pd.read_parquet(MOD / "votes.parquet").dropna(subset=["vote"])
    votes = votes[votes.congress.isin(CONGRESSES)]
    mem = pd.read_parquet(MOD / "members.parquet")
    frames = []
    for (cong, ch), df in votes.groupby(["congress", "chamber"]):
        m = mem[(mem.congress == cong) & (mem.chamber == ch)]
        init = pd.Series(np.where(m.party_code == 200.0, 0.5,
                                  np.where(m.party_code == 100.0, -0.5, 0.0)),
                         index=m.icpsr.to_numpy())
        ip = IdealPoint(k=1, min_epochs=10).fit(df, init_positions=init)
        x = ip.member_positions()["dim1"]
        c_bar = float(ip._c.mean())
        inv = {i: r for r, i in ip.rollcall_index.items()}
        rc = pd.DataFrame({"rc_key": [inv[i] for i in range(len(inv))],
                           "a": ip._a[:, 0], "b": ip._b[:, 0]})
        raw_cut = -(rc.b + c_bar) / rc.a
        rc["cut_std"] = (raw_cut - x.median()) / x.std()
        rc["direction"] = (rc.a > 0).astype(int)  # 1: yeas on the right
        rc["identified"] = rc.a.abs() >= MIN_DISCRIM
        rc["congress"], rc["chamber"] = cong, ch
        rc["rollnumber"] = rc.rc_key.str.split("_").str[2].astype(int)
        frames.append(rc)
        print(f"  {ch} {cong}: {len(rc)} rollcalls, "
              f"{int(rc.identified.sum())} identified")
    return pd.concat(frames, ignore_index=True)


def amendment_sponsors(rc: pd.DataFrame) -> pd.Series:
    """Amender's party per rollcall, via BILLSTATUS roll references.

    Clerk roll numbers RESET each session while Voteview's are
    congress-cumulative, so a session-2 reference "Roll no. 350" means
    Voteview rollnumber 350 + (session-1 rollcall count). The E5 build
    sidestepped this with a date-equality filter, which silently limited
    matches to session-1 votes; here the offset is applied explicitly and
    date equality is kept as verification. Ambiguous joins dropped."""
    am = pd.read_parquet(MOD / "amendments.parquet")
    am = am[(am.roll_no >= 0) & (am.action_date != "")
            & am.sponsor_party.isin(["D", "R"])].copy()
    am["chamber"] = am.amdt_type.map({"HAMDT": "House", "SAMDT": "Senate"})
    rcm = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "date"]].copy()
    rcm["year"] = pd.to_datetime(rcm.date).dt.year
    first_year = 1787 + 2 * rcm.congress
    s1 = (rcm[rcm.year == first_year]
          .groupby(["congress", "chamber"])["rollnumber"].max().rename("offset"))
    am["year"] = pd.to_datetime(am.action_date).dt.year
    am = am.merge(s1.reset_index(), on=["congress", "chamber"], how="left")
    am["offset"] = am["offset"].fillna(0)
    session2 = am.year > (1787 + 2 * am.congress)
    am["rollnumber"] = am.roll_no + np.where(session2, am.offset, 0)

    key = ["congress", "chamber", "rollnumber"]
    j = rc[key + ["date"]].copy()
    j["vote_date"] = j.date.astype(str).str[:10]
    j = j.merge(am[key + ["action_date", "sponsor_party"]], on=key, how="left")
    j = j[j.action_date.isna() | (j.action_date == j.vote_date)]
    j = j.drop_duplicates(key, keep=False)
    return rc.merge(j[key + ["sponsor_party"]].rename(
        columns={"sponsor_party": "amdt_sponsor"}), on=key, how="left")["amdt_sponsor"]


def attach_features(rc: pd.DataFrame) -> pd.DataFrame:
    links = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "vote_question",
         "bill_category", "bill_type", "bill_no"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "title", "summaries",
         "policy_area", "sponsor_party"]]
    rcm = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "date", "vote_desc"]]
    rc = (rc.merge(links, on=["congress", "chamber", "rollnumber"], how="left")
            .merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
            .merge(rcm, on=["congress", "chamber", "rollnumber"], how="left"))
    rc["qbucket"] = question_bucket(rc["vote_question"])
    rc["amdt_sponsor"] = amendment_sponsors(rc).to_numpy()
    # the acting legislator whose party orients the vote: the AMENDER on
    # matched amendment votes, the bill sponsor otherwise
    rc["actor_party"] = np.where(
        (rc.qbucket == "amendment") & rc.amdt_sponsor.notna(),
        rc.amdt_sponsor, rc.sponsor_party.fillna(""))
    rc["amdt_matched"] = ((rc.qbucket == "amendment")
                          & rc.amdt_sponsor.notna()).astype(float)
    q, d, s = [], [], []
    for row in rc.itertuples():
        q.append(row.vote_question if isinstance(row.vote_question, str) else "")
        d.append(row.vote_desc if isinstance(row.vote_desc, str) else "")
        if isinstance(row.summaries, str):
            s.append(embed_mod.pre_vote_text(row.summaries, row.title,
                                             str(row.date))[0])
        else:
            s.append(row.title if isinstance(row.title, str) else "")
    rc["t_question"], rc["t_desc"], rc["t_summary"] = q, d, s
    rc["text"] = (rc.t_question + ". " + rc.t_desc + ". " + rc.t_summary).str[:1500]
    emb = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet")
    ecols = [c for c in emb.columns if c.startswith("e")]
    rc = rc.merge(emb[["congress", "chamber", "rollnumber"] + ecols],
                  on=["congress", "chamber", "rollnumber"], how="left")
    rc[ecols] = rc[ecols].fillna(0.0)
    # temporal holdout within congress-chamber
    rc = rc.sort_values(["congress", "chamber", "date", "rollnumber"])
    rank = rc.groupby(["congress", "chamber"]).cumcount()
    size = rc.groupby(["congress", "chamber"])["rollnumber"].transform("size")
    rc["test"] = (rank + 0.5) / size >= 0.80
    return rc, ecols


def meta_matrix(rc: pd.DataFrame, fit_levels=None):
    if fit_levels is None:
        fit_levels = {"q": sorted(rc.qbucket.unique()),
                      "s": ["D", "R"], "c": sorted(rc.bill_category.fillna("none").unique())}
    Q = np.stack([(rc.qbucket == q).to_numpy(float) for q in fit_levels["q"]], 1)
    # two distinct actors, both informative in different places: the
    # bill's sponsor (location cue on all vote types) and the amender
    # (direction cue on matched amendment votes; zero elsewhere)
    S = np.stack([(rc.sponsor_party.fillna("") == s).to_numpy(float)
                  for s in fit_levels["s"]], 1)
    A = np.stack([(rc.amdt_sponsor.fillna("") == s).to_numpy(float)
                  for s in fit_levels["s"]], 1)
    C = np.stack([(rc.bill_category.fillna("none") == c).to_numpy(float)
                  for c in fit_levels["c"]], 1)
    # sponsor x question interactions: the sponsor's side predicts a
    # passage vote's direction (82%) but is a coin flip on amendments
    # (we observe the BILL's sponsor, not the amender's party) — main
    # effects alone average that away and understate metadata
    SQ = np.hstack([S[:, [i]] * Q for i in range(S.shape[1])])
    blocks = [Q, S, A, C, SQ, rc.amdt_matched.to_numpy(float)[:, None],
              (rc.chamber == "Senate").to_numpy(float)[:, None]]
    return np.hstack(blocks), fit_levels


def evaluate_sets(rc, ecols):
    tr, te = rc[~rc.test], rc[rc.test]
    tr_id, te_id = tr[tr.identified], te[te.identified]
    M_tr, levels = meta_matrix(tr)
    M_te, _ = meta_matrix(te, levels)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=5)
    Xt_tr = tfidf.fit_transform(tr.text)
    svd = TruncatedSVD(n_components=128, random_state=42)
    Zt_tr = svd.fit_transform(Xt_tr)
    Zt_te = svd.transform(tfidf.transform(te.text))
    E_tr, E_te = tr[ecols].to_numpy(), te[ecols].to_numpy()

    sets = {
        "constant": (None, None),
        "metadata": (M_tr, M_te),
        "tfidf_svd": (Zt_tr, Zt_te),
        "embeddings": (E_tr, E_te),
        "embeddings_meta": (np.hstack([E_tr, M_tr]), np.hstack([E_te, M_te])),
    }
    alphas = np.logspace(-1, 3, 9)
    results, preds_store = {}, {}
    id_tr, id_te = (~rc.test) & rc.identified, rc.test & rc.identified
    for name, (A_tr, A_te) in sets.items():
        if A_tr is None:
            cut_pred = np.full(len(te_id), tr_id.cut_std.mean())
            dir_pred = np.full(len(te), tr.direction.mean())
        else:
            ridge = RidgeCV(alphas=alphas).fit(
                A_tr[tr.identified.to_numpy()], tr_id.cut_std)
            cut_pred = ridge.predict(A_te[te.identified.to_numpy()])
            logit = LogisticRegression(max_iter=2000, C=1.0).fit(
                A_tr, tr.direction)
            dir_pred = logit.predict_proba(A_te)[:, 1]
        mae = float(np.abs(cut_pred - te_id.cut_std).mean())
        r = float(np.corrcoef(cut_pred, te_id.cut_std)[0, 1]) if A_tr is not None else 0.0
        acc = float(((dir_pred > 0.5) == te.direction).mean())
        # direction is weakly defined on unidentified (near-lopsided)
        # rollcalls; the identified-only accuracy is the meaningful one
        id_mask = te.identified.to_numpy()
        acc_id = float(((dir_pred[id_mask] > 0.5) == te.direction[id_mask]).mean())
        results[name] = {"cut_mae": round(mae, 3), "cut_r": round(r, 3),
                         "dir_acc": round(acc, 3), "dir_acc_identified": round(acc_id, 3),
                         "n_test_cut": int(len(te_id)), "n_test_dir": int(len(te))}
        preds_store[name] = cut_pred
        print(f"  {name:18s} cutpoint MAE {mae:.3f}  r {r:.3f}  "
              f"dir acc {acc:.3f}  (identified {acc_id:.3f})")

    # text-component ablation (TF-IDF pipeline, cutpoint task)
    ablation = {}
    for comp in ("t_question", "t_desc", "t_summary"):
        tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=5)
        Z_tr = TruncatedSVD(n_components=128, random_state=42).fit_transform(
            tf.fit_transform(tr[comp]))
        # note: SVD refit per component; transform test with same objects
        sv = TruncatedSVD(n_components=128, random_state=42)
        X0 = tf.fit_transform(tr[comp])
        Z_tr = sv.fit_transform(X0)
        Z_te = sv.transform(tf.transform(te[comp]))
        ridge = RidgeCV(alphas=alphas).fit(Z_tr[tr.identified.to_numpy()],
                                           tr_id.cut_std)
        pred = ridge.predict(Z_te[te.identified.to_numpy()])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Z_tr, tr.direction)
        dpr = lr.predict_proba(Z_te)[:, 1]
        id_mask = te.identified.to_numpy()
        acc_id = float(((dpr[id_mask] > 0.5) == te.direction[id_mask]).mean())
        ablation[comp] = {"cut_mae": round(float(np.abs(pred - te_id.cut_std).mean()), 3),
                          "cut_r": round(float(np.corrcoef(pred, te_id.cut_std)[0, 1]), 3),
                          "dir_acc_identified": round(acc_id, 3)}
        print(f"  ablation {comp:12s} MAE {ablation[comp]['cut_mae']:.3f} "
              f"r {ablation[comp]['cut_r']:.3f} dir {acc_id:.3f}")

    # interpretable term coefficients. Three restrictions, each earned by
    # a face-validity failure of the naive version: (1) SUMMARY text only
    # — question/desc terms ("recommit", "suspend") encode procedure and
    # era, not content; (2) alphabetic tokens only — year tokens ("2009")
    # proxy for which party held the majority, i.e. agenda composition,
    # not language; (3) identified rollcalls only — direction is undefined
    # on lopsided votes. What remains is the content map: which bill
    # LANGUAGE predicts a conservative-yea vs liberal-yea coalition.
    # passage votes only: rules resolutions carry procedure-speak summaries
    # ("provides for consideration of...") that would dominate the content
    # map; the claim under test is about substantive bill language
    tr_sub = tr[tr.identified & (tr.qbucket == "passage")
                & (tr.t_summary.str.len() > 100)]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stops = list(ENGLISH_STOP_WORDS | {"nbsp", "amp", "quot", "html"})
    tf2 = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=15, stop_words=stops,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b")
    X2_tr = tf2.fit_transform(tr_sub.t_summary)
    dir_lr = LogisticRegression(max_iter=2000, C=0.5).fit(X2_tr, tr_sub.direction)
    ridge2 = RidgeCV(alphas=alphas).fit(X2_tr, tr_sub.cut_std)
    terms = pd.DataFrame({"term": tf2.get_feature_names_out(),
                          "dir_coef": dir_lr.coef_[0],
                          "cut_coef": np.asarray(ridge2.coef_).ravel()})
    terms.to_parquet(OUT / "cutpoint_terms.parquet", index=False)

    keep = ["congress", "chamber", "rollnumber", "date", "qbucket",
            "policy_area", "cut_std", "direction", "identified", "test"]
    store = rc[keep].copy()
    for name, pr in preds_store.items():
        col = pd.Series(np.nan, index=rc.index)
        col.loc[id_te[id_te].index] = pr
        store[f"pred_{name}"] = col
    store.to_parquet(OUT / "cutpoint_rollcalls.parquet", index=False)
    (OUT / "cutpoint_pred.json").write_text(json.dumps(
        {"sets": results, "component_ablation": ablation}, indent=2))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cutpoint_targets_cache.parquet"
    if cache.exists():
        print("loading cached spatial-fit targets")
        rc = pd.read_parquet(cache)
    else:
        print("fitting 1D spatial models per congress-chamber...")
        rc = fit_targets()
        rc.to_parquet(cache, index=False)
    rc, ecols = attach_features(rc)
    print(f"targets: {len(rc)} rollcalls, {int(rc.identified.sum())} identified, "
          f"{int((rc.test & rc.identified).sum())} identified test")
    evaluate_sets(rc, ecols)


if __name__ == "__main__":
    main()
