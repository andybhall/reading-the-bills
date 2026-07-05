"""Two-tower forecast models: member embedding x bill-content embedding.

    logit(m, r) = a_r . x_m  +  b_r  +  c_m  +  gamma . z_mr

where a_r = W_a phi(r) and b_r = w_b phi(r) are AMORTIZED from rollcall
features phi(r) — so unlike the Regime A ideal-point model, predictions
exist for never-before-seen rollcalls. z_mr = member-bill interactions
(same party as sponsor, opposite party).

phi(r) =
  - TF-IDF (1-2 grams, fit on TRAIN rollcalls only) of bill title +
    policy area + legislative subjects, SVD-compressed to 128 dims
  - question-bucket one-hots, bill-category one-hots, sponsor-party flags

Two variants isolate the text contribution:
  - meta_tower_8d: phi WITHOUT the text block (question/category/sponsor only)
  - text_tower_8d: full phi

Leakage caveat (documented in Notes/decisions.md): titles, policy areas, and
subject terms are mostly assigned at introduction but can be revised later;
a strict pre-vote-only variant using dated summary versions is future work.
"""

import hashlib

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from harness import EPS
from models_forecast import (MajorityQuestionRate, MemberQuestionRate,
                             MemberQuestionRateRecency, _is_majority,
                             _majority_party, question_bucket)
from models_idealpoint import SEED, ES_FRACTION, _cell_hash_unit, _device, _rollcall_key

MOD = Path(__file__).resolve().parent.parent / "Modified Data"
RC_KEY = ["congress", "chamber", "rollnumber"]


def _load_bill_text() -> pd.DataFrame:
    bills = pd.read_parquet(MOD / "bills.parquet")
    import json as _json
    subj = bills["subjects"].apply(lambda s: " ".join(_json.loads(s)))
    bills["text"] = (bills["title"].fillna("") + ". " + bills["policy_area"].fillna("")
                     + ". " + subj)
    return bills[["congress", "bill_type", "bill_no", "text", "sponsor_party"]]


class TextTower:
    # Class-level defaults: artifacts pickled before an option existed have no
    # such key in their instance __dict__, so attribute lookup falls back here.
    # Required for the frozen prospective model (frozen 2026-06-12T03:25Z,
    # before these options were added) — never modify that pickle.
    use_cosponsors = False
    # use_majority: member-in-majority indicator + logit of the pooled
    # (majority x chamber x question) rate as z features. Majority status is
    # congress-agnostic, so this structure survives an unseen congress —
    # the M-D attempt at the congress-out transfer failure.
    use_majority = False
    # use_billctx: member/party behavior on strictly-prior rollcalls of the
    # SAME bill (amendment series reveal the coalition vote by vote).
    # Prior outcomes are public the moment they occur; lookups draw only on
    # train-window rollcalls, so eval labels are structurally unreachable.
    use_billctx = False
    # cal_mode "bucket": separate (temperature, bias) per question bucket on
    # the internal dev slice (buckets with thin dev support fall back to the
    # global pair). "global" is the registered-model / frozen-pickle default.
    cal_mode = "global"
    # mq_recency: the member x question offset uses exponentially decayed
    # vote weights (tau=365d) instead of flat averages
    mq_recency = False

    def __init__(self, k: int = 8, use_text: bool = True, svd_dim: int = 128,
                 lr: float = 0.02, weight_decay: float = 1e-4,
                 batch_size: int = 1 << 17, max_epochs: int = 60, patience: int = 5,
                 min_epochs: int = 10, bills_df: pd.DataFrame | None = None,
                 mq_offset: bool = False, calibrate: bool = False,
                 use_emb: bool = False, name: str | None = None,
                 emb_file: str = "rollcall_text_embeddings_v2.parquet",
                 mlp_head: bool = False, es_mode: str = "random"):
        # es_mode "temporal": the internal dev slice is the LAST 5% of train
        # rollcalls by date (whole rollcalls), so early stopping and
        # calibration see future-like data — random cells are in-sample for
        # rollcall parameters and miscalibrate forecast models (diagnosed
        # 2026-06-11 on the MLP head: 82% acc but 0.64 log loss).
        self.es_mode = es_mode
        # meta_features=False: phi is the text embedding ONLY (no question/
        # category/sponsor blocks) — used by issue probes so the learned
        # text->discrimination map cannot lean on agenda/sponsorship cues
        self.meta_features = True
        # use_cosponsors: as-of-vote-date cosponsor counts by party (phi)
        # plus a member-party x cosponsor-balance alignment term (z).
        # Dated counts only — cosponsors accruing after the vote are unseen.
        self.use_cosponsors = False
        # use_emb: sentence-transformer embeddings of strictly pre-vote
        # rollcall text (08_embed_bills.py) as the text block instead of
        # TF-IDF. mlp_head: 2-layer projection from phi to rollcall params
        # instead of linear (lets text dims interact).
        self.use_emb, self.emb_file, self.mlp_head = use_emb, emb_file, mlp_head
        self.k, self.use_text, self.svd_dim = k, use_text, svd_dim
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.min_epochs = min_epochs  # bilinear terms start slow; don't stop early
        self._bills_override = bills_df  # for tests; default loads bills.parquet
        # mq_offset: include logit of the member x question-bucket empirical
        # rate as a learnable-weight offset (gives the tower the count
        # baseline's member-level structure as a floor to improve on).
        # calibrate: post-hoc temperature + bias fit on the internal
        # early-stop slice (in-period cells; transfer to future rollcalls is
        # imperfect but corrects global overconfidence).
        self.mq_offset, self.calibrate = mq_offset, calibrate
        self.name = name or (f"text_tower_{k}d" if use_text else f"meta_tower_{k}d")

    # ---- rollcall feature tower ------------------------------------------
    def _rollcall_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        extra = ["date"] if "date" in df.columns else []
        rc = df.drop_duplicates(RC_KEY)[RC_KEY + ["vote_question", "bill_category",
                                                  "bill_type", "bill_no"] + extra].copy()
        rc["rc_key"] = _rollcall_key(rc)
        rc = rc.merge(self._bills, on=["congress", "bill_type", "bill_no"], how="left")
        rc["text"] = rc["text"].fillna("")
        rc["sponsor_party"] = rc["sponsor_party"].fillna("")
        rc["qbucket"] = question_bucket(rc["vote_question"])
        rc["bill_category"] = rc["bill_category"].fillna("none")
        return rc

    def _phi(self, rc: pd.DataFrame, fit: bool) -> np.ndarray:
        blocks = []
        if self.use_emb:
            if fit:
                emb = pd.read_parquet(MOD / self.emb_file)
                emb["rc_key"] = _rollcall_key(emb)
                cols = [c for c in emb.columns if c.startswith("e")]
                self._emb_lookup = emb.set_index("rc_key")[cols]
            E = self._emb_lookup.reindex(rc["rc_key"]).fillna(0.0).to_numpy()
            blocks.append(E.astype(np.float32) * 5.0)  # unit-norm rows; rescale
        if self.use_text:
            if fit:
                self._tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                              sublinear_tf=True)
                X = self._tfidf.fit_transform(rc["text"])
                self._svd = TruncatedSVD(n_components=self.svd_dim, random_state=SEED)
                Z = self._svd.fit_transform(X)
                self._z_scale = Z.std(axis=0) + 1e-8
            else:
                Z = self._svd.transform(self._tfidf.transform(rc["text"]))
            blocks.append(Z / self._z_scale)
        if self.use_cosponsors:
            if fit:
                cosp = pd.read_parquet(MOD / "cosponsors.parquet")
                cosp["sponsorship_date"] = pd.to_datetime(cosp["sponsorship_date"])
                self._cosp_dates = {
                    (party, congress, btype, bno): np.sort(g["sponsorship_date"].to_numpy())
                    for (party, congress, btype, bno), g in
                    cosp[cosp.party.isin(["D", "R"])]
                    .groupby(["party", "congress", "bill_type", "bill_no"])}
            counts = {}
            for party in ("D", "R"):
                n = np.zeros(len(rc), dtype=np.float32)
                for i, row in enumerate(rc.itertuples()):
                    dates = self._cosp_dates.get(
                        (party, row.congress, row.bill_type,
                         int(row.bill_no) if pd.notna(row.bill_no) else -1))
                    if dates is not None and hasattr(row, "date"):
                        n[i] = np.searchsorted(dates, np.datetime64(row.date))
                counts[party] = n
            balance = ((counts["R"] - counts["D"])
                       / (counts["R"] + counts["D"] + 1.0)).astype(np.float32)
            self._last_balance = pd.Series(balance, index=rc["rc_key"].to_numpy())
            blocks.append(np.stack([np.log1p(counts["D"]), np.log1p(counts["R"]),
                                    balance], 1))
        if self.meta_features:
            if fit:
                self._q_levels = sorted(rc["qbucket"].unique())
                self._c_levels = sorted(rc["bill_category"].unique())
            blocks.append(np.stack([(rc["qbucket"] == q).to_numpy(float) for q in self._q_levels], 1))
            blocks.append(np.stack([(rc["bill_category"] == c).to_numpy(float) for c in self._c_levels], 1))
            blocks.append(np.stack([(rc["sponsor_party"] == p).to_numpy(float) for p in ("D", "R")], 1))
        return np.hstack(blocks).astype(np.float32)

    def _alignment(self, df: pd.DataFrame) -> np.ndarray:
        """Member-party sign x as-of-date cosponsor balance of the bill."""
        sign = df["party_code"].map({200.0: 1.0, 100.0: -1.0}).fillna(0.0).to_numpy()
        bal = self._last_balance.reindex(_rollcall_key(df)).fillna(0.0).to_numpy()
        return (sign * bal).astype(np.float32)

    def _offset(self, df: pd.DataFrame) -> np.ndarray:
        """Logit of the member x question-bucket empirical training rate."""
        p = np.clip(self._mq.predict_proba(df.drop(columns=["vote"], errors="ignore")),
                    EPS, 1 - EPS)
        return np.log(p / (1 - p)).astype(np.float32)

    @staticmethod
    def temporal_es_mask(train_df: pd.DataFrame) -> np.ndarray:
        """Rows on the last 5% of train rollcalls by date per congress-
        chamber — the future-like internal slice used for early stopping,
        calibration, and residual stacking. Strictly inside train."""
        rc_dates = train_df.drop_duplicates(RC_KEY)[RC_KEY + ["date"]].copy()
        rc_dates["q"] = (rc_dates.sort_values("date")
                         .groupby(["congress", "chamber"]).cumcount())
        sizes = rc_dates.groupby(["congress", "chamber"])["rollnumber"].transform("size")
        late = rc_dates[(rc_dates["q"] + 0.5) / sizes >= 0.95]
        late_keys = set(_rollcall_key(late))
        return _rollcall_key(train_df).isin(late_keys).to_numpy()

    # ---- within-bill context ----------------------------------------------
    BILL_KEY = ["congress", "chamber", "bill_type", "bill_no"]

    @staticmethod
    def _ctx_ord(df: pd.DataFrame) -> np.ndarray:
        """Strict chronological order within a chamber: days since epoch
        dominate, rollnumber breaks same-day ties (rollnumbers increase
        through a congress). Used with side='left' searchsorted so a
        rollcall never sees itself."""
        days = pd.to_datetime(df["date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
        return days * 100000 + df["rollnumber"].to_numpy()

    def _fit_bill_context(self, train_df: pd.DataFrame) -> None:
        v = train_df[train_df["bill_no"].notna()]
        rc = v.drop_duplicates(RC_KEY)[RC_KEY + self.BILL_KEY[2:] + ["date"]].copy()
        rc["rc_key"] = _rollcall_key(rc)
        rc["ord"] = self._ctx_ord(rc)
        rates = (v.groupby(RC_KEY + ["party_code"])["vote"].mean()
                 .unstack("party_code").reindex(columns=[100.0, 200.0]))
        rates.columns = ["d_rate", "r_rate"]
        rc = rc.merge(rates.reset_index(), on=RC_KEY, how="left")
        rc = rc.sort_values("ord")
        self._ctx_bills = {
            key: (g["ord"].to_numpy(), g["rc_key"].to_numpy(),
                  g["d_rate"].to_numpy(), g["r_rate"].to_numpy())
            for key, g in rc.groupby(self.BILL_KEY, sort=False)}
        self._ctx_votes = v.assign(rc_key=_rollcall_key(v))[["rc_key", "icpsr", "vote"]]

    def _bill_context(self, df: pd.DataFrame) -> np.ndarray:
        """Five columns per vote row: has-prior indicator, member's own
        vote on the latest strictly-prior same-bill rollcall (signed, 0 if
        absent), the member's party's yea share on that rollcall (centered
        at 0), log1p(days since that rollcall), and a same-day flag. All
        from train-window history only; staleness columns let a model
        discount old priors (amendment series are same-day in training,
        but a forecast rollcall's visible prior can be weeks old)."""
        n = len(df)
        has_prior = np.zeros(n, np.float32)
        prior_rc = np.full(n, "", dtype=object)
        prior_rate = np.zeros(n, np.float32)
        log_days = np.zeros(n, np.float32)
        same_day = np.zeros(n, np.float32)
        mask = df["bill_no"].notna().to_numpy()
        q = (df.loc[mask, self.BILL_KEY + ["icpsr", "party_code", "date", "rollnumber"]]
             .copy().reset_index(drop=True))
        q["ord"] = self._ctx_ord(q)
        rows = np.flatnonzero(mask)  # rows[i] = position in df of q's row i
        party = q["party_code"].to_numpy()
        for key, g in q.groupby(self.BILL_KEY, sort=False):
            tbl = self._ctx_bills.get(key)
            if tbl is None:
                continue
            ords, keys, d_rate, r_rate = tbl
            pos = np.searchsorted(ords, g["ord"].to_numpy(), side="left") - 1
            ok = pos >= 0
            gi = g.index.to_numpy()  # positions in q (positional after reset)
            take = rows[gi[ok]]
            has_prior[take] = 1.0
            prior_rc[take] = keys[pos[ok]]
            pg = party[gi[ok]]
            rate = np.where(pg == 100.0, d_rate[pos[ok]],
                            np.where(pg == 200.0, r_rate[pos[ok]], np.nan))
            prior_rate[take] = np.nan_to_num(rate - 0.5, nan=0.0)
            days = (g["ord"].to_numpy()[ok] // 100000) - (ords[pos[ok]] // 100000)
            log_days[take] = np.log1p(days).astype(np.float32)
            same_day[take] = (days == 0).astype(np.float32)
        own = pd.DataFrame({"rc_key": prior_rc, "icpsr": df["icpsr"].to_numpy(),
                            "n": np.arange(n)})
        own = own.merge(self._ctx_votes, on=["rc_key", "icpsr"], how="left")
        own_signed = np.nan_to_num(2.0 * own.sort_values("n")["vote"].to_numpy() - 1.0,
                                   nan=0.0).astype(np.float32)
        return np.stack([has_prior, own_signed, prior_rate, log_days, same_day], 1)

    def _majority_z(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """In-majority indicator + logit of the pooled majority x chamber x
        question rate. Keyed on majority status (from member composition),
        never on congress, so both columns are meaningful on an unseen
        congress — unlike the congress-keyed mq offset, which collapses."""
        if fit:
            self._maj = MajorityQuestionRate().fit(df)
        in_maj = _is_majority(df, _majority_party(df)).astype(np.float32)
        p = np.clip(self._maj.predict_proba(df.drop(columns=["vote"], errors="ignore")),
                    EPS, 1 - EPS)
        return np.stack([in_maj, np.log(p / (1 - p)).astype(np.float32)], 1)

    # ---- member-bill interactions ----------------------------------------
    @staticmethod
    def _z(df: pd.DataFrame, sponsor_party: pd.Series) -> np.ndarray:
        member_p = df["party_code"].map({100.0: "D", 200.0: "R"}).fillna("")
        sp = sponsor_party.to_numpy()
        same = ((member_p.to_numpy() == sp) & (sp != "")).astype(np.float32)
        opp = ((member_p.to_numpy() != sp) & (sp != "")
               & (member_p.to_numpy() != "")).astype(np.float32)
        return np.stack([same, opp], 1)

    def fit(self, train_df: pd.DataFrame):
        torch.manual_seed(SEED)
        dev = _device()
        self._bills = self._bills_override if self._bills_override is not None \
            else _load_bill_text()

        rc = self._rollcall_frame(train_df)
        phi = self._phi(rc, fit=True)
        rc_index = {key: i for i, key in enumerate(rc["rc_key"])}

        self.member_index = {m: i for i, m in enumerate(sorted(train_df["icpsr"].unique()))}
        m_idx = train_df["icpsr"].map(self.member_index).to_numpy()
        r_idx = _rollcall_key(train_df).map(rc_index).to_numpy()
        sponsor_by_rc = rc.set_index("rc_key")["sponsor_party"]
        z = self._z(train_df, sponsor_by_rc.reindex(_rollcall_key(train_df)).fillna(""))
        if self.use_cosponsors:
            z = np.concatenate([z, self._alignment(train_df)[:, None]], 1)
        if self.use_billctx:
            self._fit_bill_context(train_df)
            z = np.concatenate([z, self._bill_context(train_df)], 1)
        if self.use_majority:  # before mq offset: gamma init trusts LAST column
            z = np.concatenate([z, self._majority_z(train_df, fit=True)], 1)
        if self.mq_offset:
            self._mq = (MemberQuestionRateRecency() if self.mq_recency
                        else MemberQuestionRate()).fit(train_df)
            off = self._offset(train_df)
            z = np.concatenate([z, off[:, None]], 1)
        y = train_df["vote"].to_numpy()
        self.global_rate = float(y.mean())

        if self.es_mode == "temporal" and "date" in train_df.columns:
            es = self.temporal_es_mask(train_df)
        else:
            es = _cell_hash_unit(train_df, f"earlystop{SEED}") < ES_FRACTION
        d = phi.shape[1]
        x = torch.nn.Embedding(len(self.member_index), self.k).to(dev)
        c = torch.nn.Embedding(len(self.member_index), 1).to(dev)
        torch.nn.init.normal_(x.weight, std=0.1)
        torch.nn.init.zeros_(c.weight)
        if self.mlp_head:
            W_a = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                      torch.nn.Linear(128, self.k)).to(dev)
            w_b = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(),
                                      torch.nn.Linear(128, 1)).to(dev)
        else:
            W_a = torch.nn.Linear(d, self.k).to(dev)
            w_b = torch.nn.Linear(d, 1).to(dev)
        gamma = torch.nn.Linear(z.shape[1], 1, bias=False).to(dev)
        if self.mq_offset:  # start at "trust the empirical rate fully"
            with torch.no_grad():
                gamma.weight[0, -1] = 1.0
        params = [x.weight, c.weight, *W_a.parameters(), *w_b.parameters(),
                  *gamma.parameters()]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        phi_t = torch.as_tensor(phi, device=dev)

        def tensors(mask):
            return (torch.as_tensor(m_idx[mask], dtype=torch.long, device=dev),
                    torch.as_tensor(r_idx[mask], dtype=torch.long, device=dev),
                    torch.as_tensor(z[mask], device=dev),
                    torch.as_tensor(y[mask], dtype=torch.float32, device=dev))

        mt, rt, zt, yt = tensors(~es)
        mv, rv, zv, yv = tensors(es)

        def logits(mi, ri, zi):
            f = phi_t[ri]
            return ((x(mi) * W_a(f)).sum(-1) + w_b(f).squeeze(-1)
                    + c(mi).squeeze(-1) + gamma(zi).squeeze(-1))

        best, best_state, bad = np.inf, None, 0
        n = len(yt)
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                i = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss_fn(logits(mt[i], rt[i], zt[i]), yt[i]).backward()
                opt.step()
            with torch.no_grad():
                es_loss = float(loss_fn(logits(mv, rv, zv), yv))
            if es_loss < best - 1e-5:
                best, bad = es_loss, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        with torch.no_grad():
            for p, saved in zip(params, best_state):
                p.data.copy_(saved)
        self._modules = (x, c, W_a, w_b, gamma)
        self.temperature, self.bias = 1.0, 0.0
        self.bucket_cal = {}

        def fit_temp(lv_sub, yv_sub):
            t = torch.nn.Parameter(torch.ones(1, device=dev))
            b0 = torch.nn.Parameter(torch.zeros(1, device=dev))
            copt = torch.optim.LBFGS([t, b0], lr=0.1, max_iter=50)

            def closure():
                copt.zero_grad()
                loss = loss_fn(lv_sub / t.clamp(min=0.05) + b0, yv_sub)
                loss.backward()
                return loss

            copt.step(closure)
            return float(t.detach().clamp(min=0.05)), float(b0.detach())

        if self.calibrate:  # temperature + bias on the ES slice
            with torch.no_grad():
                lv = logits(mv, rv, zv)
            self.temperature, self.bias = fit_temp(lv, yv)
            if self.cal_mode == "bucket":
                qb = question_bucket(train_df["vote_question"]).to_numpy()[es]
                for bucket in np.unique(qb):
                    sub = torch.as_tensor(qb == bucket, device=dev)
                    if int(sub.sum()) >= 2000:  # thin buckets keep global
                        self.bucket_cal[bucket] = fit_temp(lv[sub], yv[sub])
        self.es_log_loss, self.epochs_run, self._dev = best, epoch + 1, dev
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        x, c, W_a, w_b, gamma = self._modules
        rc = self._rollcall_frame(eval_df)
        phi = torch.as_tensor(self._phi(rc, fit=False), device=self._dev)
        rc_index = {key: i for i, key in enumerate(rc["rc_key"])}
        r_idx = _rollcall_key(eval_df).map(rc_index).to_numpy()
        sponsor_by_rc = rc.set_index("rc_key")["sponsor_party"]
        z = self._z(eval_df, sponsor_by_rc.reindex(_rollcall_key(eval_df)).fillna(""))
        if self.use_cosponsors:
            z = np.concatenate([z, self._alignment(eval_df)[:, None]], 1)
        if self.use_billctx:
            z = np.concatenate([z, self._bill_context(eval_df)], 1)
        if self.use_majority:
            z = np.concatenate([z, self._majority_z(eval_df, fit=False)], 1)
        if self.mq_offset:
            z = np.concatenate([z, self._offset(eval_df)[:, None]], 1)

        m = eval_df["icpsr"].map(self.member_index)
        known = m.notna().to_numpy()
        mi = torch.as_tensor(m.fillna(0).astype(int).to_numpy(), dtype=torch.long,
                             device=self._dev)
        ri = torch.as_tensor(r_idx, dtype=torch.long, device=self._dev)
        zt = torch.as_tensor(z, device=self._dev)
        with torch.no_grad():
            f = phi[ri]
            xm = x(mi) * torch.as_tensor(known[:, None].astype(np.float32),
                                         device=self._dev)
            cm = c(mi).squeeze(-1) * torch.as_tensor(known.astype(np.float32),
                                                     device=self._dev)
            logit = ((xm * W_a(f)).sum(-1) + w_b(f).squeeze(-1) + cm
                     + gamma(zt).squeeze(-1))
            logit = logit.cpu().numpy()
        if getattr(self, "bucket_cal", None):
            qb = question_bucket(eval_df["vote_question"]).to_numpy()
            temp = np.full(len(logit), self.temperature)
            bias = np.full(len(logit), self.bias)
            for bucket, (t, b0) in self.bucket_cal.items():
                temp[qb == bucket] = t
                bias[qb == bucket] = b0
            logit = logit / temp + bias
        else:
            logit = logit / self.temperature + self.bias
        return 1 / (1 + np.exp(-logit))


REGISTRY = {
    "meta_tower_8d": lambda: TextTower(k=8, use_text=False),
    "text_tower_8d": lambda: TextTower(k=8, use_text=True),
    "text_tower_mq_8d": lambda: TextTower(k=8, use_text=True, mq_offset=True,
                                          calibrate=True, name="text_tower_mq_8d"),
    # strict pre-vote LLM embeddings instead of TF-IDF (leakage-clean text);
    # pinned to v1 bill-level embeddings for leaderboard reproducibility
    "emb_tower_mq_8d": lambda: TextTower(k=8, use_text=False, use_emb=True,
                                         mq_offset=True, calibrate=True,
                                         emb_file="rollcall_text_embeddings.parquet",
                                         name="emb_tower_mq_8d"),
    # both text blocks together (v1 embeddings)
    "embtfidf_tower_mq_8d": lambda: TextTower(k=8, use_text=True, use_emb=True,
                                              mq_offset=True, calibrate=True,
                                              emb_file="rollcall_text_embeddings.parquet",
                                              name="embtfidf_tower_mq_8d"),
    # v2 rollcall-level embeddings (question + vote_desc + pre-vote bill text)
    "emb2_tower_mq_8d": lambda: TextTower(k=8, use_text=False, use_emb=True,
                                          mq_offset=True, calibrate=True,
                                          name="emb2_tower_mq_8d"),
    "emb2_tower_mq_16d": lambda: TextTower(k=16, use_text=False, use_emb=True,
                                           mq_offset=True, calibrate=True,
                                           name="emb2_tower_mq_16d"),
    "emb2_mlp_mq_16d": lambda: TextTower(k=16, use_text=False, use_emb=True,
                                         mq_offset=True, calibrate=True, mlp_head=True,
                                         name="emb2_mlp_mq_16d"),
    "emb2tfidf_tower_mq_16d": lambda: TextTower(k=16, use_text=True, use_emb=True,
                                                mq_offset=True, calibrate=True,
                                                name="emb2tfidf_tower_mq_16d"),
    # temporal internal dev slice: early-stop + calibrate on future-like data
    "emb2_mlp_mq_16d_tcal": lambda: TextTower(k=16, use_text=False, use_emb=True,
                                              mq_offset=True, calibrate=True,
                                              mlp_head=True, es_mode="temporal",
                                              name="emb2_mlp_mq_16d_tcal"),
    "emb2tfidf_mq_16d_tcal": lambda: TextTower(k=16, use_text=True, use_emb=True,
                                               mq_offset=True, calibrate=True,
                                               es_mode="temporal",
                                               name="emb2tfidf_mq_16d_tcal"),
}


REGISTRY["emb3_mlp_mq_16d_tcal"] = lambda: TextTower(
    k=16, use_text=False, use_emb=True, mq_offset=True, calibrate=True,
    mlp_head=True, es_mode="temporal",
    emb_file="rollcall_text_embeddings_v3.parquet",
    name="emb3_mlp_mq_16d_tcal")  # v3: Qwen3-Embedding-0.6B, 6000-char budget


def _with_cosponsors():
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                  calibrate=True, mlp_head=True, es_mode="temporal",
                  name="emb2cosp_mlp_mq_16d_tcal")
    m.use_cosponsors = True
    return m


def _with_majority():
    # M-D: champion + majority-status z features (in-majority indicator and
    # pooled majority x question rate) — congress-agnostic structure aimed
    # at the congress-out transfer failure
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                  calibrate=True, mlp_head=True, es_mode="temporal",
                  name="emb2maj_mlp_mq_16d_tcal")
    m.use_majority = True
    return m


REGISTRY["emb2maj_mlp_mq_16d_tcal"] = _with_majority


def _with_billctx():
    # E1 (sprint): champion + strictly-prior same-bill context features,
    # jointly trained. NEGATIVE result (shortcut learning starves the
    # backbone; see Notes/experiment_ledger.md E1) — kept for reproducibility
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                  calibrate=True, mlp_head=True, es_mode="temporal",
                  name="emb2ctx_mlp_mq_16d_tcal")
    m.use_billctx = True
    return m


REGISTRY["emb2ctx_mlp_mq_16d_tcal"] = _with_billctx


class CtxStack:
    """E1b: champion backbone trained untouched, then a small logistic
    residual on within-bill context features (incl. staleness terms and
    their interactions with the prior-vote signal), fit ONLY on the
    temporal internal dev slice with the backbone logit as a fixed offset.
    The backbone cannot be starved by construction; the corrector has ~9
    parameters against ~150k dev rows.

    Note the dev slice is reused (backbone early-stop + calibration +
    stacking); it remains strictly inside train, so val/test discipline is
    intact — flagged in the experiment ledger."""

    def __init__(self, base_factory=None, name: str = "ctx_stack_16d_tcal"):
        self._base_factory = base_factory or (lambda: TextTower(
            k=16, use_text=False, use_emb=True, mq_offset=True, calibrate=True,
            mlp_head=True, es_mode="temporal", name="ctx_stack_base"))
        self.name = name

    @staticmethod
    def _design(feats: np.ndarray) -> np.ndarray:
        has_prior, own, rate, logd, sameday = feats.T
        return np.stack([has_prior, own, rate, logd, sameday,
                         own * logd, own * sameday, rate * logd], 1)

    def fit(self, train_df: pd.DataFrame):
        self.base = self._base_factory().fit(train_df)
        self.base._fit_bill_context(train_df)
        dev = TextTower.temporal_es_mask(train_df)
        dev_df = train_df[dev]
        X = self._design(self.base._bill_context(dev_df))
        p = np.clip(self.base.predict_proba(dev_df.drop(columns=["vote"])),
                    EPS, 1 - EPS)
        off = torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(dev_df["vote"].to_numpy(), dtype=torch.float32)
        lin = torch.nn.Linear(X.shape[1], 1)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.1, max_iter=100)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(off + lin(Xt).squeeze(-1), yt)
            loss.backward()
            return loss

        opt.step(closure)
        self._lin = lin
        self.stack_weights = lin.weight.detach().numpy().ravel().tolist()
        self.global_rate = self.base.global_rate
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.base.predict_proba(eval_df), EPS, 1 - EPS)
        X = torch.as_tensor(self._design(self.base._bill_context(eval_df)),
                            dtype=torch.float32)
        with torch.no_grad():
            z = np.log(p / (1 - p)) + self._lin(X).squeeze(-1).numpy()
        return 1 / (1 + np.exp(-z))


REGISTRY["ctx_stack_16d_tcal"] = lambda: CtxStack()


def _bucket_cal_champion():
    # E3: champion architecture with per-question-bucket calibration
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                  calibrate=True, mlp_head=True, es_mode="temporal",
                  name="emb2_mlp_mq_16d_bcal")
    m.cal_mode = "bucket"
    return m


REGISTRY["emb2_mlp_mq_16d_bcal"] = _bucket_cal_champion


class LogitBlend:
    """E4: logistic regression on component towers' logits (weights + bias,
    3 params), fit only on the temporal internal dev slice. Generalizes a
    convex blend; if the components are redundant the weights collapse
    onto the stronger one."""

    def __init__(self, factories, name: str):
        self._factories, self.name = factories, name

    def fit(self, train_df: pd.DataFrame):
        self.models = [f().fit(train_df) for f in self._factories]
        # registry factories are lambdas — unpicklable and unneeded once the
        # components exist; dropping them lets the fitted blend be frozen
        self._factories = None
        dev = TextTower.temporal_es_mask(train_df)
        dev_df = train_df[dev]
        feats = dev_df.drop(columns=["vote"])
        L = np.stack([np.log(np.clip(m.predict_proba(feats), EPS, 1 - EPS))
                      - np.log(1 - np.clip(m.predict_proba(feats), EPS, 1 - EPS))
                      for m in self.models], 1).astype(np.float32)
        lin = torch.nn.Linear(L.shape[1], 1)
        with torch.no_grad():  # start at "all weight on the first component"
            lin.weight.zero_()
            lin.weight[0, 0] = 1.0
            lin.bias.zero_()
        Lt = torch.as_tensor(L)
        yt = torch.as_tensor(dev_df["vote"].to_numpy(), dtype=torch.float32)
        opt = torch.optim.LBFGS(lin.parameters(), lr=0.1, max_iter=100)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(lin(Lt).squeeze(-1), yt)
            loss.backward()
            return loss

        opt.step(closure)
        self._lin = lin
        self.blend_weights = lin.weight.detach().numpy().ravel().tolist()
        self.global_rate = self.models[0].global_rate
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        L = np.stack([np.log(np.clip(m.predict_proba(eval_df), EPS, 1 - EPS))
                      - np.log(1 - np.clip(m.predict_proba(eval_df), EPS, 1 - EPS))
                      for m in self.models], 1).astype(np.float32)
        with torch.no_grad():
            z = self._lin(torch.as_tensor(L)).squeeze(-1).numpy()
        return 1 / (1 + np.exp(-z))


REGISTRY["blend_mlp_tfidf_tcal"] = lambda: LogitBlend(
    [REGISTRY["emb2_mlp_mq_16d_tcal"], REGISTRY["emb2tfidf_mq_16d_tcal"]],
    name="blend_mlp_tfidf_tcal")


REGISTRY["blend3_mlp_tfidf_emb3_tcal"] = lambda: LogitBlend(
    [REGISTRY["emb2_mlp_mq_16d_tcal"], REGISTRY["emb2tfidf_mq_16d_tcal"],
     REGISTRY["emb3_mlp_mq_16d_tcal"]],
    name="blend3_mlp_tfidf_emb3_tcal")


REGISTRY["emb4_mlp_mq_16d_tcal"] = lambda: TextTower(
    k=16, use_text=False, use_emb=True, mq_offset=True, calibrate=True,
    mlp_head=True, es_mode="temporal",
    emb_file="rollcall_text_embeddings_v4.parquet",
    name="emb4_mlp_mq_16d_tcal")  # v4: v2 + amendment purpose text (E5)


def _recency_champion():
    # E2: champion with recency-weighted member x question offset
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=True,
                  calibrate=True, mlp_head=True, es_mode="temporal",
                  name="emb2_mlp_mqr_16d_tcal")
    m.mq_recency = True
    return m


REGISTRY["emb2_mlp_mqr_16d_tcal"] = _recency_champion


REGISTRY["emb2cosp_mlp_mq_16d_tcal"] = _with_cosponsors


REGISTRY["emb_placebo_mlp_mq_16d_tcal"] = lambda: TextTower(
    k=16, use_text=False, use_emb=True, mq_offset=True, calibrate=True,
    mlp_head=True, es_mode="temporal",
    emb_file="rollcall_text_embeddings_placebo.parquet",
    name="emb_placebo_mlp_mq_16d_tcal")  # control: byte-copy of v2 under a
# different filename — isolates file/code-path effects from text content
# (E5 family regressions were insensitive to content; diagnosed 2026-07-03)


REGISTRY["notext_mq_16d_tcal"] = lambda: TextTower(
    k=16, use_text=False, use_emb=False, mq_offset=True, calibrate=True,
    mlp_head=True, es_mode="temporal",
    name="notext_mq_16d_tcal")  # champion architecture minus ALL text:
# metadata + member history only — the counterfactual for "what does
# reading the bill add" (P5v6 protest-detection comparison)
