"""Literature baselines for the forecast regimes (Paper A horse race).

The 2011-2016 text-based vote-prediction thread, reimplemented faithfully
but simplified, on our harness and splits:

GBSpatial ("gb_spatial_*") — Gerrish-Blei (2011) two-stage analog.
    Stage 1: fit the standard 1D logistic ideal-point model on train votes
    (member position x_i, intercept c_i; per-rollcall discrimination a_j and
    difficulty b_j). Stage 2: ridge-regress (a_j, b_j) onto rollcall text;
    an eval rollcall — never seen in training — gets TEXT-PREDICTED
    parameters scored against the train-fitted ideal points. GB learned the
    text-to-parameter map jointly with LDA topics; two-stage ridge on
    TF-IDF->SVD (LSA) is the closest classical analog ("tfidf" mode). The
    "emb2" mode swaps in modern sentence embeddings of rollcall-level text,
    isolating what the representation upgrade alone buys the CLASSICAL
    architecture. Ridge alpha is chosen on an internal 10% slice of train
    rollcalls (never val/test), minimizing downstream vote log loss.

kraft_bilinear_16d — Kraft-Jain-Rush (2016) analog: learned member
    embedding dot a linear map of a FIXED text embedding, plus member
    intercept. Implemented as a TextTower with meta_features off, no
    member x question offset, no MLP, no calibration: exactly the bilinear
    core. Deviations from 2016, both documented and both favoring the
    baseline: modern sentence embeddings instead of averaged word2vec, the
    tower's 2-dim sponsor-party-alignment term, k=16 to match the champion's
    capacity, and our temporal early-stop discipline.

nominate_context_logit — the strongest "NOMINATE + metadata, no text"
    null: logistic regression of vote on frozen DW-NOMINATE (dim1, dim2)
    interacted with congress, question bucket, and sponsor-party alignment.
    Note DW-NOMINATE scores are estimated by Voteview from members' FULL
    careers (including eval-period votes) — an accepted advantage to the
    baseline, same caveat as nominate_logit in Regime A.
"""

import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from models_forecast import question_bucket
from models_idealpoint import IdealPoint, _device, _rollcall_key
from models_texttower import MOD, RC_KEY, TextTower, _load_bill_text

SEED = 42
EPS = 1e-7
ES_FRACTION = 0.02


def _hash_unit(keys: pd.Series, salt: str) -> np.ndarray:
    return (keys + "|" + salt).map(
        lambda k: int(hashlib.sha256(k.encode()).hexdigest()[:12], 16) / 16**12).to_numpy()




class GBSpatial:
    def __init__(self, text_mode: str = "tfidf", k: int = 1, svd_dim: int = 128,
                 alphas=(0.1, 1.0, 10.0, 100.0),
                 emb_file: str = "rollcall_text_embeddings_v2.parquet",
                 bills_df: pd.DataFrame | None = None, name: str | None = None,
                 calibrate: bool = False):
        self.text_mode, self.k, self.svd_dim, self.alphas = text_mode, k, svd_dim, alphas
        self.emb_file, self._bills_override = emb_file, bills_df
        # calibrate: scalar temperature + bias fit on the internal ridge-dev
        # slice. GB-2011's generative inference regularized bill parameters
        # through priors; the two-stage ridge analog lacks that and emits
        # overconfident extrapolations. The calibrated variant is the
        # "best-case classical" — it isolates discrimination failure from
        # calibration failure without touching val/test.
        self.calibrate = calibrate
        suffix = "_tcal" if calibrate else ""
        self.name = name or f"gb_spatial_{text_mode}{suffix}"

    def _rc_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("vote_question", "bill_type", "bill_no") if c in df.columns]
        rc = df.drop_duplicates(RC_KEY)[RC_KEY + cols].copy()
        rc["rc_key"] = _rollcall_key(rc)
        if self.text_mode == "tfidf":
            rc = rc.merge(self._bills[["congress", "bill_type", "bill_no", "text"]],
                          on=["congress", "bill_type", "bill_no"], how="left")
            # question bucket prepended: GB studied passage votes, where the
            # question is implicit; our rollcalls span many question types
            rc["text"] = (question_bucket(rc["vote_question"]).astype(str)
                          + ". " + rc["text"].fillna(""))
        return rc

    def _phi(self, rc: pd.DataFrame, fit: bool) -> np.ndarray:
        if self.text_mode == "emb2":
            if fit:
                emb = pd.read_parquet(MOD / self.emb_file)
                emb["rc_key"] = _rollcall_key(emb)
                cols = [c for c in emb.columns if c.startswith("e")]
                self._emb_lookup = emb.set_index("rc_key")[cols]
            return self._emb_lookup.reindex(rc["rc_key"]).fillna(0.0).to_numpy(np.float32)
        if fit:
            self._tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                          sublinear_tf=True)
            X = self._tfidf.fit_transform(rc["text"])
            self._svd = TruncatedSVD(n_components=min(self.svd_dim, X.shape[1] - 1),
                                     random_state=SEED)
            Z = self._svd.fit_transform(X)
            self._z_scale = Z.std(axis=0) + 1e-8
        else:
            Z = self._svd.transform(self._tfidf.transform(rc["text"]))
        return (Z / self._z_scale).astype(np.float32)

    def _predict_params(self, ridge_a: Ridge, ridge_b: Ridge, X: np.ndarray):
        """Text-predicted rollcall parameters with GUARANTEED shapes (n, k)
        and (n,). sklearn ravels a single-column y, so for k=1 predict()
        returns 1-D; without the reshape, (x * A) silently broadcasts
        (n_votes, 1) x (n_votes,) into an (n_votes, n_votes) matrix — 2.5TB
        at production scale, killed by jetsam (diagnosed 2026-07-03)."""
        a = np.asarray(ridge_a.predict(X)).reshape(len(X), self.k)
        b = np.asarray(ridge_b.predict(X)).ravel()
        return a, b

    def _member_terms(self, df: pd.DataFrame):
        m = df["icpsr"].map(self.ip.member_index)
        known = m.notna().to_numpy()
        mi = m.fillna(0).astype(int).to_numpy()
        x = np.where(known[:, None], self.ip._x[mi], 0.0)
        c = np.where(known, self.ip._c[mi, 0], 0.0)
        return x, c

    def _vote_log_loss(self, df: pd.DataFrame, key_pos: dict,
                       a_pred: np.ndarray, b_pred: np.ndarray) -> float:
        x, c = self._member_terms(df)
        idx = _rollcall_key(df).map(key_pos).to_numpy()
        A, B = a_pred[idx], b_pred[idx]
        assert A.shape == (len(df), self.k) and B.shape == (len(df),), \
            f"parameter shape drift: A {A.shape}, B {B.shape}"
        z = (x * A).sum(1) + B + c
        p = np.clip(1 / (1 + np.exp(-z)), EPS, 1 - EPS)
        y = df["vote"].to_numpy()
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, train_df: pd.DataFrame):
        self._bills = self._bills_override if self._bills_override is not None \
            else _load_bill_text()
        print(f"  [{self.name}] stage 1: ideal point (k={self.k})...", flush=True)
        self.ip = IdealPoint(k=self.k, min_epochs=15).fit(train_df)
        print(f"  [{self.name}] stage 2: text features...", flush=True)

        rc = self._rc_frame(train_df)
        X = self._phi(rc, fit=True)
        ridx = rc["rc_key"].map(self.ip.rollcall_index).to_numpy()
        A, B = self.ip._a[ridx], self.ip._b[ridx, 0]

        # alpha on an internal slice of train ROLLCALLS, judged by the loss
        # that matters (vote-level BCE with text-predicted parameters)
        print(f"  [{self.name}] stage 3: ridge alpha selection...", flush=True)
        dev = _hash_unit(rc["rc_key"], f"gbdev{SEED}") < 0.10
        dev_votes = train_df[_rollcall_key(train_df).isin(set(rc.loc[dev, "rc_key"]))]
        dev_pos = {key: i for i, key in enumerate(rc.loc[dev, "rc_key"])}
        best, best_params = (np.inf, None), None
        for alpha in self.alphas:
            ra = Ridge(alpha=alpha).fit(X[~dev], A[~dev])
            rb = Ridge(alpha=alpha).fit(X[~dev], B[~dev])
            pa, pb = self._predict_params(ra, rb, X[dev])
            ll = self._vote_log_loss(dev_votes, dev_pos, pa, pb)
            if ll < best[0]:
                best, best_params = (ll, alpha), (pa, pb)
        self.alpha_dev_log_loss, self.alpha = best
        self._ridge_a = Ridge(alpha=self.alpha).fit(X, A)
        self._ridge_b = Ridge(alpha=self.alpha).fit(X, B)

        self.temperature, self.bias = 1.0, 0.0
        if self.calibrate:
            # fit on the alpha loop's HELD-OUT dev predictions — the final
            # ridges above saw the dev rollcalls, so their dev logits are
            # partly in-sample and would understate the needed temperature
            pa, pb = best_params
            x, c = self._member_terms(dev_votes)
            idx = _rollcall_key(dev_votes).map(dev_pos).to_numpy()
            z = (x * pa[idx]).sum(1) + pb[idx] + c
            zt = torch.as_tensor(z, dtype=torch.float32)
            yt = torch.as_tensor(dev_votes["vote"].to_numpy(), dtype=torch.float32)
            T = torch.nn.Parameter(torch.ones(1))
            b0 = torch.nn.Parameter(torch.zeros(1))
            opt = torch.optim.LBFGS([T, b0], lr=0.1, max_iter=50)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            def closure():
                opt.zero_grad()
                loss = loss_fn(zt / T.clamp(min=0.05) + b0, yt)
                loss.backward()
                return loss

            opt.step(closure)
            self.temperature = float(T.detach().clamp(min=0.05))
            self.bias = float(b0.detach())
        self.global_rate = float(train_df["vote"].mean())
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        rc = self._rc_frame(eval_df)
        X = self._phi(rc, fit=False)
        pa, pb = self._predict_params(self._ridge_a, self._ridge_b, X)
        key_pos = {key: i for i, key in enumerate(rc["rc_key"])}
        x, c = self._member_terms(eval_df)
        idx = _rollcall_key(eval_df).map(key_pos).to_numpy()
        A, B = pa[idx], pb[idx]
        assert A.shape == (len(eval_df), self.k), f"parameter shape drift: {A.shape}"
        z = (x * A).sum(1) + B + c
        return 1 / (1 + np.exp(-(z / self.temperature + self.bias)))


class NominateContextLogit:
    """Logistic regression: vote ~ NOMINATE dims x (congress, question
    bucket, sponsor alignment). Convex, fit by Adam with an internal 2%
    early-stop slice. Unseen congress -> its dummies are zero and the
    global NOMINATE main effects carry (usable on congressout118)."""

    name = "nominate_context_logit"

    def __init__(self, lr: float = 0.05, batch_size: int = 1 << 17,
                 max_epochs: int = 40, patience: int = 3,
                 bills_df: pd.DataFrame | None = None):
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self._bills_override = bills_df

    def _sponsor_align(self, df: pd.DataFrame) -> pd.Series:
        rc = df.drop_duplicates(RC_KEY)[RC_KEY + ["bill_type", "bill_no"]].copy()
        rc = rc.merge(self._bills[["congress", "bill_type", "bill_no", "sponsor_party"]],
                      on=["congress", "bill_type", "bill_no"], how="left")
        rc["rc_key"] = _rollcall_key(rc)
        sp = rc.set_index("rc_key")["sponsor_party"].reindex(_rollcall_key(df)).fillna("")
        member_p = df["party_code"].map({100.0: "D", 200.0: "R"}).fillna("").to_numpy()
        return pd.Series(np.where(sp.to_numpy() == "", "none",
                                  np.where(sp.to_numpy() == member_p, "same", "opp")),
                         index=df.index)

    def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        d1 = df["nominate_dim1"].fillna(0.0).to_numpy(np.float32)
        d2 = df["nominate_dim2"].fillna(0.0).to_numpy(np.float32)
        qb = question_bucket(df["vote_question"])
        al = self._sponsor_align(df)
        if fit:
            self._q_levels = sorted(qb.unique())
            self._g_levels = sorted(df["congress"].unique())
        q1 = np.stack([(qb == q).to_numpy(np.float32) for q in self._q_levels], 1)
        g1 = np.stack([(df["congress"] == g).to_numpy(np.float32) for g in self._g_levels], 1)
        a1 = np.stack([(al == a).to_numpy(np.float32) for a in ("same", "opp", "none")], 1)
        return np.hstack([d1[:, None], d2[:, None], g1, q1, a1,
                          d1[:, None] * q1, d1[:, None] * a1, d1[:, None] * g1,
                          d2[:, None] * a1]).astype(np.float32)

    def fit(self, train_df: pd.DataFrame):
        torch.manual_seed(SEED)
        dev = _device()
        self._bills = self._bills_override if self._bills_override is not None \
            else _load_bill_text()
        X = self._design(train_df, fit=True)
        y = train_df["vote"].to_numpy(np.float32)
        es = _hash_unit(_rollcall_key(train_df) + "|" + train_df["icpsr"].astype(str),
                        f"nclogit{SEED}") < ES_FRACTION
        lin = torch.nn.Linear(X.shape[1], 1).to(dev)
        torch.nn.init.zeros_(lin.weight)
        opt = torch.optim.Adam(lin.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        Xt = torch.as_tensor(X[~es], device=dev)
        yt = torch.as_tensor(y[~es], device=dev)
        Xv = torch.as_tensor(X[es], device=dev)
        yv = torch.as_tensor(y[es], device=dev)
        best, best_state, bad = np.inf, None, 0
        n = len(yt)
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                i = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss_fn(lin(Xt[i]).squeeze(-1), yt[i]).backward()
                opt.step()
            with torch.no_grad():
                es_loss = float(loss_fn(lin(Xv).squeeze(-1), yv))
            if es_loss < best - 1e-5:
                best, bad = es_loss, 0
                best_state = [p.detach().clone() for p in lin.parameters()]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        with torch.no_grad():
            for p, saved in zip(lin.parameters(), best_state):
                p.data.copy_(saved)
        self._lin, self._dev = lin, dev
        self.es_log_loss, self.epochs_run = best, epoch + 1
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        X = torch.as_tensor(self._design(eval_df, fit=False), device=self._dev)
        with torch.no_grad():
            return torch.sigmoid(self._lin(X).squeeze(-1)).cpu().numpy()


def _kraft():
    m = TextTower(k=16, use_text=False, use_emb=True, mq_offset=False,
                  mlp_head=False, calibrate=False, es_mode="temporal",
                  name="kraft_bilinear_16d")
    m.meta_features = False
    return m


REGISTRY = {
    "gb_spatial_tfidf": lambda: GBSpatial(text_mode="tfidf"),
    "gb_spatial_emb2": lambda: GBSpatial(text_mode="emb2"),
    "gb_spatial_tfidf_tcal": lambda: GBSpatial(text_mode="tfidf", calibrate=True),
    "gb_spatial_emb2_tcal": lambda: GBSpatial(text_mode="emb2", calibrate=True),
    "kraft_bilinear_16d": _kraft,
    "nominate_context_logit": lambda: NominateContextLogit(),
}
