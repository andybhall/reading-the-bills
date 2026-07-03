"""Latent ideal-point models fit by SGD (PyTorch), harness-compatible.

    P(yea) = sigmoid( a_r . x_m + b_r + c_m )

x_m : k-dim member ideal point (indexed by icpsr, shared across congresses)
c_m : member intercept (base yea propensity)
a_r : k-dim rollcall discrimination, b_r : rollcall difficulty
      (indexed by congress-chamber-rollnumber)

This is the logistic (IRT) version of the spatial voting model; with member
positions frozen at DW-NOMINATE scores it reduces to the classical model
(NominateLogit below). Identification (rotation/scale) is irrelevant for
prediction; for interpretation we align signs post hoc.

Fallbacks: unseen member -> x=0, c=0 (predicts the rollcall's base rate);
unseen rollcall -> global training rate (cannot occur in Regime A, where
every rollcall retains training votes).

Early stopping uses an internal 2% slice of TRAIN (hash-based, seed-keyed);
the benchmark's val/test sets are never touched during fitting.
"""

import hashlib

import numpy as np
import pandas as pd
import torch

SEED = 42
ES_FRACTION = 0.02


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cell_hash_unit(df: pd.DataFrame, salt: str) -> np.ndarray:
    keys = (df["congress"].astype(str) + "|" + df["chamber"] + "|"
            + df["rollnumber"].astype(str) + "|" + df["icpsr"].astype(str) + "|" + salt)
    return keys.map(lambda k: int(hashlib.sha256(k.encode()).hexdigest()[:12], 16) / 16**12).to_numpy()


def _rollcall_key(df: pd.DataFrame) -> pd.Series:
    return (df["congress"].astype(str) + "_" + df["chamber"] + "_"
            + df["rollnumber"].astype(str))


class IdealPoint:
    """k-dimensional logistic ideal-point model."""

    def __init__(self, k: int = 2, lr: float = 0.05, weight_decay: float = 1e-4,
                 batch_size: int = 1 << 17, max_epochs: int = 60, patience: int = 3,
                 device: str | None = None, min_epochs: int = 0):
        self.k, self.lr, self.weight_decay = k, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        # min_epochs: bilinear factors start near zero and move slowly at
        # first; on small data (one batch/epoch) patience can fire before any
        # learning happens — whether it does depended on device RNG luck
        # (diagnosed 2026-07-03). Default 0 preserves the exact behavior of
        # the registered ideal_point_* benchmark models.
        self.min_epochs = min_epochs
        # device: force "cpu" to keep Metal out of the process; None uses
        # MPS when available (the default and the registered-model behavior)
        self.device = device
        self.name = f"ideal_point_{k}d"

    # -- index helpers ------------------------------------------------------
    def _encode(self, df, fit: bool):
        rc = _rollcall_key(df)
        if fit:
            self.member_index = {m: i for i, m in enumerate(sorted(df["icpsr"].unique()))}
            self.rollcall_index = {r: i for i, r in enumerate(sorted(rc.unique()))}
        m = df["icpsr"].map(self.member_index).to_numpy()
        r = rc.map(self.rollcall_index).to_numpy()
        return m, r

    def fit(self, train_df: pd.DataFrame, init_positions: pd.Series | None = None):
        """init_positions: optional icpsr-indexed Series seeding dim 1 of each
        member's position (e.g. +0.5 R / -0.5 D). Breaks the sign symmetry so
        the orientation of the learned axis cannot twist across eras — a real
        failure mode we hit in the Senate (see logs/2026-06-11_session02.md).
        Affects only the optimization basin, not what the data can express."""
        torch.manual_seed(SEED)
        dev = torch.device(self.device) if self.device else _device()
        m_idx, r_idx = self._encode(train_df, fit=True)
        y = train_df["vote"].to_numpy()
        self.global_rate = float(y.mean())

        # internal early-stop slice (never the benchmark val/test sets)
        es = _cell_hash_unit(train_df, f"earlystop{SEED}") < ES_FRACTION
        n_members, n_rollcalls = len(self.member_index), len(self.rollcall_index)

        x = torch.nn.Embedding(n_members, self.k).to(dev)
        c = torch.nn.Embedding(n_members, 1).to(dev)
        a = torch.nn.Embedding(n_rollcalls, self.k).to(dev)
        b = torch.nn.Embedding(n_rollcalls, 1).to(dev)
        for emb in (x, a):
            torch.nn.init.normal_(emb.weight, std=0.1)
        for emb in (c, b):
            torch.nn.init.zeros_(emb.weight)
        if init_positions is not None:
            seed_vals = np.zeros(n_members, dtype=np.float32)
            for icpsr, i in self.member_index.items():
                if icpsr in init_positions.index:
                    seed_vals[i] = init_positions.loc[icpsr]
            with torch.no_grad():
                x.weight[:, 0] = torch.as_tensor(seed_vals, device=dev)
        params = [x.weight, c.weight, a.weight, b.weight]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def to_dev(arr, dtype):
            return torch.as_tensor(arr, dtype=dtype, device=dev)

        mt, rt = to_dev(m_idx[~es], torch.long), to_dev(r_idx[~es], torch.long)
        yt = to_dev(y[~es], torch.float32)
        mv, rv = to_dev(m_idx[es], torch.long), to_dev(r_idx[es], torch.long)
        yv = to_dev(y[es], torch.float32)

        def logits(mi, ri):
            return (x(mi) * a(ri)).sum(-1) + b(ri).squeeze(-1) + c(mi).squeeze(-1)

        best, best_state, bad = np.inf, None, 0
        n = len(yt)
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(logits(mt[idx], rt[idx]), yt[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                es_loss = float(loss_fn(logits(mv, rv), yv))
            if es_loss < best - 1e-5:
                best, bad = es_loss, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        for p, saved in zip(params, best_state):
            p.data.copy_(saved)
        self._x, self._c = x.weight.detach().cpu().numpy(), c.weight.detach().cpu().numpy()
        self._a, self._b = a.weight.detach().cpu().numpy(), b.weight.detach().cpu().numpy()
        self.es_log_loss, self.epochs_run = best, epoch + 1
        return self

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        rc = _rollcall_key(eval_df)
        m = eval_df["icpsr"].map(self.member_index)
        r = rc.map(self.rollcall_index)
        known_r = r.notna().to_numpy()
        known_m = m.notna().to_numpy()
        mi = m.fillna(0).astype(int).to_numpy()
        ri = r.fillna(0).astype(int).to_numpy()
        xm = np.where(known_m[:, None], self._x[mi], 0.0)
        cm = np.where(known_m, self._c[mi, 0], 0.0)
        logit = (xm * self._a[ri]).sum(1) + self._b[ri, 0] + cm
        p = 1 / (1 + np.exp(-logit))
        return np.where(known_r, p, self.global_rate)

    def member_positions(self) -> pd.DataFrame:
        """Learned ideal points + intercepts, for interpretation/validation."""
        inv = {i: m for m, i in self.member_index.items()}
        out = pd.DataFrame(self._x, columns=[f"dim{j+1}" for j in range(self.k)])
        out["intercept"] = self._c[:, 0]
        out.insert(0, "icpsr", [inv[i] for i in range(len(inv))])
        return out


class NominateLogit(IdealPoint):
    """Classical reference: member positions FROZEN at DW-NOMINATE (dim1, dim2);
    only rollcall parameters are learned. Members missing NOMINATE scores
    (~0 in this period) sit at the origin."""

    def __init__(self, **kw):
        super().__init__(k=2, **kw)
        self.name = "nominate_logit"

    def fit(self, train_df: pd.DataFrame):
        scores = (train_df.groupby("icpsr")[["nominate_dim1", "nominate_dim2"]]
                  .last().fillna(0.0))
        return self._fit_frozen(train_df, scores)

    def _fit_frozen(self, train_df, scores):
        torch.manual_seed(SEED)
        dev = torch.device(self.device) if self.device else _device()
        m_idx, r_idx = self._encode(train_df, fit=True)
        y = train_df["vote"].to_numpy()
        self.global_rate = float(y.mean())
        es = _cell_hash_unit(train_df, f"earlystop{SEED}") < ES_FRACTION

        pos = np.zeros((len(self.member_index), 2), dtype=np.float32)
        for icpsr, i in self.member_index.items():
            if icpsr in scores.index:
                pos[i] = scores.loc[icpsr].to_numpy()
        x = torch.as_tensor(pos, device=dev)
        n_rollcalls = len(self.rollcall_index)
        a = torch.nn.Embedding(n_rollcalls, 2).to(dev)
        b = torch.nn.Embedding(n_rollcalls, 1).to(dev)
        torch.nn.init.normal_(a.weight, std=0.1)
        torch.nn.init.zeros_(b.weight)
        opt = torch.optim.AdamW([a.weight, b.weight], lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        mt = torch.as_tensor(m_idx[~es], dtype=torch.long, device=dev)
        rt = torch.as_tensor(r_idx[~es], dtype=torch.long, device=dev)
        yt = torch.as_tensor(y[~es], dtype=torch.float32, device=dev)
        mv = torch.as_tensor(m_idx[es], dtype=torch.long, device=dev)
        rv = torch.as_tensor(r_idx[es], dtype=torch.long, device=dev)
        yv = torch.as_tensor(y[es], dtype=torch.float32, device=dev)

        def logits(mi, ri):
            return (x[mi] * a(ri)).sum(-1) + b(ri).squeeze(-1)

        best, best_state, bad = np.inf, None, 0
        n = len(yt)
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(logits(mt[idx], rt[idx]), yt[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                es_loss = float(loss_fn(logits(mv, rv), yv))
            if es_loss < best - 1e-5:
                best, bad = es_loss, 0
                best_state = [a.weight.detach().clone(), b.weight.detach().clone()]
            else:
                bad += 1
                if bad >= self.patience:
                    break
        a.weight.data.copy_(best_state[0])
        b.weight.data.copy_(best_state[1])
        self._x, self._c = pos, np.zeros((len(self.member_index), 1), dtype=np.float32)
        self._a, self._b = a.weight.detach().cpu().numpy(), b.weight.detach().cpu().numpy()
        self.es_log_loss, self.epochs_run = best, epoch + 1
        return self


REGISTRY = {
    "ideal_point_1d": lambda: IdealPoint(k=1),
    "ideal_point_2d": lambda: IdealPoint(k=2),
    "ideal_point_8d": lambda: IdealPoint(k=8),
    "nominate_logit": lambda: NominateLogit(),
}
