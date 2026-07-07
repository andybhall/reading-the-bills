"""Portable bill-vote predictor: ONNX MiniLM encoder + NumPy tower.

No PyTorch, no sentence-transformers, no 218 MB pickle. Reproduces the
frozen model's MiniLM-tower forecast for a hypothetical 119th-Congress
final-passage vote, verified against the torch implementation to <1e-4
in absolute predicted probability. Loaded once per process and cached.
"""

import json
import os
from functools import lru_cache

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MAXLEN = 256
PARTY_FULL = {"D": 100.0, "R": 200.0}


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    d = np.load(os.path.join(MODEL_DIR, "tower.npz"), allow_pickle=True)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_truncation(max_length=MAXLEN)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "minilm_fp16.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"])
    with open(os.path.join(MODEL_DIR, "rosters.json")) as f:
        rosters = json.load(f)
    train_emb = np.load(os.path.join(MODEL_DIR, "train_emb_sample.npy")
                        ).astype(np.float32)
    return {"d": {k: d[k] for k in d.files}, "tok": tok, "sess": sess,
            "rosters": rosters, "train_emb": train_emb}


def _embed(text):
    """Token embeddings from the fp16 ONNX transformer, then masked mean
    pool + L2 normalize in NumPy (the SentenceTransformer pipeline)."""
    S = _load()
    e = S["tok"].encode(text)
    am = np.array([e.attention_mask], dtype=np.int64)
    te = S["sess"].run(None, {"input_ids": np.array([e.ids], dtype=np.int64),
                              "attention_mask": am})[0][0].astype(np.float32)
    mask = am[0][:, None].astype(np.float32)
    mean = (te * mask).sum(0) / max(mask.sum(), 1e-9)
    return mean / max(np.linalg.norm(mean), 1e-12)


def predict(text, chamber, sponsor):
    S = _load(); d = S["d"]
    rc_text = ("On Passage. " + " ".join(text.split()))[:1500]
    emb = _embed(rc_text)

    # amortized rollcall params g_a (16d), g_b (scalar) from phi
    phi = np.zeros(399, np.float32)
    phi[:384] = emb.astype(np.float32) * 5.0
    phi[384 + int(d["q_idx"])] = 1.0
    phi[384 + int(d["nq"]) + int(d["c_idx"])] = 1.0
    phi[384 + int(d["nq"]) + int(d["nc"]) + (0 if sponsor == "D" else 1)] = 1.0
    g_a = d["a2w"] @ np.maximum(d["a0w"] @ phi + d["a0b"], 0) + d["a2b"]
    g_b = float(d["b2w"] @ np.maximum(d["b0w"] @ phi + d["b0b"], 0) + d["b2b"])

    keys = d["keys"]
    mask = np.array([k.startswith(chamber + ":") for k in keys])
    xi, ci = d["xi"][mask], d["ci"][mask]
    mqoff, party = d["mqoff"][mask], d["party"][mask]
    same = (party == sponsor).astype(np.float32)
    opp = ((party != sponsor) & (party != "I")).astype(np.float32)
    gam = d["gamma"]
    logit = xi @ g_a + g_b + ci + gam[0] * same + gam[1] * opp + gam[2] * mqoff
    logit = logit / float(d["temperature"]) + float(d["bias"])
    p = 1.0 / (1.0 + np.exp(-logit))

    roster = S["rosters"][chamber]
    xs = np.array([r["x"] for r in roster], np.float32)
    members = [{"name": r["name"], "state": r["state"], "party": r["party"],
                "x": r["x"], "p": round(float(pi), 4)}
               for r, pi in zip(roster, p)]
    members.sort(key=lambda m: m["x"])

    # cutpoint: logistic fit of predicted prob on member position
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    b1, b0 = np.polyfit(xs, z, 1)
    cut = -b0 / b1 if abs(b1) > 0.05 else None
    if cut is not None and not (xs.min() <= cut <= xs.max()):
        cut = None

    def pstats(code):
        m = party == {"D": "D", "R": "R", "I": "I"}[code]
        n = int(m.sum())
        return {"n": n, "yea_share": float(p[m].mean()) if n else 0.0,
                "n_yea": int((p[m] > 0.5).sum())}

    n = len(p); n_yea = int((p > 0.5).sum()); thresh = (n // 2) + 1
    d_share = float(p[party == "D"].mean()) if (party == "D").any() else 0.0
    r_share = float(p[party == "R"].mean()) if (party == "R").any() else 0.0
    sponsor_share = d_share if sponsor == "D" else r_share
    other_share = r_share if sponsor == "D" else d_share
    inconsistent = sponsor_share < other_share
    sims = S["train_emb"] @ (emb / np.linalg.norm(emb).clip(1e-9))
    top10 = float(np.sort(sims)[-10:].mean())

    return {
        "chamber": chamber, "sponsor_party": sponsor, "n_members": n,
        "predicted_yea_count": n_yea, "predicted_nay_count": n - n_yea,
        "predicted_yea_share": round(float(p.mean()), 4),
        "predicted_pass": bool(n_yea >= thresh), "pass_threshold": thresh,
        "cutpoint": None if cut is None else round(float(cut), 2),
        "cutpoint_note": ("The bill divides members at this point on the "
                          "liberal(-) / conservative(+) scale."
                          if cut is not None else
                          "Predicted votes barely depend on ideology "
                          "(a lopsided or valence vote)."),
        "similarity_to_floor_agenda": round(top10, 3),
        "extrapolation_warning": (
            "The model predicts the sponsor's own party supporting this "
            "bill less than the opposition -- an internal inconsistency "
            "that indicates the bill is unlike those that reached floor "
            "votes in the training window (floor agendas are curated by "
            "the majority). Treat the direction of this forecast as "
            "unreliable." if inconsistent else None),
        "democrats": pstats("D"), "republicans": pstats("R"),
        "independents": pstats("I"), "members": members,
        "x_range": [round(float(xs.min()), 2), round(float(xs.max()), 2)],
    }
