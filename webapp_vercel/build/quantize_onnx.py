"""Dynamic int8 quantization of minilm.onnx for the serverless size
budget, with a parity re-check: int8 embeddings vs the fp32 ONNX and vs
SentenceTransformer, plus the downstream effect on predicted vote
probabilities. Accept only if cosine > 0.999 and the max predicted-prob
shift is < 0.02 (invisible in the UI; this is a demo forecaster, not the
frozen benchmark).

Run: python3 webapp_vercel/build/quantize_onnx.py
"""

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Code"
OUT = ROOT / "webapp_vercel" / "model"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "webapp_vercel" / "api"))
MAXLEN = 256


def main():
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
    from tokenizers import Tokenizer

    fp32 = OUT / "minilm.onnx"
    int8 = OUT / "minilm_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"fp32 {fp32.stat().st_size/1e6:.0f} MB -> "
          f"int8 {int8.stat().st_size/1e6:.0f} MB")

    T = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    T.enable_truncation(max_length=MAXLEN)

    def embed(sess, text):
        e = T.encode(text)
        return sess.run(None, {"input_ids": np.array([e.ids], np.int64),
                               "attention_mask": np.array([e.attention_mask],
                                                          np.int64)})[0][0]

    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    from sentence_transformers import SentenceTransformer
    em = importlib.import_module("08_embed_bills")
    st = SentenceTransformer(em.MODEL, device="cpu")

    probes = [
        "On Passage. To amend the Internal Revenue Code to expand the child "
        "tax credit and provide disaster tax relief, and for other purposes.",
        "On Passage. To require universal background checks for firearm "
        "sales and establish a national red-flag standard.",
        "On Passage. A bill to designate a United States post office.",
        "On Passage. To accelerate clean energy with tax credits for wind, "
        "solar, and battery manufacturing and a clean electricity standard."]
    worst_cos = 1.0
    for p in probes:
        st_ref = st.encode([p[:1500]], normalize_embeddings=True)[0]
        e8 = embed(s8, p[:1500])
        cos = float(st_ref @ e8 / (np.linalg.norm(st_ref) * np.linalg.norm(e8)))
        worst_cos = min(worst_cos, cos)
        print(f"  int8 vs ST  cos={cos:.5f}   ({p[15:44]!r})")

    # downstream: predicted-probability shift, int8 vs fp32, via the tower
    import _predictor as pred
    # temporarily point the predictor at each model by swapping the session
    def predict_with(sess, text, chamber, sponsor):
        S = pred._load(); d = S["d"]
        rc = ("On Passage. " + " ".join(text.split()))[:1500]
        emb = embed(sess, rc)
        phi = np.zeros(399, np.float32); phi[:384] = emb * 5.0
        phi[384 + int(d["q_idx"])] = 1.0
        phi[384 + int(d["nq"]) + int(d["c_idx"])] = 1.0
        phi[384 + int(d["nq"]) + int(d["nc"]) + (0 if sponsor == "D" else 1)] = 1.0
        g_a = d["a2w"] @ np.maximum(d["a0w"] @ phi + d["a0b"], 0) + d["a2b"]
        g_b = float(d["b2w"] @ np.maximum(d["b0w"] @ phi + d["b0b"], 0) + d["b2b"])
        m = np.array([k.startswith(chamber + ":") for k in d["keys"]])
        party = d["party"][m]
        same = (party == sponsor).astype(np.float32)
        opp = ((party != sponsor) & (party != "I")).astype(np.float32)
        gg = d["gamma"]
        lo = d["xi"][m] @ g_a + g_b + d["ci"][m] + gg[0]*same + gg[1]*opp + gg[2]*d["mqoff"][m]
        lo = lo / float(d["temperature"]) + float(d["bias"])
        return 1/(1+np.exp(-lo))

    worst_shift = 0.0
    for text in [p[12:] for p in probes]:
        for ch in ("House", "Senate"):
            for sp in ("D", "R"):
                a = predict_with(s32, text, ch, sp)
                b = predict_with(s8, text, ch, sp)
                worst_shift = max(worst_shift, float(np.max(np.abs(a - b))))
    print(f"WORST int8-vs-ST cosine: {worst_cos:.5f}")
    print(f"WORST predicted-prob shift int8 vs fp32: {worst_shift:.4f}")
    ok = worst_cos > 0.999 and worst_shift < 0.02
    print("VERDICT:", "PASS -- ship int8" if ok else "FAIL -- keep fp32")


if __name__ == "__main__":
    main()
