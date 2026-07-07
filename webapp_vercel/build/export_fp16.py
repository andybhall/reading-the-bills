"""Re-export MiniLM as a fp16 ONNX that outputs TOKEN embeddings (pooling
+ normalize done in NumPy at runtime). Exporting token embeddings avoids
the baked-in-pooling graph that broke fp16 conversion, and fp16 halves
the model (~45 MB) for the serverless size budget while keeping cosine
parity > 0.9995 with SentenceTransformer.

Run: python3 webapp_vercel/build/export_fp16.py
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Code"))
MAXLEN = 256

for stale in ("minilm_int8.onnx", "minilm_fp16.onnx", "minilm.onnx",
              "minilm_tokemb.onnx"):
    p = OUT / stale
    if p.exists():
        os.remove(p)


class TokEmb(torch.nn.Module):
    def __init__(self, b):
        super().__init__(); self.b = b

    def forward(self, input_ids, attention_mask):
        tt = torch.zeros_like(input_ids)
        return self.b(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=tt).last_hidden_state


def main():
    from sentence_transformers import SentenceTransformer
    em = importlib.import_module("08_embed_bills")
    st = SentenceTransformer(em.MODEL, device="cpu").eval()
    enc = TokEmb(st[0].auto_model.eval()).eval()
    tok = st.tokenizer
    tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))
    d = tok(["On Passage. A bill."], padding=True, truncation=True,
            max_length=MAXLEN, return_tensors="pt")

    fp32 = OUT / "minilm_tokemb.onnx"
    torch.onnx.export(
        enc, (d["input_ids"], d["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["token_emb"], opset_version=17, dynamo=False,
        dynamic_axes={"input_ids": {0: "b", 1: "l"},
                      "attention_mask": {0: "b", 1: "l"},
                      "token_emb": {0: "b", 1: "l"}})

    import onnx
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(str(fp32)),
                                           keep_io_types=True)
    onnx.save(m16, str(OUT / "minilm_fp16.onnx"))
    os.remove(fp32)
    print(f"fp16 token-emb model: "
          f"{(OUT/'minilm_fp16.onnx').stat().st_size/1e6:.1f} MB")

    import onnxruntime as ort
    from tokenizers import Tokenizer
    T = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    T.enable_truncation(max_length=MAXLEN)
    sess = ort.InferenceSession(str(OUT / "minilm_fp16.onnx"),
                                providers=["CPUExecutionProvider"])

    def embed(text):
        e = T.encode(text)
        am = np.array([e.attention_mask], np.int64)
        te = sess.run(None, {"input_ids": np.array([e.ids], np.int64),
                             "attention_mask": am})[0][0].astype(np.float32)
        mask = am[0][:, None].astype(np.float32)
        mean = (te * mask).sum(0) / max(mask.sum(), 1e-9)
        return mean / max(np.linalg.norm(mean), 1e-12)

    probes = ["To expand the child tax credit and provide disaster relief.",
              "Universal background checks and a national red-flag standard.",
              "A very long bill. " * 400,
              "Designate a United States post office building."]
    wc = 1.0
    for p in probes:
        ref = st.encode([("On Passage. " + p)[:1500]],
                        normalize_embeddings=True)[0]
        got = embed(("On Passage. " + p)[:1500])
        wc = min(wc, float(ref @ got /
                           (np.linalg.norm(ref) * np.linalg.norm(got))))
    print(f"fp16 + numpy-pool worst cos vs ST: {wc:.6f}  "
          f"{'PASS' if wc > 0.9995 else 'FAIL'}")


if __name__ == "__main__":
    main()
