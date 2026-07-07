"""Export all-MiniLM-L6-v2 to a single ONNX graph that maps token ids to
the final sentence-transformers embedding (masked mean pool + L2 norm
baked in), and verify it reproduces SentenceTransformer.encode() to
cosine > 0.9999 on probe texts. Also saves the tokenizer.json for the
`tokenizers` runtime.

Run: python3 webapp_vercel/build/export_onnx.py
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Code"
OUT = ROOT / "webapp_vercel" / "model"
sys.path.insert(0, str(CODE))
MAXLEN = 256


class Encoder(torch.nn.Module):
    """BertModel -> masked mean pool -> L2 normalize (the exact ST pipeline)."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        tok = torch.zeros_like(input_ids)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=tok).last_hidden_state       # (B,L,384)
        m = attention_mask.unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)          # (B,384)
        return mean / mean.norm(dim=1, keepdim=True).clamp(min=1e-12)


def main():
    from sentence_transformers import SentenceTransformer
    em = importlib.import_module("08_embed_bills")
    st = SentenceTransformer(em.MODEL, device="cpu").eval()
    bert = st[0].auto_model.eval()
    enc = Encoder(bert).eval()

    tok = st.tokenizer
    OUT.mkdir(exist_ok=True, parents=True)
    tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))

    # dummy input for export
    d = tok(["On Passage. A bill to provide for certain matters."],
            padding=True, truncation=True, max_length=MAXLEN,
            return_tensors="pt")
    with torch.no_grad():
        ref_graph = enc(d["input_ids"], d["attention_mask"]).numpy()

    onnx_path = OUT / "minilm.onnx"
    torch.onnx.export(
        enc, (d["input_ids"], d["attention_mask"]), str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"], opset_version=17,
        dynamic_axes={"input_ids": {0: "b", 1: "l"},
                      "attention_mask": {0: "b", 1: "l"},
                      "embedding": {0: "b"}}, dynamo=False)
    mb = onnx_path.stat().st_size / 1e6
    print(f"exported minilm.onnx ({mb:.0f} MB) + tokenizer.json")

    # ---- parity: ONNX (with tokenizers lib) vs SentenceTransformer ----
    import onnxruntime as ort
    from tokenizers import Tokenizer
    T = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    T.enable_truncation(max_length=MAXLEN)
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])

    def onnx_embed(text):
        e = T.encode(text)
        ids = np.array([e.ids], dtype=np.int64)
        am = np.array([e.attention_mask], dtype=np.int64)
        return sess.run(None, {"input_ids": ids, "attention_mask": am})[0][0]

    probes = [
        "On Passage. To amend the Internal Revenue Code to expand the child "
        "tax credit and provide disaster tax relief, and for other purposes.",
        "On Passage. To require universal background checks for firearm sales "
        "and establish a national red-flag standard.",
        "On Passage. " + ("A very long bill. " * 400),   # exercises truncation
        "On Passage. A bill to designate a post office."]
    worst_cos, worst_abs = 1.0, 0.0
    for p in probes:
        ref = st.encode([p[:1500]], normalize_embeddings=True)[0]
        got = onnx_embed(p[:1500])
        cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
        mad = float(np.max(np.abs(ref - got)))
        worst_cos = min(worst_cos, cos); worst_abs = max(worst_abs, mad)
        print(f"  cos={cos:.6f}  max|Δ|={mad:.2e}  ({p[15:45]!r}...)")
    ok = worst_cos > 0.9999 and worst_abs < 1e-3
    print(f"WORST cos={worst_cos:.6f}  max|Δ|={worst_abs:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
