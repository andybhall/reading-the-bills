"""Embed rollcall-level text with a sentence-transformer, with strict
pre-vote leakage discipline.

v2 (current): text is constructed at the ROLLCALL level, not the bill
level: vote_question + vote_desc (the motion/amendment/nominee actually
being voted on; knowable at vote time by definition of the forecasting
task) + the latest CRS summary version with action_date strictly before
the vote date (else bill title). This gives text to nominations (100%
vote_desc coverage: nominee + position) and amendments (the amendment's
own substance), which bill-level v1 text missed entirely.

HTML stripped; truncated to 1,500 chars. Each unique text embedded once
(all-MiniLM-L6-v2, 384 dims, local — no API; model held fixed from v1 so
gains are attributable to text construction, not the encoder).

Output: Modified Data/rollcall_text_embeddings_v2.parquet (v1 file kept
for reproducibility of earlier leaderboard rows).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
MODEL = "all-MiniLM-L6-v2"   # v2 default; override with --model for v3+
MAX_CHARS = 1500             # override with --max-chars

TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", s)).strip()


def pre_vote_text(summaries_json: str, title: str, vote_date) -> tuple[str, str]:
    """Latest summary strictly before the vote date, else the title."""
    best_date, best_text = None, None
    for s in json.loads(summaries_json):
        d = s.get("action_date") or ""
        if d and d < vote_date and s.get("text"):
            if best_date is None or d > best_date:
                best_date, best_text = d, s["text"]
    if best_text:
        return strip_html(best_text)[:MAX_CHARS], "summary"
    return (title or "")[:MAX_CHARS], "title"


def main():
    from sentence_transformers import SentenceTransformer
    global MAX_CHARS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--version", default="v2",
                    help="output suffix: rollcall_text_embeddings_{version}.parquet")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    MAX_CHARS = args.max_chars

    links = pd.read_parquet(MOD / "rollcall_bills.parquet")
    bills = pd.read_parquet(MOD / "bills.parquet")
    rc = pd.read_parquet(MOD / "rollcalls.parquet")[
        ["congress", "chamber", "rollnumber", "vote_desc"]]
    df = links.merge(bills, on=["congress", "bill_type", "bill_no"], how="left")
    df = df.merge(rc, on=["congress", "chamber", "rollnumber"], how="left", validate="1:1")
    df["vote_date"] = df["date"].astype(str)

    texts, sources = [], []
    for row in df.itertuples():
        if row.bill_category == "legislation" and isinstance(row.summaries, str):
            bill_text, src = pre_vote_text(row.summaries, row.title, row.vote_date)
        elif isinstance(getattr(row, "title", None), str) and row.title:
            bill_text, src = row.title[:MAX_CHARS], "title"
        else:
            bill_text, src = "", "desc_only"
        question = row.vote_question if isinstance(row.vote_question, str) else ""
        desc = row.vote_desc if isinstance(row.vote_desc, str) else ""
        t = ". ".join(p for p in (question, desc, bill_text) if p)[:MAX_CHARS]
        if not t:
            src = "none"
        texts.append(t)
        sources.append(src)
    df["text"], df["text_source"] = texts, sources

    uniq = sorted(set(t for t in texts if t))
    print(f"{len(df)} rollcalls; {len(uniq)} unique texts to embed")
    model = SentenceTransformer(args.model, device=args.device,
                                trust_remote_code=True)
    bs = 256 if args.max_chars <= 2000 else 32
    emb = model.encode(uniq, batch_size=bs, show_progress_bar=False,
                       normalize_embeddings=True)
    lookup = {t: e for t, e in zip(uniq, emb)}
    dim = emb.shape[1]
    zeros = np.zeros(dim, dtype=np.float32)
    mat = np.stack([lookup.get(t, zeros) for t in texts])

    out = pd.concat([df[["congress", "chamber", "rollnumber", "text_source"]].reset_index(drop=True),
                     pd.DataFrame(mat, columns=[f"e{j}" for j in range(dim)])], axis=1)
    out.to_parquet(MOD / f"rollcall_text_embeddings_{args.version}.parquet", index=False)

    report = {
        "version": args.version + " (rollcall-level: question + vote_desc + pre-vote bill text)",
        "model": args.model, "max_chars": args.max_chars, "dim": int(dim), "rollcalls": int(len(out)),
        "source_counts": out["text_source"].value_counts().to_dict(),
        "empty_text_share": round(float((df.text == "").mean()), 4),
        "summary_share_legislation": round(float(
            (df[df.bill_category == "legislation"].text_source == "summary").mean()), 4),
    }
    (MOD / "checks" / "embeddings_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
