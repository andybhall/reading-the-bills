"""Build v4 rollcall text embeddings: v2 + amendment purpose text.

v4 differs from v2 ONLY on amendment rollcalls matched to a BILLSTATUS
amendment (join on parsed House "Roll no." / Senate "Record Vote Number"
references; see 18_parse_amendments.py). For those rollcalls the text is
rebuilt with the amendment's purpose inserted between the vote fields and
the parent bill's pre-vote summary:

    question. vote_desc. Amendment purpose: <purpose>. <pre-vote bill text>

Everything else copies its v2 embedding row untouched — enforced by
construction (start from v2, overwrite the matched subset) and verified by
row counts printed at the end. Encoder and char budget identical to v2
(MiniLM, 1500 chars), so E5 isolates the CONTENT change alone.

Output: Modified Data/rollcall_text_embeddings_v4.parquet
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
embed_mod = importlib.import_module("08_embed_bills")
from models_forecast import question_bucket  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"


def main():
    rc = pd.read_parquet(MOD / "rollcalls.parquet")
    rc = rc[rc.congress >= 108].copy()
    rc["qb"] = question_bucket(rc["vote_question"])
    amrc = rc[rc.qb == "amendment"].copy()

    am = pd.read_parquet(MOD / "amendments.parquet")
    am = am[(am.roll_no >= 0) & (am.purpose != "") & (am.action_date != "")].copy()
    am["chamber"] = am["amdt_type"].map({"HAMDT": "House", "SAMDT": "Senate"})
    am = am.rename(columns={"roll_no": "rollnumber"})
    # the join REQUIRES the action date to equal the rollcall date: clerk
    # roll numbers reset each session, Voteview's are congress-continuous
    # (a roll+congress join alone mismatches ~40% — one-session offsets)
    amrc["vote_date"] = amrc["date"].astype(str).str[:10]
    m = amrc.merge(am[["congress", "chamber", "rollnumber", "action_date", "purpose"]],
                   left_on=["congress", "chamber", "rollnumber", "vote_date"],
                   right_on=["congress", "chamber", "rollnumber", "action_date"],
                   how="inner")
    # ambiguous joins (two amendments claiming one rollcall) are dropped —
    # a wrong purpose is worse than no purpose
    m = m.drop_duplicates(["congress", "chamber", "rollnumber"], keep=False)
    link = pd.read_parquet(MOD / "rollcall_bills.parquet")[
        ["congress", "chamber", "rollnumber", "bill_type", "bill_no"]]
    bills = pd.read_parquet(MOD / "bills.parquet")[
        ["congress", "bill_type", "bill_no", "title", "summaries"]]
    m = (m.merge(link, on=["congress", "chamber", "rollnumber"], how="left")
          .merge(bills, on=["congress", "bill_type", "bill_no"], how="left"))

    # v4c template: the purpose fills the vote_desc SLOT (exactly the
    # template Senate amendment rows already inhabit), and ONLY desc-less
    # rollcalls change — minimal embedding-distribution shift. The earlier
    # "Amendment purpose:" prefix template regressed even with verified
    # text (E5b): two text templates in one embedding space, and the
    # amendment-poor dev slice never rewards the new region.
    m = m[m["vote_desc"].isna()].copy()
    texts = []
    for row in m.itertuples():
        if isinstance(row.summaries, str):
            bill_text, _ = embed_mod.pre_vote_text(row.summaries, row.title,
                                                   str(row.date))
        else:
            bill_text = row.title if isinstance(row.title, str) else ""
        q = row.vote_question if isinstance(row.vote_question, str) else ""
        parts = [p for p in (q, row.purpose, bill_text or "") if p]
        texts.append(". ".join(parts)[:embed_mod.MAX_CHARS])
    m["text"] = texts
    print(f"rebuilding text for {len(m)} desc-less matched amendment rollcalls")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(embed_mod.MODEL, device="cpu")
    emb = st.encode(m["text"].tolist(), batch_size=256,
                    normalize_embeddings=True, show_progress_bar=True)
    new = pd.concat([m[["congress", "chamber", "rollnumber"]].reset_index(drop=True),
                     pd.DataFrame(emb, columns=[f"e{j}" for j in range(emb.shape[1])])],
                    axis=1)
    new.insert(3, "text_source", "v4_amendment")

    v2 = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet")
    keys = set(map(tuple, new[["congress", "chamber", "rollnumber"]]
                   .itertuples(index=False)))
    keep = ~v2[["congress", "chamber", "rollnumber"]].apply(tuple, axis=1).isin(keys)
    v4 = pd.concat([v2[keep], new], ignore_index=True)
    v4.to_parquet(MOD / "rollcall_text_embeddings_v4.parquet", index=False)
    print(json.dumps({
        "v2_rows": int(len(v2)), "v4_rows": int(len(v4)),
        "replaced": int((~keep).sum()), "added_new": int(len(new) - (~keep).sum()),
        "unchanged": int(keep.sum()),
    }, indent=2))
    assert len(v4) >= len(v2), "v4 must not lose rollcalls"


if __name__ == "__main__":
    main()
