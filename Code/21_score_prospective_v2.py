"""Score the frozen v2 prospective artifact (blend3) on post-snapshot votes.

Mirror of 14_score_prospective.py for the v2 artifact, with the blend's
extra needs handled:
1. Fresh 119th-Congress Voteview download; keep rollcalls strictly after
   the v2 snapshot date.
2. Fetch + parse BILLSTATUS for new bills; build v2 (MiniLM, 1500 chars)
   AND v3 (Qwen3-Embedding-0.6B, 6000 chars) rollcall text embeddings for
   new rollcalls and append to the respective parquet files.
3. Refresh each pickled component tower's embedding lookup AND its bills
   table (sponsor party for new bills; the v1 scorer left component
   _bills stale — improved here).
4. Score with the sha256-verified pickle; append-only ledger + report.

The v1 artifact continues to be scored by 14_score_prospective.py; both
ledgers are reported per protocol rule 4.

Output: Modified Data/results/frozen/prospective_ledger_v2.parquet (+ report)
"""

import hashlib
import importlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

score_v1 = importlib.import_module("14_score_prospective")
embed_mod = importlib.import_module("08_embed_bills")

from harness import metrics  # noqa: E402
from models_idealpoint import _rollcall_key  # noqa: E402
from models_texttower import _load_bill_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
ART = MOD / "results" / "frozen"

V3_MODEL = "Qwen/Qwen3-Embedding-0.6B"
V3_MAX_CHARS = 6000


def append_v3_embeddings(panel: pd.DataFrame) -> None:
    """v3 analog of the v2 append in 14_score_prospective.attach_text_features:
    same text recipe, Qwen encoder, 6000-char budget."""
    from sentence_transformers import SentenceTransformer
    bills = pd.read_parquet(MOD / "bills.parquet")
    p = panel.merge(bills[["congress", "bill_type", "bill_no", "title", "summaries"]],
                    on=["congress", "bill_type", "bill_no"], how="left",
                    suffixes=("", "_b"))
    texts = []
    saved = embed_mod.MAX_CHARS
    embed_mod.MAX_CHARS = V3_MAX_CHARS
    try:
        for row in p.itertuples():
            if isinstance(getattr(row, "summaries", None), str):
                bill_text, _ = embed_mod.pre_vote_text(row.summaries, row.title,
                                                       str(row.date))
            else:
                bill_text = row.title if isinstance(getattr(row, "title", None), str) else ""
            q = row.vote_question if isinstance(row.vote_question, str) else ""
            d = row.vote_desc if isinstance(row.vote_desc, str) else ""
            texts.append(". ".join(x for x in (q, d, bill_text or "") if x)[:V3_MAX_CHARS])
    finally:
        embed_mod.MAX_CHARS = saved
    p["text"] = texts

    st = SentenceTransformer(V3_MODEL, device="cpu")
    rc_text = p.drop_duplicates(["congress", "chamber", "rollnumber"])[
        ["congress", "chamber", "rollnumber", "text"]]
    emb = st.encode(rc_text["text"].tolist(), batch_size=8, normalize_embeddings=True)
    new = pd.concat([rc_text[["congress", "chamber", "rollnumber"]].reset_index(drop=True),
                     pd.DataFrame(emb, columns=[f"e{j}" for j in range(emb.shape[1])])],
                    axis=1)
    new.insert(3, "text_source", "prospective")
    f = MOD / "rollcall_text_embeddings_v3.parquet"
    existing = pd.read_parquet(f)
    keys = set(map(tuple, existing[["congress", "chamber", "rollnumber"]]
                   .itertuples(index=False)))
    add = new[~new[["congress", "chamber", "rollnumber"]].apply(tuple, axis=1).isin(keys)]
    if len(add):
        pd.concat([existing, add], ignore_index=True).to_parquet(f, index=False)
        print(f"appended {len(add)} new v3 rollcall embeddings")


def main():
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"], \
        "v2 pickle hash mismatch — artifact has been modified"
    with open(pkl, "rb") as f:
        model = pickle.load(f)

    files = score_v1.fetch_fresh_119()
    panel = score_v1.build_panel(files, meta["data_snapshot_max_vote_date"])
    if panel.empty:
        print("no rollcalls after snapshot yet")
        return
    panel = score_v1.attach_text_features(panel)  # v2 embeddings append
    append_v3_embeddings(panel)

    # refresh component lookups + bills tables for the new rollcalls
    fresh_bills = _load_bill_text()
    for tower in model.models:
        tower._bills = fresh_bills
        if getattr(tower, "use_emb", False):
            emb = pd.read_parquet(MOD / tower.emb_file)
            emb["rc_key"] = _rollcall_key(emb)
            cols = [c for c in emb.columns if c.startswith("e")]
            tower._emb_lookup = emb.set_index("rc_key")[cols]

    p = model.predict_proba(panel.drop(columns=["vote"]))
    ledger = panel[["congress", "chamber", "rollnumber", "icpsr", "date",
                    "vote_question", "vote"]].copy()
    ledger["p_yea"] = p
    ledger["scored_utc"] = datetime.now(timezone.utc).isoformat()
    ledger.to_parquet(ART / "prospective_ledger_v2.parquet", index=False)

    m = metrics(ledger["vote"].to_numpy(), ledger["p_yea"].to_numpy())
    report = {"snapshot": meta["data_snapshot_max_vote_date"],
              "rollcalls_scored": int(ledger.groupby(
                  ["congress", "chamber", "rollnumber"]).ngroups),
              "votes_scored": int(len(ledger)),
              "date_range": [str(ledger.date.min().date()),
                             str(ledger.date.max().date())],
              "metrics": {k: round(v, 4) for k, v in m.items()}}
    (ART / "prospective_report_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
