"""Score the frozen prospective model on rollcalls AFTER the data snapshot.

Pipeline (idempotent; run any time after new votes occur):
1. Re-download current 119th-Congress votes/rollcalls/members from Voteview.
2. Keep rollcalls dated strictly after the frozen snapshot date.
3. Fetch + parse BILLSTATUS for any new bills; build v2 rollcall text
   (question + vote_desc + pre-vote summary) and MiniLM embeddings.
4. Score every yea/nay vote with the frozen model (pickle verified by
   sha256) and append to the prospective ledger; report cumulative metrics.

The frozen model is NEVER refit here. New members fall back to x=0 /
member-rate fallbacks by construction.

Output: Modified Data/results/frozen/prospective_ledger.parquet (+ report)
"""

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import importlib
embed_mod = importlib.import_module("08_embed_bills")
parse_mod = importlib.import_module("07_parse_bills")

from harness import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
RAW = ROOT / "Original Data"
ART = MOD / "results" / "frozen"
VV = "https://voteview.com/static/data/out"


def fetch_fresh_119():
    out = RAW / "voteview_prospective"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    files = {}
    for kind, url in [("votes", f"{VV}/votes/H119_votes.csv"),
                      ("votes_s", f"{VV}/votes/S119_votes.csv"),
                      ("rollcalls", f"{VV}/rollcalls/HSall_rollcalls.csv"),
                      ("members", f"{VV}/members/HSall_members.csv")]:
        dest = out / f"{stamp}_{kind}.csv"
        if not dest.exists():
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
        files[kind] = dest
    return files


def build_panel(files, snapshot: str) -> pd.DataFrame:
    votes = pd.concat([pd.read_csv(files["votes"]), pd.read_csv(files["votes_s"])])
    rc = pd.read_csv(files["rollcalls"], low_memory=False)
    rc = rc[(rc.congress == 119)].copy()
    rc["date"] = pd.to_datetime(rc["date"])
    rc = rc[rc["date"] > pd.Timestamp(snapshot)]
    if rc.empty:
        return pd.DataFrame()
    mem = pd.read_csv(files["members"], low_memory=False)
    mem = mem[(mem.congress == 119) & (mem.chamber != "President")]
    mem = mem.drop_duplicates(["congress", "chamber", "icpsr"], keep="last")

    votes = votes[votes["icpsr"] < 99000]
    votes["vote"] = np.where(votes["cast_code"].isin([1, 2, 3]), 1.0,
                             np.where(votes["cast_code"].isin([4, 5, 6]), 0.0, np.nan))
    votes = votes.dropna(subset=["vote"])
    panel = votes.merge(rc[["congress", "chamber", "rollnumber", "date",
                            "bill_number", "vote_question", "vote_desc"]],
                        on=["congress", "chamber", "rollnumber"], how="inner")
    panel = panel.merge(mem[["congress", "chamber", "icpsr", "party_code"]],
                        on=["congress", "chamber", "icpsr"], how="left")
    return panel


def attach_text_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Parse bill linkage, fetch any new BILLSTATUS, build v2 text + embeddings,
    and append rows to rollcall_text_embeddings_v2.parquet for new rollcalls."""
    import subprocess
    link_mod = importlib.import_module("04_link_bills")
    parsed = panel["bill_number"].apply(link_mod.parse_bill_number)
    parsed["bill_no"] = parsed["bill_no"].astype("Int64")
    panel = pd.concat([panel.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    # fetch any bills not in the cache
    need = panel[panel.bill_category == "legislation"][
        ["congress", "bill_type", "bill_no"]].drop_duplicates()
    cache = RAW / "govinfo" / "billstatus"
    sess = requests.Session()
    fetched = 0
    for row in need.itertuples():
        name = f"BILLSTATUS-{row.congress}{row.bill_type}{int(row.bill_no)}.xml"
        dest = cache / name
        if not dest.exists():
            url = (f"https://www.govinfo.gov/bulkdata/BILLSTATUS/{row.congress}/"
                   f"{row.bill_type}/{name}")
            r = sess.get(url, timeout=60)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                fetched += 1
    print(f"fetched {fetched} new BILLSTATUS files")
    if fetched:
        subprocess.run(["python3", str(ROOT / "Code" / "07_parse_bills.py")], check=True)

    bills = pd.read_parquet(MOD / "bills.parquet")
    panel = panel.merge(bills[["congress", "bill_type", "bill_no", "title", "summaries"]],
                        on=["congress", "bill_type", "bill_no"], how="left")
    panel["vote_date"] = panel["date"].astype(str)
    texts = []
    for row in panel.itertuples():
        if isinstance(getattr(row, "summaries", None), str):
            bill_text, _ = embed_mod.pre_vote_text(row.summaries, row.title, row.vote_date)
        else:
            bill_text = row.title if isinstance(getattr(row, "title", None), str) else ""
        q = row.vote_question if isinstance(row.vote_question, str) else ""
        d = row.vote_desc if isinstance(row.vote_desc, str) else ""
        texts.append(". ".join(p for p in (q, d, bill_text or "") if p)[:embed_mod.MAX_CHARS])
    panel["text"] = texts

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(embed_mod.MODEL, device="cpu")
    rc_text = panel.drop_duplicates(["congress", "chamber", "rollnumber"])[
        ["congress", "chamber", "rollnumber", "text"]]
    emb = st.encode(rc_text["text"].tolist(), batch_size=256, normalize_embeddings=True)
    new_rows = pd.concat([rc_text[["congress", "chamber", "rollnumber"]].reset_index(drop=True),
                          pd.DataFrame(emb, columns=[f"e{j}" for j in range(emb.shape[1])])],
                         axis=1)
    new_rows.insert(3, "text_source", "prospective")
    emb_file = MOD / "rollcall_text_embeddings_v2.parquet"
    existing = pd.read_parquet(emb_file)
    keys = set(map(tuple, existing[["congress", "chamber", "rollnumber"]].itertuples(index=False)))
    add = new_rows[~new_rows[["congress", "chamber", "rollnumber"]]
                   .apply(tuple, axis=1).isin(keys)]
    if len(add):
        pd.concat([existing, add], ignore_index=True).to_parquet(emb_file, index=False)
        print(f"appended {len(add)} new rollcall embeddings")
    return panel


def main():
    meta = json.loads((ART / "prospective_model_meta.json").read_text())
    pkl = ART / "prospective_model.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"], \
        "frozen model pickle hash mismatch — artifact has been modified"
    with open(pkl, "rb") as f:
        model = pickle.load(f)

    files = fetch_fresh_119()
    panel = build_panel(files, meta["data_snapshot_max_vote_date"])
    if panel.empty:
        print("no rollcalls after snapshot yet")
        return
    panel = attach_text_features(panel)
    # reload embeddings into the frozen model's lookup for new rollcalls
    emb = pd.read_parquet(MOD / "rollcall_text_embeddings_v2.parquet")
    from models_texttower import _rollcall_key
    emb["rc_key"] = _rollcall_key(emb)
    cols = [c for c in emb.columns if c.startswith("e")]
    model._emb_lookup = emb.set_index("rc_key")[cols]

    p = model.predict_proba(panel.drop(columns=["vote"]))
    ledger = panel[["congress", "chamber", "rollnumber", "icpsr", "date",
                    "vote_question", "vote"]].copy()
    ledger["p_yea"] = p
    ledger["scored_utc"] = datetime.now(timezone.utc).isoformat()
    ledger.to_parquet(ART / "prospective_ledger.parquet", index=False)

    m = metrics(ledger["vote"].to_numpy(), ledger["p_yea"].to_numpy())
    report = {"snapshot": meta["data_snapshot_max_vote_date"],
              "rollcalls_scored": int(ledger.groupby(["congress", "chamber", "rollnumber"]).ngroups),
              "votes_scored": int(len(ledger)),
              "date_range": [str(ledger.date.min().date()), str(ledger.date.max().date())],
              "metrics": {k: round(v, 4) for k, v in m.items()}}
    (ART / "prospective_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
