"""Extract the frozen MiniLM-MLP tower into a portable NumPy predictor.

The Vercel deployment cannot ship PyTorch + the 218 MB blend pickle. But
for a hypothetical "On Passage" / "legislation" vote, the tower's forward
pass reduces to small dense algebra: the only bill-varying inputs are the
384-d sentence embedding and the sponsor-party choice. This script pulls
the tower's weights out of the frozen pickle, precomputes every
member-fixed quantity for the 119th Congress, saves them as a compact
.npz + rosters .json, reimplements the forward pass in pure NumPy, and
verifies the NumPy logits match torch predict_proba to <1e-4 on probe
bills across both sponsors and both chambers.

Run: python3 webapp_vercel/build/extract_tower.py
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Code"
ART = ROOT / "Modified Data" / "results" / "frozen"
MEAS = ROOT / "Modified Data" / "results" / "measures"
OUT = ROOT / "webapp_vercel" / "model"
sys.path.insert(0, str(CODE))

PARTY = {100.0: "D", 200.0: "R", 328.0: "I"}


def lin(layer):
    return (layer.weight.detach().cpu().numpy().astype(np.float32),
            layer.bias.detach().cpu().numpy().astype(np.float32))


def display_name(bioname):
    parts = str(bioname).split(",")
    last = parts[0].strip().title()
    first = parts[1].strip().split()[0].title() if len(parts) > 1 else ""
    return f"{first} {last}".strip() if first else last


def main():
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"]
    blend = pickle.load(open(pkl, "rb"))
    t = blend.models[0]
    x, c, W_a, w_b, gamma = t._modules

    # --- global weights ---
    a0w, a0b = lin(W_a[0]); a2w, a2b = lin(W_a[2])
    b0w, b0b = lin(w_b[0]); b2w, b2b = lin(w_b[2])
    gam = gamma.weight.detach().cpu().numpy().astype(np.float32).ravel()  # (3,)
    xw = x.weight.detach().cpu().numpy().astype(np.float32)               # (1460,16)
    cw = c.weight.detach().cpu().numpy().astype(np.float32).ravel()       # (1460,)
    q_idx = t._q_levels.index("passage")
    c_idx = t._c_levels.index("legislation")
    nq, nc = len(t._q_levels), len(t._c_levels)

    # --- per-119th-member fixed quantities ---
    members_meta, rosters = {}, {}
    xi_all, ci_all, mqoff_all, party_all, keyorder = [], [], [], [], []
    for chamber in ("House", "Senate"):
        pos = pd.read_parquet(MEAS / f"members_{chamber.lower()}119.parquet")
        mem = pd.read_parquet(ROOT / "Modified Data" / "members.parquet")
        mem = mem[(mem.congress == 119) & (mem.chamber == chamber)][
            ["icpsr", "party_code", "bioname", "state_abbrev"]
            ].drop_duplicates("icpsr")
        pos = pos.drop(columns=[c for c in ("bioname", "party_code",
                                            "state_abbrev") if c in pos.columns])
        r = (pos.merge(mem, on="icpsr", how="inner")
             .drop_duplicates("icpsr")
             # a few members carry two Voteview icpsr in one congress
             # (e.g. Kevin Kiley); keep one row per person for the UI
             .drop_duplicates(["bioname", "state_abbrev"])
             .reset_index(drop=True))
        # mq-offset for a hypothetical On Passage vote, per member
        probe = pd.DataFrame({
            "congress": 119, "chamber": chamber, "icpsr": r.icpsr,
            "party_code": r.party_code, "vote_question": "On Passage"})
        pmq = np.clip(t._mq.predict_proba(probe), 1e-7, 1 - 1e-7)
        mqoff = np.log(pmq / (1 - pmq)).astype(np.float32)

        rows = []
        for j, rr in enumerate(r.itertuples()):
            idx = t.member_index.get(rr.icpsr)
            xi = xw[idx] if idx is not None else np.zeros(16, np.float32)
            ci = cw[idx] if idx is not None else np.float32(0.0)
            key = f"{chamber}:{int(rr.icpsr)}"
            keyorder.append(key)
            xi_all.append(xi); ci_all.append(ci)
            mqoff_all.append(mqoff[j]); party_all.append(PARTY.get(rr.party_code, "I"))
            rows.append({"key": key, "name": display_name(rr.bioname),
                         "state": rr.state_abbrev,
                         "party": PARTY.get(rr.party_code, "I"),
                         "x": round(float(rr.x), 3)})
        rosters[chamber] = rows

    npz = dict(
        a0w=a0w, a0b=a0b, a2w=a2w, a2b=a2b, b0w=b0w, b0b=b0b, b2w=b2w, b2b=b2b,
        gamma=gam, temperature=np.float32(t.temperature), bias=np.float32(t.bias),
        q_idx=np.int64(q_idx), c_idx=np.int64(c_idx), nq=np.int64(nq), nc=np.int64(nc),
        xi=np.stack(xi_all), ci=np.array(ci_all, np.float32),
        mqoff=np.array(mqoff_all, np.float32),
        party=np.array(party_all), keys=np.array(keyorder))
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez(OUT / "tower.npz", **npz)
    (OUT / "rosters.json").write_text(json.dumps(rosters))

    # subsample of training-corpus embeddings for the floor-agenda
    # similarity chip (the full 23k x 384 lookup is 35 MB; an 8k fp16
    # sample is ~6 MB and the chip is only a coarse 3-bucket signal)
    E = t._emb_lookup.to_numpy().astype(np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)
    rng = np.random.default_rng(20260707)
    samp = E[rng.choice(len(E), size=min(8000, len(E)), replace=False)]
    np.save(OUT / "train_emb_sample.npy", samp.astype(np.float16))
    print(f"saved train_emb_sample.npy "
          f"({(OUT/'train_emb_sample.npy').stat().st_size/1e6:.1f} MB, "
          f"{len(samp)} rows)")
    print(f"saved tower.npz ({(OUT/'tower.npz').stat().st_size/1024:.0f} KB) "
          f"+ rosters.json ({sum(len(v) for v in rosters.values())} members)")

    # ---- NumPy forward pass (mirrors TextTower.predict_proba) ----
    def numpy_predict(emb, chamber, sponsor):
        d = np.load(OUT / "tower.npz", allow_pickle=True)
        mask = np.array([k.startswith(chamber + ":") for k in d["keys"]])
        phi = np.zeros(399, np.float32)
        phi[:384] = emb.astype(np.float32) * 5.0
        phi[384 + int(d["q_idx"])] = 1.0
        phi[384 + int(d["nq"]) + int(d["c_idx"])] = 1.0
        phi[384 + int(d["nq"]) + int(d["nc"]) + (0 if sponsor == "D" else 1)] = 1.0
        h_a = np.maximum(d["a0w"] @ phi + d["a0b"], 0)
        g_a = d["a2w"] @ h_a + d["a2b"]            # (16,)
        h_b = np.maximum(d["b0w"] @ phi + d["b0b"], 0)
        g_b = float(d["b2w"] @ h_b + d["b2b"])     # scalar
        xi = d["xi"][mask]; ci = d["ci"][mask]
        mqoff = d["mqoff"][mask]; party = d["party"][mask]
        same = (party == sponsor).astype(np.float32)
        opp = ((party != sponsor) & (party != "I")).astype(np.float32)
        gam = d["gamma"]
        z = gam[0] * same + gam[1] * opp + gam[2] * mqoff
        logit = xi @ g_a + g_b + ci + z
        logit = logit / float(d["temperature"]) + float(d["bias"])
        return 1 / (1 + np.exp(-logit))

    # ---- parity vs torch ----
    import importlib
    app = importlib.import_module("dashboard_app")
    app.load()
    probes = [
        ("To amend the Internal Revenue Code to expand the child tax credit "
         "and provide disaster tax relief, and for other purposes.", "R"),
        ("To require universal background checks for firearm sales and "
         "establish a national red-flag standard.", "D"),
        ("A bill to designate the facility of the United States Postal "
         "Service as the John Smith Post Office Building.", "D")]
    worst = 0.0
    for text, sp in probes:
        for chamber in ("House", "Senate"):
            rc_text = ("On Passage. " + " ".join(text.split()))[:1500]
            emb = app.STATE["encoder"].encode([rc_text],
                                              normalize_embeddings=True)[0]
            r = app.predict_bill(text, chamber, sp)  # torch path
            # order-independent parity: compare sorted probability vectors
            # (rounded-x ties sort differently on the two sides, which is a
            # display detail, not a computation difference)
            torch_p = np.sort(np.array([m["p"] for m in r["members"]]))
            np_out = np.sort(numpy_predict(emb, chamber, sp))
            diff = float(np.max(np.abs(np_out - torch_p)))
            worst = max(worst, diff)
            print(f"  {chamber:6s} sp={sp}  max|Δp|={diff:.2e}  "
                  f"(np mean {np_out.mean():.3f} vs torch {torch_p.mean():.3f})")
    print(f"WORST max|Δp| across probes: {worst:.2e}  "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
