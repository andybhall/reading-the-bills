"""Bill-forecast dashboard: paste bill text, get a predicted cutpoint and
member-by-member vote predictions for the current (119th) Congress.

    python3 Code/dashboard_app.py          # http://127.0.0.1:5001

How it works (documented in the paper's dashboard appendix):
- The pasted text is assembled into the same rollcall-text template the
  frozen model was trained on ("On Passage. <text>", 1500-char budget)
  and embedded with the same MiniLM encoder.
- Predictions come from the frozen v2 champion's MiniLM-MLP component
  tower (the blend's other towers need corpus-level TF-IDF context that a
  single pasted bill does not have; the MiniLM tower alone carries most
  of the blend weight and its solo accuracy is within a point of the
  blend's).
- The cutpoint is the position on the 119th-Congress ideal-point scale at
  which a member's predicted probability crosses one half, from a
  logistic fit of predicted votes on member positions — mirroring how
  cutpoints are estimated from realized votes in the paper.
- Every number is a model forecast for a hypothetical final-passage
  vote; nothing here observes the future.

Deployment note: CPU-only, ~1s per request after warm start; suitable
for a small container host (the model artifact is 222MB and loads once).
"""

import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "Modified Data"
ART = MOD / "results" / "frozen"
MEAS = MOD / "results" / "measures"

app = Flask(__name__)
STATE = {}


def load():
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"]
    with open(pkl, "rb") as f:
        blend = pickle.load(f)
    tower = blend.models[0]  # MiniLM-MLP component
    from sentence_transformers import SentenceTransformer
    import importlib
    embed_mod = importlib.import_module("08_embed_bills")
    STATE["tower"] = tower
    STATE["encoder"] = SentenceTransformer(embed_mod.MODEL, device="cpu")
    STATE["max_chars"] = 1500
    rosters = {}
    for chamber in ("House", "Senate"):
        pos = pd.read_parquet(MEAS / f"members_{chamber.lower()}119.parquet")
        mem = pd.read_parquet(MOD / "members.parquet")
        mem = mem[(mem.congress == 119) & (mem.chamber == chamber)][
            ["icpsr", "party_code", "bioname", "state_abbrev"]]
        rosters[chamber] = pos.merge(mem, on="icpsr", how="inner",
                                     suffixes=("_fit", ""))
    STATE["rosters"] = rosters
    # training-corpus embeddings for the out-of-distribution warning:
    # bills unlike anything that reached the floor get extrapolated
    # predictions, and the user should know (agenda selection is a
    # documented boundary of the instrument)
    E = tower._emb_lookup.to_numpy()
    STATE["train_emb"] = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)
    print("dashboard ready: model + encoder + 119th rosters loaded")


def predict_bill(text: str, chamber: str, sponsor_party: str) -> dict:
    tower = STATE["tower"]
    roster = STATE["rosters"][chamber]
    rc_text = ("On Passage. " + " ".join(text.split()))[:STATE["max_chars"]]
    emb = STATE["encoder"].encode([rc_text], normalize_embeddings=True)[0]

    key = "119_" + chamber + "_999999"
    STATE["tower"]._emb_lookup.loc[key] = emb.astype(np.float32)
    # sponsor party carries much of a bill's learned direction (the model
    # was trained with it; direction-from-text-alone is weak, a documented
    # limitation) — inject a synthetic bill row so the sponsor features
    # activate exactly as in training
    bill_type = "hr" if chamber == "House" else "s"
    synth = pd.DataFrame([{"congress": 119, "bill_type": bill_type,
                           "bill_no": 999999.0, "text": rc_text,
                           "sponsor_party": sponsor_party}])
    saved_bills = tower._bills
    tower._bills = pd.concat([saved_bills, synth], ignore_index=True)

    df = pd.DataFrame({
        "congress": 119, "chamber": chamber, "rollnumber": 999999,
        "icpsr": roster.icpsr, "party_code": roster.party_code,
        "vote_question": "On Passage", "bill_category": "legislation",
        "bill_type": bill_type, "bill_no": 999999.0,
    })
    try:
        p = np.asarray(tower.predict_proba(df))
    finally:
        tower._bills = saved_bills
        STATE["tower"]._emb_lookup.drop(index=key, inplace=True)

    out = roster[["bioname", "state_abbrev", "party_code", "x"]].copy()
    out["p_yea"] = p
    # cutpoint: logistic fit of predicted probability on member position
    from numpy.polynomial import polynomial as P
    x, q = out.x.to_numpy(), np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    b1, b0 = np.polyfit(x, z, 1)[0], np.polyfit(x, z, 1)[1]
    # report a cutpoint only if the slope is meaningful AND the crossing
    # lies within the chamber (a logistic fit to near-unanimous
    # predictions can cross 0.5 far outside the member range)
    cut = -b0 / b1 if abs(b1) > 0.05 else None
    if cut is not None and not (x.min() <= cut <= x.max()):
        cut = None

    def party_stats(pc):
        d = out[out.party_code == pc]
        return {"n": int(len(d)), "yea_share": float(d.p_yea.mean()),
                "n_yea": int((d.p_yea > 0.5).sum())}

    def swing(d):
        return [{"name": f"{str(r.bioname).split(',')[0].title()} ({r.state_abbrev})",
                 "p_yea": round(float(r.p_yea), 3)}
                for r in d.itertuples()]

    closest = out.reindex((out.p_yea - 0.5).abs().sort_values().index).head(12)
    sims = STATE["train_emb"] @ (emb / np.linalg.norm(emb).clip(1e-9))
    top10 = float(np.sort(sims)[-10:].mean())
    d_share = float(out.loc[out.party_code == 100.0, "p_yea"].mean())
    r_share = float(out.loc[out.party_code == 200.0, "p_yea"].mean())
    sponsor_share = d_share if sponsor_party == "D" else r_share
    other_share = r_share if sponsor_party == "D" else d_share
    # self-consistency: a prediction that the sponsor's own party supports
    # the bill less than the opposition almost always means the bill is
    # off the training agenda (floor agendas are majority-curated), and
    # the direction should not be trusted
    inconsistent = sponsor_share < other_share

    return {
        "chamber": chamber,
        "sponsor_party": sponsor_party,
        "similarity_to_floor_agenda": round(top10, 3),
        "extrapolation_warning": (
            "The model predicts the sponsor's own party supporting this "
            "bill less than the opposition — an internal inconsistency "
            "that indicates the bill is unlike those that reached floor "
            "votes in the training window (floor agendas are curated by "
            "the majority). Treat the direction of this forecast as "
            "unreliable." if inconsistent else None),
        "predicted_yea_share": float(out.p_yea.mean()),
        "predicted_yea_count": int((out.p_yea > 0.5).sum()),
        "n_members": int(len(out)),
        "cutpoint": None if cut is None else round(float(cut), 2),
        "cutpoint_note": ("bill divides members at this position on the "
                          "liberal(-)/conservative(+) scale"
                          if cut is not None else
                          "predicted votes barely depend on position "
                          "(lopsided or valence vote)"),
        "democrats": party_stats(100.0),
        "republicans": party_stats(200.0),
        "closest_calls": swing(closest),
    }


PAGE = """<!doctype html><html><head><title>Rollcall Forecaster</title>
<style>
 body{font-family:Helvetica,Arial,sans-serif;max-width:780px;margin:2em auto;color:#222}
 textarea{width:100%;height:160px;font-size:13px}
 button{font-size:15px;padding:6px 18px;margin-top:8px}
 .res{background:#f6f6f6;border-radius:8px;padding:1em;margin-top:1em;white-space:pre-wrap;font-family:monospace;font-size:13px}
 .note{color:#666;font-size:12px}
</style></head><body>
<h2>Rollcall Forecaster</h2>
<p>Paste the text (summary or full text) of a hypothetical bill; the frozen
model from the paper predicts a final-passage vote in the current Congress.</p>
<form id=f>
<textarea name=text placeholder="A bill to ..."></textarea><br>
<select name=chamber><option>House</option><option>Senate</option></select>
<select name=sponsor><option value="D">Democratic sponsor</option>
<option value="R">Republican sponsor</option></select>
<button>Forecast</button>
</form>
<div id=out class=res style="display:none"></div>
<p class=note>Predictions are model forecasts for a hypothetical
final-passage rollcall, from the hash-pinned artifact described in the
paper. Cutpoint scale: 119th-Congress ideal points (negative = liberal).</p>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await fetch('/predict', {method:'POST', body:fd});
  const j = await r.json();
  const o = document.getElementById('out');
  o.style.display = 'block';
  o.textContent = JSON.stringify(j, null, 2);
};
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    chamber = request.form.get("chamber", "House")
    sponsor = request.form.get("sponsor", "D")
    if len(text) < 40:
        return jsonify({"error": "paste at least a sentence of bill text"}), 400
    if sponsor not in ("D", "R"):
        return jsonify({"error": "sponsor must be D or R"}), 400
    return jsonify(predict_bill(text, chamber, sponsor))


if __name__ == "__main__":
    load()
    app.run(host="127.0.0.1", port=5001, debug=False)
