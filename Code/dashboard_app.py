"""Bill-forecast web app: paste a bill's text, get a member-by-member
vote forecast for the current (119th) Congress from the frozen model.

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
  logistic fit of predicted votes on member positions -- mirroring how
  cutpoints are estimated from realized votes in the paper.
- Every number is a model forecast for a hypothetical final-passage
  vote; nothing here observes the future.

Deployment: CPU-only, ~0.3s per request after warm start (~30s cold
load of the 218MB artifact + encoder); a single Gunicorn worker on a
small container host is sufficient. The prediction path mutates shared
model state, so it is guarded by a lock and must run single-worker (or
be refactored to clone per request) under a threaded server.
"""

import hashlib
import importlib
import json
import pickle
import threading
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
LOCK = threading.Lock()  # predict_bill mutates shared tower state

PARTY = {100.0: "D", 200.0: "R", 328.0: "I"}


def load():
    meta = json.loads((ART / "prospective_model_v2_meta.json").read_text())
    pkl = ART / "prospective_model_v2.pkl"
    assert hashlib.sha256(pkl.read_bytes()).hexdigest() == meta["pickle_sha256"]
    with open(pkl, "rb") as f:
        blend = pickle.load(f)
    tower = blend.models[0]  # MiniLM-MLP component
    from sentence_transformers import SentenceTransformer
    embed_mod = importlib.import_module("08_embed_bills")
    STATE["tower"] = tower
    STATE["encoder"] = SentenceTransformer(embed_mod.MODEL, device="cpu")
    STATE["max_chars"] = 1500
    STATE["freeze_date"] = meta.get("frozen_utc", "")[:10]
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
    print("web app ready: model + encoder + 119th rosters loaded")


def _display_name(bioname: str) -> tuple[str, str]:
    """(short, full) from Voteview 'LAST, First Middle (Nick)'."""
    parts = str(bioname).split(",")
    last = parts[0].strip().title()
    first = parts[1].strip().split()[0].title() if len(parts) > 1 else ""
    return last, (f"{first} {last}".strip() if first else last)


def predict_bill(text: str, chamber: str, sponsor_party: str) -> dict:
    tower = STATE["tower"]
    roster = STATE["rosters"][chamber]
    rc_text = ("On Passage. " + " ".join(text.split()))[:STATE["max_chars"]]
    emb = STATE["encoder"].encode([rc_text], normalize_embeddings=True)[0]

    key = "119_" + chamber + "_999999"
    bill_type = "hr" if chamber == "House" else "s"
    synth = pd.DataFrame([{"congress": 119, "bill_type": bill_type,
                           "bill_no": 999999.0, "text": rc_text,
                           "sponsor_party": sponsor_party}])
    df = pd.DataFrame({
        "congress": 119, "chamber": chamber, "rollnumber": 999999,
        "icpsr": roster.icpsr, "party_code": roster.party_code,
        "vote_question": "On Passage", "bill_category": "legislation",
        "bill_type": bill_type, "bill_no": 999999.0,
    })
    # sponsor party carries much of a bill's learned direction (the model
    # was trained with it; direction-from-text-alone is weak, a documented
    # limitation) -- inject a synthetic bill row so the sponsor features
    # activate exactly as in training. The tower's _emb_lookup and _bills
    # are shared mutable state, so serialize the whole read-modify-restore.
    with LOCK:
        tower._emb_lookup.loc[key] = emb.astype(np.float32)
        saved_bills = tower._bills
        tower._bills = pd.concat([saved_bills, synth], ignore_index=True)
        try:
            p = np.asarray(tower.predict_proba(df))
        finally:
            tower._bills = saved_bills
            tower._emb_lookup.drop(index=key, inplace=True)

    out = roster[["bioname", "state_abbrev", "party_code", "x"]].copy()
    out["p_yea"] = p

    # cutpoint: logistic fit of predicted probability on member position
    x, q = out.x.to_numpy(), np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    b1, b0 = np.polyfit(x, z, 1)
    # report a cutpoint only if the slope is meaningful AND the crossing
    # lies within the chamber (a logistic fit to near-unanimous
    # predictions can cross 0.5 far outside the member range)
    cut = -b0 / b1 if abs(b1) > 0.05 else None
    if cut is not None and not (x.min() <= cut <= x.max()):
        cut = None

    def party_stats(pc):
        d = out[out.party_code == pc]
        return {"n": int(len(d)),
                "yea_share": float(d.p_yea.mean()) if len(d) else 0.0,
                "n_yea": int((d.p_yea > 0.5).sum())}

    members = []
    for r in out.itertuples():
        short, full = _display_name(r.bioname)
        members.append({
            "name": full, "last": short, "state": r.state_abbrev,
            "party": PARTY.get(r.party_code, "I"),
            "x": round(float(r.x), 3), "p": round(float(r.p_yea), 4)})
    members.sort(key=lambda m: m["x"])

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
    n_yea = int((out.p_yea > 0.5).sum())
    n = int(len(out))
    threshold = (n // 2) + 1  # simple majority of those voting

    return {
        "chamber": chamber,
        "sponsor_party": sponsor_party,
        "n_members": n,
        "predicted_yea_count": n_yea,
        "predicted_nay_count": n - n_yea,
        "predicted_yea_share": round(float(out.p_yea.mean()), 4),
        "predicted_pass": bool(n_yea >= threshold),
        "pass_threshold": threshold,
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
        "democrats": party_stats(100.0),
        "republicans": party_stats(200.0),
        "independents": party_stats(328.0),
        "members": members,
        "x_range": [round(float(out.x.min()), 2), round(float(out.x.max()), 2)],
    }


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Bill Vote Forecaster</title>
<style>
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;
 --dem:#3b6fb0;--rep:#c0392b;--ind:#7a7a7a;--yea:#1f9d6b;--nay:#c0392b;
 --card:#fff;--accent:#111}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
 color:var(--ink);background:var(--bg);margin:0;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--mut);font-size:14px;margin:0 0 22px;max-width:70ch}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:18px;margin-bottom:18px}
label{font-size:13px;font-weight:600;display:block;margin-bottom:6px}
textarea{width:100%;height:150px;font-size:14px;padding:10px;border:1px solid var(--line);
 border-radius:8px;resize:vertical;font-family:inherit}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-top:12px}
.row>div{flex:1;min-width:150px}
select{width:100%;font-size:14px;padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff}
button{font-size:15px;font-weight:600;padding:10px 22px;background:var(--accent);color:#fff;
 border:0;border-radius:8px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.examples{margin-top:10px;font-size:12px;color:var(--mut)}
.examples a{color:var(--dem);cursor:pointer;text-decoration:underline;margin-right:12px}
#out{display:none}
.verdict{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.badge{font-size:22px;font-weight:700;padding:8px 18px;border-radius:10px;color:#fff}
.pass{background:var(--yea)} .fail{background:var(--nay)}
.count{font-size:15px;color:var(--mut)}
.count b{color:var(--ink);font-size:19px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:16px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
.pcard{border:1px solid var(--line);border-radius:10px;padding:12px}
.pcard .lab{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.pcard .big{font-size:22px;font-weight:700;margin:2px 0}
.pcard .det{font-size:12px;color:var(--mut)}
.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:7px}
.bar>span{display:block;height:100%}
.warn{background:#fef3c7;border:1px solid #f59e0b;color:#7c5b00;border-radius:10px;
 padding:12px 14px;font-size:13px;margin-top:14px}
.chip{display:inline-block;font-size:12px;padding:3px 10px;border-radius:20px;font-weight:600}
.chip.hi{background:#dcfce7;color:#166534}.chip.mid{background:#fef9c3;color:#854d0e}
.chip.lo{background:#fee2e2;color:#991b1b}
svg{width:100%;height:auto;display:block}
.axlab{font-size:11px;fill:var(--mut)}
.tools{display:flex;gap:10px;margin:14px 0 8px;flex-wrap:wrap;align-items:center}
.tools input{padding:7px 10px;border:1px solid var(--line);border-radius:7px;font-size:13px;flex:1;min-width:160px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:7px 8px;border-bottom:2px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--dem)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.tbar{position:relative;height:14px;background:var(--line);border-radius:3px;width:120px;display:inline-block;vertical-align:middle}
.tbar>span{position:absolute;top:0;left:0;height:100%;border-radius:3px}
.tbar>i{position:absolute;top:-2px;bottom:-2px;left:50%;width:1px;background:#999}
.pill{font-size:11px;font-weight:700;padding:1px 7px;border-radius:20px;color:#fff}
.ty{background:var(--yea)}.tn{background:var(--nay)}
.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:14px}
.spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;
 border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
</style></head><body><div class=wrap>
<h1>Bill Vote Forecaster</h1>
<p class=sub>Paste the summary or text of a hypothetical bill. A frozen
language-model-based model from <i>Reading the Bills</i> forecasts how every
member of the current (119th) Congress would vote on final passage, and where
the bill would divide the chamber. Every number is a model prediction, not an
observed outcome.</p>

<div class=panel>
<label for=text>Bill text or summary</label>
<textarea id=text placeholder="A bill to ..."></textarea>
<div class=examples>Try:
<a data-ex=tax>a bipartisan tax bill</a>
<a data-ex=guns>a gun-safety bill</a>
<a data-ex=border>a border-security bill</a>
<a data-ex=climate>a climate bill</a></div>
<div class=row>
 <div><label for=chamber>Chamber</label>
  <select id=chamber><option>House</option><option>Senate</option></select></div>
 <div><label for=sponsor>Sponsoring party</label>
  <select id=sponsor><option value=D>Democratic</option>
  <option value=R>Republican</option></select></div>
 <div style="flex:0 0 auto"><button id=go>Forecast vote</button></div>
</div>
</div>

<div id=out>
 <div class=panel>
  <div class=verdict>
   <span id=badge class=badge></span>
   <span class=count><b id=yeacount></b> predicted yea &nbsp;/&nbsp;
     <b id=naycount></b> nay &nbsp;<span id=passthresh></span></span>
   <span id=support></span>
  </div>
  <div id=warn class=warn style=display:none></div>
  <div class=grid>
   <div class=pcard><div class=lab style="color:var(--dem)">Democrats</div>
     <div class=big id=dbig></div><div class=det id=ddet></div>
     <div class=bar><span id=dbar style="background:var(--dem)"></span></div></div>
   <div class=pcard><div class=lab style="color:var(--rep)">Republicans</div>
     <div class=big id=rbig></div><div class=det id=rdet></div>
     <div class=bar><span id=rbar style="background:var(--rep)"></span></div></div>
   <div class=pcard><div class=lab>Cutpoint</div>
     <div class=big id=cbig></div><div class=det id=cdet></div></div>
  </div>
 </div>

 <div class=panel>
  <h3 style="margin:0 0 2px;font-size:16px">Predicted vote by member ideology</h3>
  <div class=det style="color:var(--mut);font-size:13px;margin-bottom:8px">
   Each dot is one member, placed by first-dimension ideal point
   (liberal&rarr;conservative) and predicted probability of voting yea. The
   dashed line is the cutpoint. Hover for names.</div>
  <div id=plot></div>
 </div>

 <div class=panel>
  <h3 style="margin:0 0 0;font-size:16px">Every member</h3>
  <div class=tools>
   <input id=search placeholder="Search member or state...">
   <span class=det id=tcount></span></div>
  <div style="overflow-x:auto"><table id=tbl>
   <thead><tr>
    <th data-k=name>Member</th><th data-k=party>Party</th>
    <th data-k=state>State</th><th data-k=x>Ideology</th>
    <th data-k=p>Predicted P(yea)</th></tr></thead>
   <tbody></tbody></table></div>
 </div>
</div>

<div class=foot>
 Forecasts come from the hash-pinned frozen artifact described in the paper
 (MiniLM-tower component), scored for a hypothetical final-passage rollcall
 in the 119th Congress. The cutpoint is on the 119th-Congress ideal-point
 scale (negative = liberal). The model reads only the pasted text plus the
 sponsoring party; it has never seen the bill. Off-agenda inputs are flagged.
</div>
</div>

<script>
var EX = {
 tax:"To amend the Internal Revenue Code to expand the child tax credit, increase the standard deduction for working families, restore full expensing for research and development, and provide disaster tax relief, and for other purposes.",
 guns:"To require a background check for every firearm sale, establish a national red-flag standard allowing courts to temporarily remove firearms from persons found to pose a danger, and for other purposes.",
 border:"To provide additional funding for border security infrastructure and personnel, impose new limits on asylum eligibility, expand expedited removal, and require employers to use E-Verify, and for other purposes.",
 climate:"To accelerate the transition to clean energy by extending tax credits for wind, solar, and battery manufacturing, setting a clean electricity standard, and investing in climate resilience, and for other purposes."};
var LAST=null, SORT={k:'x',dir:1};
document.querySelectorAll('.examples a').forEach(function(a){
 a.onclick=function(){document.getElementById('text').value=EX[a.dataset.ex];};});

document.getElementById('go').onclick=async function(){
 var text=document.getElementById('text').value.trim();
 if(text.length<40){alert('Please paste at least a sentence of bill text.');return;}
 var btn=this; btn.disabled=true; btn.innerHTML='<span class=spin></span>Forecasting';
 try{
  var fd=new FormData();
  fd.append('text',text);
  fd.append('chamber',document.getElementById('chamber').value);
  fd.append('sponsor',document.getElementById('sponsor').value);
  var r=await fetch('/predict',{method:'POST',body:fd});
  var j=await r.json();
  if(j.error){alert(j.error);}else{LAST=j;render(j);}
 }catch(e){alert('Request failed: '+e);}
 btn.disabled=false; btn.textContent='Forecast vote';
};

function pct(x){return Math.round(x*100);}
function render(j){
 document.getElementById('out').style.display='block';
 var b=document.getElementById('badge');
 b.textContent=j.predicted_pass?'Predicted to PASS':'Predicted to FAIL';
 b.className='badge '+(j.predicted_pass?'pass':'fail');
 document.getElementById('yeacount').textContent=j.predicted_yea_count;
 document.getElementById('naycount').textContent=j.predicted_nay_count;
 document.getElementById('passthresh').textContent='(needs '+j.pass_threshold+' of '+j.n_members+')';
 var s=j.similarity_to_floor_agenda, cls=s>0.7?'hi':(s>0.55?'mid':'lo'),
     lab=s>0.7?'typical of floor agenda':(s>0.55?'somewhat unusual':'off the floor agenda');
 document.getElementById('support').innerHTML='<span class="chip '+cls+'">'+lab+'</span>';
 var w=document.getElementById('warn');
 if(j.extrapolation_warning){w.style.display='block';w.textContent='⚠ '+j.extrapolation_warning;}
 else{w.style.display='none';}
 var d=j.democrats,rp=j.republicans;
 document.getElementById('dbig').textContent=pct(d.yea_share)+'% yea';
 document.getElementById('ddet').textContent=d.n_yea+' of '+d.n+' members';
 document.getElementById('dbar').style.width=pct(d.yea_share)+'%';
 document.getElementById('rbig').textContent=pct(rp.yea_share)+'% yea';
 document.getElementById('rdet').textContent=rp.n_yea+' of '+rp.n+' members';
 document.getElementById('rbar').style.width=pct(rp.yea_share)+'%';
 if(j.cutpoint===null){document.getElementById('cbig').textContent='—';
  document.getElementById('cdet').textContent=j.cutpoint_note;}
 else{document.getElementById('cbig').textContent=j.cutpoint;
  document.getElementById('cdet').textContent=j.cutpoint_note;}
 drawPlot(j); drawTable(j);
}

var SVGNS='http://www.w3.org/2000/svg';
function el(tag,attrs,text){
 var e=document.createElementNS(SVGNS,tag);
 for(var k in attrs){if(attrs[k]!==null&&attrs[k]!==undefined)e.setAttribute(k,attrs[k]);}
 if(text!==undefined)e.textContent=text;
 return e;
}
function drawPlot(j){
 var W=920,H=340,mL=44,mR=14,mT=14,mB=42, iw=W-mL-mR, ih=H-mT-mB;
 var xr=j.x_range, xmin=xr[0]-0.3, xmax=xr[1]+0.3;
 function sx(x){return mL+(x-xmin)/(xmax-xmin)*iw;}
 function sy(p){return mT+(1-p)*ih;}
 var col={D:'#3b6fb0',R:'#c0392b',I:'#7a7a7a'};
 var svg=el('svg',{viewBox:'0 0 '+W+' '+H,preserveAspectRatio:'xMidYMid meet'});
 [0,.25,.5,.75,1].forEach(function(p){var y=sy(p);
  svg.appendChild(el('line',{x1:mL,y1:y,x2:W-mR,y2:y,stroke:(p==.5?'#bbb':'#eee'),
    'stroke-width':1,'stroke-dasharray':(p==.5?'2,2':null)}));
  svg.appendChild(el('text',{x:mL-6,y:y+3,'text-anchor':'end','class':'axlab'},''+p));});
 if(j.cutpoint!==null){var cx=sx(j.cutpoint);
  svg.appendChild(el('line',{x1:cx,y1:mT,x2:cx,y2:mT+ih,stroke:'#111','stroke-width':1.5,'stroke-dasharray':'5,3'}));
  svg.appendChild(el('text',{x:cx+4,y:mT+11,'class':'axlab',style:'fill:#111;font-weight:600'},'cutpoint '+j.cutpoint));}
 j.members.forEach(function(m){
  var c=el('circle',{cx:sx(m.x).toFixed(1),cy:sy(m.p).toFixed(1),r:3.4,
    fill:col[m.party],'fill-opacity':0.72});
  c.appendChild(el('title',{},m.name+' ('+m.party+'-'+m.state+') — P(yea)='+m.p.toFixed(2)));
  svg.appendChild(c);});
 svg.appendChild(el('text',{x:mL,y:H-8,'class':'axlab'},'← more liberal'));
 svg.appendChild(el('text',{x:W-mR,y:H-8,'text-anchor':'end','class':'axlab'},'more conservative →'));
 svg.appendChild(el('text',{x:mL-34,y:mT+ih/2,'class':'axlab','text-anchor':'middle',
   transform:'rotate(-90 '+(mL-34)+' '+(mT+ih/2)+')'},'Predicted P(yea)'));
 var plot=document.getElementById('plot'); plot.innerHTML=''; plot.appendChild(svg);
}

function drawTable(j){
 var q=document.getElementById('search').value.toLowerCase();
 var rows=j.members.filter(function(m){
  return !q || m.name.toLowerCase().indexOf(q)>=0 || m.state.toLowerCase().indexOf(q)>=0;});
 rows.sort(function(a,b){var k=SORT.k,va=a[k],vb=b[k];
  if(typeof va==='string'){return SORT.dir*va.localeCompare(vb);}
  return SORT.dir*(va-vb);});
 var col={D:'#3b6fb0',R:'#c0392b',I:'#7a7a7a'};
 var h='';
 rows.forEach(function(m){
  var yea=m.p>=0.5;
  h+='<tr><td><span class=dot style="background:'+col[m.party]+'"></span>'+m.name+'</td>'+
     '<td>'+m.party+'</td><td>'+m.state+'</td><td>'+m.x.toFixed(2)+'</td>'+
     '<td><span class=tbar><span style="width:'+pct(m.p)+'%;background:'+(yea?'var(--yea)':'var(--nay)')+'"></span><i></i></span> '+
     '<span class="pill '+(yea?'ty':'tn')+'">'+(yea?'Yea':'Nay')+'</span> '+m.p.toFixed(2)+'</td></tr>';});
 document.querySelector('#tbl tbody').innerHTML=h;
 document.getElementById('tcount').textContent=rows.length+' members'+(q?' matching':'');
}

document.getElementById('search').oninput=function(){if(LAST)drawTable(LAST);};
document.querySelectorAll('#tbl th').forEach(function(th){
 th.onclick=function(){var k=th.dataset.k;
  SORT.dir=(SORT.k===k)?-SORT.dir:1; SORT.k=k; if(LAST)drawTable(LAST);};});
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": "tower" in STATE, "freeze_date": STATE.get("freeze_date")})


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    chamber = request.form.get("chamber", "House")
    sponsor = request.form.get("sponsor", "D")
    if len(text) < 40:
        return jsonify({"error": "Please paste at least a sentence of bill text."}), 400
    if chamber not in ("House", "Senate"):
        return jsonify({"error": "chamber must be House or Senate"}), 400
    if sponsor not in ("D", "R"):
        return jsonify({"error": "sponsor must be D or R"}), 400
    return jsonify(predict_bill(text, chamber, sponsor))


if __name__ == "__main__":
    load()
    app.run(host="127.0.0.1", port=5001, debug=False)
