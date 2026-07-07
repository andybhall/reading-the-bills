"""Vercel Python entrypoint: serves the UI and the forecast API.

  GET  /            -> the single-page app (HTML embedded below)
  GET  /healthz     -> readiness JSON
  POST /api/predict -> {"text","chamber","sponsor"} -> vote forecast

SELF-CONTAINED, GENERATED FILE -- do not edit. It inlines api/_predictor.py
and index.html; edit those and re-run build/gen_entrypoint.py. Declared as
the deployment entrypoint in pyproject.toml; model files are bundled via
vercel.json includeFiles.
"""
from http.server import BaseHTTPRequestHandler

# ===================== inlined api/_predictor.py =====================
import json
import os
from functools import lru_cache

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MAXLEN = 256
PARTY_FULL = {"D": 100.0, "R": 200.0}


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    d = np.load(os.path.join(MODEL_DIR, "tower.npz"), allow_pickle=True)
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_truncation(max_length=MAXLEN)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "minilm_fp16.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"])
    with open(os.path.join(MODEL_DIR, "rosters.json")) as f:
        rosters = json.load(f)
    train_emb = np.load(os.path.join(MODEL_DIR, "train_emb_sample.npy")
                        ).astype(np.float32)
    return {"d": {k: d[k] for k in d.files}, "tok": tok, "sess": sess,
            "rosters": rosters, "train_emb": train_emb}


def _embed(text):
    """Token embeddings from the fp16 ONNX transformer, then masked mean
    pool + L2 normalize in NumPy (the SentenceTransformer pipeline)."""
    S = _load()
    e = S["tok"].encode(text)
    am = np.array([e.attention_mask], dtype=np.int64)
    te = S["sess"].run(None, {"input_ids": np.array([e.ids], dtype=np.int64),
                              "attention_mask": am})[0][0].astype(np.float32)
    mask = am[0][:, None].astype(np.float32)
    mean = (te * mask).sum(0) / max(mask.sum(), 1e-9)
    return mean / max(np.linalg.norm(mean), 1e-12)


def predict(text, chamber, sponsor):
    S = _load(); d = S["d"]
    rc_text = ("On Passage. " + " ".join(text.split()))[:1500]
    emb = _embed(rc_text)

    # amortized rollcall params g_a (16d), g_b (scalar) from phi
    phi = np.zeros(399, np.float32)
    phi[:384] = emb.astype(np.float32) * 5.0
    phi[384 + int(d["q_idx"])] = 1.0
    phi[384 + int(d["nq"]) + int(d["c_idx"])] = 1.0
    phi[384 + int(d["nq"]) + int(d["nc"]) + (0 if sponsor == "D" else 1)] = 1.0
    g_a = d["a2w"] @ np.maximum(d["a0w"] @ phi + d["a0b"], 0) + d["a2b"]
    g_b = float(d["b2w"] @ np.maximum(d["b0w"] @ phi + d["b0b"], 0) + d["b2b"])

    keys = d["keys"]
    mask = np.array([k.startswith(chamber + ":") for k in keys])
    xi, ci = d["xi"][mask], d["ci"][mask]
    mqoff, party = d["mqoff"][mask], d["party"][mask]
    same = (party == sponsor).astype(np.float32)
    opp = ((party != sponsor) & (party != "I")).astype(np.float32)
    gam = d["gamma"]
    logit = xi @ g_a + g_b + ci + gam[0] * same + gam[1] * opp + gam[2] * mqoff
    logit = logit / float(d["temperature"]) + float(d["bias"])
    p = 1.0 / (1.0 + np.exp(-logit))

    roster = S["rosters"][chamber]
    xs = np.array([r["x"] for r in roster], np.float32)
    members = [{"name": r["name"], "state": r["state"], "party": r["party"],
                "x": r["x"], "p": round(float(pi), 4)}
               for r, pi in zip(roster, p)]
    members.sort(key=lambda m: m["x"])

    # cutpoint: logistic fit of predicted prob on member position
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    b1, b0 = np.polyfit(xs, z, 1)
    cut = -b0 / b1 if abs(b1) > 0.05 else None
    if cut is not None and not (xs.min() <= cut <= xs.max()):
        cut = None

    def pstats(code):
        m = party == {"D": "D", "R": "R", "I": "I"}[code]
        n = int(m.sum())
        return {"n": n, "yea_share": float(p[m].mean()) if n else 0.0,
                "n_yea": int((p[m] > 0.5).sum())}

    n = len(p); n_yea = int((p > 0.5).sum()); thresh = (n // 2) + 1
    d_share = float(p[party == "D"].mean()) if (party == "D").any() else 0.0
    r_share = float(p[party == "R"].mean()) if (party == "R").any() else 0.0
    sponsor_share = d_share if sponsor == "D" else r_share
    other_share = r_share if sponsor == "D" else d_share
    inconsistent = sponsor_share < other_share
    sims = S["train_emb"] @ (emb / np.linalg.norm(emb).clip(1e-9))
    top10 = float(np.sort(sims)[-10:].mean())

    return {
        "chamber": chamber, "sponsor_party": sponsor, "n_members": n,
        "predicted_yea_count": n_yea, "predicted_nay_count": n - n_yea,
        "predicted_yea_share": round(float(p.mean()), 4),
        "predicted_pass": bool(n_yea >= thresh), "pass_threshold": thresh,
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
        "democrats": pstats("D"), "republicans": pstats("R"),
        "independents": pstats("I"), "members": members,
        "x_range": [round(float(xs.min()), 2), round(float(xs.max()), 2)],
    }

# ===================== UI + HTTP handler =====================
PAGE = '<!doctype html><html lang=en><head><meta charset=utf-8>\n<meta name=viewport content="width=device-width,initial-scale=1">\n<title>Bill Vote Forecaster</title>\n<style>\n:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;\n --dem:#3b6fb0;--rep:#c0392b;--ind:#7a7a7a;--yea:#1f9d6b;--nay:#c0392b;\n --card:#fff;--accent:#111}\n*{box-sizing:border-box}\nbody{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;\n color:var(--ink);background:var(--bg);margin:0;line-height:1.5}\n.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 80px}\nh1{font-size:26px;margin:0 0 4px}\n.sub{color:var(--mut);font-size:14px;margin:0 0 22px;max-width:70ch}\n.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;\n padding:18px;margin-bottom:18px}\nlabel{font-size:13px;font-weight:600;display:block;margin-bottom:6px}\ntextarea{width:100%;height:150px;font-size:14px;padding:10px;border:1px solid var(--line);\n border-radius:8px;resize:vertical;font-family:inherit}\n.row{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-top:12px}\n.row>div{flex:1;min-width:150px}\nselect{width:100%;font-size:14px;padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff}\nbutton{font-size:15px;font-weight:600;padding:10px 22px;background:var(--accent);color:#fff;\n border:0;border-radius:8px;cursor:pointer}\nbutton:disabled{opacity:.5;cursor:default}\n.examples{margin-top:10px;font-size:12px;color:var(--mut)}\n.examples a{color:var(--dem);cursor:pointer;text-decoration:underline;margin-right:12px}\n#out{display:none}\n.verdict{display:flex;align-items:center;gap:18px;flex-wrap:wrap}\n.badge{font-size:22px;font-weight:700;padding:8px 18px;border-radius:10px;color:#fff}\n.pass{background:var(--yea)} .fail{background:var(--nay)}\n.count{font-size:15px;color:var(--mut)}\n.count b{color:var(--ink);font-size:19px}\n.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:16px}\n@media(max-width:640px){.grid{grid-template-columns:1fr}}\n.pcard{border:1px solid var(--line);border-radius:10px;padding:12px}\n.pcard .lab{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.03em}\n.pcard .big{font-size:22px;font-weight:700;margin:2px 0}\n.pcard .det{font-size:12px;color:var(--mut)}\n.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:7px}\n.bar>span{display:block;height:100%}\n.warn{background:#fef3c7;border:1px solid #f59e0b;color:#7c5b00;border-radius:10px;\n padding:12px 14px;font-size:13px;margin-top:14px}\n.chip{display:inline-block;font-size:12px;padding:3px 10px;border-radius:20px;font-weight:600}\n.chip.hi{background:#dcfce7;color:#166534}.chip.mid{background:#fef9c3;color:#854d0e}\n.chip.lo{background:#fee2e2;color:#991b1b}\nsvg{width:100%;height:auto;display:block}\n.axlab{font-size:11px;fill:var(--mut)}\n.tools{display:flex;gap:10px;margin:14px 0 8px;flex-wrap:wrap;align-items:center}\n.tools input{padding:7px 10px;border:1px solid var(--line);border-radius:7px;font-size:13px;flex:1;min-width:160px}\ntable{width:100%;border-collapse:collapse;font-size:13px}\nth{text-align:left;padding:7px 8px;border-bottom:2px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap}\nth:hover{color:var(--dem)}\ntd{padding:6px 8px;border-bottom:1px solid var(--line)}\n.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}\n.tbar{position:relative;height:14px;background:var(--line);border-radius:3px;width:120px;display:inline-block;vertical-align:middle}\n.tbar>span{position:absolute;top:0;left:0;height:100%;border-radius:3px}\n.tbar>i{position:absolute;top:-2px;bottom:-2px;left:50%;width:1px;background:#999}\n.pill{font-size:11px;font-weight:700;padding:1px 7px;border-radius:20px;color:#fff}\n.ty{background:var(--yea)}.tn{background:var(--nay)}\n.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:14px}\n.spin{display:inline-block;width:15px;height:15px;border:2px solid #fff;border-top-color:transparent;\n border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}\n@keyframes sp{to{transform:rotate(360deg)}}\n</style></head><body><div class=wrap>\n<h1>Bill Vote Forecaster</h1>\n<p class=sub>Paste a bill\'s summary &mdash; a paragraph or two describing what it\ndoes. A frozen language-model-based model from <i>Reading the Bills</i>\nforecasts how every member of the current (119th) Congress would vote on final\npassage, and where the bill would divide the chamber. Every number is a model\nprediction, not an observed outcome.</p>\n\n<div class=panel>\n<label for=text>Bill text or summary</label>\n<textarea id=text placeholder="A bill to ..."></textarea>\n<div class=examples>Try:\n<a data-ex=tax>a bipartisan tax bill</a>\n<a data-ex=guns>a gun-safety bill</a>\n<a data-ex=border>a border-security bill</a>\n<a data-ex=climate>a climate bill</a></div>\n<div class=examples>Or paste a real one: browse\n<a href="https://www.govtrack.us/congress/bills/browse?sort=-introduced_date"\n target=_blank rel=noopener>recently introduced bills on GovTrack</a>\n(or search <a href="https://www.congress.gov/" target=_blank rel=noopener>Congress.gov</a>),\nopen one, and copy its <b>summary</b> &mdash; the plain-language description, not\nthe full legislative text.</div>\n<div class=row>\n <div><label for=chamber>Chamber</label>\n  <select id=chamber><option>House</option><option>Senate</option></select></div>\n <div><label for=sponsor>Sponsoring party</label>\n  <select id=sponsor><option value=D>Democratic</option>\n  <option value=R>Republican</option></select></div>\n <div style="flex:0 0 auto"><button id=go>Forecast vote</button></div>\n</div>\n</div>\n\n<div id=out>\n <div class=panel>\n  <div class=verdict>\n   <span id=badge class=badge></span>\n   <span class=count><b id=yeacount></b> predicted yea &nbsp;/&nbsp;\n     <b id=naycount></b> nay &nbsp;<span id=passthresh></span></span>\n   <span id=support></span>\n  </div>\n  <div id=warn class=warn style=display:none></div>\n  <div class=grid>\n   <div class=pcard><div class=lab style="color:var(--dem)">Democrats</div>\n     <div class=big id=dbig></div><div class=det id=ddet></div>\n     <div class=bar><span id=dbar style="background:var(--dem)"></span></div></div>\n   <div class=pcard><div class=lab style="color:var(--rep)">Republicans</div>\n     <div class=big id=rbig></div><div class=det id=rdet></div>\n     <div class=bar><span id=rbar style="background:var(--rep)"></span></div></div>\n   <div class=pcard><div class=lab>Cutpoint</div>\n     <div class=big id=cbig></div><div class=det id=cdet></div></div>\n  </div>\n </div>\n\n <div class=panel>\n  <h3 style="margin:0 0 2px;font-size:16px">Predicted vote by member ideology</h3>\n  <div class=det style="color:var(--mut);font-size:13px;margin-bottom:8px">\n   Each dot is one member, placed by first-dimension ideal point\n   (liberal&rarr;conservative) and predicted probability of voting yea. The\n   dashed line is the cutpoint. Hover for names.</div>\n  <div id=plot></div>\n </div>\n\n <div class=panel>\n  <h3 style="margin:0 0 0;font-size:16px">Every member</h3>\n  <div class=tools>\n   <input id=search placeholder="Search member or state...">\n   <span class=det id=tcount></span></div>\n  <div style="overflow-x:auto"><table id=tbl>\n   <thead><tr>\n    <th data-k=name>Member</th><th data-k=party>Party</th>\n    <th data-k=state>State</th><th data-k=x>Ideology</th>\n    <th data-k=p>Predicted P(yea)</th></tr></thead>\n   <tbody></tbody></table></div>\n </div>\n</div>\n\n<div class=foot>\n Forecasts come from the hash-pinned frozen artifact described in the paper\n (MiniLM-tower component), scored for a final-passage rollcall\n in the 119th Congress. The cutpoint is on the 119th-Congress ideal-point\n scale (negative = liberal). The model reads only the pasted text plus the\n sponsoring party; it has never seen the bill. It was trained on plain-language\n bill summaries and reads a short (~250-word) window, so a summary works better\n than raw legislative text. Off-agenda inputs are flagged.\n</div>\n</div>\n\n<script>\nvar EX = {\n tax:"To amend the Internal Revenue Code to expand the child tax credit, increase the standard deduction for working families, restore full expensing for research and development, and provide disaster tax relief, and for other purposes.",\n guns:"To require a background check for every firearm sale, establish a national red-flag standard allowing courts to temporarily remove firearms from persons found to pose a danger, and for other purposes.",\n border:"To provide additional funding for border security infrastructure and personnel, impose new limits on asylum eligibility, expand expedited removal, and require employers to use E-Verify, and for other purposes.",\n climate:"To accelerate the transition to clean energy by extending tax credits for wind, solar, and battery manufacturing, setting a clean electricity standard, and investing in climate resilience, and for other purposes."};\nvar LAST=null, SORT={k:\'x\',dir:1};\ndocument.querySelectorAll(\'.examples a\').forEach(function(a){\n a.onclick=function(){document.getElementById(\'text\').value=EX[a.dataset.ex];};});\n\ndocument.getElementById(\'go\').onclick=async function(){\n var text=document.getElementById(\'text\').value.trim();\n if(text.length<40){alert(\'Please paste at least a sentence of bill text.\');return;}\n var btn=this; btn.disabled=true; btn.innerHTML=\'<span class=spin></span>Forecasting\';\n try{\n  var r=await fetch(\'/api/predict\',{method:\'POST\',headers:{\'content-type\':\'application/json\'},\n    body:JSON.stringify({text:text,chamber:document.getElementById(\'chamber\').value,\n      sponsor:document.getElementById(\'sponsor\').value})});\n  var j=await r.json();\n  if(j.error){alert(j.error);}else{LAST=j;render(j);}\n }catch(e){alert(\'Request failed: \'+e);}\n btn.disabled=false; btn.textContent=\'Forecast vote\';\n};\n\nfunction pct(x){return Math.round(x*100);}\nfunction render(j){\n document.getElementById(\'out\').style.display=\'block\';\n var b=document.getElementById(\'badge\');\n b.textContent=j.predicted_pass?\'Predicted to PASS\':\'Predicted to FAIL\';\n b.className=\'badge \'+(j.predicted_pass?\'pass\':\'fail\');\n document.getElementById(\'yeacount\').textContent=j.predicted_yea_count;\n document.getElementById(\'naycount\').textContent=j.predicted_nay_count;\n document.getElementById(\'passthresh\').textContent=\'(needs \'+j.pass_threshold+\' of \'+j.n_members+\')\';\n var s=j.similarity_to_floor_agenda, cls=s>0.7?\'hi\':(s>0.55?\'mid\':\'lo\'),\n     lab=s>0.7?\'typical of floor agenda\':(s>0.55?\'somewhat unusual\':\'off the floor agenda\');\n document.getElementById(\'support\').innerHTML=\'<span class="chip \'+cls+\'">\'+lab+\'</span>\';\n var w=document.getElementById(\'warn\');\n if(j.extrapolation_warning){w.style.display=\'block\';w.textContent=\'⚠ \'+j.extrapolation_warning;}\n else{w.style.display=\'none\';}\n var d=j.democrats,rp=j.republicans;\n document.getElementById(\'dbig\').textContent=pct(d.yea_share)+\'% yea\';\n document.getElementById(\'ddet\').textContent=d.n_yea+\' of \'+d.n+\' members\';\n document.getElementById(\'dbar\').style.width=pct(d.yea_share)+\'%\';\n document.getElementById(\'rbig\').textContent=pct(rp.yea_share)+\'% yea\';\n document.getElementById(\'rdet\').textContent=rp.n_yea+\' of \'+rp.n+\' members\';\n document.getElementById(\'rbar\').style.width=pct(rp.yea_share)+\'%\';\n document.getElementById(\'cbig\').textContent=(j.cutpoint===null)?\'—\':j.cutpoint;\n document.getElementById(\'cdet\').textContent=j.cutpoint_note;\n drawPlot(j); drawTable(j);\n}\n\nvar SVGNS=\'http://www.w3.org/2000/svg\';\nfunction el(tag,attrs,text){\n var e=document.createElementNS(SVGNS,tag);\n for(var k in attrs){if(attrs[k]!==null&&attrs[k]!==undefined)e.setAttribute(k,attrs[k]);}\n if(text!==undefined)e.textContent=text;\n return e;\n}\nfunction drawPlot(j){\n var W=920,H=340,mL=44,mR=14,mT=14,mB=42, iw=W-mL-mR, ih=H-mT-mB;\n var xr=j.x_range, xmin=xr[0]-0.3, xmax=xr[1]+0.3;\n function sx(x){return mL+(x-xmin)/(xmax-xmin)*iw;}\n function sy(p){return mT+(1-p)*ih;}\n var col={D:\'#3b6fb0\',R:\'#c0392b\',I:\'#7a7a7a\'};\n var svg=el(\'svg\',{viewBox:\'0 0 \'+W+\' \'+H,preserveAspectRatio:\'xMidYMid meet\'});\n [0,.25,.5,.75,1].forEach(function(p){var y=sy(p);\n  svg.appendChild(el(\'line\',{x1:mL,y1:y,x2:W-mR,y2:y,stroke:(p==.5?\'#bbb\':\'#eee\'),\n    \'stroke-width\':1,\'stroke-dasharray\':(p==.5?\'2,2\':null)}));\n  svg.appendChild(el(\'text\',{x:mL-6,y:y+3,\'text-anchor\':\'end\',\'class\':\'axlab\'},\'\'+p));});\n if(j.cutpoint!==null){var cx=sx(j.cutpoint);\n  svg.appendChild(el(\'line\',{x1:cx,y1:mT,x2:cx,y2:mT+ih,stroke:\'#111\',\'stroke-width\':1.5,\'stroke-dasharray\':\'5,3\'}));\n  svg.appendChild(el(\'text\',{x:cx+4,y:mT+11,\'class\':\'axlab\',style:\'fill:#111;font-weight:600\'},\'cutpoint \'+j.cutpoint));}\n j.members.forEach(function(m){\n  var c=el(\'circle\',{cx:sx(m.x).toFixed(1),cy:sy(m.p).toFixed(1),r:3.4,\n    fill:col[m.party],\'fill-opacity\':0.72});\n  c.appendChild(el(\'title\',{},m.name+\' (\'+m.party+\'-\'+m.state+\') — P(yea)=\'+m.p.toFixed(2)));\n  svg.appendChild(c);});\n svg.appendChild(el(\'text\',{x:mL,y:H-8,\'class\':\'axlab\'},\'← more liberal\'));\n svg.appendChild(el(\'text\',{x:W-mR,y:H-8,\'text-anchor\':\'end\',\'class\':\'axlab\'},\'more conservative →\'));\n svg.appendChild(el(\'text\',{x:mL-34,y:mT+ih/2,\'class\':\'axlab\',\'text-anchor\':\'middle\',\n   transform:\'rotate(-90 \'+(mL-34)+\' \'+(mT+ih/2)+\')\'},\'Predicted P(yea)\'));\n var plot=document.getElementById(\'plot\'); plot.innerHTML=\'\'; plot.appendChild(svg);\n}\n\nfunction drawTable(j){\n var q=document.getElementById(\'search\').value.toLowerCase();\n var rows=j.members.filter(function(m){\n  return !q || m.name.toLowerCase().indexOf(q)>=0 || m.state.toLowerCase().indexOf(q)>=0;});\n rows.sort(function(a,b){var k=SORT.k,va=a[k],vb=b[k];\n  if(typeof va===\'string\'){return SORT.dir*va.localeCompare(vb);}\n  return SORT.dir*(va-vb);});\n var col={D:\'#3b6fb0\',R:\'#c0392b\',I:\'#7a7a7a\'};\n var h=\'\';\n rows.forEach(function(m){\n  var yea=m.p>=0.5;\n  h+=\'<tr><td><span class=dot style="background:\'+col[m.party]+\'"></span>\'+m.name+\'</td>\'+\n     \'<td>\'+m.party+\'</td><td>\'+m.state+\'</td><td>\'+m.x.toFixed(2)+\'</td>\'+\n     \'<td><span class=tbar><span style="width:\'+pct(m.p)+\'%;background:\'+(yea?\'var(--yea)\':\'var(--nay)\')+\'"></span><i></i></span> \'+\n     \'<span class="pill \'+(yea?\'ty\':\'tn\')+\'">\'+(yea?\'Yea\':\'Nay\')+\'</span> \'+m.p.toFixed(2)+\'</td></tr>\';});\n document.querySelector(\'#tbl tbody\').innerHTML=h;\n document.getElementById(\'tcount\').textContent=rows.length+\' members\'+(q?\' matching\':\'\');\n}\n\ndocument.getElementById(\'search\').oninput=function(){if(LAST)drawTable(LAST);};\ndocument.querySelectorAll(\'#tbl th\').forEach(function(th){\n th.onclick=function(){var k=th.dataset.k;\n  SORT.dir=(SORT.k===k)?-SORT.dir:1; SORT.k=k; if(LAST)drawTable(LAST);};});\n</script></body></html>\n'


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/healthz":
            return self._json({"ok": True}, 200)
        return self._write(PAGE.encode(), "text/html; charset=utf-8", 200)

    def do_POST(self):
        try:
            import json as _json
            n = int(self.headers.get("content-length") or 0)
            body = _json.loads(self.rfile.read(n) or b"{}")
            text = (body.get("text") or "").strip()
            chamber = body.get("chamber", "House")
            sponsor = body.get("sponsor", "D")
            if len(text) < 40:
                return self._json({"error": "Please paste at least a "
                                            "sentence of bill text."}, 400)
            if chamber not in ("House", "Senate"):
                return self._json({"error": "chamber must be House or "
                                            "Senate"}, 400)
            if sponsor not in ("D", "R"):
                return self._json({"error": "sponsor must be D or R"}, 400)
            self._json(predict(text, chamber, sponsor), 200)
        except Exception as e:
            self._json({"error": f"server error: {type(e).__name__}"}, 500)

    def _json(self, obj, status):
        import json as _json
        self._write(_json.dumps(obj).encode(), "application/json", status)

    def _write(self, b, ctype, status):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
