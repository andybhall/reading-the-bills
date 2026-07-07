"""Generate api/predict.py (the Vercel entrypoint) as a single
self-contained module: the portable predictor (_predictor.py) + the UI
HTML (index.html) + the HTTP handler, all inlined. Inlining avoids a
sibling import that Vercel's static tracer fails to bundle. _predictor.py
and index.html remain the editable sources of truth.

Run: python3 webapp_vercel/build/gen_entrypoint.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "index.html").read_text()
predictor_src = (ROOT / "api" / "_predictor.py").read_text()

header = '''"""Vercel Python entrypoint: serves the UI and the forecast API.

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
'''

footer = '''
# ===================== UI + HTTP handler =====================
PAGE = ''' + repr(html) + '''


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
'''

# strip the predictor's own module docstring (keep its imports + code)
body = predictor_src
if body.startswith('"""'):
    body = body[body.index('"""', 3) + 3:].lstrip("\n")

(ROOT / "api" / "predict.py").write_text(header + body + footer)
print(f"generated self-contained api/predict.py "
      f"(predictor {len(body)}B + HTML {len(html)}B)")
