# Bill Vote Forecaster — web app

Paste the text or summary of a hypothetical bill and get a
member-by-member vote forecast for the current (119th) Congress, using
the frozen model from *Reading the Bills*. Every number is a model
prediction; nothing observes the future.

## What it does

- Assembles the pasted text into the same rollcall-text template the
  model was trained on (`"On Passage. <text>"`, 1,500-char budget) and
  embeds it with the same frozen MiniLM encoder.
- Scores it through the frozen v2 champion's MiniLM-MLP component tower
  (hash-verified on load). The other blend towers need corpus-level
  TF-IDF context a single pasted bill lacks; the MiniLM tower carries
  most of the blend weight and its solo accuracy is within a point of
  the full blend's.
- Reports: predicted pass/fail and yea count, party breakdowns, the
  cutpoint on the 119th-Congress ideal-point scale, a member-by-member
  scatter of ideal point vs. predicted P(yea) (mirroring the paper's
  Figure 2), and a searchable/sortable member table.
- Flags off-agenda inputs two ways: a similarity-to-floor-agenda chip,
  and a sponsor–coalition mismatch warning (if the model predicts the
  sponsor's own party supporting the bill less than the opposition, the
  direction is unreliable — floor agendas are majority-curated and the
  model never saw off-agenda bills).

## Run locally

```bash
pip install -r requirements-app.txt
python3 dashboard_app.py          # http://127.0.0.1:5001
```

Cold start loads a 218 MB artifact + the MiniLM encoder (~30 s);
predictions are ~0.3 s each thereafter, CPU-only.

## Deploy

The prediction path mutates shared model state under a lock, so it must
run **single-worker** (or be refactored to clone the tower per request).
For a small container host:

```bash
gunicorn --workers 1 --threads 4 --timeout 120 \
  --preload -b 0.0.0.0:$PORT 'dashboard_app:app'
```

`--preload` runs `load()` once before forking. The app also exposes
`GET /healthz` for readiness checks. Suitable for Hugging Face Spaces
(Docker), Fly.io, Render, or any container host with ~1 GB RAM; not
Vercel (Torch/native deps, long-lived process).

## Provenance

Forecasts come from the hash-pinned artifact described in the paper's
dashboard appendix. Member positions are the 119th-Congress ideal points
from the paper's own fits. The frozen model's SHA-256 and data-snapshot
date are recorded in
`Modified Data/results/frozen/prospective_model_v2_meta.json` and
verified on every load.
