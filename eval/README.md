# eval/

Receipts and tooling behind `/how-its-built`. Nothing here is imported by the site; everything here imports the site (`app.py`) so the eval always measures the exact code that is live.

```
eval/
├── capture_v0.py      # writes v0/prompt.md (no key needed) and v0/raw_response.json (needs OPENAI_API_KEY)
├── v0/
│   ├── prompt.md          # verbatim prompts, generated — do not hand-edit
│   └── raw_response.json  # one real request/response pair (image bytes stripped)
├── testset/           # (Phase 2b) 30–50 photos chosen along known axes
├── labels.csv         # (Phase 2b) Shirley's hand labels
├── results/           # (Phase 2c) vN.csv per eval run + analysis notes
└── run_eval.py        # (Phase 2b) runs every testset photo N times through the live classifier code
```

Run anything with the repo venv: `.venv/bin/python eval/<script>.py`. Cost note: one `capture_v0.py` live run = 1–2 API calls; a full eval = photos × runs × (1–2) calls.
