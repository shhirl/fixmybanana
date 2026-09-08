#!/usr/bin/env python3
"""
Eval runner: scores every photo in eval/testset/ with the SAME classifier code the live site
uses (app.analyze_handstand_posture), N times each, and writes one CSV per version.

  .venv/bin/python eval/run_eval.py --version v0 --runs 5
  .venv/bin/python eval/run_eval.py --version v0 --runs 1 --limit 3      # smoke test, 3 photos
  .venv/bin/python eval/run_eval.py --version v0 --runs 5 --resume       # continue an interrupted run

Reads OPENAI_API_KEY from the environment or from a .env file in the repo root (gitignored).
Output: eval/results/<version>.csv — one row per (photo, run). Never edit by hand.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# USD per 1M tokens (input, output). Update when models/prices change; note the date on the page.
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4-turbo": (10.00, 30.00), "gpt-4-turbo-2024-04-09": (10.00, 30.00),
          "gpt-5.6-terra": (None, None), "gpt-5.6-sol": (None, None)}
PRICES_AS_OF = "2026-09-08 (gpt-4o/4-turbo list prices; gpt-5.6 unknown — fill in)"

FIELDS = ["photo", "run", "version", "timestamp", "model", "label", "form_quality", "raw_text", "feedback",
          "n_calls", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd", "http_status"]


def load_dotenv():
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def score_one(app, path):
    """Run the site's classifier once; return a result row (without photo/run/version)."""
    calls = []
    real_post = app.requests.post

    def recording_post(url, headers=None, json=None, timeout=None):
        t0 = time.time()
        r = real_post(url, headers=headers, json=json, timeout=timeout)
        entry = {"model": json.get("model"), "status": r.status_code, "latency": time.time() - t0}
        if r.status_code == 200:
            entry["usage"] = r.json().get("usage", {})
        calls.append(entry)
        return r

    class _OK:  # skip the per-request /v1/models key check the site does (saves one call per photo)
        status_code = 200
        text = "{}"

    with mock.patch.object(app.requests, "post", recording_post), \
            mock.patch.object(app.requests, "get", lambda *a, **k: _OK()):
        res = app.analyze_handstand_posture(path)

    ok = [c for c in calls if c["status"] == 200]
    model = ok[0]["model"] if ok else (calls[-1]["model"] if calls else "")
    pt = sum(c.get("usage", {}).get("prompt_tokens", 0) for c in ok)
    ct = sum(c.get("usage", {}).get("completion_tokens", 0) for c in ok)
    pin, pout = PRICES.get(model, (None, None))
    cost = round(pt / 1e6 * pin + ct / 1e6 * pout, 6) if pin is not None else ""
    fq = res["form_quality"]
    label = {"good": "good form", "bad": "banana back", "unclear": "unclear", "error": "error"}.get(fq, fq)
    return dict(model=model, label=label, form_quality=fq, raw_text=res["analysis"],
                feedback=res.get("detailed_feedback") or "", n_calls=len(calls),
                latency_s=round(sum(c["latency"] for c in calls), 2), prompt_tokens=pt,
                completion_tokens=ct, cost_usd=cost,
                http_status=";".join(str(c["status"]) for c in calls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="name of this code version, e.g. v0 — becomes results/<version>.csv")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (smoke test)")
    ap.add_argument("--only", default="", help="substring filter on photo filename")
    ap.add_argument("--resume", action="store_true", help="skip (photo, run) pairs already in the CSV")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    a = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (env or .env in repo root)")
    import app  # noqa: E402

    labels = list(csv.DictReader(open(os.path.join(ROOT, "eval", "labels.csv"), newline="")))
    photos = [r["photo"] for r in labels if (not a.only or a.only in r["photo"])]
    if a.limit:
        photos = photos[:a.limit]

    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "eval", "results", f"{a.version}.csv")
    done = set()
    if a.resume and os.path.exists(out_path):
        done = {(r["photo"], int(r["run"])) for r in csv.DictReader(open(out_path, newline=""))}
    write_header = not (a.resume and os.path.exists(out_path))
    f = open(out_path, "a" if a.resume else "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    total_cost, n = 0.0, 0
    for photo in photos:
        path = os.path.join(ROOT, "eval", "testset", photo)
        for run in range(1, a.runs + 1):
            if (photo, run) in done:
                continue
            for attempt in range(3):
                row = score_one(app, path)
                if row["form_quality"] == "error" and "429" in row["http_status"]:
                    print(f"  429 on {photo} run {run}; waiting 30s (attempt {attempt + 1}/3)")
                    time.sleep(30)
                    continue
                break
            row.update(photo=photo, run=run, version=a.version,
                       timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            w.writerow(row); f.flush()
            n += 1
            total_cost += float(row["cost_usd"] or 0)
            print(f"{photo} run {run}: {row['label']:<12} {row['model']} {row['latency_s']}s ${row['cost_usd']}")
            time.sleep(a.sleep)
    f.close()
    print(f"\nwrote {out_path}: {n} new rows, est. cost ${total_cost:.4f} (prices as of {PRICES_AS_OF})")
    print("next: .venv/bin/python eval/summarize.py --version", a.version)


if __name__ == "__main__":
    main()
