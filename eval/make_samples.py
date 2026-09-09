#!/usr/bin/env python3
"""
Build eval/samples.json: the two "try a sample" photos on the home page and their recorded results.

Results are copied verbatim from eval/results/v0.csv (run 1 of the 8 Sep 2026 eval), so a sample click
shows a real, recorded model answer without making a live API call. Credits come from eval/labels.csv.

  .venv/bin/python eval/make_samples.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = [
    ("banana", "A proper banana", "017_side_freestanding_beach-bikini.jpg"),
    ("straight", "Nice and straight", "001_side_freestanding_studio.jpg"),
]
RUN = "1"


def main():
    with open(os.path.join(ROOT, "eval", "labels.csv"), newline="") as f:
        labels = {r["photo"]: r for r in csv.DictReader(f)}
    with open(os.path.join(ROOT, "eval", "results", "v0.csv"), newline="") as f:
        runs = {(r["photo"], r["run"]): r for r in csv.DictReader(f)}
    out = []
    for sid, title, photo in SAMPLES:
        r, lab = runs[(photo, RUN)], labels[photo]
        credit, _, url = lab["source"].rpartition(", http")
        out.append({
            "id": sid, "title": title, "photo": photo,
            "analysis": r["label"], "form_quality": r["form_quality"],
            "detailed_feedback": r["feedback"] or None,
            "eval_version": r["version"], "run": int(RUN), "model": r["model"],
            "recorded_at": r["timestamp"], "latency_s": float(r["latency_s"]), "cost_usd": float(r["cost_usd"]),
            "shirley_score": int(lab["shirley_score_0_10"]), "shirley_label": lab["shirley_label"],
            "credit": credit, "source_url": "http" + url,
        })
    with open(os.path.join(ROOT, "eval", "samples.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote eval/samples.json ({len(out)} samples)")


if __name__ == "__main__":
    main()
