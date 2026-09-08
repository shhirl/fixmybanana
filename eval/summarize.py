#!/usr/bin/env python3
"""
Summarise one eval run against Shirley's labels.

  .venv/bin/python eval/summarize.py --version v0            # prints the summary
  .venv/bin/python eval/summarize.py --version v0 --md       # also writes eval/results/v0-summary.md

Numbers reported (all on handstand photos only, unless stated):
  agreement       % of runs whose label equals Shirley's label
  majority agree  % of photos whose majority label (over the runs) equals Shirley's
  within-1        n/a for v0 (binary output); reported once a version returns a 0-10 score
  consistency     % of photos where all runs gave the same label
  controls        how the model handled non-handstand photos (v0 has no "not a handstand" output)
  errors/unclear  runs that returned neither label
  cost/latency    mean per analysis (both calls when banana back)
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(version):
    labels = {r["photo"]: r for r in csv.DictReader(open(os.path.join(ROOT, "eval", "labels.csv"), newline=""))}
    runs = list(csv.DictReader(open(os.path.join(ROOT, "eval", "results", f"{version}.csv"), newline="")))
    return labels, runs


def summarize(version):
    labels, runs = load(version)
    by_photo = defaultdict(list)
    for r in runs:
        by_photo[r["photo"]].append(r)
    out = []
    P = out.append

    hs = [p for p in by_photo if labels[p]["is_handstand"] == "yes"]
    ctrl = [p for p in by_photo if labels[p]["is_handstand"] == "no"]
    hs_runs = [r for p in hs for r in by_photo[p]]
    n_runs = len(hs_runs)
    valid = [r for r in hs_runs if r["label"] in ("good form", "banana back")]
    agree = sum(r["label"] == labels[r["photo"]]["shirley_label"] for r in valid)
    maj_agree, consistent, spread_rows = 0, 0, []
    for p in hs:
        c = Counter(r["label"] for r in by_photo[p])
        maj = c.most_common(1)[0][0]
        maj_agree += maj == labels[p]["shirley_label"]
        consistent += len(c) == 1
        spread_rows.append((p, labels[p]["shirley_score_0_10"], labels[p]["shirley_label"], dict(c)))
    unclear = sum(r["label"] not in ("good form", "banana back") for r in hs_runs)
    cost = [float(r["cost_usd"]) for r in runs if r["cost_usd"]]
    lat = [float(r["latency_s"]) for r in runs if r["latency_s"]]

    P(f"# {version} — eval summary\n")
    P(f"Photos: {len(hs)} handstands + {len(ctrl)} controls · runs per photo: {max(len(v) for v in by_photo.values())} · total analyses: {len(runs)}\n")
    P("| metric | value |\n|---|---|")
    P(f"| run-level agreement with Shirley (handstands) | {agree}/{len(valid)} = **{agree / max(len(valid), 1):.0%}** |")
    P(f"| photo-level (majority of runs) agreement | {maj_agree}/{len(hs)} = **{maj_agree / max(len(hs), 1):.0%}** |")
    P(f"| consistency: same label on every run | {consistent}/{len(hs)} = **{consistent / max(len(hs), 1):.0%}** |")
    P(f"| unclear / error runs (handstands) | {unclear}/{n_runs} |")
    P(f"| mean cost per analysis | ${sum(cost) / max(len(cost), 1):.4f} (total ${sum(cost):.2f}) |")
    P(f"| mean latency per analysis | {sum(lat) / max(len(lat), 1):.1f} s |")
    models = Counter(r["model"] for r in runs)
    P(f"| models that answered | {dict(models)} |\n")

    # confusion on majority labels
    conf = Counter()
    for p in hs:
        maj = Counter(r["label"] for r in by_photo[p]).most_common(1)[0][0]
        conf[(labels[p]["shirley_label"], maj)] += 1
    P("## Confusion (photo-level, Shirley → model majority)\n")
    P("| Shirley \\ model | good form | banana back | other |\n|---|---|---|---|")
    for s in ("good form", "banana back"):
        P(f"| {s} | {conf[(s, 'good form')]} | {conf[(s, 'banana back')]} | {sum(v for k, v in conf.items() if k[0] == s and k[1] not in ('good form', 'banana back'))} |")

    # by axis
    P("\n## Agreement by axis (photo-level majority)\n")
    for axis in ("view", "support", "quality", "mirror"):
        P(f"**{axis}**  ")
        groups = defaultdict(list)
        for p in hs:
            groups[labels[p][axis]].append(p)
        for val, ps in sorted(groups.items()):
            ok = sum(Counter(r["label"] for r in by_photo[p]).most_common(1)[0][0] == labels[p]["shirley_label"] for p in ps)
            P(f"- {val}: {ok}/{len(ps)} ({ok / len(ps):.0%})")
        P("")

    # controls
    P("## Controls (not a handstand) — what did the model say?\n")
    P("| photo | labels over runs |\n|---|---|")
    for p in ctrl:
        P(f"| {p} | {dict(Counter(r['label'] for r in by_photo[p]))} |")

    # disagreements + inconsistent photos, the raw material for error analysis
    P("\n## Photos to read for error analysis (majority disagrees with Shirley, or runs disagree with each other)\n")
    P("| photo | Shirley score | Shirley label | model labels over runs |\n|---|---|---|---|")
    for p, s, sl, c in spread_rows:
        maj = max(c, key=c.get)
        if maj != sl or len(c) > 1:
            P(f"| {p} | {s} | {sl} | {c} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--md", action="store_true", help="write eval/results/<version>-summary.md")
    a = ap.parse_args()
    text = summarize(a.version)
    print(text)
    if a.md:
        p = os.path.join(ROOT, "eval", "results", f"{a.version}-summary.md")
        open(p, "w").write(text + "\n")
        print("\nwrote", p)


if __name__ == "__main__":
    main()
