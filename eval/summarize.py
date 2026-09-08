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


def summary_json(version):
    """Machine-readable summary consumed by the /how-its-built page (app.py reads eval/results/summary.json)."""
    labels, runs = load(version)
    by_photo = defaultdict(list)
    for r in runs:
        by_photo[r["photo"]].append(r)
    hs = [p for p in labels if labels[p]["is_handstand"] == "yes" and p in by_photo]
    ctrl = [p for p in labels if labels[p]["is_handstand"] == "no" and p in by_photo]
    hs_runs = [r for p in hs for r in by_photo[p]]
    valid = [r for r in hs_runs if r["label"] in ("good form", "banana back")]
    agree = sum(r["label"] == labels[r["photo"]]["shirley_label"] for r in valid)
    maj = {p: Counter(r["label"] for r in by_photo[p]).most_common(1)[0][0] for p in hs}
    maj_agree = sum(maj[p] == labels[p]["shirley_label"] for p in hs)
    consistent = sum(len(set(r["label"] for r in by_photo[p])) == 1 for p in hs)
    unclear = sum(r["label"] not in ("good form", "banana back") for r in hs_runs)
    cost = [float(r["cost_usd"]) for r in runs if r["cost_usd"]]
    lat = [float(r["latency_s"]) for r in runs if r["latency_s"]]
    pt1 = [int(r["prompt_tokens"]) for r in runs if r["n_calls"] == "1" and r["prompt_tokens"]]
    def source_kind(row):
        src = row["source"]
        return "synthetic" if "synthetic" in src else ("derived" if "derived" in src else "real (Commons)")
    axes = {}
    for axis in ("view", "support", "quality", "mirror", "source"):
        groups = defaultdict(list)
        for p in hs:
            groups[source_kind(labels[p]) if axis == "source" else labels[p][axis]].append(p)
        axes[axis] = {v: {"n": len(ps), "agree": sum(maj[p] == labels[p]["shirley_label"] for p in ps)} for v, ps in groups.items()}
    per_photo = [{"photo": p, "id": p[:3], "shirley_score": int(labels[p]["shirley_score_0_10"] or 0),
                  "shirley_label": labels[p]["shirley_label"], "runs": [r["label"] for r in by_photo[p]],
                  "view": labels[p]["view"], "support": labels[p]["support"], "quality": labels[p]["quality"]}
                 for p in sorted(hs, key=lambda p: (int(labels[p]["shirley_score_0_10"] or 0), p))]
    controls = [{"photo": p, "id": p[:3], "note": labels[p]["notes"].split(":", 1)[-1].split(",")[0].strip(),
                 "runs": [r["label"] for r in by_photo[p]]} for p in ctrl]
    inter_n = sum(1 for p in labels if labels[p]["is_handstand"] == "yes")
    inter_agree = sum(labels[p]["shirley_label"] == labels[p]["claude_label"] for p in labels if labels[p]["is_handstand"] == "yes")
    conf = Counter((labels[p]["shirley_label"], labels[p]["claude_label"]) for p in labels if labels[p]["is_handstand"] == "yes")
    # Cohen's kappa for the two labellers, and a 95% Wilson interval on photo-level agreement,
    # so the page can say what a difference between versions would need to be to mean anything.
    import math
    def kappa(conf, n):
        if not n:
            return None
        po = (conf.get(("good form", "good form"), 0) + conf.get(("banana back", "banana back"), 0)) / n
        a_good = (conf.get(("good form", "good form"), 0) + conf.get(("good form", "banana back"), 0)) / n
        b_good = (conf.get(("good form", "good form"), 0) + conf.get(("banana back", "good form"), 0)) / n
        pe = a_good * b_good + (1 - a_good) * (1 - b_good)
        return round((po - pe) / (1 - pe), 2) if pe < 1 else None
    def wilson(k, n, z=1.96):
        if not n:
            return (0, 0)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (round(100 * (c - h)), round(100 * (c + h)))
    model_conf = Counter((labels[p]["shirley_label"], maj[p]) for p in hs)
    ci = wilson(maj_agree, len(hs))
    return {
        "version": version, "n_photos": len(hs) + len(ctrl),
        "majority_ci95": list(ci), "kappa_model_vs_shirley": kappa(model_conf, len(hs)),
        "kappa_interrater": kappa(conf, inter_n), "n_handstands": len(hs), "n_controls": len(ctrl),
        "n_runs": max(len(v) for v in by_photo.values()), "n_analyses": len(runs), "n_hs_runs": len(hs_runs),
        "agreement_pct": round(100 * agree / max(len(valid), 1)), "agreement_n": f"{agree}/{len(valid)}",
        "majority_pct": round(100 * maj_agree / max(len(hs), 1)), "majority_n": f"{maj_agree}/{len(hs)}",
        "consistency_pct": round(100 * consistent / max(len(hs), 1)), "consistency_n": f"{consistent}/{len(hs)}",
        "unclear_runs": unclear, "cost_per_analysis": round(sum(cost) / max(len(cost), 1), 4), "total_cost": round(sum(cost), 2),
        "latency_s": round(sum(lat) / max(len(lat), 1), 1), "prompt_tokens_1call": round(sum(pt1) / max(len(pt1), 1)),
        "models": dict(Counter(r["model"] for r in runs)), "axes": axes, "per_photo": per_photo, "controls": controls,
        "interrater_pct": round(100 * inter_agree / max(inter_n, 1)), "interrater_n": f"{inter_agree}/{inter_n}",
        "interrater_confusion": {f"{a}|{b}": n for (a, b), n in conf.items()},
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--md", action="store_true", help="write eval/results/<version>-summary.md")
    ap.add_argument("--json", action="store_true", help="update eval/results/summary.json (read by the /how-its-built page)")
    a = ap.parse_args()
    text = summarize(a.version)
    print(text)
    if a.md:
        p = os.path.join(ROOT, "eval", "results", f"{a.version}-summary.md")
        open(p, "w").write(text + "\n")
        print("\nwrote", p)
    if a.json:
        import json
        p = os.path.join(ROOT, "eval", "results", "summary.json")
        allv = json.load(open(p)) if os.path.exists(p) else {}
        allv[a.version] = summary_json(a.version)
        json.dump(allv, open(p, "w"), indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()
