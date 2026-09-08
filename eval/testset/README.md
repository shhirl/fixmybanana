# Test set — what to collect and how to label it (Shirley's job, ~2 hours)

This folder holds the fixed set of photos every version is scored against. It is built **once**, then never changes, so that differences between v0, v1, v3… are caused by the code, not the photos.

## 1. How many, and along which axes

Target **30–50 photos**. Don't pick randomly; cover each axis below on purpose, and note on the page why each axis matters.

| Axis | Values | Why it matters | Aim for |
|---|---|---|---|
| view | `side`, `front`, `back`, `angled` | The prompt says "side-on". Front/back views hide the arch, so the model should be *less* confident there. | ~60% side, rest spread |
| support | `freestanding`, `wall` | Wall handstands often look "straighter" than they are; wall-facing vs back-to-wall differ. | ~half / half |
| quality | `good`, `dark`, `blurry`, `low_res` | Phone photos in gyms are dark and grainy. Does the model still answer, and should it? | ~10 imperfect |
| mirror | `yes`, `no` | Mirror selfies flip the image and add a second body. | 3–5 mirror shots |
| control | `not_handstand` | A cat, a plank, a yoga pose, an empty gym. The model must refuse, not score. | 3–5 controls |
| banana range | scores `0–10` | Need clearly straight, clearly banana, and ambiguous middles. Ambiguous ones are the most informative. | spread across the range |

Sources: your own photos, friends who say yes, your students (ask first), Creative-Commons handstand photos (note the licence in `labels.csv` → `source`).

## 2. File naming

`NNN_<view>_<support>_<short-note>.jpg` — e.g. `007_side_wall_gym-dark.jpg`, `031_front_freestanding_mirror.jpg`, `045_control_cat.jpg`.
Numbers are just order; they never change once assigned. JPEG or PNG, longest side ≤ 1600 px (resize bigger ones — smaller files, cheaper calls, and consistent with what phones upload).

## 3. Labelling — fill `eval/labels.csv`, one row per photo

| column | what to put |
|---|---|
| `photo` | filename exactly as in this folder |
| `view` / `support` / `quality` / `mirror` | the axis values above |
| `is_handstand` | `yes` or `no` (controls are `no`) |
| `shirley_score_0_10` | your banana score: 0 = ruler-straight, 10 = full croissant. Blank for controls. |
| `shirley_label` | `good form` or `banana back` — the v0 binary. Rule of thumb: score ≤ 3 → good form, ≥ 6 → banana back, 4–5 → your call, but write it down. |
| `feedback_usable` | after seeing the model's feedback for this photo: `pass` if a coach would say it's correct and actionable, `fail` otherwise. Leave blank until an eval has run. |
| `notes` | anything a second labeller would need ("hips slightly forward but ribs tucked — borderline") |
| `source` | `own`, `friend-ok`, `student-ok`, or a CC licence + URL |

Label **before** looking at any model output, so the model can't anchor you. Do it in one sitting if you can; consistency matters more than perfection.

## 4. Privacy — decide before pushing

Photos of other people go public with the repo. Options: (a) only photos you have explicit permission for, (b) keep `testset/` out of git (`.gitignore`) and publish just `labels.csv` + results, with 3–5 sample photos in `eval/testset/sample/`. Write the decision in `TODO.md` "Decisions made".

## 5. Done when

- [ ] 30–50 files here, named per §2
- [ ] every file has a row in `eval/labels.csv` with score, label, axes, source
- [ ] privacy decision recorded
- [ ] then run `eval/run_eval.py --version v0 --runs 5` (Phase 2b builds this script)
