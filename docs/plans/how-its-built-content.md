# `/how-its-built` — content plan and first-draft copy

Drafted 2026-09-08 from the eval plan, the prompt/schema doc, `eval/`, `CHANGELOG.md` and `TODO.md`. This is the copy for `templates/how_its_built.html`; the template is built from this file, not the other way round. Placeholders look like `{{name}}` and are listed in §"Numbers → placeholders" at the bottom.

## Audience and goal

- Reader: a hiring manager or CTO deciding whether to bring in a freelance AI consultant. They have five minutes and have seen a hundred "I built a GPT wrapper" posts.
- Goal: show judgment about reliability, cost and failure modes on a deliberately small, silly project — the thinking transfers, the banana doesn't.
- Voice: "the builder who teaches". First person, plain, honest, short. Numbers published even when bad. No hype words (no "robust", "cutting-edge", "leverage").

## Page outline

Top of page, under the site header: a one-line versions nav — **v0 · v1 · v2 · v3 · v4 · v5** — where each item is an anchor link (`#v0` … `#v5`). Versions that don't exist yet render greyed out with "soon", not as links. Anchors are on the version H2s below and never change, so LinkedIn posts can link to one step.

### Intro

Purpose: say what the site is and why anyone should read a page about it, in 4–5 sentences.

Draft copy:

> fixmybanana.com is a joke with an API behind it: upload a handstand photo and an OpenAI vision model tells you whether you have "good form" or a "banana back" (the arched spine every beginner has). I built it for fun and used it at handstand class.
>
> Then I asked the question I ask clients: how do I know it works? This page is my answer, kept as a lab notebook. Each version below shows what I changed, what it cost, and how it scored against the same fixed set of {{n_photos}} photos that I labelled by hand. Some of the numbers are bad. That's the point — you can't improve what you haven't measured, and I'd rather show you the measuring than the marketing.

Visual: none. Small text under it: "Code, prompts, test set and every result CSV: github.com/shhirl/fixmybanana."

### How the tool works

Purpose: make the mechanism concrete so the eval numbers later mean something; disclose the prompt verbatim.

Draft copy:

> One upload triggers up to two calls to OpenAI's chat-completions API, both with the photo attached.
>
> 1. **Classifier.** Temperature 0, five output tokens, two text-only examples, told to answer exactly "good form" or "banana back". The app then checks whether the word "banana" or "good" appears in the reply; anything else is shown to the user raw as "unclear".
> 2. **Coaching feedback.** Only if call 1 said banana back: temperature 0.3, up to 200 tokens, 2–3 sentences on spine, hips, shoulders and body line.
>
> Models are tried in order until one returns a 200: `{{v0_model_list}}`. Uploads are rate-limited (5 per IP per day, 50 per day total) and deleted after 24 hours.

Visual: a collapsible block "Show the exact prompts (v0)" containing the classifier system prompt, the two few-shot pairs, the final user message, and the feedback prompt — copied verbatim from `eval/v0/prompt.md`, with a line saying that file is generated from `app.py` by `eval/capture_v0.py`, so it can't drift from what's live. Second collapsible: "One raw response" from `eval/v0/raw_response.json` once it exists.

### v0 — the first draft, measured anyway  `#v0`

Purpose: the baseline. Establish the four-part block format and show that the "obvious" build has real, measurable problems.

**1. What changed and why.** Draft copy:

> Nothing changed — this is the site as it was first built: a free-text prompt, a string match on the answer, no schema, no eval, no logging of what the model actually said. I ran the eval on it anyway. A bad baseline is more useful than no baseline, because every later version is judged by the same {{n_photos}} photos.

**2. Screenshot or snippet.** Visual: `docs/screenshots/2026-09-08-baseline/result-desktop-working.png` (a healthy v0 result: "Banana Alert!" card plus the AI Analysis box), captioned "v0 result card, 8 Sep 2026". Optionally `home-desktop-top.png` as a small thumbnail beside it. The prompt itself lives in the collapsible above; link to it rather than repeat it.

**3. Eval on the fixed set.** Draft copy:

> Each of the {{n_handstands}} handstand photos was sent through the live classifier code {{n_runs}} times ({{n_analyses}} analyses in total). My hand label is the reference.
>
> - Agrees with me on {{v0_agreement_pct}} of runs; on {{v0_majority_pct}} of photos if you take the majority of the five runs.
> - Gives the same answer on all five runs for only {{v0_consistency_pct}} of photos. Temperature 0 did not make it deterministic.
> - {{v0_unclear_runs}} of {{n_hs_runs}} runs returned neither label — the model wrote "I can't determine the form" and the app showed that raw text as the verdict.
> - Non-handstand controls ({{n_controls}} photos: a plank, a surfer, a statue, an astronaut…): v0 has no way to say "not a handstand", so it {{v0_controls_behaviour}}.
> - Where it fails: {{v0_axis_worst}} (see the axis table).

Visual: a small table with the six metric rows above (source: `eval/results/v0-summary.md`, top table) and a second small table "agreement by axis" (view / support / quality / mirror, from the same file). Link: "full per-run CSV: eval/results/v0.csv".

**4. Cost per analysis.** Draft copy:

> Mean {{v0_cost_per_analysis}} per upload on `{{v0_model_used}}` (about {{v0_prompt_tokens}} input tokens per call, mostly the image), {{v0_latency_s}} s mean latency. The whole eval run cost {{v0_total_cost}}. Prices as of {{prices_as_of}}.

### v1 — structured output  `#v1`  *(placeholder block, greyed "soon")*

Purpose: reserve the anchor and tell the reader what's coming so the sequence makes sense.

Draft copy (one line): "Next: force a JSON schema — usability flag, view, support, 0–10 score, five body segments, confidence with a reason. Same photos, same table. Design already written; not yet wired in."

### v2 — the test set and the error analysis  `#v2`

Purpose: the most-read section. Show how the photos were chosen, how they were labelled, how honest the labels are, and what the failures look like when you actually read them.

**1. What changed and why.** Draft copy:

> No code changed in v2. The change is that there is now a fixed test set and a way to read failures instead of guessing at them. I built the set before looking at any model output, along axes I already knew mattered from teaching handstands — not by random sampling, because random uploads would be 90% side-view gym photos and I'd learn nothing about the edges.

**2. Test set design.** Draft copy: "{{n_photos}} photos: {{n_handstands}} handstands and {{n_controls}} not-a-handstand controls. Chosen along these axes:"

Visual: the axes table (from `eval/testset/README.md` §1, trimmed to three columns):

| Axis | Values (count in set) | Why it's there |
|---|---|---|
| View | side {{n_view_side}}, front {{n_view_front}}, back {{n_view_back}}, angled {{n_view_angled}} | The prompt says "side-on". Front and back views hide the arch; the model should be less sure there, not more. |
| Support | freestanding {{n_support_free}}, wall {{n_support_wall}} | Wall handstands often look straighter than they are. |
| Photo quality | good {{n_quality_good}}, dark {{n_quality_dark}}, low-res {{n_quality_lowres}}, blurry {{n_quality_blurry}} | Real uploads are grainy gym phone shots. Should the model still answer? |
| Mirror | {{n_mirror}} | Mirror selfies flip the image and add a second body. |
| Controls | {{n_controls}} non-handstands | A plank, a surfer, a statue. The model must refuse, not score. |
| Banana range | my scores spread 1–9, thickest at 3–7 | Clear cases are easy; the ambiguous middle is where a tool earns trust. |

Then a disclosure paragraph, not hidden:

> Where the photos came from: {{n_commons}} are Creative-Commons or public-domain images from Wikimedia Commons (licence and author for each in the repo). {{n_derived}} are darkened, blurred or downscaled copies of photos already in the set, to test robustness on the same pose. {{n_synthetic}} are AI-generated, because Commons has almost no wall-assisted or mirror handstands; they're marked `synthetic` in the labels and one of them has anatomy no human could produce — I kept it as a deliberate edge case. No photos of students or friends: the set is public and reproducible, so it only contains images I'm allowed to republish.

Visual: a contact sheet of the set — `docs/screenshots/eval-review/testset-sheet-1.jpg` (or a fresh 6×9 thumbnail grid generated from `eval/testset/`), with synthetic ones outlined.

**3. Labelling.** Draft copy:

> I scored every handstand 0–10 (0 = ruler, 10 = croissant) before seeing any model output. The binary label is derived by a fixed rule — 0–5 is good form, 6–10 is banana back — so it can't drift from the score. The cut sits at 6 because the v0 prompt defines banana back as a *clear* arch and the v1 schema calls 3–5 a "slight curve".
>
> To check my own consistency I had a second labeller (Claude, given the same photos and the same rule) score the set independently. We agreed on the binary label for {{interrater_pct}} of handstands ({{interrater_n}}). Most disagreements are photos I scored 4–6 — the borderline the rule is forced to cut through. That number is a ceiling: a model can't reliably agree with me more often than a second careful reader does.

Visual: a small 2×2 confusion table, Shirley × second labeller (computed from `labels.csv`: `shirley_label` vs `claude_label`).

**4. Same photo, five scores.** Draft copy:

> The chart below is the one I'd show a client first. Each row is a photo; the marks are the label from each of the five runs. Rows with mixed colours are photos where the same model, same prompt, same image gave different answers. {{v0_consistency_pct}} of photos were stable. Any single upload on the live site is one draw from this.

Visual: chart "same photo, five scores" — one row per handstand photo, five marks per row (good form / banana back / unclear), sorted by my score, with my label as a marker at the row end. Data: `eval/results/v0.csv` (rows) joined to `labels.csv` (order). Rendered as a static PNG committed to `static/` or as inline SVG; either way, a data table fallback.

**5. Error taxonomy.** Draft copy:

> I read every run that disagreed with me or with itself before writing any of the numbers above. They group into a handful of failure modes; counts are photos (out of {{n_handstands}}), not runs.

Visual: a table — failure mode / count / example photo / what I think causes it. Names to confirm after reading the full run (the first 20 photos suggest all five):

| Failure mode (placeholder name) | Count | Example | Likely cause |
|---|---|---|---|
| Refuses and the app shows the refusal | {{fm_refusal_n}} | 004, 020 | Five-token limit plus string match; no fallback path. |
| Coin-flip on borderline photos (my score 3–5) | {{fm_borderline_n}} | 010, 011, 013, 014 | Prompt has no middle category; the model splits the difference across runs. |
| Under-calls a clear arch on outdoor / silhouette shots | {{fm_undercall_n}} | 006 | Low contrast on the torso; the model reads the leg line, not the spine. |
| Guesses on front / back views instead of hedging | {{fm_frontview_n}} | 03x front | Prompt only describes side view; no way to say "can't judge". |
| Scores a non-handstand | {{fm_control_n}} | 04x controls | No "not a handstand" label exists in v0. |

Each row links to the photo's row in `labels.csv` and its five raw rows in `v0.csv`. Draft closing line: "The next change is the one that removes the biggest row — that's v3, and it will be measured on the same photos."

### What broke in production

Purpose: a short boxed aside — one real outage, one real lesson, framed as the kind of thing a consultant is hired to prevent.

Draft copy:

> On 8 Sep 2026, while I was taking baseline screenshots, every upload returned "All Vision Models Failed". The site had been silently broken for an unknown time. The cause was boring: the OpenAI account had run out of prepaid credits. The real problem was that the code caught the error and threw the body away, so the only visible symptom was a generic message and a fallback loop through four model names that all failed the same way. Fix: add credits (two minutes), log status and response body for each failed call, and show users a message that distinguishes "we're out of quota" from "you're rate-limited". Lesson I'd bring to a client: the cheapest reliability work is making failures say why they failed.

Visual: `docs/screenshots/2026-09-08-baseline/result-desktop-top.png` (the failed verdict) beside `result-desktop-working.png`, captioned "16:xx vs 17:20 the same day". Link to PR #8.

### Not done / next — what I'd do with a client budget

Purpose: reads as maturity; the gaps are named, ordered, and each has a reason it was skipped.

Draft copy:

> Things I know are missing, in the order I'd do them if this were paid work:
>
> - **v1 structured output.** A JSON schema so the model has to declare the view, whether the photo is usable, a 0–10 score and a confidence with a reason. Removes the "unclear" failure mode by construction. Designed; not wired in.
> - **Fix the top failure mode and re-run** (v3). One change, same set, before/after table. Eval runs cost cents, so there's no excuse not to.
> - **A human second labeller.** The second reader above is an AI; a coach or a second gymnast would be a stronger check on my labels.
> - **100–200 examples per failure mode.** Five examples tell you a failure exists, not how often. Sourcing that many is the expensive part.
> - **Production sampling.** Log score, confidence and latency for real uploads (with consent), re-label a weekly sample, chart the score distribution over time (v5). Right now I only know how the model behaves on my photos, not on yours.
> - **Cheaper or second model** (v4), judged on the same table plus cost and latency, not on vibes.
> - **Skipped on purpose: an automated LLM-as-judge.** It needs a hundred-plus labelled examples to validate against, and it drifts. At this scale, reading every failure myself is cheaper and more honest.

Visual: none.

### Method sources

Purpose: show the method is borrowed from people who do this for a living, not invented for a portfolio.

Draft copy: two bullets, plain links:
- Hamel Husain and Shreya Shankar, "AI Evals: Everything You Need to Know" (hamel.dev) — error analysis before metrics, binary labels first, sample by dimensions, why LLM-judges need validation.
- "Rating Roulette: Self-Inconsistency in LLM-as-a-Judge Frameworks" (arXiv 2510.27106) — why the same input gets different scores, why temperature 0 isn't a fix, and why to aggregate runs.

### Photo credits

Purpose: honour the CC licences without a 54-line list on the page.

Draft copy: "Test-set photos are from Wikimedia Commons under CC0, CC BY, CC BY-SA and public-domain licences, plus {{n_synthetic}} AI-generated images. Every photo's author, licence and source page is in `eval/labels.csv` (column `source`) — the credits live with the data so they can't go stale." Link to the file on GitHub. If a Commons contributor asks for on-page attribution, add a collapsible full list generated from the CSV.

### Footer CTA

Draft copy: "I'm Shirley He — I build AI products and teach people how to judge them. More work at whirleyworld.com · say hello on LinkedIn (/in/shhirl) · code on GitHub (@shhirl)." Reuse the existing site footer; this line sits above it.

## Numbers → placeholders

Every `{{placeholder}}` above, where it comes from, and what to do if it isn't there yet. Summary fields refer to `eval/results/v0-summary.md` written by `eval/summarize.py --version v0 --md`; "labels" means counts from `eval/labels.csv`.

| Placeholder | Source | Fallback if unknown |
|---|---|---|
| `n_photos`, `n_handstands`, `n_controls` | labels: rows / `is_handstand`=yes / =no (54 / 47 / 7 today) | use today's counts |
| `n_runs`, `n_analyses`, `n_hs_runs` | summary header line ("runs per photo", "total analyses"); handstand runs = 47 × 5 | 5 / 270 / 235 |
| `v0_agreement_pct` | summary: "run-level agreement with Shirley" | "not yet run" and hide the row |
| `v0_majority_pct` | summary: "photo-level (majority of runs) agreement" | same |
| `v0_consistency_pct` | summary: "consistency: same label on every run" | same |
| `v0_unclear_runs` | summary: "unclear / error runs (handstands)" | same |
| `v0_controls_behaviour` | summary: Controls table, summarised in words ("called all 7 good form", "split") | "not yet run" |
| `v0_axis_worst` | summary: "Agreement by axis", lowest bucket with n ≥ 3 | omit bullet |
| `v0_cost_per_analysis`, `v0_total_cost` | summary: "mean cost per analysis" (partial run today: $0.0049, $0.49) | show partial with "first 20 photos" note |
| `v0_latency_s` | summary: "mean latency per analysis" (partial: 2.8 s) | same |
| `v0_model_used`, `v0_model_list` | summary: "models that answered"; `eval/v0/prompt.md` model list | gpt-4o; list from CHANGELOG |
| `v0_prompt_tokens` | `eval/results/v0.csv` mean `prompt_tokens` for 1-call rows (~994) | ~1,000 |
| `prices_as_of` | `PRICES_AS_OF` in `eval/run_eval.py` | "8 Sep 2026" |
| `n_view_*`, `n_support_*`, `n_quality_*` | labels, handstands only: side 34 / front 9 / back 3 / angled 1; free 41 / wall 6; good 37 / dark 6 / low_res 3 / blurry 1 | use these |
| `n_mirror` | labels: `mirror`=yes among handstands (1) plus mirror controls (1) — see questions | "2 (one is a control)" |
| `n_commons`, `n_derived`, `n_synthetic` | labels `source`: 46 / 3 / 5 | use these |
| `interrater_pct`, `interrater_n` | labels: `shirley_label` = `claude_label` over handstands — **CSV computes 27/47 = 57% today; TODO.md says 66% (31/47)** | see questions; whichever is confirmed |
| `fm_*_n` | count of photos per failure mode after reading the full `v0-summary.md` "photos to read" table | leave the table with "tbc" and ship |

## What makes this credible — checklist for the page

- [ ] Receipts in the repo: verbatim prompt file generated from the live code path, one raw response, every eval CSV, the summary script. The page links to each, not to a summary of them.
- [ ] Fixed test set, built before any model output was read, never edited after the first run. Say so on the page and in `eval/testset/README.md`.
- [ ] Published even if bad: v0 numbers go up as they are; no "preliminary" hedge, no rounding up.
- [ ] Two labellers, with the disagreement rate shown and named as a ceiling on any model's agreement.
- [ ] Synthetic and derived photos disclosed in the set description, marked in the CSV, and outlined on the contact sheet.
- [ ] Licences honoured: every photo's author and licence in `labels.csv`; page credits link to it.
- [ ] Failures read by hand before metrics; taxonomy shows counts and example photo IDs a reader can open.
- [ ] Cost stated per analysis and per eval run, with the price date.
- [ ] Gaps listed with the reason each was skipped, not just "future work".
- [ ] One real production failure described honestly, with the fix.

## Open questions for Shirley (content only)

1. **Inter-rater number.** `labels.csv` today gives 27/47 = 57% Shirley–Claude agreement; TODO.md says 66% (31/47). The 66% may predate the 011/027 corrections and synthetic scores in commit c7f73bd. Which do we publish? (I'd publish whatever the CSV computes at page-build time and say so.)
2. **Mirror axis is thin.** Only one mirror handstand (054, synthetic) plus one mirror control (053). Keep the axis row and say "2, both synthetic", or drop the row and mention mirrors under "Not done"?
3. **Photo 050** (synthetic, implausible anatomy): keep as a deliberate edge case, as the draft assumes?
4. **Naming the second labeller.** The draft says openly that the second labeller is Claude. Comfortable with that, or say "an AI second reader"?
5. **Intro origin story.** "used it at handstand class" — true? Replace with the real reason it exists.
6. **Prompt and raw response on the page verbatim** — including the model list with `gpt-5.6-*` ids? `raw_response.json` isn't captured yet.
7. **Footer CTA wording and whether to mention SWISS / Constructor** (the eval plan mentions both; the draft doesn't).
