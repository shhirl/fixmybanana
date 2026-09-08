# fixmybanana.com — Eval & "How It's Built" Plan

Handoff brief. Everything below was decided in conversation with Shirley on 8 Sep 2026\. Use it as the working spec.

## 1\. Context

**Owner:** Shirley He — data scientist → AI builder; AI products & strategy at SWISS; freelance AI consultant; part-time instructor at Constructor Zurich. Positioning: "the builder who teaches" — translator between technical and business audiences. Goal for this project: attract companies hiring freelance AI consultants by showing judgment about reliability, cost and failure modes, not just the ability to call an API.

**Project:** [www.fixmybanana.com](http://www.fixmybanana.com) — Banana Handstand Detector. User uploads a handstand photo, OpenAI analyses form and rates it on a "banana scale". Built for fun. Site currently has: landing page with the tool, "How it works" section, feedback \+ email form, footer linking to whirleyworld.com (portfolio hub), LinkedIn, Instagram, GitHub (@shhirl). Domain on Cloudflare; other projects on Railway.

**Problem:** The site works as a joke but not as evidence. It reads as "photo → OpenAI → score". It needs to show how Shirley thinks.

**Constraint:** Present something as soon as possible. Minimal work, maximum credibility. Anything not done is recorded on the page as a known gap.

## 2\. Decisions

### Site structure

- `fixmybanana.com` stays the tool. Do NOT turn the landing page into a portfolio page.  
- Add one header link: **"How it's built"**.  
- Under the tool result, one line: "Evaluated on N photos · agrees with me X% · see how" (link to the page).  
- `fixmybanana.com/how-its-built` — **one page**, static, hosted next to the tool. No tabs, no blog subdomain, no CMS, no separate eval page, no "About me" (footer link covers it).  
- GitHub repo (public): code, eval script, labelled set (or sample), `CHANGELOG.md`, eval results as CSVs. Linked from the page. The page summarises; the repo has the receipts.  
- `whirleyworld.com`: one project card → links to `/how-its-built`, not the tool.  
- LinkedIn posts link to `/how-its-built` (or a version anchor) with one honest number in the post text.

### Page format: versioned lab notebook

Short intro (what it is, why a banana), then one block per version in chronological order, each with the same four parts:

1. What changed and why (2–3 sentences)  
2. Screenshot or code snippet (prompt, schema, raw response)  
3. Eval on the **fixed** test set: pass rate, ±1-banana agreement, score spread across 5 runs  
4. Cost per analysis

Anchors per version (`#v0`, `#v1`, …) so posts can link to a specific step. Close with a "Not done / next" section, kept current.

Ship the page at v0 \+ v2 (baseline \+ eval). Each later version becomes a new LinkedIn post.

### Version roadmap

| Version | Content |
| :---- | :---- |
| v0 | First draft as it exists today. Free-text prompt. Screenshots \+ verbatim prompt \+ raw response. Run the eval on it anyway — the bad baseline is the point. |
| v1 | Structured output: force a JSON schema (score, primary issue, confidence, reasoning). Same photos; show what got more consistent and what didn't. |
| v2 | Hand-labelled test set \+ error analysis. How photos were chosen, how labelled, failure taxonomy with counts. Most-read section. |
| v3 | Prompt fixes driven by the taxonomy. Before/after table on the same set. |
| v4 | Model switch (second or cheaper model). Same set, same table, plus cost and latency. Frame as a judgment call. |
| v5 | Monitoring: log score, confidence, latency for real uploads; score-distribution chart over time; weekly re-labelled sample. Can be short with one real chart. |

## 3\. Eval method (v2 core)

1. **Test set: 30–50 photos, deliberately varied, not random.** Sample along known axes: freestanding vs wall-assisted; side vs front view; mirror shots; poor lighting / phone quality; a few "not a handstand" controls. State on the page why each axis is included.  
2. **Hand-label, binary first.** Shirley is the domain expert. Per photo: her own banana score \+ pass/fail on "feedback is correct and usable". Report pass rate and how often the model lands within ±1 banana of her.  
3. **Error analysis before metrics.** Read every failure, group into a taxonomy with counts (e.g. over-scores wall-assisted, misses hip pike from front view, hallucinates alignment). Write evaluators for errors found, not errors imagined.  
4. **Consistency: run each photo 3–5 times, show the spread.** LLM scoring is non-deterministic; temperature 0 doesn't fully fix it and can degrade quality; aggregating runs helps. Chart: "same photo, five scores".  
5. **Fix the top failure mode, re-run, show before/after.** One change, same set, same table. Note eval-run cost (a few hundred calls ≈ cents).

Publish the numbers even if bad. "Agrees with me 68%, fails systematically on front-view photos, here's why" beats a clean 95%.

## 4\. "Not done / next" (list on the page — reads as maturity)

- Second labeller to check Shirley's own consistency (inter-rater)  
- Automated LLM-as-judge — skipped because it needs 100+ labelled examples, validation against human labels, and ongoing maintenance  
- Production sampling of real uploads  
- 100–200 examples per failure mode  
- Frame as: "what I'd do next with a client budget"

## 5\. Immediate to-dos (before touching anything)

- [ ] Screenshot current site: landing, upload, result  
- [ ] Save current prompt and one raw model response verbatim  
- [ ] Tag current commit `v0`  
- [ ] Assemble the 30–50 photo test set along the axes above  
- [ ] Label it  
- [ ] Run v0 eval (5 runs per photo), record results as CSV  
- [ ] Draft `/how-its-built` with v0 \+ v2 blocks and the "Not done" section  
- [ ] Add header link \+ one-line eval summary under the tool result

## 6\. Sources behind the method

- Hamel Husain & Shreya Shankar, "AI Evals: Everything You Need to Know" (hamel.dev/blog/posts/evals-faq) — error analysis first, binary labels, sampling by dimensions, LLM-judge cost/validation  
- "Rating Roulette: Self-Inconsistency in LLM-as-a-Judge Frameworks" (arXiv 2510.27106) — low intra-rater reliability, aggregate runs, temperature-0 trade-off

