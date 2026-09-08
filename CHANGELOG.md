# Changelog

One entry per version shown on `/how-its-built`. Newest first. Each version has a git tag.

## v0 — September 2025 (original build; tag `v0` marks its last commit, 2026-07-23)

The site as it was first built in September–October 2025 and used since, with small updates through July 2026. Free-text prompt, no structured output, no eval.

- One page: upload a photo → verdict card (good form / banana back / error) → shareable score card.
- Two OpenAI chat-completions calls, both with the photo attached:
  1. **Classifier** — temperature 0, max 5 tokens, two text-only few-shot examples, must answer `good form` or `banana back`. Any other answer is shown raw as "unclear".
  2. **Coaching feedback** — only for banana back; temperature 0.3, max 200 tokens, 2–3 sentences on spine, hips, shoulders, body line.
- Models tried in order until one returns 200: `gpt-4o`, `gpt-4-turbo`, `gpt-4-turbo-2024-04-09`.
- Rate limits: 5 uploads/day per IP, 50/day global. Uploads purged after 24h.
- Receipts: `eval/v0/prompt.md` (verbatim prompts), `eval/v0/raw_response.json` (one real response), screenshots in the page itself (`static/how-its-built/`).

### Fixes on 2026-09-08 (not a new version)

- PR #8: log OpenAI status + body per failed call; friendlier user message; `gpt-5.6-terra` and `gpt-5.6-sol` appended as fallbacks.
