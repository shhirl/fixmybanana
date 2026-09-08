# fixmybanana — TODO & decisions log

Living doc. Add new items at the top of each section. Move done items to "Done" with a date.

---

## Now / next

- [ ] **Small PR: `friendly_ai_error` should distinguish OpenAI `insufficient_quota` (no credits → "temporarily unavailable") from a real rate limit ("try again tomorrow").** Check `insufficient_quota` in the 429 body. Side finding from the 2026-09-08 outage: `gpt-5.6-terra` / `gpt-5.6-sol` are valid ids (useful for v4).
- [ ] **`/how-its-built` initiative — started 2026-09-08.** Spec: `docs/plans/fixmybanana-eval-plan.md`; ordered steps: `docs/plans/2026-09-08-execution-plan.md` (Phase 2); v1 JSON schema + prompt: `docs/plans/fixmybanana-prompt-and-schema.md`. Baseline screenshots in `docs/screenshots/2026-09-08-baseline/`; code baseline = tags `v0` / `baseline-2026-09-08`. Phase 1 done. Phase 2a in progress: `eval/v0/prompt.md` + `CHANGELOG.md` done; **Shirley: run `OPENAI_API_KEY=sk-... .venv/bin/python eval/capture_v0.py <photo.jpg>` and commit `eval/v0/raw_response.json`.** Then Phase 2b: Claude builds `eval/run_eval.py`; **Test set fully labelled 2026-09-08: Shirley scored all 47 handstands (labels derived by rule); Claude/Shirley binary agreement 62%.** Next: Shirley creates `.env` with `OPENAI_API_KEY`, Claude runs `eval/run_eval.py --version v0 --runs 5`.
- [ ] **Housekeeping:** remote branch `improve-feedback-response` (v1) is unmerged and superseded by v2 (merged) — safe to delete on GitHub. `brand-and-share` and `improve-feedback-response-v2` are merged and can be deleted too.
- [ ] **Merge PR #7** (`dedupe-footer-link`) — removes the redundant whirleyworld.com footer link ("Built by Shirley He" already links there). One-line change; merging deploys it.
- [ ] **Phone-test the share card on the live site.** Upload a photo → tap "Share it" → native share sheet should open with the score-card PNG attached. Desktop should download the PNG + copy share text. (Client-side JS — never manually tested; costs one OpenAI call + one of the 5/day rate-limit slots.)
- [ ] **Optional safety upgrades discussed 2026-07-23:** smoke test as `test_app.py` + GitHub Action (checks: app imports, `GET /` is 200, all templates render, purge deletes >24h files), branch protection requiring it, and Railway PR preview environments (Railway settings → Environments).
- [ ] **Record demo GIF for the README.** QuickTime or Kap → record the upload→result flow → export GIF → save as `static/demo.gif` → uncomment the image tag in README.md.
- [ ] **Pin the repo on GitHub.** Manual step (no API for profile pins): github.com/shhirl → "Customize your pins" on the profile page → check fixmybanana.

- [ ] **Persistent storage for uploaded images + analysis results.**
  *2026-07-23 note:* uploads now auto-purge after 24h (privacy promise on homepage). If you later pick S3/R2 or a Railway Volume here, either drop the purge or update the homepage privacy note to match.
  Today: images save to local `uploads/` (ephemeral on Railway — wiped on every redeploy). No metadata, no log of who uploaded what.
  Pick one path:
  - **S3 / Cloudflare R2** (cheap, durable, signed URLs for serving back)
  - **Railway Volume** (simplest — just a persistent disk, no code changes beyond mount path)
  - **Don't store at all** — analyze in memory, discard. Privacy-first.
  Pair the storage choice with a metadata table (SQLite for now, Postgres if it grows): `id, timestamp, filename, form_quality, detailed_feedback, ip_or_session`. Lets you actually see usage and revisit edge cases the model got wrong.

- [ ] **Filename collision risk in `uploads/`.**
  `secure_filename()` keeps the original name, so two users uploading `handstand.jpg` overwrite each other. Fix when implementing persistent storage above — prefix with UTC timestamp + short random suffix.

- [ ] **Watch the Formspree free-tier cap (50 submissions/month).**
  If a tweet/post sends a spike, submissions past 50 in a calendar month are rejected by Formspree until reset. Either upgrade ($10/mo for 1k) or swap to Resend (3k/mo free). Submissions past the cap are LOST, not queued.

## Ideas / maybe

- [ ] Show a tiny "you're submission #N" counter on the result page — light social proof.
- [ ] Mobile camera capture: `<input type="file" accept="image/*" capture="environment">` so phone users go straight to camera.
- [ ] If rate-limiter memory backend becomes a problem (e.g., users hitting limits mid-session after a Railway redeploy), swap `storage_uri="memory://"` to Redis. Railway has a Redis plugin.

## Decisions made

- **2026-09-08 — Binary label rule: banana score 0–5 = `good form`, 6–10 = `banana back`.**
  Why: the v0 prompt defines banana back as a *clear* arch, and the v1 schema bands 3–5 as "slight curve" and 6–8 as "proper banana". Putting the cut at 6 keeps the binary eval consistent with both. Shirley scores; the label is derived by script (`eval/testset/README.md` §3).
  How to apply: never hand-edit `shirley_label` without also changing the score, or the two drift. 4–5 photos are tagged "borderline" in `notes`.
- **2026-09-08 — Test-set photos are Creative-Commons images from Wikimedia Commons, committed to the public repo with attribution.**
  Why: the eval must be reproducible by anyone reading `/how-its-built`, and Shirley has no rights to redistribute Google-Images results or photos of students. CC0/CC BY/CC BY-SA/public-domain images can be republished if credited.
  How to apply: every row in `eval/labels.csv` carries licence + author + source URL; the page gets a credits line. Don't add a photo without a licence you can name. Files are capped at 1600 px longest side to keep the repo small (~14 MB for 49 files).
- **2026-07-23 — Branch + PR for behavior changes; direct-to-main only for docs.**
  Why: Railway auto-deploys every push to `main`, so `git push` = "deploy to live site." The healthcheck (`/`) catches boot failures but not 200-but-broken bugs (template/JS breakage). A PR is the deliberate "ready to be live?" checkpoint and keeps main always-deployable.
  How to apply:
  - Touching `app.py` or `templates/` → work on a short-lived branch, open a PR, merge = conscious deploy.
  - README/TODO/typos → direct to main is fine.
  - Something broke anyway → Railway dashboard → roll back to previous deploy (one click).
  - Nice-to-haves not yet done: Railway PR preview environments (Settings → Environments), smoke-test CI (`test_app.py` + GitHub Action) + branch protection.

- **2026-04-26 — GitHub auth via `gh` CLI (HTTPS), not SSH host aliases.**
  Two accounts (`shhirl`, `shirleysbot`) both stored in `gh` keyring. `gh auth setup-git` wires git's HTTPS credentials through `gh`, so the *currently active* `gh` account is the one `git push` uses.
  Switch with `gh auth switch -u shhirl` or `gh auth switch -u shirleysbot`.
  Trade-off: it's global state — easy to forget which account is active. If that bites, revisit `includeIf` in `~/.gitconfig` to auto-pick identity by folder, and/or per-account SSH keys with `~/.ssh/config` host aliases.
  This repo also has `git config --local user.name/email` set to shhirl so commit author can't drift even if `gh` is on the wrong account.

- **2026-04-26 — Feedback handled by Formspree (form ID `mqewoaln`), not our backend.**
  Why: zero backend code, built-in spam protection, free tier (50/mo) covers current scale, and submissions land directly in shirley.he09@gmail.com without us touching Railway's ephemeral filesystem. Formspree dashboard doubles as a searchable archive.
  How to apply: the homepage feedback `<form>` posts to `https://formspree.io/f/mqewoaln`. Don't add a backend `/feedback` route — it would just duplicate Formspree. If we ever need to *process* submissions programmatically (auto-tag, run AI on them, etc.), revisit Resend or a backend route. The earlier Resend-vs-webhook-vs-Formspree analysis is in chat history if needed.
  Also: hidden `_next` field redirects back to `/?submitted=1#feedback` so the existing thank-you banner still fires; `_subject` makes the Gmail subject line scannable; `_gotcha` honeypot adds a free spam layer on top of Formspree's built-in.

## Done

- **2026-07-23** — Brand + viral-loop pass: (1) footer on every page — "Built by Shirley He" linking to LinkedIn (`/in/shhirl`), Instagram (`@shirleywhirlhe`), GitHub (`shhirl`), and whirleyworld.com; (2) one-line privacy note on homepage, backed by a real `purge_old_uploads()` that deletes uploads older than 24h on each new upload; (3) shareable banana score card on the result page — canvas-rendered 1080×1080 PNG (verdict + user photo + fixmybanana.com), Web Share API with file support on mobile, download + clipboard fallback on desktop; (4) README rewritten with live link, stack, privacy section, and a commented-out demo-GIF slot.
- **2026-05-12** — Rate-limited `/upload` to 5/day per IP + 50/day globally via `Flask-Limiter`. Belt-and-suspenders with the OpenAI hard cap ($20/day) set in the OpenAI billing dashboard. ProxyFix wraps the WSGI app so Railway's edge proxy doesn't make every request look like one IP. Custom 429 template matches the banana aesthetic. In-memory storage backend (resets on redeploy — fine for now).
- **2026-04-26** — Swapped feedback handling to Formspree. Removed the `/feedback` Flask route, `FEEDBACK_FILE` constant, datetime import, and `feedback.jsonl` from `.gitignore`. Form now posts directly to `https://formspree.io/f/mqewoaln` with `_next`/`_subject`/`_gotcha` hidden fields. Added `maxlength="5000"` on the textarea so users see the limit instead of having it silently truncated server-side.
- **2026-04-26** — Added feedback section to homepage (textarea + required email), `/feedback` POST route, thank-you banner on redirect, `.gitignore` for user data. *(Superseded by Formspree swap above — the route is gone, but the section/UX remains.)*
