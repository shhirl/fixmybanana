# fixmybanana — TODO & decisions log

Living doc. Add new items at the top of each section. Move done items to "Done" with a date.

---

## Now / next

- [ ] **Persistent storage for uploaded images + analysis results.**
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

- **2026-05-12** — Rate-limited `/upload` to 5/day per IP + 50/day globally via `Flask-Limiter`. Belt-and-suspenders with the OpenAI hard cap ($20/day) set in the OpenAI billing dashboard. ProxyFix wraps the WSGI app so Railway's edge proxy doesn't make every request look like one IP. Custom 429 template matches the banana aesthetic. In-memory storage backend (resets on redeploy — fine for now).
- **2026-04-26** — Swapped feedback handling to Formspree. Removed the `/feedback` Flask route, `FEEDBACK_FILE` constant, datetime import, and `feedback.jsonl` from `.gitignore`. Form now posts directly to `https://formspree.io/f/mqewoaln` with `_next`/`_subject`/`_gotcha` hidden fields. Added `maxlength="5000"` on the textarea so users see the limit instead of having it silently truncated server-side.
- **2026-04-26** — Added feedback section to homepage (textarea + required email), `/feedback` POST route, thank-you banner on redirect, `.gitignore` for user data. *(Superseded by Formspree swap above — the route is gone, but the section/UX remains.)*
