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

- [ ] **Feedback inbox.**
  `feedback.jsonl` is local-only and not committed. On Railway it'll be wiped on redeploy. Decide: (a) email-on-submit via SendGrid/Resend, (b) write to same persistent store as uploads, or (c) post to a Slack/Discord webhook. (c) is the lowest-effort "actually see it" option.

## Ideas / maybe

- [ ] Show a tiny "you're submission #N" counter on the result page — light social proof.
- [ ] Rate-limit `/upload` and `/feedback` (basic per-IP) before this gets discovered by anyone weird.
- [ ] Mobile camera capture: `<input type="file" accept="image/*" capture="environment">` so phone users go straight to camera.

## Decisions made

- **2026-04-26 — Feedback form is a plain HTML POST, not AJAX.**
  Reason: simpler, works without JS, matches the existing form pattern (`/upload`). Trade-off: full page reload on submit, but the redirect with `?submitted=1#feedback` lands the user back at the section with a thank-you banner, which feels fine for a low-frequency action.

- **2026-04-26 — Feedback stored as JSON Lines (`feedback.jsonl`), not a DB.**
  Reason: zero setup, easy to grep/cat. Will migrate to the same store as uploads when that decision is made (see "Persistent storage" above).

- **2026-04-26 — `feedback.jsonl` and `uploads/` are gitignored.**
  Reason: contains user emails and uploaded photos — never goes in the repo.

## Done

- **2026-04-26** — Added feedback section to homepage (textarea + required email), `/feedback` POST route, thank-you banner on redirect, `.gitignore` for user data.
