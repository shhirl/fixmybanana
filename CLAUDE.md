# fixmybanana

Flask app (Python 3.11, gunicorn) live at www.fixmybanana.com. Hosted on Railway, Cloudflare in front for DNS/proxy. Code at github.com/shhirl/fixmybanana.

## ⚠️ Deploy safety — main IS production

Railway auto-deploys every push to `main`. There is no staging. Follow this workflow:

- **Behavior changes (`app.py`, `templates/`, `static/` used by pages): never push directly to main.** Create a short-lived branch, open a PR, and let Shirley merge — merging is the deploy decision and it's hers.
- **Docs-only changes (README.md, TODO.md, CLAUDE.md, typos): direct to main is fine.**
- Don't push at all unless asked. Committing locally when asked is fine.
- If a deploy breaks the site: Railway dashboard → previous deploy → rollback (one click). The `/` healthcheck already blocks boot-failure deploys from replacing the running one, but NOT 200-but-broken bugs (template/JS errors) — that's what the PR checkpoint is for.

## Conventions

- `TODO.md` is the living decisions log — add new decisions there (dated, with Why / How to apply), move finished work to its Done section.
- Feedback form posts to Formspree (`mqewoaln`) — no backend feedback route; don't add one.
- Uploads are ephemeral by design: `purge_old_uploads()` deletes files >24h old, backing the homepage privacy note. If storage strategy changes, the privacy note must change with it.
- Two GitHub accounts in `gh` keyring (`shhirl`, `shirleysbot`); repo-local git config pins author to shhirl. Check `gh auth status` before pushing.

## Verify before declaring done

No test suite yet. Minimum smoke check: with the venv's python, import `app`, `test_client().get('/')` returns 200, and render each template in `templates/` with representative args (see TODO.md Done 2026-07-23 for the exact checks used).
