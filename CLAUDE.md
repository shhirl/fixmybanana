# fixmybanana

Flask app (Python 3.11, gunicorn) live at www.fixmybanana.com. Hosted on Railway, Cloudflare in front for DNS/proxy. Code at github.com/shhirl/fixmybanana.

## Private working notes — read first

The decision log, plans, screenshots and the full working rules live in a **private** repo, cloned into `notes/` (gitignored here):

```bash
git clone https://github.com/shhirl/fixmybanana-notes notes
```

Then read `notes/CLAUDE.md` (full rules and session checklists) and `notes/TODO.md` ("Now / next"). Nothing in `notes/` may be copied into this public repo.

## ⚠️ Deploy safety — main IS production

Railway auto-deploys every push to `main`. There is no staging.

- **Behavior changes (`app.py`, `templates/`, `static/` used by pages): never push directly to main.** Branch → PR → Shirley merges; merging is the deploy decision and it's hers.
- **Docs-only changes (README.md, CHANGELOG.md, CLAUDE.md, typos): direct to main is fine.**
- Don't push at all unless asked.
- Rollback: Railway dashboard → previous deploy → rollback, then revert the commit so main matches what's live.

## Conventions

- Feedback form posts to Formspree (`mqewoaln`) — no backend feedback route; don't add one.
- Uploads are ephemeral by design: `purge_old_uploads()` deletes files >24h old, backing the homepage privacy note.
- `eval/` is public on purpose: it is the evidence behind `/how-its-built`. Eval scripts read `OPENAI_API_KEY` from a gitignored `.env` (never commit or print it).
- Repo-local git config pins the author to shhirl; check `gh auth status` before pushing.

## Verify before declaring done

No test suite yet. Minimum smoke check with the venv's python: import `app`, `test_client().get('/')` and `get('/how-its-built')` return 200, and render each template in `templates/` with representative args.
