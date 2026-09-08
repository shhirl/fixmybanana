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

## Visual-change safety net (added 2026-09-08)

The site is about to get a new page and other visual changes. Rules so any change is easy to undo:

- **One visual change per branch/PR.** Branch names: `<topic>` (e.g. `eval-page`, `nav-links`). Never bundle "new page" with "restyle homepage" in one PR — a single `git revert` should undo exactly one thing.
- **Tags mark known-good looks.** `baseline-2026-09-08` = the live site before this work started. After each visual PR merges and deploys OK, tag it: `git tag -a look-YYYY-MM-DD-<topic> -m "..."` then `git push origin --tags`.
- **Screenshots live in `docs/screenshots/YYYY-MM-DD-<label>/`** (see `docs/README.md`). Take a `before-` set before merging a visual PR and an `after-` set once deployed.
- **To see what changed since baseline:** `git diff baseline-2026-09-08 --stat` (or `-- templates/ static/ app.py`).
- **To roll back one file to how it looked:** `git checkout baseline-2026-09-08 -- templates/index.html` → commit on a branch → PR → merge.
- **To undo a whole merged PR:** `git revert -m 1 <merge-commit>` on a branch → PR → merge. (Or Railway dashboard → previous deploy → rollback for an instant, code-free fix; then still revert in git so main matches what's live.)
- **Planning docs live in `docs/plans/`.** The current one is `docs/plans/fixmybanana-eval-plan.md` — read it before touching the eval-page work.

## Session handover

Every session must leave the repo in a state another session can pick up cold:

- **Fresh clone (Shirley wipes the folder after each session and may continue from another machine).** After `git clone https://github.com/shhirl/fixmybanana`, run these before anything else:
  ```bash
  git config --local user.name shhirl
  git config --local user.email 44226442+shhirl@users.noreply.github.com
  gh auth status            # must show shhirl active; else: gh auth switch -u shhirl
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  ```
  Everything the project needs is in git — there is no local-only state to carry over.
- **Start:** `git pull`, `git status`, `gh pr list`, read `TODO.md` → "Now / next", then `docs/plans/2026-09-08-execution-plan.md` (the ordered plan) and any other `docs/plans/*.md`.
- **End:** commit everything (WIP is fine on a branch), push the branch, update `TODO.md` "Now / next" with what's in flight and the exact next step, then push. Nothing may exist only on this machine.
- **Local venv:** `.venv/` (gitignored). Recreate with `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Smoke check: `.venv/bin/python -c "import app; print(app.app.test_client().get('/').status_code)"` → `200`.
- **macOS note:** this app cannot read `~/Downloads` or `~/Desktop` (Files & Folders permission). Files Shirley wants Claude to use must be copied into the repo (`docs/plans/`, `docs/screenshots/`) first, or the permission granted in System Settings → Privacy & Security → Files and Folders.
