# Execution plan — fix live analysis, then add the eval page (written 2026-09-08)

Read this first in any new session. It says what to do, in what order, and how each step can be undone.
The detailed *what to build* lives in `fixmybanana-eval-plan.md` in this folder (Shirley's doc).

## Ground rules (from CLAUDE.md)
- `main` = live site. Every code change goes on its own branch → PR → Shirley merges → Railway deploys.
- One topic per PR. Screenshots before and after every visual PR into `docs/screenshots/YYYY-MM-DD-<label>/`.
- After each successful deploy, tag it `look-YYYY-MM-DD-<topic>`. Undo = revert that PR's merge commit.

## Phase 0 — Intake (Shirley, ~5 min)
- [ ] Six baseline PNGs saved in `docs/screenshots/2026-09-08-baseline/` (names in that folder's README), committed, pushed.
- [ ] `fixmybanana-eval-plan.md` copied into `docs/plans/`, committed, pushed.
- [ ] Claude reads the eval plan and turns its requirements into a checklist appended to this file under Phase 2.

## Phase 1 — Fix the broken analysis (do BEFORE adding a page)
Adding a page to a site whose core feature returns "All Vision Models Failed" is pointless; fix this first.
- [ ] Shirley: check OpenAI dashboard → usage and billing limits (the $20/day hard cap is the top suspect).
- [ ] Branch `fix-vision-errors`: in `app.py` log `response.status_code` and the first ~300 chars of `response.text` for each failed model, and show the user a friendlier message than the raw internal string. Update model ids if the failures are 404 "model not found".
- [ ] Verify: local smoke test (`.venv/bin/python -c "import app; ..."` → 200), then one real upload on the PR preview or after merge.
- [ ] PR → merge → confirm a real upload returns a verdict → tag `look-2026-MM-DD-fix-vision`.
- Undo: revert the merge commit. Nothing visual changes in this PR, so no screenshot set needed.

## Phase 2 — Build the new page (per the eval plan)
- [ ] Branch `eval-page`.
- [ ] Add route in `app.py` + new template in `templates/` extending `base.html` (same card-on-tiled-background look, same footer). Keep the upload-preview background-tiling feature untouched.
- [ ] Add a link to the new page from the homepage only if the plan asks for one; if so, that is a separate tiny PR (`nav-link-eval`) so it can be reverted alone.
- [ ] Verify: all templates render, `GET /` and `GET /<new-page>` return 200, rate-limit page still renders.
- [ ] Screenshots: `docs/screenshots/YYYY-MM-DD-before-eval-page/` (the live site) and `.../after-eval-page/` (once deployed), desktop + mobile.
- [ ] PR → Shirley merges → tag `look-YYYY-MM-DD-eval-page`.
- Undo: `git revert -m 1 <merge-commit>` on a branch → PR → merge. Or Railway one-click rollback, then still revert in git.

## Phase 3 — Safety upgrades (optional, after the above)
- [ ] `test_app.py` smoke test + GitHub Action so a PR cannot merge if a template breaks.
- [ ] Railway PR preview environments, so each PR gets its own URL to screenshot before merging.
- [ ] Merge or close PR #7; delete merged remote branches.

## End of every session
Commit, push the branch, update TODO.md "Now / next" with the exact next step, push. Nothing may exist only on this machine.
