# docs/

Planning docs and visual snapshots for fixmybanana. Nothing in here is served by the app.

## plans/

Planning / spec documents in Markdown. One file per initiative, e.g. `fixmybanana-eval-plan.md`.
Keep the plan in the repo (not only in Downloads) so any future session can read it.

## screenshots/

Visual snapshots of the live site, so you can always see what it looked like before a change.

Layout: one folder per snapshot, named `YYYY-MM-DD-<label>/`.

```
docs/screenshots/
└── 2026-09-08-baseline/     # live site before the eval-page work started
    ├── home-desktop.png
    ├── home-mobile.png
    ├── result-desktop.png
    ├── result-mobile.png
    └── 429-desktop.png
```

File naming: `<page>-<viewport>.png` where page is `home`, `result`, `429`, or the new page's name,
and viewport is `desktop` or `mobile`. Any extra shots: `home-desktop-feedback-section.png` etc.

Rules:
- Take a new snapshot folder **before** merging any PR that changes the look (`YYYY-MM-DD-before-<pr-topic>/`),
  and one after it deploys (`YYYY-MM-DD-after-<pr-topic>/`). Then a bad change is a side-by-side, not a memory.
- Keep each image under ~1 MB (export as PNG at 1x, or JPEG for photo-heavy shots). Git is fine with that.
- The `baseline-2026-09-08` git tag is the code counterpart of `2026-09-08-baseline/` screenshots.
