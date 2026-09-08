# 2026-09-08 baseline — what the live site looked like before the eval-page work

Shirley captured six desktop screenshots (Chrome, incognito, ~1900px wide) on 2026-09-08.
Save them in this folder with these names. Until the PNGs are here, this text is the record.

| File | What it shows |
|---|---|
| `home-desktop-top.png` | `/` above the fold: white card on tiled sunglasses-banana background. H1 "🍌 Are You Going Bananas? 🍌", subtitle "The Banana Handstand Detector", dashed upload drop zone ("Drop your handstand photo here / or click to browse"), 🔒 privacy line, green "How It Works" box (3 numbered steps + 💡 accuracy tip). |
| `home-desktop-bottom.png` | `/` scrolled down: yellow-bordered "Loved it? Got ideas? 🍌" feedback box (textarea, email field, purple "↗ Send it" button), then the white footer card "Built by Shirley He · LinkedIn · Instagram · GitHub · whirleyworld.com". Note: whirleyworld.com link is still there (PR #7 removes it, unmerged). |
| `home-desktop-filepicker.png` | macOS file dialog open over `/` (click-to-browse path). Not a site visual; kept for flow completeness. |
| `home-desktop-selected.png` | `/` after choosing a file: preview thumbnail inside the drop zone, "File selected: <name>" + "Click to choose a different file", grey "Selected: <name>" bar, purple "↗ Analyze My Handstand" button. **The page background switches from bananas to a tile of the uploaded photo** — deliberate feature, keep it. |
| `result-desktop-top.png` | `/upload` result: H1 "Handstand Analysis Results", yellow "❓ Analysis Result" box, "📸 Your Handstand Photo" with the uploaded image, tiled-photo background persists. **In this capture the verdict reads "Classification: All Vision Models Failed. Please Try Again." — the live analysis was broken at capture time (see TODO.md).** A healthy result shows good-form / banana-back + feedback instead. |
| `result-desktop-working.png` | `/upload` result **after credits were added** (17:20): yellow "🍌 Banana Alert!" card, "Classification: Banana Back", explanatory sentence, then an "🤖 AI Analysis" box with the 2–3 sentence coaching feedback in italics, then the photo. This is what a healthy v0 result looks like; use it on `/how-its-built` #v0. |
| `result-desktop-bottom.png` | `/upload` scrolled down: "Good form vs banana back" example illustration with caption, purple "↗ 📸 Analyze Another Photo" button, same footer. |

Still missing from this set: mobile viewport shots of `/` and `/upload`, and the `/429` rate-limit page.
