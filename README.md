# 🍌 fixmybanana

Is your handstand secretly a banana? Upload a photo and find out.

**👉 Try it live: [www.fixmybanana.com](https://www.fixmybanana.com)**

<!-- TODO(shirley): record the demo GIF and save it as static/demo.gif, then delete this comment.
     Quick recipe: QuickTime or Kap (getkap.co) → record the upload→result flow on the live site
     → export/convert to GIF (Kap exports GIF directly) → save to static/demo.gif → commit.
     The image tag below will light up automatically. -->
<!-- ![Demo: upload a handstand photo, get your banana verdict](static/demo.gif) -->

AI-powered handstand posture analyzer that detects "banana back" (that telltale arched-spine C-shape) from a photo, explains what's off, and hands you a shareable banana score card.

## How it works

1. **Upload** a side-on photo of your handstand
2. **GPT-4o vision** classifies it as *good form* or *banana back*
3. **Banana backs** get specific coaching feedback on spinal alignment, hip position, and body line
4. **Share** your banana score card and challenge your friends

## Stack

- Python 3.11 / Flask, served by gunicorn
- Deployed on [Railway](https://railway.app) with Cloudflare in front
- OpenAI vision API for classification + coaching feedback
- Flask-Limiter rate limiting (5 uploads/day per IP) to bound API costs

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python app.py   # → http://localhost:1010
```

## Privacy

Uploaded photos are sent to OpenAI for analysis and auto-deleted from the server within 24 hours. Nothing is shared or sold.

## File structure

```
fixmybanana/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile               # Railway deployment config
├── railway.json           # Railway build/deploy settings
├── templates/             # HTML templates
│   ├── base.html          # Layout, styles, footer
│   ├── index.html         # Upload page
│   ├── result.html        # Verdict + shareable score card
│   └── 429.html           # Rate-limit page
├── static/                # Images (demo, background, reactions)
├── uploads/               # User uploads (auto-created, purged after 24h)
└── notes/                 # private working notes (separate repo, gitignored)
```

---

Built by [Shirley He](https://www.whirleyworld.com) · [LinkedIn](https://www.linkedin.com/in/shhirl) · [Instagram](https://www.instagram.com/shirleywhirlhe) · [GitHub](https://github.com/shhirl)
