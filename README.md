# 🌍 AI Tour Planner — Gemini + MongoDB + Vercel

Answer 10 quick questions → get a full personalized, day-by-day AI tour plan
(must-visit places, itinerary, stays, food, ₹ budget, packing & safety tips).

**Your two application files:**
- `index.html` — the entire frontend (wizard, results, history, dark mode) in one file
- `api/index.py` — the entire backend (Gemini calls + MongoDB) in one file

**Plus 3 tiny config files Vercel itself needs:**
- `vercel.json` — routes `/api/*` to Python, everything else to `index.html`, and allows 60s function time for Gemini
- `requirements.txt` — Python dependencies (installed by Vercel automatically)
- `README.md` — this file

---

## 🚀 Deploy to Vercel (5 minutes)

### Step 1 — Push to GitHub
Upload these files to a GitHub repository (exact folder structure below):

```
your-repo/
├── index.html
├── vercel.json
├── requirements.txt
└── api/
    └── index.py
```

### Step 2 — Get your keys
| Key | Where to get it | Needed? |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey (free, starts with `AIza...`) | ✅ Required |
| `MONGODB_URI`    | MongoDB Atlas (free M0 cluster, steps below) | Optional — enables cloud saving, share links, ratings & cross-device history |

> ⚠️ Do not paste keys into the code. Vercel Environment Variables are the safe way.

### Step 3 — Import into Vercel
1. Go to https://vercel.com → **Add New → Project** → import your GitHub repo.
2. Framework preset: **Other** (auto-detected).
3. Open **Environment Variables** and add:
   - `GEMINI_API_KEY` = your Gemini key
   - `MONGODB_URI` = your Atlas connection string (optional)
   - `MONGODB_DB` = `ai_tour_planner` (optional, this is the default)
4. Click **Deploy** → you get `https://your-project.vercel.app` 🎉

### (Alternative) Deploy with Vercel CLI
```bash
npm i -g vercel
cd your-repo
vercel          # follow the prompts
vercel --prod
```

---

## 🍃 MongoDB Atlas setup (free, 3 minutes)

1. Sign up at https://www.mongodb.com/cloud/atlas → **Create** a free **M0** cluster.
2. **Database Access** → Add user (e.g. `tourapp`) with a password.
3. **Network Access** → Add IP `0.0.0.0/0` (Vercel functions call from dynamic IPs).
4. **Connect → Drivers** → copy the connection string:
   `mongodb+srv://tourapp:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
5. Paste it as the `MONGODB_URI` environment variable in Vercel.

Nothing else to create — the app auto-creates the `plans` collection and indexes.

---

## 💻 Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...        # Windows: set GEMINI_API_KEY=AIza...
export MONGODB_URI=mongodb+srv://...  # optional
python api/index.py
```
Open http://localhost:5000 — the Flask app serves the frontend AND the API together.

---

## 🔌 API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/api/plan` | Generate a tour plan (Gemini multi-model fallback) and save it |
| `GET`  | `/api/health` | Health check (Gemini + MongoDB status) |
| `GET`  | `/api/plans?limit=12` | Recent saved plans (metadata only) |
| `GET`  | `/api/plan/<share_id>` | Fetch one plan (share link: `?plan=<share_id>`) |
| `DELETE` | `/api/plan/<share_id>` | Delete a plan |
| `POST` | `/api/plan/<share_id>/feedback` | Rate a plan `{"rating": 1}` or `{"rating": -1}` |

Optional env vars: `GEMINI_MODELS` (comma-separated fallback list, default
`gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-2.5-flash, ...`),
`GEMINI_TIMEOUT` (per-model seconds, default 50), `MONGODB_DB`.

If every configured model 404s, the backend **auto-discovers** whichever
`generateContent` models your key actually supports — so it keeps working as
Google renames models.

---

## ✨ Features

- 10-question guided wizard with live "Noted" confirmations, quick-pick chips and validation
- Auto-saved draft — close the tab, come back and **Resume**
- Gemini multi-model fallback (+ auto model discovery) on the server
- Beautifully rendered plan: sections, day blocks, budget tables, bold names
- Day-by-day itinerary cards → tap a day to jump to its full details
- Shareable links (`?plan=ID`) powered by MongoDB — open on any device
- Trip history: offline (this device) + cloud trips ☁️ merged in one screen
- Copy / Download (.md) / Print-to-PDF, 👍👎 ratings, one-tap Regenerate
- Dark mode 🌙, Enter-to-continue, friendly error hints + retry, fully mobile responsive

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "GEMINI_API_KEY is not configured" | Vercel → Settings → Environment Variables → add it, then **Redeploy** |
| Generation takes long / times out | The function uses the max 60s Hobby allows. Retry, or plan fewer days. `maxDuration: 60` is in `vercel.json` |
| 429 / rate limit errors | Free Gemini keys have per-minute limits — wait a minute and retry |
| Share links don't work | `MONGODB_URI` is missing or Atlas Network Access isn't `0.0.0.0/0` |
| Local `pymongo` error | `pip install -r requirements.txt` (dnspython is required for `mongodb+srv://` URIs) |
