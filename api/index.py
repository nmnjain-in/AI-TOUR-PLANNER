"""
============================================================================
 AI TOUR PLANNER - Backend (Vercel Serverless Function + MongoDB Atlas)
============================================================================
 Deploy this file as:  api/index.py
 (Vercel runs every Python file inside /api as a serverless function -
  this single Flask app powers ALL API routes below.)

 ENVIRONMENT VARIABLES (Vercel -> Settings -> Environment Variables):
   GEMINI_API_KEY   (required) Google Gemini API key
                    get one free: https://aistudio.google.com/app/apikey
   MONGODB_URI      (optional) MongoDB Atlas connection string
                    enables cloud saving, shareable links, history,
                    ratings & view counters. App works fine without it.
   MONGODB_DB       (optional) database name (default: ai_tour_planner)
   GEMINI_MODELS    (optional) comma separated model fallback list
   GEMINI_TIMEOUT   (optional) per-model timeout seconds (default 50)

 API ENDPOINTS:
   GET    /api/health                    -> service health check
   POST   /api/plan                       -> generate a tour plan (Gemini)
   GET    /api/plans                      -> recent saved plans (metadata)
   GET    /api/plan/<share_id>            -> fetch one saved plan
   DELETE /api/plan/<share_id>            -> delete a saved plan
   POST   /api/plan/<share_id>/feedback   -> rate a plan (+1 / -1)

 LOCAL DEVELOPMENT:
   pip install -r requirements.txt
   python api/index.py        ->  http://localhost:5000
   (the Flask app also serves index.html locally, so the full app
    works end-to-end on http://localhost:5000)
============================================================================
"""

import os
import re
import time
import secrets
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify, request, send_file

try:
    from pymongo import MongoClient
    _PYMONGO = True
except Exception:                     # pymongo missing -> app runs without Mongo
    _PYMONGO = False

# ============================== CONFIGURATION ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MONGODB_URI    = os.environ.get("MONGODB_URI", "").strip()
MONGODB_DB     = (os.environ.get("MONGODB_DB", "ai_tour_planner").strip()
                  or "ai_tour_planner")

try:
    GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "50"))
except ValueError:
    GEMINI_TIMEOUT = 50
GEMINI_TIMEOUT = max(10, min(55, GEMINI_TIMEOUT))

# Gemini models tried in order (first that works wins). Override with the
# GEMINI_MODELS env var. If every configured model 404s, the server
# auto-discovers whatever generateContent models the API key supports.
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
GEMINI_MODELS = [m.strip() for m in
                 os.environ.get("GEMINI_MODELS", ",".join(DEFAULT_MODELS)).split(",")
                 if m.strip()]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

ROOT_HTML = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "index.html")

app = Flask(__name__)

# ------------------------------ module caches ------------------------------
_mongo_client  = None
_mongo_indexed = False
_mongo_fail_until = 0.0        # don't retry a dead Atlas for 60s
_discovered    = None          # cached model auto-discovery


# ============================== SMALL HELPERS ==============================

def _txt(value, limit=400):
    """Clean a user supplied string: collapse whitespace, cap length."""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _int(value, default, lo, hi):
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else (dt or None)


def _json(payload, code=200):
    resp = jsonify(payload)
    resp.status_code = code
    return resp


def _parse_dmy(s):
    try:
        return datetime.strptime(str(s).strip(), "%d/%m/%Y")
    except Exception:
        return None


# ============================== MONGODB LAYER ==============================

def _mongo():
    """Return a live Mongo database handle, or None when unavailable.

    Connection is cached per warm lambda instance; failures get a
    60 second cooldown so a dead Atlas does not slow every request.
    """
    global _mongo_client, _mongo_indexed, _mongo_fail_until
    if not _PYMONGO or not MONGODB_URI:
        return None
    now = time.time()
    if _mongo_client is None:
        if now < _mongo_fail_until:
            return None
        try:
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=4000,
                connectTimeoutMS=4000,
                socketTimeoutMS=8000,
                maxPoolSize=4,
                appName="ai-tour-planner",
            )
            client.admin.command("ping")
            _mongo_client = client
        except Exception:
            _mongo_fail_until = now + 60.0
            return None
    try:
        db = _mongo_client[MONGODB_DB]
        if not _mongo_indexed:
            db.plans.create_index([("created_at", -1)])
            _mongo_indexed = True
        return db
    except Exception:
        return None


def _save_plan(doc):
    try:
        db = _mongo()
        if db is None:
            return False
        db.plans.insert_one(dict(doc))
        return True
    except Exception:
        return False


def _plan_meta(doc):
    prof = doc.get("profile") or {}
    return {
        "share_id": str(doc.get("share_id") or doc.get("_id")),
        "destination": prof.get("city") or prof.get("destination") or "Trip",
        "display": (prof.get("travel") or {}).get("display", ""),
        "model": doc.get("model"),
        "rating": doc.get("rating", 0),
        "views": doc.get("views", 0),
        "created_at": _iso(doc.get("created_at")),
        "profile": prof,
    }


# ============================ GEMINI INTEGRATION ============================

def _headers():
    return {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}


def _discover_models():
    """Ask the API which generateContent models this key can actually use."""
    global _discovered
    if _discovered is not None:
        return _discovered
    _discovered = []
    try:
        r = requests.get(f"{GEMINI_BASE}/models?pageSize=200",
                          headers=_headers(), timeout=10)
        if r.status_code == 200:
            names = []
            for m in r.json().get("models", []):
                name = str(m.get("name", "")).replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods and re.match(
                        r"^gemini-[\d.]+-(flash|flash-lite|pro)$", name):
                    names.append(name)

            def key(n):
                nums = re.findall(r"\d+", n)
                major = int(nums[0]) if nums else 0
                minor = int(nums[1]) if len(nums) > 1 else 0
                prefers_flash = 1 if "flash" in n else 0
                return (major, minor, prefers_flash)

            names.sort(key=key, reverse=True)
            _discovered = names[:4]
    except Exception:
        _discovered = []
    return _discovered


def _gemini_once(model, prompt):
    """Call one Gemini model. Returns dict(ok, text|error, status)."""
    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    gen_cfg = {"temperature": 0.7, "topP": 0.95, "maxOutputTokens": 8192}
    # gemini-2.5+ models think by default and burn the time budget -
    # disable it for the fast flash family.
    if re.search(r"gemini-(2\.5|[3-9]\.)", model):
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": gen_cfg}

    def post():
        return requests.post(url, headers=_headers(), json=payload,
                             timeout=GEMINI_TIMEOUT)

    try:
        r = post()
    except requests.exceptions.Timeout:
        return {"ok": False, "status": 0, "error": f"timed out after {GEMINI_TIMEOUT}s"}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)[:200]}

    # some model versions reject thinkingConfig - retry once without it
    if r.status_code == 400 and "thinkingConfig" in gen_cfg:
        gen_cfg.pop("thinkingConfig", None)
        try:
            r = post()
        except Exception as e:
            return {"ok": False, "status": 0, "error": str(e)[:200]}

    if r.status_code != 200:
        return {"ok": False, "status": r.status_code,
                "error": f"HTTP {r.status_code}: {r.text[:180]}"}

    try:
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            fb = (data.get("promptFeedback") or {}).get("blockReason")
            why = f" (blocked: {fb})" if fb else ""
            return {"ok": False, "status": 200,
                    "error": f"no candidates returned{why}"}
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict)).strip()
        if not text:
            return {"ok": False, "status": 200, "error": "empty response"}
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "status": 200, "error": f"parse error: {e}"}


def call_gemini(prompt):
    """Try every configured model (then auto-discovered ones) until one works."""
    if not GEMINI_API_KEY:
        return {"ok": False, "setup": True,
                "error": "GEMINI_API_KEY is not configured on the server."}

    candidates = list(GEMINI_MODELS)
    tried = set()
    discovered_used = False
    last_err = "no model was attempted"
    idx = 0

    while idx < len(candidates):
        model = candidates[idx]
        idx += 1
        if model in tried:
            continue
        tried.add(model)
        res = _gemini_once(model, prompt)
        if res["ok"]:
            return {"ok": True, "model": model, "text": res["text"]}
        last_err = f"{model}: {res['error']}"
        # unknown model -> pull the real model list once, try those too
        if res["status"] == 404 and not discovered_used:
            discovered_used = True
            for m in _discover_models():
                if m not in candidates:
                    candidates.append(m)

    # everything configured failed for other reasons - last resort discovery
    if not discovered_used:
        for m in _discover_models():
            if m in tried:
                continue
            res = _gemini_once(m, prompt)
            if res["ok"]:
                return {"ok": True, "model": m, "text": res["text"]}
            last_err = f"{m}: {res['error']}"

    return {"ok": False, "error": f"All Gemini models failed. Last error: {last_err}"}


# ============================ PROFILE SANITIZING ============================

def sanitize_profile(raw):
    """Normalize + harden any incoming trip profile into a safe dict."""
    raw = raw or {}
    p = {}

    # 1 - city + tour theme
    p["city"] = _txt(raw.get("city"), 80)
    p["destination"] = _txt(raw.get("destination"), 300)

    # 2 - group
    g = raw.get("group") or {}
    gtype = "Group" if str(g.get("type", "")).strip().lower().startswith("g") else "Solo"
    count = _int(g.get("count"), 1, 1, 100)
    if gtype == "Solo":
        count = 1
    p["group_type"] = gtype
    p["count"] = count
    p["group_display"] = f"{gtype} ({count} Traveler{'s' if count > 1 else ''})"

    # 3 - age group
    p["age_group"] = _txt(raw.get("age_group"), 120) or "Mixed Age Group"

    # 4 - duration / calendar dates
    t = raw.get("travel") or {}
    if str(t.get("type", "")).strip().lower() == "dates":
        s, e = _parse_dmy(t.get("start")), _parse_dmy(t.get("end"))
        if s and e and e > s:
            dur = min((e - s).days, 120)
            p["travel"] = {
                "type": "dates",
                "start": f"{s:%d/%m/%Y}", "end": f"{e:%d/%m/%Y}",
                "duration": dur,
                "display": f"{s:%d/%m/%Y} to {e:%d/%m/%Y} "
                           f"({dur} Days with Full Year)",
            }
        else:
            p["travel"] = {"type": "days", "duration": 5, "display": "5 Days Trip"}
    else:
        dur = _int(t.get("duration"), 5, 1, 120)
        p["travel"] = {"type": "days", "duration": dur, "display": f"{dur} Days Trip"}

    # calendar labels (Gemini puts real dates on every day when known)
    p["day_labels"] = []
    if p["travel"]["type"] == "dates":
        s = _parse_dmy(p["travel"]["start"])
        for i in range(p["travel"]["duration"]):
            d = s + timedelta(days=i)
            p["day_labels"].append(f"Day {i + 1} ({d:%a, %d %b %Y})")

    # 5 - travelers
    m = raw.get("members") or {}
    src = m.get("list") or m.get("members") or []
    members = []
    for x in src[:count]:
        gdr = str(_txt(x.get("gender"), 10)).lower()
        gender = ("Male" if gdr.startswith("m") else
                 "Female" if gdr.startswith("f") else "Other")
        members.append({
            "name": _txt(x.get("name"), 60) or f"Traveler {len(members) + 1}",
            "age": _int(x.get("age"), 25, 1, 120),
            "gender": gender,
        })
    while len(members) < count:
        members.append({"name": f"Traveler {len(members) + 1}",
                        "age": 25, "gender": "Other"})
    males = sum(1 for x in members if x["gender"] == "Male")
    females = sum(1 for x in members if x["gender"] == "Female")
    summary = f"\n      - Count: {males} Males, {females} Females"
    for x in members:
        summary += f"\n      - {x['name']} ({x['age']} yrs, {x['gender']})"
    p["members"] = {"males": males, "females": females, "list": members}
    p["members_summary"] = summary

    # 6 - relationship / 7 - budget
    p["relationship"] = _txt(raw.get("relationship"), 160) or "Mixed Group"
    b = raw.get("budget") or {}
    p["budget"] = _txt(b.get("range") if isinstance(b, dict) else b, 200) \
        or "Mid Range"
    amount = re.sub(r"\D", "", str(b.get("amount", "")))[:9] \
        if isinstance(b, dict) else ""
    p["budget_amount"] = amount
    p["budget_display"] = p["budget"] + (f" | Target Total: \u20b9{amount}"
                                         if amount else "")

    # 8 - transport (+ personal vehicle details)
    t = raw.get("transport") or {}
    mode = _txt(t.get("mode") if isinstance(t, dict) else t, 120) or "Personal Vehicle"
    vehicle = _txt(t.get("vehicle"), 12).upper() if isinstance(t, dict) else ""
    if vehicle not in ("CAR", "SUV", "BIK", "VAN"):
        vehicle = ""
    seats = _txt(t.get("seats"), 10) if isinstance(t, dict) else ""
    display = mode
    if vehicle:
        display += f" - [{vehicle}]"
    if seats:
        display += f" (Seater: {seats})"
    p["transport"] = mode
    p["vehicle"] = vehicle
    p["seats"] = seats
    p["transport_display"] = display

    # 9 - food
    f = raw.get("food") or {}
    diet = _txt(f.get("diet") if isinstance(f, dict) else f, 160) or "Pure Vegetarian"
    allergy = _txt(f.get("allergy"), 200) if isinstance(f, dict) else ""
    p["food"] = diet
    p["allergies"] = allergy
    p["food_display"] = diet + (f" | Notes: {allergy}" if allergy else "")

    # 10 - stay + city echo done above
    p["stay"] = _txt(raw.get("stay"), 160) or "Any place is fine (Best budget combo)"

    return p


def validate_profile(p):
    problems = []
    if len(p["destination"]) < 3:
        problems.append("Please describe your tour theme/destination (Q1).")
    if p["count"] < 1:
        problems.append("Group size must be at least 1 traveler.")
    if p["travel"]["duration"] < 1:
        problems.append("Trip duration must be at least 1 day.")
    return problems


# ================================ THE PROMPT ================================

def build_prompt(p):
    city = p["city"] or p["destination"].split(",")[0]
    cal = ""
    if p["day_labels"]:
        cal = "\nCalendar day labels (use these exact labels for each day):\n" + \
              "\n".join(f"  - {d}" for d in p["day_labels"])
    concise = ("\n- Because the trip is long, keep each time-of-day line to ONE short sentence."
               if p["travel"]["duration"] > 14 else "")

    return f"""You are a master Travel Guide and Tour Planner AI with deep knowledge of
destinations, routes, seasons, hidden gems, local food and realistic Indian prices.

==================================================
              TRAVELER REQUIREMENTS
==================================================
 1. Destination City / Region : {city}
 2. Tour Theme / Focus        : {p['destination']}
 3. Group Size                : {p['group_display']}
 4. Age Group                 : {p['age_group']}
 5. Duration / Dates          : {p['travel']['display']}
 6. Traveler Profiles         :{p['members_summary']}
 7. Group Dynamic             : {p['relationship']}
 8. Budget Category           : {p['budget_display']}
 9. Transport                 : {p['transport_display']}
10. Food Diet                 : {p['food_display']}
11. Stay Preference           : {p['stay']}
{cal}

Build a comprehensive, realistic, highly detailed day-by-day travel plan for
EXACTLY {p['travel']['duration']} day(s).

==================================================
          REQUIRED OUTPUT FORMAT (Markdown)
==================================================
First line must be:
# {city} - <short catchy trip title>
Then a 2-3 line trip summary paragraph.

Then these sections, with EXACTLY these "## " headings, in this order:

## \U0001F4CD 1. Must-Visit Places & Hidden Gems
- 6-10 specific named attractions/viewpoints/routes with best visiting time
  and a one-line reason to go.

## \U0001F5FA\uFE0F 2. Day-by-Day Itinerary
One block per day (all {p['travel']['duration']} days), each exactly:
**Day N - <Short Title>**
**Morning:** <places, start time, travel time, entry fee if any>
**Afternoon:** <places, lunch stop, travel time>
**Evening:** <activity, dinner spot, night vibe>

## \U0001F3E8 3. Recommended Stays
- 2-3 realistic property names for each night-stop region that match the stay
  preference and budget, with \u20b9 price per night.

## \U0001F37D\uFE0F 4. Food & Restaurant Recommendations
- Specific eatery names plus the exact local dishes to try, strictly matching
  the diet preference ({p['food_display']}).

## \U0001F4B0 5. Estimated Budget (\u20b9 INR)
- A markdown table with columns: Category | Details | Estimated Cost (\u20b9)
  and rows: Stay, Food, Local Transport/Fuel, Entry Tickets, Miscellaneous.
- Then finish the section with two bold lines:
**Per Person: \u20b9<amount>**
**Total Estimated Cost: \u20b9<amount>**

## \U0001F392 6. Essential Packing & Safety
- Weather-suitable gear, clothing etiquette for cultural/religious sites,
  first-aid, and advice tailored to this exact age group and group dynamic.

## \u2728 7. Smart Local Tips & Hidden Secrets
- 5-7 insider tips: timing hacks, ticket booking, scams to avoid, photo spots.

RULES:
- Realistic Indian Rupee (\u20b9) pricing aligned with: {p['budget']}.
- Bold every place/property/dish name and every price.
- Short lines and bullet points only - no walls of text.{concise}
- No code blocks. No asking questions - just deliver the plan."""


# ============================== API ENDPOINTS ==============================

def api_home():
    """GET / - serve the frontend locally, info JSON everywhere else."""
    if os.path.exists(ROOT_HTML):
        try:
            return send_file(ROOT_HTML)
        except Exception:
            pass
    return _json({
        "success": True,
        "service": "AI Tour Planner API",
        "gemini_configured": bool(GEMINI_API_KEY),
        "mongodb_available": bool(_mongo()),
        "endpoints": [
            "GET  /api/health", "POST /api/plan", "GET  /api/plans",
            "GET  /api/plan/<id>", "DELETE /api/plan/<id>",
            "POST /api/plan/<id>/feedback",
        ],
    })


def api_health():
    return _json({
        "success": True,
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "mongodb_available": bool(_mongo()),
        "models": GEMINI_MODELS,
        "time": datetime.now(timezone.utc).isoformat(),
    })


def api_generate():
    """POST /api/plan - build prompt, call Gemini, save to MongoDB."""
    raw = request.get_json(silent=True) or {}
    profile = sanitize_profile(raw)
    problems = validate_profile(profile)
    if problems:
        return _json({"success": False, "error": " | ".join(problems)}, 400)

    result = call_gemini(build_prompt(profile))
    if not result.get("ok"):
        payload = {"success": False, "error": result.get("error", "Gemini failed")}
        if result.get("setup"):
            payload["setup"] = True
            payload["error"] = ("Server is missing its Gemini API key. Add "
                                "GEMINI_API_KEY in Vercel -> Settings -> "
                                "Environment Variables, then redeploy.")
            payload["hint"] = "Free key: https://aistudio.google.com/app/apikey"
        else:
            payload["hint"] = ("Free-tier Gemini keys have rate limits - wait a "
                               "minute and retry, or check the model list.")
        return _json(payload, 502)

    share_id, saved = None, False
    doc = {
        "_id": "",
        "share_id": "",
        "profile": profile,
        "plan_text": result["text"],
        "model": result["model"],
        "rating": 0,
        "views": 0,
        "created_at": datetime.now(timezone.utc),
    }
    if _mongo() is not None:
        doc["_id"] = doc["share_id"] = secrets.token_urlsafe(6)
        saved = _save_plan(doc)
        if not saved:
            doc["share_id"] = ""

    return _json({
        "success": True,
        "share_id": doc["share_id"] or None,
        "mongo_saved": saved,
        "plan": result["text"],
        "model": result["model"],
        "profile": profile,
    })


def api_list_plans():
    """GET /api/plans - recent saved plans (metadata only)."""
    db = _mongo()
    if db is None:
        return _json({"success": False, "plans": [],
                      "reason": "MongoDB not configured"})
    try:
        limit = _int(request.args.get("limit"), 12, 1, 50)
        cur = db.plans.find({}, {"profile": 1, "model": 1, "rating": 1,
                                 "views": 1, "created_at": 1}) \
            .sort("created_at", -1).limit(limit)
        plans = [_plan_meta(d) for d in cur]
        return _json({"success": True, "plans": plans})
    except Exception as e:
        return _json({"success": False, "plans": [], "error": str(e)[:200]})


def api_get_plan(share_id):
    """GET /api/plan/<id> - fetch one full plan."""
    db = _mongo()
    if db is None:
        return _json({"success": False,
                      "error": "MongoDB not configured - shared links are "
                               "disabled. Add MONGODB_URI to enable them."}, 503)
    try:
        sid = _txt(share_id, 40)
        doc = db.plans.find_one({"_id": sid}) or db.plans.find_one({"share_id": sid})
        if not doc:
            return _json({"success": False, "error": "Plan not found"}, 404)
        try:
            db.plans.update_one({"_id": doc.get("_id")},
                                {"$inc": {"views": 1}})
        except Exception:
            pass
        doc.pop("_id", None)
        doc["created_at"] = _iso(doc.get("created_at"))
        return _json({"success": True, "plan": doc})
    except Exception as e:
        return _json({"success": False, "error": str(e)[:200]}, 500)


def api_delete_plan(share_id):
    """DELETE /api/plan/<id>."""
    db = _mongo()
    if db is None:
        return _json({"success": False, "deleted": False,
                      "reason": "MongoDB not configured"})
    try:
        res = db.plans.delete_one({"_id": _txt(share_id, 40)})
        return _json({"success": True, "deleted": res.deleted_count > 0})
    except Exception as e:
        return _json({"success": False, "error": str(e)[:200]})


def api_feedback(share_id):
    """POST /api/plan/<id>/feedback - {"rating": 1|-1|0}."""
    db = _mongo()
    if db is None:
        return _json({"success": False, "rating": None,
                      "reason": "MongoDB not configured"})
    try:
        body = request.get_json(silent=True) or {}
        rating = _int(body.get("rating"), 0, -1, 1)
        db.plans.update_one({"_id": _txt(share_id, 40)},
                             {"$set": {"rating": rating}})
        return _json({"success": True, "rating": rating})
    except Exception as e:
        return _json({"success": False, "error": str(e)[:200]})


# ============================ CATCH-ALL ROUTER ============================
# The Flask app may be mounted at "/", "/api" or "/api/index" depending on
# how it is invoked (Vercel warm instance, local run, CLI...). This router
# normalizes ANY of those prefixes and dispatches by clean path, so the
# endpoints work identically at /api/plan, /plan and /api/index/plan.

def _normalize(path):
    p = "/" + (path or "").strip("/")
    for prefix in ("/api/index", "/api", "/index"):
        if p == prefix:
            return "/"
        if p.startswith(prefix + "/"):
            return p[len(prefix):]
    return p


@app.after_request
def _cors(resp):
    h = resp.headers
    h["Access-Control-Allow-Origin"] = "*"
    h["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    h["Access-Control-Allow-Headers"] = "Content-Type"
    h["Cache-Control"] = "no-store"
    return resp


@app.route("/", defaults={"path": ""},
           methods=["GET", "POST", "DELETE", "OPTIONS"])
@app.route("/<path:path>",
           methods=["GET", "POST", "DELETE", "OPTIONS"])
def router(path=""):
    if request.method == "OPTIONS":
        return "", 204

    route = _normalize(request.path)
    seg = [s for s in route.strip("/").split("/") if s]
    method = request.method

    # GET  /  -> home (frontend when local, info JSON otherwise)
    if route == "/" and method == "GET":
        return api_home()

    if seg[:1] == ["health"] and method == "GET":
        return api_health()

    if seg[:1] == ["plan"]:
        rest = seg[1:]
        if not rest and method == "POST":
            return api_generate()                       # POST /plan
        if len(rest) == 1:
            if method == "GET":
                return api_get_plan(rest[0])            # GET /plan/<id>
            if method == "DELETE":
                return api_delete_plan(rest[0])         # DELETE /plan/<id>
        if len(rest) == 2 and rest[1] == "feedback" and method == "POST":
            return api_feedback(rest[0])                # POST /plan/<id>/feedback

    if seg[:1] == ["plans"] and method == "GET":
        return api_list_plans()                         # GET /plans

    return _json({"success": False,
                  "error": f"Unknown endpoint: {method} {route}"}, 404)


# ============================== LOCAL RUNNER ==============================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AI TOUR PLANNER - Flask backend (Gemini + MongoDB)")
    print("=" * 65)
    print("  Gemini key  :", "set" if GEMINI_API_KEY else "MISSING (set GEMINI_API_KEY)")
    print("  MongoDB     :", "set" if MONGODB_URI else "not set (cloud saving disabled)")
    print("  Models      :", ", ".join(GEMINI_MODELS))
    print("  Serving     : http://localhost:5000  (app + API)")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
