"""
============================================================================
 AI TOUR PLANNER  —  Backend (index.py)
============================================================================
 Flask app that:
   1. Serves the frontend (index.html) at "/", "/index.html" and any
      non-API path (SPA fallback) — fixes "Unknown endpoint" errors.
   2. Exposes a JSON API powered by Google Gemini with a multi-model
      fallback chain (gemini-3.8-flash -> ... -> gemini-2.0-flash).
   3. Stores every generated plan in MongoDB (auto-falls back to a local
      JSON file if MONGODB_URI is not set, so the app always works).

 --------------------------------------------------------------------------
 RUN LOCALLY
   pip install flask requests pymongo
   python index.py            ->  http://localhost:5000

 --------------------------------------------------------------------------
 DEPLOY ON VERCEL  (2 code files + 1 tiny config)
   Project root:
     index.html
     index.py
     vercel.json        <-- add this file:

     {
       "version": 2,
       "builds": [{ "src": "index.py", "use": "@vercel/python" }],
       "routes": [
         { "src": "/api/(.*)", "dest": "index.py" },
         { "src": "/(.*)",     "dest": "index.py" }
       ]
     }

   Then:  vercel --prod

   Set these env vars in Vercel (Project -> Settings -> Environment
   Variables) — values below are only fallbacks:
     GEMINI_API_KEY   your Google Gemini API key
     MONGODB_URI      e.g. mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/
============================================================================
"""

import os
import json
import uuid
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY",
                                "")
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/{model}:generateContent")
HEADERS = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}

MONGODB_URI = os.environ.get("MONGODB_URI", "")
FALLBACK_FILE = os.path.join("/tmp", "tour_plans.json")

VALID = {
    "age_groups": [
        "Under 18 (Kids & Teens)",
        "18-25 (Young Adults)",
        "26-35 (Adults)",
        "36-50 (Adults)",
        "51-60 (Mature Adults)",
        "60+ (Seniors)",
    ],
    "relationships": ["Friends", "Family", "Colleagues", "Couples",
                      "Family with Kids"],
    "budget_tiers": ["Budget", "Mid-range", "Premium", "Luxury"],
    "transports": ["Flight", "Bus", "Train", "Personal"],
    "vehicles": ["CAR", "SUV", "VAN", "BIKE"],
    "food_prefs": ["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian",
                   "Jain", "Mixed / No Restrictions"],
    "stays": ["Hotel (1-3 Star)", "Hotel (4-5 Star)", "Resort",
              "Homestay / Airbnb", "Hostel", "Dharamshala / Budget Stay",
              "Glamping / Camping"],
}

BUDGET_RANGES = {
    "Budget":    "INR 800 - INR 1,500 per person per day",
    "Mid-range": "INR 2,000 - INR 4,500 per person per day",
    "Premium":   "INR 5,000 - INR 12,000 per person per day",
    "Luxury":    "INR 12,000+ per person per day",
}


# --------------------------------------------------------------------------
# MONGODB (with local JSON fallback so the app never breaks)
# --------------------------------------------------------------------------
_client = None


def get_db():
    """Return the plans collection, or None if Mongo is unavailable."""
    global _client
    if not MONGODB_URI:
        return None
    try:
        if _client is None:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
            _client.admin.command("ping")
        return _client["tour_planner"]["plans"]
    except Exception:
        _client = None
        return None


def _fallback_load():
    try:
        with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _fallback_save(docs):
    try:
        with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def save_plan(doc):
    db = get_db()
    if db is not None:
        db.insert_one(dict(doc))
    else:
        docs = _fallback_load()
        docs.append(doc)
        _fallback_save(docs)
    return doc["plan_id"]


def list_plans(limit=20):
    db = get_db()
    if db is not None:
        cur = db.find({}, {"plan": 0}).sort("created_at", -1).limit(limit)
        return list(cur)
    return sorted(_fallback_load(), key=lambda x: x.get("created_at", ""),
                  reverse=True)[:limit]


def get_plan(pid):
    db = get_db()
    if db is not None:
        return db.find_one({"plan_id": pid})
    for doc in _fallback_load():
        if doc.get("plan_id") == pid:
            return doc
    return None


def delete_plan(pid):
    db = get_db()
    if db is not None:
        return db.delete_one({"plan_id": pid}).deleted_count > 0
    docs = _fallback_load()
    new = [d for d in docs if d.get("plan_id") != pid]
    _fallback_save(new)
    return len(new) != len(docs)


# --------------------------------------------------------------------------
# GEMINI API  (multi-model fallback)
# --------------------------------------------------------------------------
def call_gemini_api(prompt):
    """Try each model in order; return (text, model_used). Raise on total fail."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
    }
    last_err = None
    for model in GEMINI_MODELS:
        try:
            res = requests.post(GEMINI_URL.format(model=model),
                                headers=HEADERS, json=payload, timeout=90)
            if res.status_code != 200:
                last_err = "%s -> HTTP %d" % (model, res.status_code)
                continue
            text = (res.json()["candidates"][0]["content"]["parts"][0]["text"])
            if text and text.strip():
                return text, model
            last_err = "%s -> empty response" % model
        except Exception as exc:
            last_err = "%s -> %s" % (model, exc)
    raise RuntimeError("All Gemini models failed. Last error: %s" % last_err)


# --------------------------------------------------------------------------
# PROMPT BUILDER  (mirrors the original CLI script)
# --------------------------------------------------------------------------
def build_gemini_prompt(d):
    total_days = d["total_days"]
    if d["date_mode"] == "dates" and d.get("start_date") and d.get("end_date"):
        duration = "%d days (%s to %s)" % (total_days, d["start_date"],
                                           d["end_date"])
    else:
        duration = "%d days" % total_days

    if d["group_type"] == "solo":
        group_line = "Solo traveler (1 person)"
    else:
        group_line = "Group of %d people" % d["group_size"]

    member_lines = []
    for i, m in enumerate(d["members"], 1):
        member_lines.append("    %d. %s | Age %s | Gender %s"
                            % (i, m["name"], m["age"], m["gender"]))
    members_block = "\n".join(member_lines)

    lines = []
    lines.append("You are an expert AI travel planner with deep local "
                 "knowledge of destinations across the world.")
    lines.append("Create a complete, detailed and personalized tour plan "
                 "for the trip described below.")
    lines.append("")
    lines.append("========== TRIP DETAILS ==========")
    lines.append("Destination        : %s" % d["destination"])
    lines.append("Group Type         : %s" % group_line)
    lines.append("Age Group          : %s" % d["age_group"])
    lines.append("Trip Duration      : %s" % duration)
    lines.append("Traveler Details   :")
    lines.append(members_block)
    lines.append("Relationship       : %s" % d["relationship"])
    lines.append("Budget Category    : %s (%s)"
                 % (d["budget_tier"], BUDGET_RANGES[d["budget_tier"]]))
    if d.get("total_budget"):
        lines.append("Total Trip Budget  : INR %s (plan strictly within this "
                    "total amount)" % d["total_budget"])
    lines.append("Preferred Transport: %s" % d["transport"])
    if d["transport"] == "Personal":
        lines.append("Personal Vehicle   : %s with %s seats"
                     % (d.get("vehicle_type", "CAR"), d.get("vehicle_seats", 4)))
    lines.append("Food Preference    : %s" % d["food_preference"])
    if d.get("allergies"):
        lines.append("Allergies / Food restrictions to avoid: %s"
                     % d["allergies"])
    lines.append("Stay Preference    : %s" % d["stay_preference"])
    lines.append("")
    lines.append("========== REQUIRED OUTPUT FORMAT ==========")
    lines.append("Write the plan in markdown and include these EXACT 6 "
                 "sections with these headers:")
    lines.append("1. '# \U0001F4CD Must-Visit Places & Hidden Gems' — top "
                 "attractions plus lesser-known local gems, with a one-line "
                 "reason and typical entry ticket price in INR for each.")
    lines.append("2. '# \U0001F5FA Day-by-Day Itinerary' — a full plan for "
                 "ALL %d days, each day split into Morning / Afternoon / "
                 "Evening with specific places, timings and travel tips."
                 % total_days)
    lines.append("3. '# \U0001F3E8 Recommended Stays' — 2-3 property "
                 "suggestions matching the stay preference, with approximate "
                 "price per night in INR.")
    lines.append("4. '# \U0001F37D Food & Restaurants' — where to eat, "
                 "matching the food preference, with must-try dishes and "
                 "approximate cost per meal in INR.")
    lines.append("5. '# \U0001F4B0 Detailed Budget Breakdown' — a realistic "
                 "table (Stay / Food / Transport / Tickets / Miscellaneous) "
                 "per person and grand total in INR.")
    lines.append("6. '# \U0001F392 Packing & Safety Advice' — packing list "
                 "for this destination and season, plus safety tips and "
                 "emergency contacts.")
    lines.append("")
    lines.append("Be specific, use real place names, keep all money in INR, "
                 "and make the plan practical for the group described above.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# REQUEST VALIDATION  (mirrors the CLI input rules)
# --------------------------------------------------------------------------
def validate_request(d):
    errors = []

    def s(v):
        return str(v).strip() if v is not None else ""

    clean = {}

    destination = s(d.get("destination"))
    if len(destination) < 5:
        errors.append("Destination must be at least 5 characters.")
    clean["destination"] = destination

    group_type = s(d.get("group_type")).lower()
    if group_type not in ("solo", "group"):
        errors.append("Group type must be 'solo' or 'group'.")
        group_type = "solo"
    clean["group_type"] = group_type

    try:
        group_size = int(d.get("group_size", 1))
    except (TypeError, ValueError):
        group_size = 1
    if group_type == "solo":
        group_size = 1
    elif not (2 <= group_size <= 100):
        errors.append("Group size must be between 2 and 100 for groups.")
        group_size = 1
    clean["group_size"] = group_size

    age_group = s(d.get("age_group"))
    if age_group not in VALID["age_groups"]:
        errors.append("Invalid age group.")
    clean["age_group"] = age_group

    # --- dates: either total days, or a DD/MM/YYYY range (1-120 days) ---
    date_mode = s(d.get("date_mode")) or "days"
    total_days = None
    start_date = s(d.get("start_date")) or None
    end_date = s(d.get("end_date")) or None
    if date_mode == "dates":
        try:
            sd = datetime.strptime(start_date, "%d/%m/%Y").date()
            ed = datetime.strptime(end_date, "%d/%m/%Y").date()
            if ed < sd:
                errors.append("End date cannot be before start date.")
            total_days = (ed - sd).days + 1
            if total_days < 1 or total_days > 120:
                errors.append("Trip must be between 1 and 120 days.")
                total_days = None
        except (TypeError, ValueError):
            errors.append("Start/end dates must be valid DD/MM/YYYY.")
    else:
        try:
            total_days = int(d.get("total_days"))
            if not (1 <= total_days <= 120):
                errors.append("Total days must be between 1 and 120.")
                total_days = None
        except (TypeError, ValueError):
            errors.append("Total days must be a number between 1 and 120.")
    if total_days is None:
        errors.append("A valid trip duration is required.")
    clean["date_mode"] = date_mode
    clean["total_days"] = total_days
    clean["start_date"] = start_date
    clean["end_date"] = end_date

    # --- members: one row per traveler, ages 1-120, gender M/F/O ---
    members = d.get("members") or []
    if not isinstance(members, list) or len(members) != group_size:
        errors.append("Member details are required for all %d traveler(s)."
                      % group_size)
        members = []
    else:
        for i, m in enumerate(members, 1):
            name = s(m.get("name"))
            if len(name) < 2:
                errors.append("Member %d: name is required (min 2 characters)."
                              % i)
            try:
                age = int(m.get("age"))
                if not (1 <= age <= 120):
                    raise ValueError
            except (TypeError, ValueError):
                age = None
                errors.append("Member %d: age must be between 1 and 120." % i)
            gender = s(m.get("gender")).upper()
            if gender not in ("M", "F", "O"):
                gender = None
                errors.append("Member %d: gender must be M, F or O." % i)
            members[i - 1] = {"name": name, "age": age, "gender": gender}
    clean["members"] = members

    relationship = s(d.get("relationship"))
    if relationship not in VALID["relationships"]:
        errors.append("Invalid relationship type.")
    clean["relationship"] = relationship

    budget_tier = s(d.get("budget_tier"))
    if budget_tier not in VALID["budget_tiers"]:
        errors.append("Invalid budget tier.")
    clean["budget_tier"] = budget_tier

    total_budget = s(d.get("total_budget"))
    if total_budget:
        try:
            tb = int(float(total_budget))
            if tb <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Total budget must be a positive number.")
            total_budget = None
    clean["total_budget"] = total_budget or None

    transport = s(d.get("transport"))
    if transport not in VALID["transports"]:
        errors.append("Invalid transport option.")
    clean["transport"] = transport

    vehicle_type = s(d.get("vehicle_type")).upper()
    if transport == "Personal":
        if vehicle_type not in VALID["vehicles"]:
            errors.append("Vehicle type must be CAR, SUV, VAN or BIKE.")
            vehicle_type = "CAR"
        try:
            seats = int(d.get("vehicle_seats"))
            if not (1 <= seats <= 50):
                raise ValueError
        except (TypeError, ValueError):
            seats = None
            errors.append("Vehicle seats must be between 1 and 50.")
    else:
        vehicle_type, seats = None, None
    clean["vehicle_type"] = vehicle_type
    clean["vehicle_seats"] = seats

    food_preference = s(d.get("food_preference"))
    if food_preference not in VALID["food_prefs"]:
        errors.append("Invalid food preference.")
    clean["food_preference"] = food_preference
    clean["allergies"] = s(d.get("allergies")) or None

    stay_preference = s(d.get("stay_preference"))
    if stay_preference not in VALID["stays"]:
        errors.append("Invalid stay preference.")
    clean["stay_preference"] = stay_preference

    return (len(errors) == 0), errors, clean


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return resp


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    """Serve the frontend for every non-API path (fixes Unknown endpoint)."""
    if path.startswith("api/"):
        return jsonify({"success": False,
                        "error": "Unknown endpoint: %s /%s"
                                 % (request.method, path)}), 404
    full = os.path.join(BASE_DIR, path)
    if path and os.path.isfile(full) and not path.endswith(".py"):
        return send_file(full)
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/api/health")
def api_health():
    return jsonify({"success": True, "status": "ok",
                    "storage": "mongodb" if get_db() is not None
                               else "local-file-fallback",
                    "models": GEMINI_MODELS})


@app.route("/api/options")
def api_options():
    return jsonify({"success": True, "options": VALID,
                    "budget_ranges": BUDGET_RANGES})


@app.route("/api/generate", methods=["POST", "OPTIONS"])
def api_generate():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    ok, errors, clean = validate_request(data)
    if not ok:
        return jsonify({"success": False,
                        "error": "Please fix the highlighted answers.",
                        "details": errors}), 400

    prompt = build_gemini_prompt(clean)
    try:
        plan_text, model_used = call_gemini_api(prompt)
    except Exception as exc:
        return jsonify({"success": False,
                        "error": "Gemini API failed: %s" % exc}), 502

    doc = {
        "plan_id": uuid.uuid4().hex,
        "destination": clean["destination"],
        "group_type": clean["group_type"],
        "group_size": clean["group_size"],
        "total_days": clean["total_days"],
        "budget_tier": clean["budget_tier"],
        "transport": clean["transport"],
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": model_used,
        "request": clean,
        "plan": plan_text,
    }
    try:
        plan_id = save_plan(doc)
    except Exception:
        plan_id = None  # storage failure should never kill a generated plan

    return jsonify({"success": True, "plan_id": plan_id,
                    "model": model_used, "plan": plan_text,
                    "summary": doc["created_at"]})


@app.route("/api/history")
def api_history():
    try:
        plans = list_plans(int(request.args.get("limit", 20)))
    except Exception:
        plans = []
    for p in plans:
        p["plan_id"] = p.get("plan_id") or str(p.get("_id", ""))
        p.pop("_id", None)
    return jsonify({"success": True, "count": len(plans), "plans": plans})


@app.route("/api/plan/<pid>", methods=["GET", "DELETE"])
def api_plan(pid):
    if request.method == "DELETE":
        deleted = delete_plan(pid)
        return jsonify({"success": deleted}), (200 if deleted else 404)
    doc = get_plan(pid)
    if not doc:
        return jsonify({"success": False, "error": "Plan not found."}), 404
    doc.pop("_id", None)
    return jsonify({"success": True, "plan": doc})


# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=True)
