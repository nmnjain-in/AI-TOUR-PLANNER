"""
============================================================================
 AI TOUR PLANNER  —  Backend (index.py)
============================================================================
 Flask backend for the "AI Tour Planner" frontend (index.html).
 Serves index.html at "/", "/index.html" and any non-API path, and
 implements the EXACT API contract the frontend expects:

     POST   /api/plan                    -> generate a plan with Gemini
     GET    /api/plans?limit=12          -> list cloud trips (history)
     GET    /api/plan/<share_id>         -> open / share a saved plan
     DELETE /api/plan/<share_id>         -> delete a plan
     POST   /api/plan/<share_id>/feedback -> rate a plan (1 / -1 / 0)
     GET    /api/health                 -> service status

 Plans are stored in MongoDB (set MONGODB_URI). If MongoDB is not
 configured, a local JSON file is used so the app still works (sharing
 and cross-device history then need MONGODB_URI).

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

   Set env vars in Vercel -> Settings -> Environment Variables:
     GEMINI_API_KEY   your Google Gemini API key (free: aistudio.google.com)
     MONGODB_URI      e.g. mongodb+srv://user:pass@cluster0.xxx.mongodb.net/
============================================================================
"""

import os
import json
import uuid
import secrets
from datetime import datetime, date

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

# Exact option values used by the frontend (do not change these strings).
VALID = {
    "group_types": ["Group", "Solo"],
    "age_groups": [
        "18 - 25 (Young Adults / Students)",
        "25 - 35 (Working Professionals)",
        "35 - 50 (Family / Middle Age)",
        "50 - 65 (Senior Citizens)",
        "65+ (Elderly)",
        "Mixed Age Group (Kids + Adults + Seniors)",
    ],
    "relationships": [
        "Singles Group (Solo / Friends)",
        "Couples Group (Romantic / Honeymoon)",
        "Family with Kids",
        "Mixed Group (Singles + Couples)",
        "All Boys / All Girls Friends Trip",
    ],
    "budget_ranges": [
        "Low Range (Budget / Backpacker - ₹800-₹1500/day per person)",
        "Mid Range (Comfortable - ₹2000-₹4500/day per person)",
        "High Range (Premium / Luxury - ₹5000-₹12000/day per person)",
        "Ultra Luxury (5 Star / Private Everything)",
    ],
    "transports": [
        "Flight ✈️",
        "Bus (Volvo / Sleeper / State) 🚌",
        "Train (Vande Bharat / Express) 🚂",
        "Personal Vehicle 🚗",
    ],
    "vehicles": ["CAR", "SUV", "BIK", "VAN"],
    "diets": [
        "Pure Vegetarian 🥬",
        "Non-Veg 🍗",
        "Non-Veg + Veg (Both) 🍗🥬",
        "Jainism Type (No Onion, No Garlic) 🙏",
        "Vegan 🌱",
        "Eggetarian 🥚",
    ],
    "stays": [
        "Hotels (Standard / 3-Star / 4-Star)",
        "Dharamshalas / Ashrams / Temple Trusts 🙏",
        "Hostels (Zostel / Backpacker Dorms) 🏠",
        "Homestays (Local Family / Village Vibe) 🏡",
        "Luxury Resorts / Riverside Cottages 🏖️",
        "Camping / Tents / Glamping ⛺",
        "Any place is fine (Best budget combo)",
    ],
}


# --------------------------------------------------------------------------
# STORAGE: MongoDB with automatic local-JSON fallback
# --------------------------------------------------------------------------
_client = None


def get_db():
    """Return the plans collection, or None when Mongo is unavailable."""
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
            json.dump(docs, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _new_share_id():
    return uuid.uuid4().hex[:16]


def db_save(doc):
    db = get_db()
    if db is not None:
        db.insert_one(dict(doc))
    else:
        docs = _fallback_load()
        docs.append(doc)
        _fallback_save(docs)
    return doc["share_id"]


def db_list(limit=12):
    db = get_db()
    if db is not None:
        cur = (db.find({}, {"plan_text": 0, "profile": 0})
                 .sort("created_ts", -1).limit(limit))
        return list(cur)
    return sorted(_fallback_load(),
                  key=lambda x: x.get("created_ts", 0),
                  reverse=True)[:limit]


def db_get(share_id):
    db = get_db()
    if db is not None:
        return db.find_one({"share_id": share_id})
    for doc in _fallback_load():
        if doc.get("share_id") == share_id:
            return doc
    return None


def db_delete(share_id):
    db = get_db()
    if db is not None:
        return db.delete_one({"share_id": share_id}).deleted_count > 0
    docs = _fallback_load()
    new = [d for d in docs if d.get("share_id") != share_id]
    _fallback_save(new)
    return len(new) != len(docs)


def db_update(share_id, fields):
    db = get_db()
    if db is not None:
        db.update_one({"share_id": share_id}, {"$set": fields})
        return db.find_one({"share_id": share_id})
    docs = _fallback_load()
    for doc in docs:
        if doc.get("share_id") == share_id:
            doc.update(fields)
            _fallback_save(docs)
            return doc
    return None


# --------------------------------------------------------------------------
# GEMINI (multi-model fallback chain)
# --------------------------------------------------------------------------
def call_gemini_api(prompt):
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
            text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            if text and text.strip():
                return text, model
            last_err = "%s -> empty response" % model
        except Exception as exc:
            last_err = "%s -> %s" % (model, exc)
    raise RuntimeError(last_err)


# --------------------------------------------------------------------------
# VALIDATION  (mirrors the frontend wizard rules)
# --------------------------------------------------------------------------
def validate_profile(d):
    errors = []

    def s(v):
        return str(v).strip() if v is not None else ""

    p = {}
    p["city"] = s(d.get("city"))
    if len(p["city"]) < 2:
        errors.append("Please enter your destination city (min 2 characters).")

    p["destination"] = s(d.get("destination"))
    if len(p["destination"]) < 5:
        errors.append("Q1: describe your tour theme in at least 5 characters.")

    grp = d.get("group") or {}
    gtype = s(grp.get("type")) or ("Group" if grp.get("type") else "")
    if gtype not in VALID["group_types"]:
        errors.append("Q2: group type must be 'Group' or 'Solo'.")
        gtype = "Solo"
    try:
        count = int(grp.get("count", 1))
    except (TypeError, ValueError):
        count = 1
    if gtype == "Solo":
        count = 1
    elif not (2 <= count <= 100):
        errors.append("Q2: group size must be between 2 and 100 people.")
        count = 1
    p["group_type"] = gtype
    p["count"] = count

    p["age_group"] = s(d.get("age_group"))
    if p["age_group"] not in VALID["age_groups"]:
        errors.append("Q3: invalid age group.")

    # --- travel: total days OR a DD/MM/YYYY calendar range (1-120 days) ---
    trav = d.get("travel") or {}
    ttype = s(trav.get("type")) or "days"
    p["travel"] = {"type": ttype, "duration": None,
                   "start": "", "end": "", "display": ""}
    if ttype == "dates":
        try:
            sd = datetime.strptime(s(trav.get("start")), "%d/%m/%Y").date()
            ed = datetime.strptime(s(trav.get("end")), "%d/%m/%Y").date()
        except (TypeError, ValueError):
            errors.append("Q4: start/end dates must be valid DD/MM/YYYY.")
            sd = ed = None
        if sd and ed:
            if ed < sd:
                errors.append("Q4: end date must be after start date.")
            else:
                dur = (ed - sd).days
                if 1 <= dur <= 120:
                    p["travel"] = {"type": "dates", "duration": dur,
                                   "start": s(trav.get("start")),
                                   "end": s(trav.get("end")),
                                   "display": "%s to %s (%d Days with Full Year)"
                                              % (s(trav.get("start")),
                                                 s(trav.get("end")), dur)}
                else:
                    errors.append("Q4: trip must be between 1 and 120 days.")
    else:
        try:
            dur = int(trav.get("duration"))
            if not (1 <= dur <= 120):
                raise ValueError
        except (TypeError, ValueError):
            dur = None
            errors.append("Q4: total days must be between 1 and 120.")
        if dur:
            p["travel"] = {"type": "days", "duration": dur, "start": "",
                           "end": "", "display": "%d Days Trip" % dur}
    if p["travel"]["duration"] is None:
        errors.append("Q4: a valid trip duration is required.")

    # --- members: one entry per traveler (names optional) ---
    mem = d.get("members") or {}
    raw_list = mem.get("list") or []
    if not isinstance(raw_list, list) or len(raw_list) != count:
        errors.append("Q5: traveler details are required for all %d "
                      "traveler(s)." % count)
        raw_list = []
    members = []
    for i, m in enumerate(raw_list, 1):
        name = s(m.get("name")) or ("Traveler %d" % i)
        try:
            age = int(m.get("age"))
            if not (1 <= age <= 120):
                raise ValueError
        except (TypeError, ValueError):
            age = None
            errors.append("Q5: traveler %d needs a valid age (1-120)." % i)
        gender = s(m.get("gender")).capitalize()
        gender = gender if gender in ("Male", "Female", "Other") else None
        if not gender:
            gender = "Other"
            errors.append("Q5: traveler %d needs a gender (M/F/Other)." % i)
        members.append({"name": name, "age": age, "gender": gender})
    p["members"] = {
        "males": sum(1 for m in members if m["gender"] == "Male"),
        "females": sum(1 for m in members if m["gender"] == "Female"),
        "list": members,
    }

    p["relationship"] = s(d.get("relationship"))
    if p["relationship"] not in VALID["relationships"]:
        errors.append("Q6: invalid group composition.")

    bud = d.get("budget") or {}
    p["budget"] = s(bud.get("range"))
    if p["budget"] not in VALID["budget_ranges"]:
        errors.append("Q7: invalid budget range.")
    amount = s(bud.get("amount"))
    if amount:
        if not amount.isdigit() or int(amount) <= 0:
            errors.append("Q7: total budget must be a positive number.")
            amount = ""
    p["budget_amount"] = amount

    tra = d.get("transport") or {}
    p["transport"] = s(tra.get("mode"))
    if p["transport"] not in VALID["transports"]:
        errors.append("Q8: invalid mode of transport.")
    p["vehicle"] = s(tra.get("vehicle"))
    p["seats"] = s(tra.get("seats"))
    if "Personal Vehicle" in p["transport"]:
        if p["vehicle"] not in VALID["vehicles"]:
            errors.append("Q8: pick a personal vehicle type (CAR/SUV/BIK/VAN).")
            p["vehicle"] = ""
        if p["seats"]:
            if not p["seats"].isdigit() or not (1 <= int(p["seats"]) <= 50):
                errors.append("Q8: seats must be between 1 and 50.")
                p["seats"] = ""
    else:
        p["vehicle"] = p["vehicle"] if p["vehicle"] in VALID["vehicles"] else ""
        p["seats"] = p["seats"] if p["seats"].isdigit() else ""

    food = d.get("food") or {}
    p["food"] = s(food.get("diet"))
    if p["food"] not in VALID["diets"]:
        errors.append("Q9: invalid food preference.")
    p["allergies"] = s(food.get("allergy"))[:100]

    p["stay"] = s(d.get("stay"))
    if p["stay"] not in VALID["stays"]:
        errors.append("Q10: invalid stay preference.")

    return (len(errors) == 0), errors, p


# --------------------------------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------------------------------
def build_gemini_prompt(p):
    if p["group_type"] == "Solo":
        group_line = "Solo traveler (1 person)"
    else:
        group_line = "Group of %d people" % p["count"]

    trav = p["travel"]
    if trav["type"] == "dates":
        duration = ("%d days (%s to %s) — Gemini must date every single day"
                    % (trav["duration"], trav["start"], trav["end"]))
    else:
        duration = "%d days" % trav["duration"]

    member_lines = "\n".join(
        "    %d. %s | Age %s | %s" % (i, m["name"], m["age"], m["gender"])
        for i, m in enumerate(p["members"]["list"], 1))

    transport_line = p["transport"]
    if p["vehicle"]:
        transport_line += " - [%s]" % p["vehicle"]
    if p["seats"]:
        transport_line += " (Seater: %s)" % p["seats"]

    food_line = p["food"] + ((" | Allergies/restrictions: " + p["allergies"])
                             if p["allergies"] else "")

    budget_line = p["budget"]
    if p["budget_amount"]:
        budget_line += " | Target TOTAL budget for the whole group: ₹%s — " \
                       "plan strictly within this amount" % p["budget_amount"]

    lines = [
        "You are an expert AI travel planner with deep local knowledge. "
        "Create a complete, personalized tour plan for the profile below.",
        "",
        "================ TRIP PROFILE ================",
        "Base City           : %s" % p["city"],
        "Tour Theme / Focus  : %s" % p["destination"],
        "Group Type          : %s" % group_line,
        "Age Group           : %s" % p["age_group"],
        "Trip Duration       : %s" % duration,
        "Travelers           :",
        member_lines,
        "Group Composition   : %s" % p["relationship"],
        "Budget              : %s" % budget_line,
        "Transport           : %s" % transport_line,
        "Food Preference     : %s" % food_line,
        "Stay Preference     : %s" % p["stay"],
        "",
        "================ REQUIRED OUTPUT FORMAT ================",
        "Write the whole plan in Markdown, in this EXACT structure:",
        "",
        "# <Catchy Trip Title> — <City>",
        "",
        "## 📍 Must-Visit Places & Hidden Gems",
        "A short list of top attractions plus lesser-known local gems, each "
        "with a one-line reason and typical entry ticket price in ₹.",
        "",
        "## 🗺️ Day-by-Day Itinerary",
        "One block for EVERY day of the trip (all %d days), each block "
        "formatted exactly like this:" % trav["duration"],
        "**Day 1 — <short day title>**",
        "**Morning:** <activity, place, timing>",
        "**Afternoon:** <activity, place, timing>",
        "**Evening:** <activity, place, timing>",
        "",
        "## 🏨 Recommended Stays",
        "2-3 properties matching the stay preference, with approximate price "
        "per night in ₹ for the group.",
        "",
        "## 🍽️ Food & Restaurants",
        "Where to eat (matching the food preference), must-try dishes and "
        "approximate cost per meal in ₹.",
        "",
        "## 💰 Detailed Budget Breakdown",
        "A markdown table with columns Category | Details | Cost (₹) covering "
        "Stay, Food, Transport, Tickets, Miscellaneous, ending with a line: "
        "**Total Estimated Cost: ₹<number>** for the whole group.",
        "",
        "## 🎒 Packing & Safety Advice",
        "Packing list for this destination/season plus safety tips and "
        "emergency contacts.",
        "",
        "Rules: use real place names, keep ALL money in Indian Rupees (₹), "
        "match every recommendation to the profile above, and be practical "
        "and specific.",
    ]
    return "\n".join(lines)


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


# ---- POST /api/plan : generate a plan ------------------------------------
@app.route("/api/plan", methods=["POST", "OPTIONS"])
def api_create_plan():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    ok, errors, profile = validate_profile(data)
    if not ok:
        return jsonify({"success": False,
                        "error": "Please complete the highlighted steps.",
                        "details": errors,
                        "hint": "Go back and fix the items listed above."}), 400

    prompt = build_gemini_prompt(profile)
    try:
        plan_text, model = call_gemini_api(prompt)
    except Exception as exc:
        return jsonify({"success": False,
                        "error": "Gemini API failed: %s" % exc,
                        "hint": "Wait a moment and tap Retry."}), 502

    doc = {
        "share_id": _new_share_id(),
        "destination": profile["city"] or "Trip",
        "display": profile["travel"]["display"],
        "city": profile["city"],
        "plan_text": plan_text,
        "model": model,
        "rating": 0,
        "views": 0,
        "profile": profile,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_ts": int(datetime.utcnow().timestamp()),
    }
    try:
        db_save(doc)
    except Exception:
        pass  # storage failure must not kill a generated plan

    return jsonify({"success": True, "share_id": doc["share_id"],
                    "plan": plan_text, "model": model, "profile": profile})


# ---- GET /api/plans : cloud history --------------------------------------
@app.route("/api/plans")
def api_list_plans():
    try:
        limit = min(50, max(1, int(request.args.get("limit", 12))))
    except ValueError:
        limit = 12
    try:
        docs = db_list(limit)
    except Exception:
        docs = []
    plans = []
    for d in docs:
        d.pop("_id", None)
        d.pop("plan_text", None)
        d.pop("profile", None)
        plans.append(d)
    return jsonify({"success": True, "count": len(plans), "plans": plans})


# ---- GET/DELETE /api/plan/<share_id> -------------------------------------
@app.route("/api/plan/<share_id>", methods=["GET", "DELETE"])
def api_get_plan(share_id):
    share_id = "".join(c for c in share_id if c.isalnum() or c in "_-")
    if request.method == "DELETE":
        deleted = db_delete(share_id)
        return jsonify({"success": bool(deleted)}), (200 if deleted else 404)

    doc = db_get(share_id)
    if not doc:
        return jsonify({"success": False,
                        "error": "Plan not found.",
                        "hint": "This trip may have been deleted."}), 404
    doc.pop("_id", None)
    try:
        db_update(share_id, {"views": int(doc.get("views", 0)) + 1})
        doc["views"] = int(doc.get("views", 0)) + 1
    except Exception:
        doc["views"] = int(doc.get("views", 0))
    return jsonify({"success": True, "plan": doc})


# ---- POST /api/plan/<share_id>/feedback ----------------------------------
@app.route("/api/plan/<share_id>/feedback", methods=["POST", "OPTIONS"])
def api_feedback(share_id):
    if request.method == "OPTIONS":
        return ("", 204)
    share_id = "".join(c for c in share_id if c.isalnum() or c in "_-")
    data = request.get_json(silent=True) or {}
    try:
        rating = int(data.get("rating", 0))
        if rating not in (1, -1, 0):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False,
                        "error": "rating must be 1, -1 or 0."}), 400
    doc = db_update(share_id, {"rating": rating})
    if not doc:
        return jsonify({"success": False, "error": "Plan not found."}), 404
    return jsonify({"success": True, "rating": rating})


# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=True)
