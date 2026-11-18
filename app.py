from datetime import date, datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from google import genai
from google.genai import types
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

load_dotenv()

app = Flask(__name__)

# =========================================================
# APP CONFIGURATION
# =========================================================

DEFAULT_SECRET = "nutriai-secret-key"

app.secret_key = os.environ.get("NUTRIAI_SECRET_KEY", DEFAULT_SECRET)

if app.secret_key == DEFAULT_SECRET:
    print(
        "WARNING: Using the default secret key. "
        "Set NUTRIAI_SECRET_KEY before deploying."
    )

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///nutriai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")  # Gmail App Password
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME")

mail = Mail(app)


def send_email(to, subject, body):
    """Send email in a background thread so the page doesn't wait."""
    if not app.config["MAIL_USERNAME"] or not app.config["MAIL_PASSWORD"]:
        return

    def _send():
        with app.app_context():
            try:
                mail.send(Message(subject, recipients=[to], body=body))
            except Exception as e:
                print("Email error:", repr(e))

    threading.Thread(target=_send, daemon=True).start()


# =========================================================
# GEMINI AI CONFIGURATION
# =========================================================

_env_model = os.environ.get("GEMINI_MODEL", "").strip()

GEMINI_MODELS = []
for _m in [
    _env_model,
    "gemini-4-hydrogen",
    "gemini-2.5-flash",
]:
    if _m and _m not in GEMINI_MODELS:
        GEMINI_MODELS.append(_m)

_working_model = None
_gemini_client = None
_cached_api_key = None


def get_gemini_client():
    """Return a configured google-genai Client, or None if no API key is set."""
    global _gemini_client, _cached_api_key

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key in ("YOUR_GEMINI_API_KEY", "MY_GEMINI_API_KEY"):
        return None

    if _gemini_client is None or _cached_api_key != api_key:
        _gemini_client = genai.Client(api_key=api_key)
        _cached_api_key = api_key

    return _gemini_client


# =========================================================
# GENERAL HELPERS
# =========================================================

def safe_float(value, default=0.0):
    """Convert a value to a finite float, or return the default."""
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Convert a value to an integer, or return the default."""
    try:
        return int(round(safe_float(value, default)))
    except (ValueError, TypeError):
        return default


def extract_json(text):
    """
    Parse JSON from Gemini output. Handles markdown code fences
    and extra text around the JSON.
    """
    if not text:
        return None

    text = text.strip()

    fence = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


def call_gemini_json(prompt):
    """
    Send a prompt to Gemini and return parsed JSON (or None).
    Tries each configured model until one works.
    """
    global _working_model

    client = get_gemini_client()
    if not client:
        print("Gemini Error: GEMINI_API_KEY is not configured.")
        return None

    models_to_try = list(GEMINI_MODELS)
    if _working_model in models_to_try:
        models_to_try.remove(_working_model)
        models_to_try.insert(0, _working_model)

    for model_name in models_to_try:
        for _attempt in range(2):
            try:
                print(f"Sending request to Gemini ({model_name})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                response_text = getattr(response, "text", None)
                if not response_text:
                    print("Gemini returned an empty response.")
                    time.sleep(0.8)
                    continue

                data = extract_json(response_text)
                if data is None:
                    print("Gemini returned invalid JSON.")
                    continue

                _working_model = model_name
                print("Gemini response received.")
                return data

            except Exception as e:
                message = repr(e)
                print(f"Gemini Error ({model_name}):", message)
                lowered = message.lower()
                if (
                    "not_found" in lowered
                    or "404" in lowered
                    or "not found" in lowered
                    or "permission" in lowered
                    or "invalid_argument" in lowered
                ):
                    break
                time.sleep(1.2)

    return None


def clean_text_list(value):
    """Return a clean list of non-empty strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


# =========================================================
# DATABASE MODELS & MONGODB TRACKING
# =========================================================

_mongo_client = None
_mongo_db = None
_mongo_init_attempted = False


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_encode_credential(value):
    from urllib.parse import quote, unquote
    return quote(unquote(value), safe="")


def normalize_mongo_uri(raw_uri):
    """Strip quotes/prefixes, detect placeholders, and URL-encode special characters in password."""
    if not raw_uri:
        return None
    cleaned = raw_uri.strip()
    if re.match(r"^MONGODB_URI\s*=", cleaned, re.IGNORECASE):
        cleaned = re.sub(r"^MONGODB_URI\s*=\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    if not cleaned or cleaned.upper() == "MY_MONGODB_URI":
        return None

    m = re.match(r"^(mongodb(?:\+srv)?://)(.+)$", cleaned, re.IGNORECASE)
    if not m:
        return None
    scheme, rest = m.group(1), m.group(2)
    split_points = [idx for idx in (rest.find("/"), rest.find("?")) if idx != -1]
    split_idx = min(split_points) if split_points else -1
    authority = rest if split_idx == -1 else rest[:split_idx]
    tail = "" if split_idx == -1 else rest[split_idx:]

    at_idx = authority.rfind("@")
    if at_idx == -1:
        return f"{scheme}{authority}{tail}"

    userinfo, hosts = authority[:at_idx], authority[at_idx + 1 :]
    if not hosts or ":" not in userinfo:
        return f"{scheme}{authority}{tail}"

    raw_user, raw_pass = userinfo.split(":", 1)
    if re.search(r"<[^>]*(password|user|db_password)[^>]*>", raw_pass, re.IGNORECASE) or raw_pass.upper() == "YOUR_PASSWORD":
        print("MONGODB_URI contains an unreplaced placeholder like <db_password>; using local SQLite database.")
        return None

    return f"{scheme}{_safe_encode_credential(raw_user)}:{_safe_encode_credential(raw_pass)}@{hosts}{tail}"


def get_mongo_db():
    """Return configured MongoDB database handle if MONGODB_URI is set, else None."""
    global _mongo_client, _mongo_db, _mongo_init_attempted

    if _mongo_db is not None:
        return _mongo_db
    if _mongo_init_attempted:
        return None

    _mongo_init_attempted = True
    mongo_uri = normalize_mongo_uri(os.environ.get("MONGODB_URI", ""))
    if not mongo_uri or MongoClient is None:
        return None

    db_name = os.environ.get("MONGODB_DB_NAME", "nutriai").strip() or "nutriai"
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        mdb = client[db_name]
        mdb.users.create_index("email", unique=True)
        mdb.users.create_index("id", unique=True)
        mdb.profiles.create_index("userId", unique=True)
        mdb.profiles.create_index("email", unique=True)
        mdb.usage_logs.create_index([("createdAt", -1)])
        mdb.usage_logs.create_index("userId")
        mdb.password_resets.create_index([("email", 1), ("used", 1)])
        _mongo_client = client
        _mongo_db = mdb
        print(f"Connected to MongoDB database '{db_name}'.")
        return _mongo_db
    except Exception as exc:
        print("MongoDB unavailable, using SQLite fallback:", repr(exc))
        return None


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    diet = db.Column(db.String(50))
    goal = db.Column(db.String(50))
    created_at = db.Column(db.String(40), default=utc_now_iso)
    last_login_at = db.Column(db.String(40))
    last_active_at = db.Column(db.String(40))
    login_count = db.Column(db.Integer, default=0)
    actions_count = db.Column(db.Integer, default=0)


class UsageLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    user_name = db.Column(db.String(100), nullable=False)
    user_email = db.Column(db.String(120), nullable=False)
    event_type = db.Column(db.String(60), nullable=False)
    details = db.Column(db.String(255), default="")
    ip_address = db.Column(db.String(100), default="")
    user_agent = db.Column(db.String(255), default="")
    created_at = db.Column(db.String(40), default=utc_now_iso)


class GroceryItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class MealProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    meal_type = db.Column(db.String(50), nullable=False)
    food = db.Column(db.String(255), nullable=False)
    cost = db.Column(db.Float, default=0)
    completed = db.Column(db.Boolean, default=False)
    date = db.Column(db.Date, nullable=False)


class GeneratedMeal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    meal_type = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, default="")
    ingredients = db.Column(db.Text, nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    calories = db.Column(db.Float, default=0)
    protein = db.Column(db.Float, default=0)
    carbs = db.Column(db.Float, default=0)
    fat = db.Column(db.Float, default=0)
    cost = db.Column(db.Float, default=0)
    date = db.Column(db.Date, nullable=False)


class WaterLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    liters = db.Column(db.Float, default=0.0)
    date = db.Column(db.Date, nullable=False)


class PasswordReset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    code_hash = db.Column(db.String(128), nullable=False)
    expires_at = db.Column(db.String(40), nullable=False)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.String(40), default=utc_now_iso)


with app.app_context():
    db.create_all()
    existing_cols = {
        row[1] for row in db.session.execute(text("PRAGMA table_info(user)")).fetchall()
    }
    if "created_at" not in existing_cols:
        db.session.execute(text("ALTER TABLE user ADD COLUMN created_at VARCHAR(40)"))
    if "last_login_at" not in existing_cols:
        db.session.execute(text("ALTER TABLE user ADD COLUMN last_login_at VARCHAR(40)"))
    if "last_active_at" not in existing_cols:
        db.session.execute(text("ALTER TABLE user ADD COLUMN last_active_at VARCHAR(40)"))
    if "login_count" not in existing_cols:
        db.session.execute(text("ALTER TABLE user ADD COLUMN login_count INTEGER DEFAULT 0"))
    if "actions_count" not in existing_cols:
        db.session.execute(text("ALTER TABLE user ADD COLUMN actions_count INTEGER DEFAULT 0"))
    db.session.commit()


def sync_mongo_user(user):
    """Mirror a user record and profile metrics to MongoDB when configured."""
    mdb = get_mongo_db()
    if mdb is None or user is None:
        return
    try:
        now_iso = utc_now_iso()
        mdb.users.update_one(
            {"id": user.id},
            {
                "$set": {
                    "id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "age": user.age,
                    "height": user.height,
                    "weight": user.weight,
                    "diet": user.diet,
                    "goal": user.goal,
                    "createdAt": user.created_at or now_iso,
                    "lastLoginAt": user.last_login_at,
                    "lastActiveAt": user.last_active_at,
                    "loginCount": user.login_count or 0,
                    "actionsCount": user.actions_count or 0,
                }
            },
            upsert=True,
        )
        bmi, bmi_status = compute_bmi(user.height, user.weight)
        calorie_target, protein_target = compute_nutrition_targets(user)
        weight_val = safe_float(user.weight, 64.0) or 64.0
        water_target = round(max(2.0, min(4.0, weight_val * 0.035)), 1)
        h_val = safe_float(user.height, 0.0)
        if h_val > 0:
            h_m = h_val / 100.0
            min_kg = round(18.5 * h_m * h_m, 1)
            max_kg = round(24.9 * h_m * h_m, 1)
        else:
            min_kg, max_kg = None, None
        mdb.profiles.update_one(
            {"userId": user.id},
            {
                "$set": {
                    "userId": user.id,
                    "name": user.name,
                    "email": user.email,
                    "age": user.age,
                    "height": user.height,
                    "weight": user.weight,
                    "diet": user.diet,
                    "goal": user.goal,
                    "bmi": bmi,
                    "bmiStatus": bmi_status,
                    "healthyWeightRange": {"minKg": min_kg, "maxKg": max_kg},
                    "dailyTargets": {
                        "calorieTarget": calorie_target,
                        "proteinTarget": protein_target,
                        "waterTarget": water_target,
                    },
                    "createdAt": user.created_at or now_iso,
                    "lastLoginAt": user.last_login_at,
                    "lastActiveAt": user.last_active_at,
                    "loginCount": user.login_count or 0,
                    "actionsCount": user.actions_count or 0,
                    "updatedAt": now_iso,
                }
            },
            upsert=True,
        )
    except Exception as exc:
        print("MongoDB user sync warning:", repr(exc))


def record_usage_event(user, event_type, details=""):
    """Record account creation and app usage events in SQLite and MongoDB."""
    if user is None:
        return

    now_iso = utc_now_iso()
    ip_addr = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "")[:100]
    ua = (request.headers.get("User-Agent", "") or "")[:255]

    try:
        if not user.created_at:
            user.created_at = now_iso
        user.last_active_at = now_iso
        if event_type == "account_created":
            user.actions_count = max(1, user.actions_count or 0)
        else:
            user.actions_count = (user.actions_count or 0) + 1

        log_entry = UsageLog(
            user_id=user.id,
            user_name=user.name,
            user_email=user.email,
            event_type=event_type,
            details=details[:255],
            ip_address=ip_addr,
            user_agent=ua,
            created_at=now_iso,
        )
        db.session.add(log_entry)
        db.session.commit()

        sync_mongo_user(user)
        mdb = get_mongo_db()
        if mdb is not None:
            mdb.usage_logs.insert_one(
                {
                    "id": log_entry.id,
                    "userId": user.id,
                    "userName": user.name,
                    "userEmail": user.email,
                    "eventType": event_type,
                    "details": details[:255],
                    "ipAddress": ip_addr,
                    "userAgent": ua,
                    "createdAt": now_iso,
                }
            )
    except Exception as exc:
        db.session.rollback()
        print("Usage log error:", repr(exc))


# =========================================================
# SIMPLE INDIAN MEAL DATABASE & CONSTANTS
# =========================================================

vegetarian = [
    {"type": "Breakfast", "food": "Vegetable Poha + Tea", "cost": 25},
    {"type": "Breakfast", "food": "2 Idli + Sambar", "cost": 30},
    {"type": "Breakfast", "food": "Vegetable Upma", "cost": 25},
    {"type": "Breakfast", "food": "2 Roti + Curd", "cost": 30},
    {"type": "Breakfast", "food": "Besan Chilla + Curd", "cost": 35},
    {"type": "Lunch", "food": "Dal + Rice + Cucumber", "cost": 45},
    {"type": "Lunch", "food": "2 Roti + Dal + Sabzi", "cost": 50},
    {"type": "Lunch", "food": "Rajma Rice", "cost": 50},
    {"type": "Lunch", "food": "Vegetable Khichdi + Curd", "cost": 45},
    {"type": "Lunch", "food": "2 Roti + Chana Masala", "cost": 50},
    {"type": "Snack", "food": "Banana + Roasted Chana", "cost": 20},
    {"type": "Snack", "food": "Sprouts Chaat", "cost": 25},
    {"type": "Snack", "food": "Roasted Makhana", "cost": 30},
    {"type": "Snack", "food": "Fruit Bowl", "cost": 30},
    {"type": "Snack", "food": "Peanuts + Tea", "cost": 20},
    {"type": "Dinner", "food": "2 Roti + Dal + Sabzi", "cost": 50},
    {"type": "Dinner", "food": "Moong Dal Khichdi + Curd", "cost": 45},
    {"type": "Dinner", "food": "Vegetable Pulao + Raita", "cost": 50},
    {"type": "Dinner", "food": "2 Roti + Paneer Bhurji", "cost": 65},
    {"type": "Dinner", "food": "Dal + Rice + Salad", "cost": 45},
]

non_vegetarian = [
    {"type": "Breakfast", "food": "2 Boiled Eggs + 2 Roti", "cost": 35},
    {"type": "Breakfast", "food": "Egg Bhurji + 2 Roti", "cost": 40},
    {"type": "Breakfast", "food": "2 Egg Omelette + Toast", "cost": 40},
    {"type": "Lunch", "food": "Chicken Curry + Rice", "cost": 80},
    {"type": "Lunch", "food": "Chicken Curry + 2 Roti", "cost": 85},
    {"type": "Lunch", "food": "Egg Curry + Rice", "cost": 55},
    {"type": "Lunch", "food": "Chicken Pulao + Raita", "cost": 75},
    {"type": "Snack", "food": "Boiled Eggs + Fruit", "cost": 35},
    {"type": "Snack", "food": "Egg Sandwich", "cost": 40},
    {"type": "Snack", "food": "Roasted Chana + Banana", "cost": 20},
    {"type": "Dinner", "food": "Chicken Curry + 2 Roti", "cost": 85},
    {"type": "Dinner", "food": "Egg Curry + 2 Roti", "cost": 60},
    {"type": "Dinner", "food": "Chicken Rice Bowl", "cost": 75},
    {"type": "Dinner", "food": "Egg Bhurji + 2 Roti + Salad", "cost": 55},
]

vegan = [
    {"type": "Breakfast", "food": "Vegetable Poha", "cost": 25},
    {"type": "Breakfast", "food": "Vegetable Upma", "cost": 25},
    {"type": "Breakfast", "food": "Besan Chilla + Green Chutney", "cost": 30},
    {"type": "Breakfast", "food": "2 Roti + Vegetable Sabzi", "cost": 35},
    {"type": "Lunch", "food": "Dal + Rice + Salad", "cost": 45},
    {"type": "Lunch", "food": "Rajma Rice", "cost": 50},
    {"type": "Lunch", "food": "Chana Masala + 2 Roti", "cost": 50},
    {"type": "Lunch", "food": "Vegetable Khichdi", "cost": 40},
    {"type": "Snack", "food": "Banana + Peanuts", "cost": 20},
    {"type": "Snack", "food": "Sprouts Chaat", "cost": 25},
    {"type": "Snack", "food": "Roasted Chana", "cost": 15},
    {"type": "Snack", "food": "Seasonal Fruit", "cost": 25},
    {"type": "Dinner", "food": "Dal + 2 Roti + Sabzi", "cost": 50},
    {"type": "Dinner", "food": "Rajma Rice", "cost": 50},
    {"type": "Dinner", "food": "Vegetable Pulao", "cost": 45},
    {"type": "Dinner", "food": "Chana Masala + 2 Roti", "cost": 50},
]

eggetarian = [
    {"type": "Breakfast", "food": "2 Boiled Eggs + Poha", "cost": 40},
    {"type": "Breakfast", "food": "Egg Omelette + 2 Roti", "cost": 40},
    {"type": "Breakfast", "food": "Egg Bhurji + Toast", "cost": 40},
    {"type": "Breakfast", "food": "2 Idli + Sambar + Egg", "cost": 40},
    {"type": "Lunch", "food": "Dal + Rice + Boiled Egg", "cost": 50},
    {"type": "Lunch", "food": "Egg Curry + Rice", "cost": 55},
    {"type": "Lunch", "food": "2 Roti + Dal + Egg Bhurji", "cost": 55},
    {"type": "Lunch", "food": "Rajma Rice + Boiled Egg", "cost": 55},
    {"type": "Snack", "food": "Boiled Eggs + Banana", "cost": 35},
    {"type": "Snack", "food": "Egg Sandwich", "cost": 40},
    {"type": "Snack", "food": "Roasted Chana + Fruit", "cost": 25},
    {"type": "Dinner", "food": "Egg Curry + 2 Roti", "cost": 60},
    {"type": "Dinner", "food": "Egg Bhurji + 2 Roti + Salad", "cost": 55},
    {"type": "Dinner", "food": "Dal Rice + Boiled Eggs", "cost": 55},
    {"type": "Dinner", "food": "Vegetable Pulao + Egg", "cost": 55},
]

jain = [
    {"type": "Breakfast", "food": "Moong Dal Chilla + Mint Chutney", "cost": 30},
    {"type": "Breakfast", "food": "Jain Poha (No Onion/Garlic) + Curd", "cost": 25},
    {"type": "Lunch", "food": "Toor Dal + Jeera Rice + Lauki Sabzi", "cost": 50},
    {"type": "Lunch", "food": "2 Roti + Paneer Bhurji (No Onion/Garlic)", "cost": 65},
    {"type": "Snack", "food": "Roasted Makhana + Almonds", "cost": 30},
    {"type": "Snack", "food": "Fresh Fruit Bowl", "cost": 30},
    {"type": "Dinner", "food": "Moong Dal Khichdi + Curd", "cost": 45},
    {"type": "Dinner", "food": "2 Roti + Yellow Dal Tadka (No Garlic)", "cost": 45},
]

meals_database = {
    "Vegetarian": vegetarian,
    "Non-Vegetarian": non_vegetarian,
    "Vegan": vegan,
    "Eggetarian": eggetarian,
    "Jain": jain,
}

meal_costs = {
    "Breakfast": 55,
    "Lunch": 95,
    "Snack": 35,
    "Dinner": 110,
    "Extra": 40,
}

ALLOWED_DIETS = [
    "Vegetarian",
    "Non-Vegetarian",
    "Eggetarian",
    "Vegan",
    "Jain",
]

ALLOWED_GOALS = [
    "Weight Loss",
    "Weight Gain",
    "Muscle Gain",
    "Healthy Lifestyle",
    "Maintain Weight",
]

DASHBOARD_MEAL_MAP = {
    "Breakfast": "Breakfast",
    "Lunch": "Lunch",
    "Snack": "Snack",
    "Mid-Morning Snack": "Snack",
    "Evening Snack": "Snack",
    "Dinner": "Dinner",
}


# =========================================================
# NUTRITION & TARGET CALCULATIONS
# =========================================================

def compute_nutrition_targets(user):
    """Calculate daily calorie & protein targets via Mifflin-St Jeor + goal."""
    weight = safe_float(getattr(user, "weight", None), 62.0) or 62.0
    height = safe_float(getattr(user, "height", None), 168.0) or 168.0
    age = safe_float(getattr(user, "age", None), 25.0) or 25.0
    goal = (getattr(user, "goal", None) or "Healthy Lifestyle").strip()

    bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) - 50.0
    tdee = bmr * 1.375

    offsets = {
        "Weight Loss": -400.0,
        "Weight Gain": 400.0,
        "Muscle Gain": 300.0,
        "Healthy Lifestyle": 0.0,
        "Maintain Weight": 0.0,
    }
    calorie_target = max(1200, min(4200, int(round((tdee + offsets.get(goal, 0.0)) / 10.0) * 10)))

    protein_multiplier = 1.4
    if goal == "Muscle Gain":
        protein_multiplier = 1.8
    elif goal == "Weight Loss":
        protein_multiplier = 1.5
    elif goal == "Weight Gain":
        protein_multiplier = 1.6

    protein_target = max(45, min(220, int(round(weight * protein_multiplier))))
    return calorie_target, protein_target


def compute_bmi(height_cm, weight_kg):
    """Return (bmi, status) given height in cm and weight in kg."""
    h = safe_float(height_cm, 0.0)
    w = safe_float(weight_kg, 0.0)
    if h <= 0 or w <= 0:
        return None, ""
    h_m = h / 100.0
    bmi = round(w / (h_m * h_m), 1)
    if bmi < 18.5:
        status = "Underweight"
    elif bmi < 25:
        status = "Healthy Range"
    elif bmi < 30:
        status = "Overweight"
    else:
        status = "Obesity Range"
    return bmi, status


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# VALIDATION HELPERS
# =========================================================

def is_valid_email(email):
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return re.fullmatch(email_pattern, email) is not None


def is_strong_password(password):
    if len(password) < 8 or len(password) > 128:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[^A-Za-z0-9]", password):
        return False
    return True


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        age_raw = request.form.get("age", "")
        height_raw = request.form.get("height", "")
        weight_raw = request.form.get("weight", "")
        diet = request.form.get("diet", "").strip()
        goal = request.form.get("goal", "").strip()

        if not name:
            flash("Please enter your full name.")
            return redirect(url_for("register"))
        if len(name) < 2:
            flash("Name must contain at least 2 characters.")
            return redirect(url_for("register"))
        if len(name) > 100:
            flash("Name is too long.")
            return redirect(url_for("register"))
        if not re.fullmatch(r"[A-Za-zÀ-ÿ\s.'-]+", name):
            flash("Please enter a valid name.")
            return redirect(url_for("register"))

        if not email:
            flash("Please enter your email address.")
            return redirect(url_for("register"))
        if len(email) > 120:
            flash("Email address is too long.")
            return redirect(url_for("register"))
        if not is_valid_email(email):
            flash("Please enter a valid email address.")
            return redirect(url_for("register"))

        if not password:
            flash("Please create a password.")
            return redirect(url_for("register"))
        if not is_strong_password(password):
            flash(
                "Password must contain at least 8 characters, "
                "one uppercase letter, one lowercase letter, "
                "one number and one special character."
            )
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists. Please login.")
            return redirect(url_for("login"))

        try:
            age = int(age_raw)
            height = float(height_raw)
            weight = float(weight_raw)
        except (ValueError, TypeError):
            flash("Please enter valid age, height and weight.")
            return redirect(url_for("register"))

        if math.isnan(height) or math.isinf(height) or math.isnan(weight) or math.isinf(weight):
            flash("Please enter valid age, height and weight.")
            return redirect(url_for("register"))

        if age < 13 or age > 120:
            flash("Age must be between 13 and 120.")
            return redirect(url_for("register"))
        if height < 50 or height > 250:
            flash("Height must be between 50 and 250 cm.")
            return redirect(url_for("register"))
        if weight < 20 or weight > 300:
            flash("Weight must be between 20 and 300 kg.")
            return redirect(url_for("register"))

        if diet not in ALLOWED_DIETS:
            flash("Please select a valid diet preference.")
            return redirect(url_for("register"))

        if goal not in ALLOWED_GOALS:
            flash("Please select a valid fitness goal.")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password, method="scrypt")

        now_iso = utc_now_iso()
        user = User(
            name=name,
            email=email,
            password=hashed_password,
            age=age,
            height=height,
            weight=weight,
            diet=diet,
            goal=goal,
            created_at=now_iso,
            last_active_at=now_iso,
            login_count=0,
            actions_count=1,
        )

        try:
            db.session.add(user)
            db.session.commit()
        except Exception:
            db.session.rollback()
            flash("Something went wrong while creating your account. Please try again.")
            return redirect(url_for("register"))

        record_usage_event(
            user,
            "account_created",
            f"Registered account ({diet}, {goal})",
        )

        send_email(
            user.email,
            "Welcome to NutriAI — Registration Confirmed",
            f"Hi {user.name},\n\n"
            f"You have successfully registered on NutriAI with this email address ({user.email}).\n"
            f"Diet Preference: {diet}\n"
            f"Fitness Goal: {goal}\n\n"
            "You can now sign in and explore your personalized dashboard.\n\n"
            "— NutriAI",
        )

        if app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"):
            flash(f"Account created successfully! Confirmation email sent to {user.email}. Please login.")
        else:
            flash("Account created successfully! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter your email and password.")
            return redirect(url_for("login"))

        if not is_valid_email(email):
            flash("Please enter a valid email address.")
            return redirect(url_for("login"))

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session.clear()
            session["user_id"] = user.id
            session.permanent = True
            session["login_success"] = True

            now_iso = utc_now_iso()
            user.last_login_at = now_iso
            user.last_active_at = now_iso
            user.login_count = (user.login_count or 0) + 1
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
            record_usage_event(user, "login", "Signed in to NutriAI")

            send_email(
                user.email,
                "New login to NutriAI",
                f"Hi {user.name},\n\nYour account was just logged in. "
                "If this wasn't you, please change your password.",
            )

            return redirect(url_for("dashboard"))

        flash("Invalid email or password.")
        return redirect(url_for("login"))

    return render_template("login.html")


def _hash_reset_code(email, code):
    secret = str(app.secret_key or DEFAULT_SECRET).encode("utf-8")
    msg = f"reset:{email.strip().lower()}:{code.strip()}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    step = request.args.get("step", "request")
    email = request.args.get("email", "").strip().lower()
    code_prefill = request.args.get("code", "").strip()
    if email and code_prefill:
        step = "reset"
    preview_code = None

    if request.method == "POST":
        action = request.form.get("action", "request")
        email = request.form.get("email", "").strip().lower()

        if action == "request":
            if not email or not is_valid_email(email):
                flash("Please enter a valid email address.")
                return render_template("forgot_password.html", step="request", email=email, code="")

            user = User.query.filter_by(email=email).first()
            if not user:
                flash("No NutriAI account found with that email address.")
                return render_template("forgot_password.html", step="request", email=email, code="")

            code = str(100000 + secrets.randbelow(900000))
            code_hash = _hash_reset_code(user.email, code)
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()

            PasswordReset.query.filter_by(email=user.email, used=False).update({"used": True})
            reset_row = PasswordReset(
                user_id=user.id,
                email=user.email,
                code_hash=code_hash,
                expires_at=expires_at,
                used=False,
            )
            db.session.add(reset_row)
            db.session.commit()

            mdb = get_mongo_db()
            if mdb is not None:
                mdb.password_resets.update_many({"email": user.email, "used": 0}, {"$set": {"used": 1}})
                mdb.password_resets.insert_one(
                    {
                        "id": reset_row.id,
                        "userId": user.id,
                        "email": user.email,
                        "codeHash": code_hash,
                        "expiresAt": expires_at,
                        "used": 0,
                        "createdAt": utc_now_iso(),
                    }
                )

            smtp_ready = bool(app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"))
            if smtp_ready:
                send_email(
                    user.email,
                    "NutriAI Password Reset Code",
                    f"Hi {user.name},\n\nYour 6-digit NutriAI password reset code is: {code}\n"
                    "This code expires in 15 minutes.\n\n— NutriAI Security",
                )
                flash(f"A 6-digit password reset code has been sent to {user.email}.")
            else:
                preview_code = code
                flash("Verification code generated below. Enter it with your new password.")

            record_usage_event(user, "password_reset_requested", f"Password reset code issued for {user.email}")
            return render_template(
                "forgot_password.html",
                step="reset",
                email=user.email,
                code="",
                preview_code=preview_code,
            )

        # action == "reset"
        code = request.form.get("code", "").strip()
        new_password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not email or not is_valid_email(email):
            flash("Please enter a valid email address.")
            return render_template("forgot_password.html", step="reset", email=email, code=code)
        if not re.fullmatch(r"\d{6}", code):
            flash("Please enter the 6-digit verification code.")
            return render_template("forgot_password.html", step="reset", email=email, code=code)
        if not is_strong_password(new_password):
            flash("Password must contain at least 8 characters, uppercase, lowercase, number, and special character.")
            return render_template("forgot_password.html", step="reset", email=email, code=code)
        if new_password != confirm_password:
            flash("Passwords do not match.")
            return render_template("forgot_password.html", step="reset", email=email, code=code)

        user = User.query.filter_by(email=email).first()
        record = (
            PasswordReset.query.filter_by(email=email, used=False)
            .order_by(PasswordReset.id.desc())
            .first()
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        if not user or not record or record.code_hash != _hash_reset_code(email, code) or record.expires_at <= now_iso:
            flash("Invalid or expired verification code. Please request a new code.")
            return render_template("forgot_password.html", step="reset", email=email, code="")

        record.used = True
        user.password = generate_password_hash(new_password, method="scrypt")
        db.session.commit()
        record_usage_event(user, "password_reset_completed", f"Reset password for {user.email}")

        if app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"):
            send_email(
                user.email,
                "Your NutriAI password was changed",
                f"Hi {user.name},\n\nYour NutriAI password was successfully updated. You can now sign in with your new password.\n\n— NutriAI Security",
            )

        flash("Password updated successfully! Please login with your new password.")
        return redirect(url_for("login"))

    return render_template(
        "forgot_password.html",
        step=step,
        email=email,
        code=code_prefill,
        preview_code=preview_code,
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    today = date.today()
    meals = MealProgress.query.filter_by(user_id=user.id, date=today).all()
    generated_today = GeneratedMeal.query.filter_by(user_id=user.id, date=today).all()

    meal_status = {meal.meal_type: meal.completed for meal in meals}

    completed_meals = sum(
        1
        for meal in meals
        if meal.completed and meal.meal_type in ["Breakfast", "Lunch", "Snack", "Dinner"]
    )
    meal_percentage = min(100, int((completed_meals / 4) * 100))

    completed_cost = round(sum(meal.cost or 0 for meal in meals if meal.completed), 2)

    calorie_target, protein_target = compute_nutrition_targets(user)

    # Calculate consumed calories/protein from completed generated meals if available,
    # otherwise proportional to completed meals
    gen_by_type = {}
    for g in generated_today:
        mapped = DASHBOARD_MEAL_MAP.get(g.meal_type, g.meal_type)
        if mapped not in gen_by_type:
            gen_by_type[mapped] = g

    calories_consumed = 0.0
    protein_consumed = 0.0
    for m_type, is_done in meal_status.items():
        if is_done:
            if m_type in gen_by_type:
                calories_consumed += safe_float(gen_by_type[m_type].calories, 0.0)
                protein_consumed += safe_float(gen_by_type[m_type].protein, 0.0)
            else:
                calories_consumed += calorie_target / 4.0
                protein_consumed += protein_target / 4.0

    calorie_percentage = min(100, int(round((calories_consumed / calorie_target) * 100))) if calorie_target else 0
    protein_percentage = min(100, int(round((protein_consumed / protein_target) * 100))) if protein_target else 0

    bmi, bmi_status = compute_bmi(user.height, user.weight)
    h_val = safe_float(user.height, 0.0)
    if h_val > 0:
        h_m = h_val / 100.0
        healthy_min_kg = round(18.5 * h_m * h_m, 1)
        healthy_max_kg = round(24.9 * h_m * h_m, 1)
    else:
        healthy_min_kg, healthy_max_kg = None, None
    bmi_pointer_pct = (
        int(round(((max(14.0, min(38.0, bmi)) - 14.0) / 24.0) * 100))
        if bmi is not None
        else 0
    )

    water_log = WaterLog.query.filter_by(user_id=user.id, date=today).first()
    water_today = round(safe_float(water_log.liters if water_log else 0.0, 0.0), 2)
    weight_val = safe_float(user.weight, 64.0) or 64.0
    water_target = round(max(2.0, min(4.0, weight_val * 0.035)), 1)
    water_pct = min(100, int(round((water_today / water_target) * 100))) if water_target else 0
    water_remaining = max(0.0, round(water_target - water_today, 2))
    filled_glasses = min(8, int(round((water_today / water_target) * 8))) if water_target else 0

    recent_logs = (
        UsageLog.query.filter_by(user_id=user.id)
        .order_by(UsageLog.id.desc())
        .limit(8)
        .all()
    )
    notifications = []
    if water_today < water_target:
        notifications.append(
            {
                "icon": "💧",
                "title": "Hydration Goal Reminder",
                "message": f"You have logged {water_today} L today — {water_remaining} L left for your {water_target} L goal.",
                "time": "Today",
            }
        )
    else:
        notifications.append(
            {
                "icon": "💧",
                "title": "Daily Hydration Goal Reached",
                "message": f"Great job! You reached {water_today} L out of {water_target} L today.",
                "time": "Today",
            }
        )
    for log_item in recent_logs:
        ev = log_item.event_type
        icon = "🔒" if ev == "login" else "💧" if ev == "water_logged" else "🍽️" if "meal" in ev else "🌿"
        title = (
            "New Sign-In"
            if ev == "login"
            else "Water Intake Logged"
            if ev == "water_logged"
            else "Meal Update"
            if "meal" in ev
            else "Account Activity"
        )
        notifications.append(
            {
                "icon": icon,
                "title": title,
                "message": log_item.details,
                "time": (log_item.created_at or "")[11:16] or "Recent",
            }
        )

    show_welcome = session.pop("login_success", False)

    return render_template(
        "dashboard.html",
        user=user,
        meals=meals,
        meal_status=meal_status,
        completed_meals=completed_meals,
        meal_percentage=meal_percentage,
        completed_cost=completed_cost,
        calorie_target=calorie_target,
        protein_target=protein_target,
        calories_consumed=int(round(calories_consumed)),
        protein_consumed=int(round(protein_consumed)),
        calorie_percentage=calorie_percentage,
        protein_percentage=protein_percentage,
        water_today=water_today,
        water_target=water_target,
        water_pct=water_pct,
        water_remaining=water_remaining,
        filled_glasses=filled_glasses,
        notifications=notifications,
        bmi=bmi,
        bmi_status=bmi_status,
        healthy_min_kg=healthy_min_kg,
        healthy_max_kg=healthy_max_kg,
        bmi_pointer_pct=bmi_pointer_pct,
        show_welcome=show_welcome,
        today=today,
    )


# =========================================================
# COMPLETE / UNCOMPLETE MEAL
# =========================================================

@app.route("/complete-meal/<meal_type>", methods=["POST"])
def complete_meal(meal_type):
    if "user_id" not in session:
        return redirect(url_for("login"))

    allowed_meals = ["Breakfast", "Lunch", "Snack", "Dinner"]
    next_endpoint = request.form.get("next") or request.args.get("next") or "dashboard"
    if next_endpoint not in ("dashboard", "mealplanner"):
        next_endpoint = "dashboard"

    if meal_type not in allowed_meals:
        flash("Invalid meal type.")
        return redirect(url_for(next_endpoint))

    today = date.today()
    existing_meal = MealProgress.query.filter_by(
        user_id=session["user_id"],
        meal_type=meal_type,
        date=today,
    ).first()

    if existing_meal:
        existing_meal.completed = not existing_meal.completed
    else:
        new_meal = MealProgress(
            user_id=session["user_id"],
            meal_type=meal_type,
            food="Meal completed",
            cost=meal_costs.get(meal_type, 0),
            completed=True,
            date=today,
        )
        db.session.add(new_meal)

    try:
        db.session.commit()
        user_obj = db.session.get(User, session["user_id"])
        status_label = "completed" if (existing_meal.completed if existing_meal else True) else "pending"
        record_usage_event(user_obj, "meal_toggled", f"{meal_type} marked {status_label}")
    except Exception:
        db.session.rollback()
        flash("Could not update meal. Please try again.")

    return redirect(url_for(next_endpoint))


# =========================================================
# GEMINI AI MEAL PLANNER
# =========================================================

def clean_ai_meal(meal, meal_types):
    """Validate and clean a single meal returned by Gemini."""
    if not isinstance(meal, dict):
        return None

    meal_type = str(meal.get("type", "")).strip()
    food = str(meal.get("food", "")).strip()

    if not meal_type or not food or meal_type not in meal_types:
        return None

    nutrition = meal.get("nutrition", {})
    if not isinstance(nutrition, dict):
        nutrition = {}

    return {
        "type": meal_type,
        "food": food,
        "description": str(meal.get("description", "")).strip(),
        "cost": round(max(0.0, safe_float(meal.get("cost", 0))), 2),
        "ingredients": clean_text_list(meal.get("ingredients", [])),
        "instructions": clean_text_list(meal.get("instructions", [])),
        "nutrition": {
            "calories": round(safe_float(nutrition.get("calories", 0)), 1),
            "protein": round(safe_float(nutrition.get("protein", 0)), 1),
            "carbs": round(safe_float(nutrition.get("carbs", 0)), 1),
            "fat": round(safe_float(nutrition.get("fat", 0)), 1),
        },
    }


def generate_ai_meal_plan(goal, diet, budget, meals_per_day):
    if not get_gemini_client():
        return None

    if meals_per_day == 3:
        meal_types = ["Breakfast", "Lunch", "Dinner"]
    elif meals_per_day == 4:
        meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
    elif meals_per_day == 5:
        meal_types = [
            "Breakfast",
            "Mid-Morning Snack",
            "Lunch",
            "Evening Snack",
            "Dinner",
        ]
    else:
        return None

    budget_text = (
        str(int(budget))
        if float(budget).is_integer()
        else str(round(budget, 2))
    )

    prompt = f"""
You are NutriAI, an AI Indian meal planning assistant.

Create a complete one-day Indian meal plan for the user.

USER INFORMATION
Fitness Goal: {goal}
Diet Preference: {diet}
Daily Food Budget: ₹{budget_text}
Number of Meals: {meals_per_day}
MEAL TYPES: {", ".join(meal_types)}

IMPORTANT RULES:
1. Generate exactly {meals_per_day} meals using each of these meal types once: {", ".join(meal_types)}.
2. Every meal MUST include a short appetizing description, 3 to 8 affordable Indian ingredients, and beginner-friendly cooking instructions.
3. Cooking time should normally be between 10 and 30 minutes.
4. Respect the user's diet preference strictly (for Jain: no onion, garlic, or root vegetables).
5. Consider the user's fitness goal and keep total daily cost within or close to ₹{budget_text}.
6. Return ONLY valid JSON (no markdown, no extra text).

RETURN EXACTLY THIS JSON ARRAY STRUCTURE:
[
    {{
        "type": "Breakfast",
        "food": "Vegetable Poha with Curd",
        "description": "Light, energizing flattened rice dish tempered with mustard seeds and fresh vegetables.",
        "cost": 30,
        "ingredients": [
            "Poha - 1 cup",
            "Onion - 1 small",
            "Carrot - 1 small",
            "Green peas - 1/4 cup",
            "Oil - 1 teaspoon",
            "Curd - 1/2 cup"
        ],
        "instructions": [
            "Wash the poha and drain the water.",
            "Heat oil in a pan and saute vegetables.",
            "Add poha, salt and turmeric and mix well.",
            "Cook for 3 to 5 minutes and serve with curd."
        ],
        "nutrition": {{
            "calories": 320,
            "protein": 9,
            "carbs": 48,
            "fat": 10
        }}
    }}
]
"""

    for _attempt in range(2):
        ai_plan = call_gemini_json(prompt)

        if isinstance(ai_plan, dict):
            for key in ("meals", "meal_plan", "plan"):
                if isinstance(ai_plan.get(key), list):
                    ai_plan = ai_plan[key]
                    break

        if not isinstance(ai_plan, list):
            continue

        cleaned_plan = []
        used_types = set()

        for meal in ai_plan:
            cleaned = clean_ai_meal(meal, meal_types)
            if cleaned is None or cleaned["type"] in used_types:
                continue
            used_types.add(cleaned["type"])
            cleaned_plan.append(cleaned)

        if len(cleaned_plan) != meals_per_day:
            continue

        order = {name: i for i, name in enumerate(meal_types)}
        cleaned_plan.sort(key=lambda m: order[m["type"]])
        return cleaned_plan

    return None


@app.route("/mealplanner", methods=["GET", "POST"])
def mealplanner():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    selected_goal = user.goal or "Healthy Lifestyle"
    selected_diet = user.diet or "Vegetarian"
    selected_budget = 250.0
    selected_meals = 4
    meal_plan = []
    error = None

    today = date.today()

    if request.method == "POST":
        selected_goal = request.form.get("goal", selected_goal).strip()
        selected_diet = request.form.get("diet", selected_diet).strip()

        if selected_goal not in ALLOWED_GOALS:
            error = "Please select a valid fitness goal."

        if error is None and selected_diet not in ALLOWED_DIETS:
            error = "Please select a valid diet preference."

        if error is None:
            try:
                selected_budget = float(request.form.get("budget", 250))
                if math.isnan(selected_budget) or math.isinf(selected_budget):
                    raise ValueError
            except (ValueError, TypeError):
                selected_budget = 250.0
                error = "Please enter a valid budget."

        if error is None:
            try:
                selected_meals = int(request.form.get("meals", 4))
            except (ValueError, TypeError):
                selected_meals = 4
                error = "Please select a valid number of meals."

        if error is None and selected_budget <= 0:
            error = "Daily budget must be greater than ₹0."

        if error is None and (selected_meals < 3 or selected_meals > 5):
            error = "Meals per day must be between 3 and 5."

        if error is None:
            meal_plan = generate_ai_meal_plan(
                selected_goal,
                selected_diet,
                selected_budget,
                selected_meals,
            )

            if not meal_plan:
                meal_plan = []
                error = (
                    "AI meal generation is currently unavailable. "
                    "Please check your GEMINI_API_KEY configuration and try again."
                )
            else:
                try:
                    MealProgress.query.filter_by(user_id=user.id, date=today).delete()
                    GeneratedMeal.query.filter_by(user_id=user.id, date=today).delete()

                    for meal in meal_plan:
                        nutrition = meal.get("nutrition", {})
                        generated_meal = GeneratedMeal(
                            user_id=user.id,
                            meal_type=meal.get("type", "Meal"),
                            name=meal.get("food", "AI Generated Meal"),
                            description=meal.get("description", ""),
                            ingredients=json.dumps(meal.get("ingredients", [])),
                            instructions=json.dumps(meal.get("instructions", [])),
                            calories=safe_float(nutrition.get("calories", 0)),
                            protein=safe_float(nutrition.get("protein", 0)),
                            carbs=safe_float(nutrition.get("carbs", 0)),
                            fat=safe_float(nutrition.get("fat", 0)),
                            cost=safe_float(meal.get("cost", 0)),
                            date=today,
                        )
                        db.session.add(generated_meal)

                    tracked_types = set()
                    for meal in meal_plan:
                        dashboard_type = DASHBOARD_MEAL_MAP.get(meal.get("type", "").strip())
                        if dashboard_type is None or dashboard_type in tracked_types:
                            continue
                        tracked_types.add(dashboard_type)
                        db.session.add(
                            MealProgress(
                                user_id=user.id,
                                meal_type=dashboard_type,
                                food=meal.get("food", "").strip(),
                                cost=safe_float(meal.get("cost", 0)),
                                completed=False,
                                date=today,
                            )
                        )

                    db.session.commit()
                    record_usage_event(
                        user,
                        "meal_plan_generated",
                        f"{selected_meals} {selected_diet} meals ({selected_goal}, ₹{selected_budget})",
                    )
                except Exception as e:
                    db.session.rollback()
                    print("Meal plan save error:", repr(e))
                    error = "Your meal plan was generated but could not be saved. Please try again."
    else:
        # Load today's persisted meal plan on GET
        saved_rows = (
            GeneratedMeal.query.filter_by(user_id=user.id, date=today)
            .order_by(GeneratedMeal.id.asc())
            .all()
        )
        for row in saved_rows:
            try:
                ingredients = json.loads(row.ingredients)
            except Exception:
                ingredients = []
            try:
                instructions = json.loads(row.instructions)
            except Exception:
                instructions = []
            meal_plan.append(
                {
                    "type": row.meal_type,
                    "food": row.name,
                    "description": row.description or "",
                    "cost": round(safe_float(row.cost, 0), 2),
                    "ingredients": ingredients if isinstance(ingredients, list) else [],
                    "instructions": instructions if isinstance(instructions, list) else [],
                    "nutrition": {
                        "calories": round(safe_float(row.calories, 0), 1),
                        "protein": round(safe_float(row.protein, 0), 1),
                        "carbs": round(safe_float(row.carbs, 0), 1),
                        "fat": round(safe_float(row.fat, 0), 1),
                    },
                }
            )

        if len(meal_plan) in (3, 4, 5):
            selected_meals = len(meal_plan)

    progress_rows = MealProgress.query.filter_by(user_id=user.id, date=today).all()
    meal_status = {row.meal_type: row.completed for row in progress_rows}

    total_cost = round(sum(safe_float(item.get("cost", 0)) for item in meal_plan), 2)
    remaining_budget = round(selected_budget - total_cost, 2)
    budget_status = "within" if remaining_budget >= 0 else "over"

    return render_template(
        "mealplanner.html",
        user=user,
        meal_plan=meal_plan,
        meal_status=meal_status,
        selected_goal=selected_goal,
        selected_diet=selected_diet,
        selected_budget=int(selected_budget) if float(selected_budget).is_integer() else selected_budget,
        selected_meals=selected_meals,
        total_cost=total_cost,
        remaining_budget=remaining_budget,
        budget_status=budget_status,
        error=error,
    )


# =========================================================
# ADD MEAL INGREDIENTS TO GROCERY
# =========================================================

@app.route("/add-meal-ingredients", methods=["POST"])
def add_meal_ingredients():
    if "user_id" not in session:
        return redirect(url_for("login"))

    ingredients = request.form.getlist("ingredients")
    if not ingredients:
        flash("No ingredients found.", "error")
        return redirect(url_for("mealplanner"))

    added_count = 0
    for ingredient in ingredients:
        ingredient = ingredient.strip()[:100]
        if not ingredient:
            continue

        existing_item = GroceryItem.query.filter_by(
            user_id=session["user_id"],
            item=ingredient,
        ).first()
        if existing_item:
            continue

        db.session.add(
            GroceryItem(
                item=ingredient,
                quantity="As needed",
                price=0,
                user_id=session["user_id"],
            )
        )
        added_count += 1

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        flash("Could not add ingredients. Please try again.", "error")
        return redirect(url_for("mealplanner"))

    flash(f"🛒 {added_count} ingredient(s) added to your grocery list!", "success")
    return redirect(url_for("grocery"))


# =========================================================
# NUTRITION ANALYZER
# =========================================================

def calculate_nutrition_score(calories, protein, carbs, fat):
    score = 50

    if protein >= 20:
        score += 20
    elif protein >= 10:
        score += 12
    elif protein >= 5:
        score += 6

    if calories <= 100:
        score += 15
    elif calories <= 200:
        score += 10
    elif calories <= 300:
        score += 5

    if fat <= 5:
        score += 10
    elif fat <= 15:
        score += 5

    if carbs <= 30:
        score += 5

    score = max(0, min(score, 100))

    if score >= 85:
        message = (
            "Excellent nutritional profile. "
            "This food can fit well into a balanced diet."
        )
    elif score >= 70:
        message = (
            "Good nutritional profile. "
            "Pair it with other nutrient-rich foods for a balanced meal."
        )
    elif score >= 50:
        message = (
            "Moderate nutritional profile. "
            "Portion size and the rest of your meal matter."
        )
    else:
        message = (
            "This food is more calorie-dense or less protein-rich, "
            "so consider portion size."
        )

    return score, message


def analyze_food_with_gemini(food, quantity):
    """Use Gemini to estimate nutrition values for any food outside the local table."""
    if not get_gemini_client():
        return None

    clean_food = re.sub(r"[\r\n\t]+", " ", food).strip()[:120]
    prompt = f"""
You are a nutrition database assistant.
Estimate realistic nutritional values for:
Food: {clean_food}
Serving weight: {quantity} grams

Return ONLY valid JSON with this exact schema:
{{
    "calories": 250,
    "protein": 12.5,
    "carbs": 30.0,
    "fat": 8.2,
    "summary": "A brief 1-2 sentence nutritional breakdown of this food serving."
}}
"""
    data = call_gemini_json(prompt)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None

    calories = round(max(0.0, safe_float(data.get("calories", 0))), 1)
    protein = round(max(0.0, safe_float(data.get("protein", 0))), 1)
    carbs = round(max(0.0, safe_float(data.get("carbs", 0))), 1)
    fat = round(max(0.0, safe_float(data.get("fat", 0))), 1)

    if calories <= 0 and protein <= 0 and carbs <= 0 and fat <= 0:
        return None

    score, score_message = calculate_nutrition_score(calories, protein, carbs, fat)
    summary = str(data.get("summary") or "").strip()
    if not summary:
        summary = (
            f"{clean_food.title()} provides approximately {calories} calories for {quantity}g. "
            f"It contains {protein}g protein, {carbs}g carbohydrates and {fat}g fat."
        )

    return {
        "food": clean_food.title(),
        "quantity": quantity,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "score": score,
        "score_message": score_message,
        "summary": summary,
    }


@app.route("/nutrition", methods=["GET", "POST"])
def nutrition():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    nutrition_data = None
    error = None
    food_display = ""
    quantity_value = 100

    if request.method == "POST":
        food = request.form.get("food", "").strip()
        quantity_raw = request.form.get("quantity", "100").strip()
        food_display = food.title()

        if not food:
            error = "Please enter a food item."
        else:
            try:
                quantity = float(quantity_raw)
                if math.isnan(quantity) or math.isinf(quantity) or quantity <= 0:
                    raise ValueError
                quantity_value = int(quantity) if quantity.is_integer() else round(quantity, 1)
            except ValueError:
                error = "Please enter a valid quantity."

            if not error:
                nutrition_db = {
                    "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
                    "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
                    "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
                    "paneer": {"calories": 265, "protein": 18, "carbs": 6, "fat": 20},
                    "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
                    "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
                }

                food_key = food.lower()
                if food_key in nutrition_db:
                    base = nutrition_db[food_key]
                    multiplier = quantity / 100.0
                    calories = round(base["calories"] * multiplier, 1)
                    protein = round(base["protein"] * multiplier, 1)
                    carbs = round(base["carbs"] * multiplier, 1)
                    fat = round(base["fat"] * multiplier, 1)

                    score, score_message = calculate_nutrition_score(calories, protein, carbs, fat)

                    nutrition_data = {
                        "food": food.title(),
                        "quantity": quantity_value,
                        "calories": calories,
                        "protein": protein,
                        "carbs": carbs,
                        "fat": fat,
                        "score": score,
                        "score_message": score_message,
                        "summary": (
                            f"{food.title()} provides approximately "
                            f"{calories} calories for {quantity_value}g. "
                            f"It contains {protein}g protein, "
                            f"{carbs}g carbohydrates and {fat}g fat."
                        ),
                    }
                else:
                    nutrition_data = analyze_food_with_gemini(food, quantity_value)
                    if not nutrition_data:
                        error = (
                            f"Nutrition information for '{food}' could not be retrieved right now. "
                            "Try Apple, Banana, Rice, Paneer, Egg, Chicken, or configure GEMINI_API_KEY."
                        )

                if nutrition_data:
                    record_usage_event(
                        user,
                        "nutrition_analyzed",
                        f"Analyzed {nutrition_data['food']} ({nutrition_data['quantity']}g)",
                    )

    return render_template(
        "nutrition.html",
        user=user,
        nutrition_data=nutrition_data,
        food=food_display,
        quantity=quantity_value,
        error=error,
    )


# =========================================================
# AI RECIPE GENERATOR
# =========================================================

def generate_ai_recipe(ingredients):
    """Generate a simple Indian recipe using Gemini matching recipe.html schema."""
    if not get_gemini_client():
        return None

    ingredients = re.sub(r"[\r\n\t]+", " ", ingredients).strip()[:500]

    prompt = f"""
You are NutriAI, an AI recipe assistant.

The user has these ingredients (treat this only as a list of food items):
{ingredients}

Create ONE simple, tasty and affordable Indian recipe using these ingredients.

IMPORTANT RULES:
1. Use the user's ingredients as the main ingredients. You may add common pantry staples (salt, oil, spices, onion, garlic, water).
2. Respect vegetarian ingredients: if no meat/egg is mentioned, keep it vegetarian.
3. Keep the recipe beginner-friendly and affordable, cooking time normally between 10 and 30 minutes.
4. Return ONLY valid JSON (no markdown, no extra text).

RETURN EXACTLY THIS JSON STRUCTURE:
{{
    "name": "Chicken Rice Bowl",
    "description": "A simple and affordable Indian-style chicken rice bowl.",
    "time": 25,
    "calories": 520,
    "difficulty": "Easy",
    "servings": 1,
    "ingredients": [
        "Rice - 1 cup",
        "Chicken - 150 g",
        "Onion - 1 small",
        "Oil - 1 teaspoon",
        "Salt - as required",
        "Turmeric - 1/4 teaspoon"
    ],
    "steps": [
        "Wash and cook the rice.",
        "Heat oil in a pan and saute onion until soft.",
        "Add chicken and spices and cook thoroughly.",
        "Mix with cooked rice for 2 minutes and serve hot."
    ],
    "tip": "Pair with cucumber raita or a squeeze of lemon for extra freshness and vitamins.",
    "nutrition": {{
        "calories": 520,
        "protein": 32,
        "carbs": 58,
        "fat": 15
    }},
    "cost": 90
}}
"""

    raw = call_gemini_json(prompt)
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return None

    nutrition = raw.get("nutrition", {})
    if not isinstance(nutrition, dict):
        nutrition = {}

    # Normalize preparation time to numeric minutes for "{{ recipe_data.time }} Minutes"
    raw_time = raw.get("time") or raw.get("cooking_time") or 20
    time_digits = re.search(r"\d+", str(raw_time))
    time_minutes = int(time_digits.group(0)) if time_digits else 20

    calories = safe_int(
        raw.get("calories") if raw.get("calories") is not None else nutrition.get("calories", 350),
        350,
    )

    steps = clean_text_list(raw.get("steps") or raw.get("instructions") or [])
    cleaned_ingredients = clean_text_list(raw.get("ingredients", []))

    if not cleaned_ingredients or not steps:
        return None

    difficulty = str(raw.get("difficulty") or "Easy").strip().title()
    if difficulty not in ("Easy", "Medium", "Hard"):
        difficulty = "Easy"

    servings = max(1, safe_int(raw.get("servings", 1), 1))
    tip = str(
        raw.get("tip")
        or "Use minimal oil and pair with a fresh salad or curd for balanced nutrition."
    ).strip()

    return {
        "name": str(raw.get("name") or "AI Generated Recipe").strip(),
        "description": str(raw.get("description") or "").strip(),
        "time": time_minutes,
        "calories": calories,
        "difficulty": difficulty,
        "servings": servings,
        "ingredients": cleaned_ingredients,
        "steps": steps,
        "instructions": steps,
        "tip": tip,
        "nutrition": {
            "calories": calories,
            "protein": safe_float(nutrition.get("protein", 0)),
            "carbs": safe_float(nutrition.get("carbs", 0)),
            "fat": safe_float(nutrition.get("fat", 0)),
        },
        "cost": safe_float(raw.get("cost", 0)),
    }


@app.route("/recipe", methods=["GET", "POST"])
def recipe():
    if "user_id" not in session:
        return redirect(url_for("login"))

    recipe_data = None
    error = None
    ingredients = ""

    if request.method == "POST":
        ingredients = request.form.get("ingredients", "").strip()

        if not ingredients:
            error = "Please enter at least one ingredient."
        elif len(ingredients) > 500:
            error = "Please enter a shorter list of ingredients."
        else:
            recipe_data = generate_ai_recipe(ingredients)
            if not recipe_data:
                error = (
                    "Recipe generation is currently unavailable. "
                    "Please check your GEMINI_API_KEY configuration and try again."
                )
            else:
                user_obj = db.session.get(User, session["user_id"])
                record_usage_event(
                    user_obj,
                    "recipe_generated",
                    f"Generated {recipe_data.get('name', 'Recipe')}",
                )

    return render_template(
        "recipe.html",
        recipe_data=recipe_data,
        ingredients=ingredients,
        error=error,
    )


# =========================================================
# GROCERY ASSISTANT
# =========================================================

@app.route("/grocery", methods=["GET", "POST"])
def grocery():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":
        item = request.form.get("item", "").strip()
        quantity = request.form.get("quantity", "").strip()
        price_text = request.form.get("price", "").strip()

        if not item or not quantity or not price_text:
            flash("Please fill in item, quantity, and price.", "error")
            return redirect(url_for("grocery"))

        try:
            price = float(price_text)
            if math.isnan(price) or math.isinf(price) or price < 0:
                raise ValueError
        except ValueError:
            flash("Please enter a valid non-negative price.", "error")
            return redirect(url_for("grocery"))

        new_item = GroceryItem(
            item=item[:100],
            quantity=quantity[:50],
            price=price,
            user_id=user.id,
        )

        try:
            db.session.add(new_item)
            db.session.commit()
            record_usage_event(user, "grocery_added", f"Added {item} ({quantity})")
        except Exception:
            db.session.rollback()
            flash("Could not save grocery item. Please try again.", "error")

        return redirect(url_for("grocery"))

    grocery_items = (
        GroceryItem.query.filter_by(user_id=user.id)
        .order_by(GroceryItem.id.desc())
        .all()
    )
    total = round(sum(item.price for item in grocery_items), 2)

    return render_template(
        "grocery.html",
        user=user,
        grocery_items=grocery_items,
        total=total,
    )


@app.route("/delete-grocery/<int:item_id>", methods=["GET", "POST"])
def delete_grocery(item_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    item = GroceryItem.query.get_or_404(item_id)
    if item.user_id != session["user_id"]:
        return redirect(url_for("grocery"))

    item_name = item.item
    try:
        db.session.delete(item)
        db.session.commit()
        user_obj = db.session.get(User, session["user_id"])
        record_usage_event(user_obj, "grocery_removed", f"Removed {item_name}")
    except Exception:
        db.session.rollback()

    return redirect(url_for("grocery"))


# =========================================================
# REPORT & HYDRATION LOGGING
# =========================================================

@app.route("/report")
def report():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    today = date.today()
    bmi, bmi_status = compute_bmi(user.height, user.weight)
    calorie_target, protein_target = compute_nutrition_targets(user)

    meals_today = MealProgress.query.filter_by(user_id=user.id, date=today).all()
    generated_today = GeneratedMeal.query.filter_by(user_id=user.id, date=today).all()
    water_log = WaterLog.query.filter_by(user_id=user.id, date=today).first()

    water_today = round(safe_float(water_log.liters if water_log else 0.0, 0.0), 2)

    completed_types = [
        m.meal_type
        for m in meals_today
        if m.completed and m.meal_type in ("Breakfast", "Lunch", "Snack", "Dinner")
    ]
    completed_count = len(completed_types)

    gen_by_type = {}
    for g in generated_today:
        mapped = DASHBOARD_MEAL_MAP.get(g.meal_type, g.meal_type)
        if mapped not in gen_by_type:
            gen_by_type[mapped] = g

    calories_today = 0.0
    protein_today = 0.0
    for m_type in completed_types:
        if m_type in gen_by_type:
            calories_today += safe_float(gen_by_type[m_type].calories, 0.0)
            protein_today += safe_float(gen_by_type[m_type].protein, 0.0)
        else:
            calories_today += calorie_target / 4.0
            protein_today += protein_target / 4.0

    calories_today_int = int(round(calories_today))
    calorie_goal_pct = min(100, int(round((calories_today / calorie_target) * 100))) if calorie_target else 0
    water_goal_pct = min(100, int(round((water_today / 3.0) * 100)))
    protein_goal_pct = min(100, int(round((protein_today / protein_target) * 100))) if protein_target else 0
    meal_goal_pct = min(100, int(round((completed_count / 4.0) * 100)))

    if completed_count == 0 and water_today == 0:
        ai_summary = (
            f"Welcome to your health report, {user.name.split()[0]}! "
            f"Your daily target is {calorie_target} kcal and {protein_target}g protein for your "
            f"{user.goal or 'Healthy Lifestyle'} goal. Log meals on your Dashboard and track water above."
        )
    elif water_goal_pct < 60:
        ai_summary = (
            f"Great momentum! You have logged {calories_today_int} kcal across {completed_count} meal(s) today. "
            f"Your water intake is at {water_today} L out of 3.0 L — try adding another glass or two to stay hydrated."
        )
    else:
        ai_summary = (
            f"Great work! Your intake is {calories_today_int} kcal ({calorie_goal_pct}% of target) "
            f"with {water_today} L water logged today. Keep maintaining balanced {user.diet or ''} meals!"
        )

    return render_template(
        "report.html",
        user=user,
        bmi=bmi,
        bmi_status=bmi_status,
        calories_today=calories_today_int,
        calorie_target=calorie_target,
        water_today=water_today,
        calorie_goal_pct=calorie_goal_pct,
        water_goal_pct=water_goal_pct,
        protein_goal_pct=protein_goal_pct,
        meal_goal_pct=meal_goal_pct,
        ai_summary=ai_summary,
    )


@app.route("/report/water", methods=["POST"])
def add_water():
    if "user_id" not in session:
        return redirect(url_for("login"))

    next_endpoint = request.form.get("next") or request.args.get("next") or "report"
    if next_endpoint not in ("report", "dashboard"):
        next_endpoint = "report"

    amount = safe_float(request.form.get("amount", 0.25), 0.25)
    if amount == 0 or amount < -5.0 or amount > 5.0:
        amount = 0.25

    today = date.today()
    log = WaterLog.query.filter_by(user_id=session["user_id"], date=today).first()
    if log:
        log.liters = round(max(0.0, min(15.0, safe_float(log.liters, 0.0) + amount)), 2)
    else:
        initial = round(max(0.0, min(15.0, amount)), 2)
        log = WaterLog(user_id=session["user_id"], liters=initial, date=today)
        db.session.add(log)

    try:
        db.session.commit()
        user_obj = db.session.get(User, session["user_id"])
        delta_text = f"+{amount}" if amount > 0 else f"{amount}"
        record_usage_event(user_obj, "water_logged", f"Water intake {delta_text} L (total {log.liters} L)")
    except Exception:
        db.session.rollback()
        flash("Could not update water intake.", "error")

    return redirect(url_for(next_endpoint))


# =========================================================
# PROFILE
# =========================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name or len(name) < 2 or len(name) > 100 or not re.fullmatch(r"[A-Za-zÀ-ÿ\s.'-]+", name):
            flash("Please enter a valid full name (2-100 characters).", "error")
            return redirect(url_for("profile", edit="true"))

        try:
            age_raw = request.form.get("age", "").strip()
            height_raw = request.form.get("height", "").strip()
            weight_raw = request.form.get("weight", "").strip()

            new_age = int(age_raw) if age_raw else None
            new_height = float(height_raw) if height_raw else None
            new_weight = float(weight_raw) if weight_raw else None
        except ValueError:
            flash("Please enter valid numbers for age, height, and weight.", "error")
            return redirect(url_for("profile", edit="true"))

        if new_age is not None and not (13 <= new_age <= 120):
            flash("Age must be between 13 and 120.", "error")
            return redirect(url_for("profile", edit="true"))

        if new_height is not None and (math.isnan(new_height) or math.isinf(new_height) or not (50 <= new_height <= 250)):
            flash("Height must be between 50 and 250 cm.", "error")
            return redirect(url_for("profile", edit="true"))

        if new_weight is not None and (math.isnan(new_weight) or math.isinf(new_weight) or not (20 <= new_weight <= 300)):
            flash("Weight must be between 20 and 300 kg.", "error")
            return redirect(url_for("profile", edit="true"))

        diet = request.form.get("diet", user.diet).strip()
        goal = request.form.get("goal", user.goal).strip()

        if diet not in ALLOWED_DIETS:
            flash("Please select a valid diet preference.", "error")
            return redirect(url_for("profile", edit="true"))

        if goal not in ALLOWED_GOALS:
            flash("Please select a valid fitness goal.", "error")
            return redirect(url_for("profile", edit="true"))

        user.name = name
        user.age = new_age
        user.height = new_height
        user.weight = new_weight
        user.diet = diet
        user.goal = goal

        try:
            db.session.commit()
            record_usage_event(user, "profile_updated", f"Updated profile ({diet}, {goal})")
            flash("Profile updated successfully!", "success")
        except Exception:
            db.session.rollback()
            flash("Could not save your profile. Please try again.", "error")
            return redirect(url_for("profile", edit="true"))

        return redirect(url_for("profile"))

    edit = request.args.get("edit") == "true"
    bmi, bmi_status = compute_bmi(user.height, user.weight)
    calorie_target, protein_target = compute_nutrition_targets(user)
    weight_val = safe_float(user.weight, 64.0) or 64.0
    water_target = round(max(2.0, min(4.0, weight_val * 0.035)), 1)
    h_val = safe_float(user.height, 0.0)
    if h_val > 0:
        h_m = h_val / 100.0
        healthy_min_kg = round(18.5 * h_m * h_m, 1)
        healthy_max_kg = round(24.9 * h_m * h_m, 1)
    else:
        healthy_min_kg, healthy_max_kg = None, None
    return render_template(
        "profile.html",
        user=user,
        edit=edit,
        bmi=bmi,
        bmi_status=bmi_status,
        healthy_min_kg=healthy_min_kg,
        healthy_max_kg=healthy_max_kg,
        calorie_target=calorie_target,
        protein_target=protein_target,
        water_target=water_target,
    )







# =========================================================

@app.route("/users")
def users_activity():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("login"))

    query = request.args.get("q", "").strip().lower()
    today_prefix = date.today().isoformat()

    mdb = get_mongo_db()
    if mdb is not None:
        raw_accounts = list(mdb.users.find({}, {"_id": 0}).sort("createdAt", -1))
        raw_logs = list(mdb.usage_logs.find({}, {"_id": 0}).sort("createdAt", -1).limit(60))
    else:
        raw_accounts = [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "diet": u.diet,
                "goal": u.goal,
                "age": u.age,
                "createdAt": u.created_at or "",
                "lastLoginAt": u.last_login_at,
                "lastActiveAt": u.last_active_at,
                "loginCount": u.login_count or 0,
                "actionsCount": u.actions_count or 0,
            }
            for u in User.query.order_by(User.id.desc()).all()
        ]
        raw_logs = [
            {
                "id": l.id,
                "userId": l.user_id,
                "userName": l.user_name,
                "userEmail": l.user_email,
                "eventType": l.event_type,
                "details": l.details,
                "createdAt": l.created_at or "",
            }
            for l in UsageLog.query.order_by(UsageLog.id.desc()).limit(60).all()
        ]

    total_accounts = len(raw_accounts)
    active_today = sum(
        1
        for a in raw_accounts
        if str(a.get("lastActiveAt") or a.get("lastLoginAt") or a.get("createdAt") or "").startswith(today_prefix)
    )
    total_logins = sum(int(a.get("loginCount") or 0) for a in raw_accounts)
    total_actions = max(len(raw_logs), sum(int(a.get("actionsCount") or 0) for a in raw_accounts))

    if query:
        accounts = [
            a
            for a in raw_accounts
            if query in str(a.get("name", "")).lower()
            or query in str(a.get("email", "")).lower()
            or query in str(a.get("diet", "")).lower()
            or query in str(a.get("goal", "")).lower()
        ]
    else:
        accounts = raw_accounts

    return render_template(
        "users.html",
        user=user,
        query=request.args.get("q", ""),
        total_accounts=total_accounts,
        active_today=active_today,
        total_logins=total_logins,
        total_actions=total_actions,
        accounts=accounts,
        recent_activity=raw_logs,
    )


@app.route("/users/<int:target_user_id>")
def user_profile_detail(target_user_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))

    current_user = db.session.get(User, session["user_id"])
    if current_user is None:
        session.clear()
        return redirect(url_for("login"))

    target_user = db.session.get(User, target_user_id)
    if target_user is None:
        flash("User profile not found.", "error")
        return redirect(url_for("users_activity"))

    bmi, bmi_status = compute_bmi(target_user.height, target_user.weight)
    calorie_target, protein_target = compute_nutrition_targets(target_user)
    weight_val = safe_float(target_user.weight, 64.0) or 64.0
    water_target = round(max(2.0, min(4.0, weight_val * 0.035)), 1)
    h_val = safe_float(target_user.height, 0.0)
    if h_val > 0:
        h_m = h_val / 100.0
        healthy_min_kg = round(18.5 * h_m * h_m, 1)
        healthy_max_kg = round(24.9 * h_m * h_m, 1)
    else:
        healthy_min_kg, healthy_max_kg = None, None

    return render_template(
        "profile.html",
        user=target_user,
        edit=False,
        is_own_profile=(target_user.id == current_user.id),
        bmi=bmi,
        bmi_status=bmi_status,
        healthy_min_kg=healthy_min_kg,
        healthy_max_kg=healthy_max_kg,
        calorie_target=calorie_target,
        protein_target=protein_target,
        water_target=water_target,
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    if "user_id" in session:
        user_obj = db.session.get(User, session["user_id"])
        if user_obj is not None:
            record_usage_event(user_obj, "logout", "Signed out of NutriAI")
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
