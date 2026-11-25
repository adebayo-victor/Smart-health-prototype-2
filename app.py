"""SmartHealth: offline-first clinical support for Nigerian communities.

Flow of a patient encounter:
    translation -> triage -> diagnosis -> referral

Pipeline for intake:
    Whisper (Groq) transcribes  ->  Gemini translates  ->  LiteLLM analyzes (TC / vitals / dialect)

v5 CHW single-page app (templates/chw_app.html, served at /chw-app) talks to these routes. Every stage can run on
its own: triage and diagnosis work without a patient (nothing is stored), and /api/save_encounter writes any
stage, or the whole visit, to a patient's record when the worker is done. /api/referral now accepts the same
clinical fields, so a referral always carries the visit's translation, vitals, triage and diagnosis with it.
"""
import base64
import hmac
import io
import json
import logging
import os
import random
import re
import secrets
import string
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import wraps

import qrcode
from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from flask_sqlalchemy import SQLAlchemy
from groq import Groq
from gtts import gTTS
from openai import OpenAI
from sqlalchemy import func, inspect as sa_inspect, text as sql_text
from sqlalchemy.exc import IntegrityError
from flask.sessions import SecureCookieSessionInterface
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

# Gemini SDK (translation only). Optional import: if the package is missing the app still boots
# and translation falls back to the LiteLLM model instead of crashing every page.
try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
load_dotenv()
DEBUG = os.getenv("FLASK_DEBUG") == "1"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("smarthealth")

app = Flask(__name__)


def database_url():
    """DATABASE_URL as given by the host, made SQLAlchemy-ready."""
    url = (os.getenv("DATABASE_URL") or "sqlite:////tmp/smarthealth.db").strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def load_secret_key():
    """Stable signing key shared across workers."""
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    path = os.path.join(tempfile.gettempdir(), "smarthealth_secret_key")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    logger.warning("SECRET_KEY is not set. Using temporary shared key.")
    for _ in range(20):
        with open(path) as f:
            key = f.read().strip()
        if key:
            return key
        time.sleep(0.05)
    return secrets.token_hex(32)


_secret = load_secret_key()

app.config.update(
    SECRET_KEY=_secret,
    SQLALCHEMY_DATABASE_URI=database_url(),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={} if database_url().startswith("sqlite") else {"pool_pre_ping": True, "pool_recycle": 280},
    MAX_CONTENT_LENGTH=15 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
)
db = SQLAlchemy(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


class AutoSecureSession(SecureCookieSessionInterface):
    def get_cookie_secure(self, app):
        return request.is_secure


app.session_interface = AutoSecureSession()

# gemini-1.5-flash and gemini-2.0-flash have been retired by Google, so these defaults point at a
# current model. Check https://ai.google.dev/gemini-api/docs/models and override with env vars if needed.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")   # LiteLLM analysis (stage 3)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")    # direct Gemini SDK translation (stage 2)
STAGES = ["translation", "triage", "diagnosis", "referral"]


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ==========================================
# 2. DATABASE MODELS
# ==========================================
class MedicalCenter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=True)
    short_code = db.Column(db.String(10), unique=True, nullable=False)


class Worker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    worker_code = db.Column(db.String(20), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default="CHW")
    facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=False)
    facility = db.relationship("MedicalCenter", backref="workers")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Patient(db.Model):
    __tablename__ = "patient"
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=False, index=True)
    full_name = db.Column(db.String(100), nullable=False)
    name_key = db.Column(db.String(120), nullable=False, index=True)
    hospital_id = db.Column(db.String(40), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    age_years = db.Column(db.Integer, nullable=True)
    sex = db.Column(db.String(1), nullable=True)
    village_lga = db.Column(db.String(100), nullable=True)
    next_of_kin_name = db.Column(db.String(100), nullable=True)
    next_of_kin_phone = db.Column(db.String(20), nullable=True)
    is_pregnant = db.Column(db.Boolean, nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    source_patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)
    merged_into_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)
    merge_log = db.Column(db.Text, nullable=True)   # JSON: what a merge moved, so undo-merge can put it back
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_by_worker_id = db.Column(db.Integer, db.ForeignKey("worker.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (db.UniqueConstraint("facility_id", "hospital_id", name="uq_patient_facility_hospital_id"),)

    def to_dict(self):
        return {
            "id": self.id, "full_name": self.full_name, "hospital_id": self.hospital_id, "phone": self.phone,
            "age_years": self.age_years, "sex": self.sex, "village_lga": self.village_lga,
            "next_of_kin_name": self.next_of_kin_name, "next_of_kin_phone": self.next_of_kin_phone,
            "is_pregnant": self.is_pregnant, "allergies": self.allergies,
        }


class Encounter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True, index=True)
    patient = db.relationship("Patient", foreign_keys=[patient_id])
    facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=True, index=True)
    referred_from_id = db.Column(db.Integer, db.ForeignKey("encounter.id"), nullable=True)
    patient_name = db.Column(db.String(100), nullable=False)
    dialect = db.Column(db.String(50), nullable=True)
    statement = db.Column(db.Text, nullable=True)
    translation = db.Column(db.Text, nullable=True)
    vitals_json = db.Column(db.Text, nullable=True)
    tc_steps_translation = db.Column(db.Text, nullable=True)
    triage_tier = db.Column(db.String(10), nullable=True)
    triage_notes = db.Column(db.Text, nullable=True)
    tc_steps_triage = db.Column(db.Text, nullable=True)
    diagnosis_notes = db.Column(db.Text, nullable=True)
    diagnosis_medications = db.Column(db.Text, nullable=True)
    tc_steps_diagnosis = db.Column(db.Text, nullable=True)
    referral_reason = db.Column(db.Text, nullable=True)
    referral_facility = db.Column(db.String(100), nullable=True)
    referral_code = db.Column(db.String(20), nullable=True, unique=True)
    referral_qr = db.Column(db.Text, nullable=True)
    # Companion (patient portal) passes belong to ONE companion user, so one phone's code never shows in another's chat.
    companion_user_id = db.Column(db.Integer, db.ForeignKey("companion_user.id"), nullable=True, index=True)
    pass_hospitals = db.Column(db.Text, nullable=True)      # nearby hospitals listed on an open emergency pass
    pass_location = db.Column(db.String(100), nullable=True)  # town/area the patient spoke
    origin_worker_id = db.Column(db.Integer, db.ForeignKey("worker.id"), nullable=True)
    origin_facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=True)
    destination_facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=True)
    # pending | fulfilled | revoked (cancelled by the patient) | cancelled (cancelled by the sending worker)
    referral_status = db.Column(db.String(20), default="pending")
    current_stage = db.Column(db.String(20), default="translation")
    status = db.Column(db.String(20), default="active")
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class CompanionUser(db.Model):
    __tablename__ = "companion_user"
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(50), nullable=True)
    preferred_language = db.Column(db.String(10), default="ha")
    created_at = db.Column(db.DateTime, default=utcnow)


class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), unique=True, nullable=False, index=True)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("companion_user.id"), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)


memory_keywords = db.Table(
    "memory_keywords",
    db.Column("memory_id", db.Integer, db.ForeignKey("memory_index.id"), primary_key=True),
    db.Column("keyword_id", db.Integer, db.ForeignKey("keyword.id"), primary_key=True),
)


class MemoryIndex(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("companion_user.id"), nullable=False, index=True)
    context_chunk = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    keywords = db.relationship("Keyword", secondary=memory_keywords, backref="memories")


class Phrase(db.Model):
    """Quick-ask questions, treatment instructions and referral reasons shown in the CHW app. Edit them in the
    database; the page loads them from /api/phrases. kind: ask | instruction | reason."""
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(20), nullable=False, index=True)
    text_en = db.Column(db.String(300), nullable=False)
    text_ha = db.Column(db.Text, nullable=True)
    text_pcm = db.Column(db.Text, nullable=True)
    sort_order = db.Column(db.Integer, nullable=False, default=0)


# ==========================================
# 3. INITIALIZATION & DEMO DATA
# ==========================================
DEFAULT_SEED_PASSWORD = "password123"


def seed_demo_data():
    if MedicalCenter.query.count():
        return
    password = os.getenv("SEED_PASSWORD", DEFAULT_SEED_PASSWORD)
    if password == DEFAULT_SEED_PASSWORD:
        logger.warning("Demo workers were created with the DEFAULT password. Set SEED_PASSWORD, or change these "
                       "passwords before real patients are entered.")
    mu = MedicalCenter(name="Murtala Muhammed Specialist Hospital", location="Kano", short_code="MU")
    ak = MedicalCenter(name="Aminu Kano Teaching Hospital", location="Kano", short_code="AK")
    db.session.add_all([mu, ak])
    db.session.commit()

    w1 = Worker(username="chw_amina", full_name="Amina Bello", worker_code="MU2789", role="CHW", facility_id=mu.id)
    w2 = Worker(username="dr_yusuf", full_name="Dr. Yusuf Ali", worker_code="AK1001", role="Doctor", facility_id=ak.id)
    w3 = Worker(username="chw_ibrahim", full_name="Ibrahim Sani", worker_code="MU3001", role="CHW", facility_id=mu.id)
    for w in (w1, w2, w3):
        w.set_password(password)
    db.session.add_all([w1, w2, w3])
    db.session.commit()


DEFAULT_PHRASES = [
    ("ask", "What brings you here?", "Me ya kawo ka nan?", "Wetin bring you come here?"),
    ("ask", "Where does it hurt?", "A ina kake jin ciwo?", "Which place dey pain you?"),
    ("instruction", "Take the tablets 2 times a day for 3 days.", "Ka sha magungunan sau biyu a rana tsawon kwana uku.", "Chop di medicine two times for day, three days."),
    ("instruction", "Drink plenty of clean water.", "Ka sha ruwa mai tsabta da yawa.", "Drink plenty clean water."),
    ("instruction", "Come back fast if vomiting.", "Ka dawo da sauri idan kana amai.", "Come back quick if you dey vomit."),
    ("reason", "Cannot drink", None, None), ("reason", "Vomiting a lot", None, None),
    ("reason", "Fever many days", None, None), ("reason", "Pregnant", None, None),
    ("reason", "Very weak", None, None), ("reason", "Fits", None, None),
]


def seed_phrases():
    if Phrase.query.count():
        return
    for i, (kind, en, ha, pcm) in enumerate(DEFAULT_PHRASES):
        db.session.add(Phrase(kind=kind, text_en=en, text_ha=ha, text_pcm=pcm, sort_order=i))
    db.session.commit()


def add_missing_columns(table, wanted):
    """db.create_all() creates missing TABLES but never adds columns to an existing one.
    Add new columns on databases created before they existed (SQLite and PostgreSQL)."""
    try:
        have = {c["name"] for c in sa_inspect(db.engine).get_columns(table)}
    except Exception:
        logger.exception("Could not inspect the %s table", table)
        return
    for name, ddl in wanted.items():
        if name in have:
            continue
        try:
            db.session.execute(sql_text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
            db.session.commit()
            logger.info("Added %s.%s", table, name)
        except Exception:  # another gunicorn worker may have added it a moment ago
            db.session.rollback()


def ensure_columns():
    add_missing_columns("encounter", {"companion_user_id": "INTEGER", "pass_hospitals": "TEXT", "pass_location": "VARCHAR(100)"})
    add_missing_columns("patient", {"merge_log": "TEXT"})
    try:
        db.session.execute(sql_text("CREATE INDEX IF NOT EXISTS ix_encounter_companion_user_id ON encounter (companion_user_id)"))
        db.session.commit()
    except Exception:
        db.session.rollback()


with app.app_context():
    db.create_all()
    ensure_columns()
    seed_demo_data()
    seed_phrases()


# ==========================================
# 4. HELPERS
# ==========================================
_groq = None
_llm = None

# ---- Gemini (direct SDK): the translator ----
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# None means "Gemini unavailable" exactly as the callers expect.
gemini_model = None
_GEMINI_SAFETY = None
if not _GEMINI_KEY:
    logger.warning("GEMINI_API_KEY is NOT set. Translation will fall back to the LiteLLM model.")
elif genai is None:
    logger.warning("google-generativeai is not installed. Translation will fall back to the LiteLLM model.")
else:
    try:
        genai.configure(api_key=_GEMINI_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        # Health talk (pregnancy, bleeding, injuries) can trip default filters and return nothing.
        _GEMINI_SAFETY = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    except Exception:
        logger.exception("Could not initialise Gemini model %s. Translation will use the LiteLLM model.", GEMINI_MODEL)
        gemini_model = None

# ---- Speech-to-text settings ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
MIN_AUDIO_BYTES = 2000   # about 0.5 s; one-word answers such as a town name must still pass
STT_PROMPT_CLINIC = "Medical consultation in Nigeria. Focus only on clear speech. Ignore background noise, coughing, or pauses. Terms: malaria, zazzabi, BP, temperature, headache, stomach."
STT_PROMPT_COMPANION = "A person in Nigeria talking about their health, or naming their town, such as Kano, Ibadan, Odeda."
# Phrases Whisper tends to invent when it hears silence or noise.
WHISPER_GHOSTS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "you", "bye", "bye bye",
    "okay", "ok", "so", "uh", "um", "hmm", "mm", "mhm", "subtitles by the amaraorg community",
}
# What the browser may send as "language". Pidgin has no Whisper code; English works best for it.
STT_LANGS = {"en": "en", "pcm": "en", "ha": "ha"}

# ---- Small in-memory caches. The same instructions are played to patients over and over, so a repeat tap
# should be instant and should not cost another Gemini + gTTS round trip. Only successes are cached. ----
_CACHE_MAX = 80
_tts_cache = OrderedDict()
_tr_cache = OrderedDict()


def _cache_get(cache, key):
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _cache_put(cache, key, val):
    if val is None:
        return
    cache[key] = val
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX:
        cache.popitem(last=False)


def get_groq():
    global _groq
    if _groq is None:
        # timeout + retries: Groq occasionally returns 429/5xx or stalls on poor links
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=45.0, max_retries=2)
    return _groq


def get_llm():
    global _llm
    if _llm is None:
        # The OpenAI client defaults to a 10 minute timeout and 2 retries; a stalled proxy would freeze a worker.
        _llm = OpenAI(api_key=os.getenv("LITELLM_API_KEY"), base_url=os.getenv("LITELLM_BASE_URL"),
                      timeout=float(os.getenv("LLM_TIMEOUT", "30")), max_retries=1)
    return _llm


def llm_text(messages, temperature=0.2, max_tokens=None):
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    resp = get_llm().chat.completions.create(model=LLM_MODEL, messages=messages, temperature=temperature, **kwargs)
    return resp.choices[0].message.content.strip()


def parse_json_reply(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    return json.loads(raw[start:end + 1])


def llm_json(prompt, temperature=0.2):
    return parse_json_reply(llm_text([{"role": "user", "content": prompt}], temperature))


def _seg(seg, key, default=None):
    """Segments come back as dicts or objects depending on the SDK version."""
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)


def gemini_translate(text, target="English", lang_hint=None):
    """Translate with Gemini. Returns the translation, or None if Gemini is unavailable or failed.

    Callers decide what to do on None (the LiteLLM model is the fallback), so a Gemini outage
    never breaks the intake flow.
    """
    if not text or not text.strip() or gemini_model is None:
        return None

    if target == "English":
        lang_line = ""
        if lang_hint == "ha":
            lang_line = "The source language is Hausa."
        elif lang_hint == "en":
            lang_line = "The source language is English or Nigerian Pidgin."
        prompt = (
            "You are a forensic sociolinguist and low-resource language reconstruction specialist fluent in "
            "Hausa, Nigerian Pidgin, and English. The input below is a noisy voice-to-text transcript that may "
            "contain phonetic misspellings, missing diacritics, merged or split words, broken spacing, and "
            "jumbled word order. Silently execute this reconstruction pipeline before answering: "
            "(1) break the transcript into individual words and phonetic fragments, "
            "(2) map each token to its likely intended word and candidate meaning, "
            "(3) test adjacent words and neighboring clauses side by side—merging split syllables, separating "
            "run-together words, and reordering scrambled segments until a coherent meaning emerges, "
            "(4) check against literal nonsense to catch idioms or colloquial expressions, and "
            "(5) synthesize the FULL intended meaning into clear, natural English. "
            "HARD RULES: Output ONLY the final English translation. No quotes, no markdown, no notes, no "
            "step-by-step output, and no left-over Hausa or Pidgin words. Do NOT add, guess, or invent any "
            "medical symptoms or details that are not supported by the input. "
            "If the input is already clear English, return it unchanged. "
            f"{lang_line}\n\nInput: \"{text}\""
        )

    else:
        prompt = (
            f"You are a professional medical translator. Translate the English instruction below into {target}, "
            "in simple everyday words that a patient in Nigeria would understand. "
            "HARD RULES: Output ONLY the translation. No quotes, no markdown, no notes. "
            f"Do not add or remove any information.\n\nInput: \"{text}\""
        )

    try:
        response = gemini_model.generate_content(
            prompt,
            # Newer Gemini models "think" before answering and thinking tokens count against this limit,
            # so keep it generous or the visible answer can come back empty.
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=2048),
            safety_settings=_GEMINI_SAFETY,
            request_options={"timeout": int(os.getenv("GEMINI_TIMEOUT", "15"))},
        )
        out = (response.text or "").strip().strip('"\'')
        if not out:
            logger.warning("Gemini returned an empty translation.")
            return None
        logger.info(f"Gemini {target}: '{text[:60]}' -> '{out[:60]}'")
        return out
    except Exception as e:
        logger.error(f"Gemini translation failed ({e}).")
        return None


def gemini_translation_options(text, lang_hint=None):
    """Three ranked English readings of a noisy transcript, for the companion's meta-reasoning.

    Returns ([{"text", "confidence"}, ...] best first, spoken language "ha"/"pcm"/"en"/None), or (None, None)
    if Gemini is unavailable or its reply is unusable. The caller then carries on exactly as before with
    gemini_translate(), so this can never break the companion. gemini_translate() itself is untouched.
    """
    if not text or not text.strip() or gemini_model is None:
        return None, None

    lang_line = ""
    if lang_hint == "ha":
        lang_line = "The source language is Hausa."
    elif lang_hint == "en":
        lang_line = "The source language is English or Nigerian Pidgin."
    prompt = (
        "You are a forensic sociolinguist and low-resource language reconstruction specialist fluent in "
        "Hausa, Nigerian Pidgin, and English, working as an expert medical interpreter. The input below is a "
        "noisy voice-to-text transcript that may contain phonetic misspellings, missing diacritics, merged or "
        "split words, broken spacing, and jumbled word order. Silently reconstruct what the speaker most "
        "likely meant, then give EXACTLY 3 plausible clear, natural English translations, ranked from most to "
        "least likely. Each has an integer confidence percent. The three confidences MUST add up to exactly 100. "
        "The translations should differ only in how an unclear word or phrase is understood. "
        "If the input is already clear English, translation 1 is the input unchanged (95-100) and "
        "translation 2 a minor rephrase. "
        "HARD RULES: Do NOT add, guess, or invent any medical symptoms or details that are not supported by the "
        "input. No left-over Hausa or Pidgin words in the translations. Also say which language the speaker used: "
        '"ha" for Hausa, "pcm" for Nigerian Pidgin, "en" for English. '
        'Return ONLY valid JSON, no markdown: {"options": [{"text": "...", "confidence": 0}, '
        '{"text": "...", "confidence": 0}, {"text": "...", "confidence": 0}], "language": "ha"} '
        f"{lang_line}\n\nInput: \"{text}\""
    )

    try:
        response = gemini_model.generate_content(
            prompt,
            # Same generous limit as gemini_translate(): thinking tokens count against it.
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=2048,
                                                           response_mime_type="application/json"),
            safety_settings=_GEMINI_SAFETY,
            request_options={"timeout": int(os.getenv("GEMINI_TIMEOUT", "15"))},
        )
        data = parse_json_reply(response.text or "")
        rows = [(str(r["text"]).strip().strip('"\''), float(r["confidence"])) for r in data["options"]]
    except Exception as e:
        logger.error(f"Gemini translation options failed ({e}).")
        return None, None

    rows = sorted((r for r in rows if r[0] and r[1] >= 0), key=lambda r: r[1], reverse=True)[:3]
    total = sum(c for _, c in rows)
    if not rows or total <= 0:
        logger.warning("Gemini returned no usable translation options.")
        return None, None
    options = [{"text": t, "confidence": round(c * 100 / total)} for t, c in rows]
    lang = str(data.get("language") or "").strip().lower()
    logger.info("Gemini options (confidence): %s", [o["confidence"] for o in options])
    return options, (lang if lang in ("ha", "pcm", "en") else None)


def _prompt_echo_detected(text, prompt):
    """True if the transcription contains a suspicious multi-word slice of the prompt."""
    if not text:
        return False
    # Normalize both sides
    t_norm = re.sub(r"[^\w\s]", "", text.lower())
    t_words = t_norm.split()
    p_words = re.sub(r"[^\w\s]", "", prompt.lower()).split()
    # Build all 4-word shingles from the prompt
    shingles = {" ".join(p_words[i:i + 4]) for i in range(len(p_words) - 3)}
    # If any 4-word window of the transcription matches a prompt shingle → echo
    for i in range(len(t_words) - 3):
        if " ".join(t_words[i:i + 4]) in shingles:
            return True
    return False


def transcribe(audio_file, language=None, prompt=STT_PROMPT_CLINIC):
    """Whisper transcription — NO prompt sent to Groq to prevent echo bugs."""
    data = audio_file.read()
    if len(data) < MIN_AUDIO_BYTES:
        raise ValueError("Recording too short. Hold the button and speak clearly.")
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Recording too long. Please keep it under 2 minutes.")

    mime = (audio_file.mimetype or "audio/webm").split(";")[0].strip().lower()
    if "webm" in mime:
        ext = "webm"
    elif "ogg" in mime:
        ext = "ogg"
    elif "mp4" in mime or "m4a" in mime:
        ext = "m4a"
    elif "wav" in mime:
        ext = "wav"
    else:
        ext = "webm"
    filename = f"recording.{ext}"

    kwargs = dict(
        file=(filename, data, mime),
        model=WHISPER_MODEL,
        response_format="verbose_json",
        temperature=0.0,
        # NOTE: prompt intentionally NOT sent. Whisper echoes prompt fragments
        # as "transcription" when audio is silent/noisy. Removing it fixes that.
    )
    if language in ("en", "ha"):
        kwargs["language"] = language

    result = get_groq().audio.transcriptions.create(**kwargs)

    segments = result.get("segments") if isinstance(result, dict) else getattr(result, "segments", None)
    full_text = (result.get("text") if isinstance(result, dict) else getattr(result, "text", "")) or ""

    if segments:
        kept = []
        for s in segments:
            no_speech = _seg(s, "no_speech_prob", 0.0) or 0.0
            logprob = _seg(s, "avg_logprob", 0.0) or 0.0
            if no_speech > 0.6 and logprob < -1.0:
                continue
            kept.append((_seg(s, "text", "") or "").strip())
        text = " ".join(t for t in kept if t).strip()
    else:
        text = full_text.strip()

    norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not norm or norm in WHISPER_GHOSTS:
        raise ValueError("No speech detected. Move closer, reduce background noise and try again.")

    return text


def speak(text, lang="en"):
    key = (lang, text)
    cached = _cache_get(_tts_cache, key)
    if cached:
        return cached
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        uri = "data:audio/mp3;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        _cache_put(_tts_cache, key, uri)
        return uri
    except Exception:
        logger.exception("gTTS failed")
        return None


def make_qr_data_uri(text):
    buf = io.BytesIO()
    qrcode.make(text).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_referral_code(dest_short_code):
    while True:
        rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        code = f"{dest_short_code}-{rand}"
        if not Encounter.query.filter_by(referral_code=code).first():
            return code


def get_or_create_keyword(word):
    word = word.lower().strip()[:50]
    kw = Keyword.query.filter_by(word=word).first()
    if not kw:
        kw = Keyword(word=word)
        db.session.add(kw)
        db.session.flush()
    return kw


STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out",
    "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
    "did", "she", "use", "way", "many", "would", "there", "their", "na", "wetin", "dey", "abi", "sha", "ko",
    "ni", "ta", "da", "wu",
}


def extract_search_keywords(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return list({w for w in text.split() if w.isdigit() or (len(w) > 3 and w not in STOP_WORDS)})


def clean_phone(raw):
    phone = re.sub(r"[^\d+]", "", raw or "")
    return phone if 10 <= len(phone.lstrip("+")) <= 15 else None


def to_float(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def advance_stage(enc, stage):
    if STAGES.index(stage) > STAGES.index(enc.current_stage or "translation"):
        enc.current_stage = stage


def merge_vitals(enc, new):
    current = json.loads(enc.vitals_json) if enc.vitals_json else {}
    current.update({k: v for k, v in new.items() if v is not None})
    enc.vitals_json = json.dumps(current)
    return current


def get_facility_encounter(encounter_id):
    try:
        enc = db.session.get(Encounter, int(encounter_id)) if encounter_id else None
    except (TypeError, ValueError):
        return None
    return enc if enc and enc.facility_id == session.get("facility_id") else None


def get_facility_patient(patient_id):
    try:
        p = db.session.get(Patient, int(patient_id)) if patient_id else None
    except (TypeError, ValueError):
        return None
    return p if p and p.facility_id == session.get("facility_id") and p.is_active else None


def normalize_name(name):
    return " ".join(sorted(re.sub(r"[^\w\s]", " ", (name or "").lower()).split()))


def new_encounter(patient=None, **extra):
    enc = Encounter(
        patient_id=patient.id if patient else None,
        patient_name=patient.full_name if patient else extra.pop("patient_name", "Unnamed Patient"),
        facility_id=session["facility_id"], origin_facility_id=session["facility_id"],
        origin_worker_id=session["worker_id"], **extra,
    )
    db.session.add(enc)
    return enc


def attach_patient(enc, patient):
    enc.patient_id = patient.id
    enc.patient_name = patient.full_name


def resolve_patient(data, enc=None):
    if data.get("patient_id"):
        return get_facility_patient(data["patient_id"])
    return enc.patient if enc and enc.patient and enc.patient.facility_id == session["facility_id"] else None


def optional_phone(raw):
    raw = (raw or "").strip()
    if not raw:
        return None, None
    phone = clean_phone(raw)
    return (phone, None) if phone else (None, "Use 10 to 15 digits.")


def hospital_id_owner(facility_id, hospital_id, exclude_id=None):
    q = Patient.query.filter(Patient.facility_id == facility_id, Patient.is_active.is_(True),
                             func.lower(Patient.hospital_id) == hospital_id.lower())
    if exclude_id:
        q = q.filter(Patient.id != exclude_id)
    return q.first()


def clean_patient_fields(data, facility_id, existing=None):
    out = {}
    if "full_name" in data or existing is None:
        name = re.sub(r"\s+", " ", str(data.get("full_name") or "")).strip()
        if len(name) < 2:
            return None, "Enter the patient's full name."
        out["full_name"], out["name_key"] = name[:100], normalize_name(name[:100])
    for key in ("phone", "next_of_kin_phone"):
        if key in data:
            out[key], err = optional_phone(data.get(key))
            if err:
                return None, err
    if "age_years" in data:
        raw = data.get("age_years")
        if raw in (None, ""):
            out["age_years"] = None
        else:
            try:
                age = int(raw)
            except (TypeError, ValueError):
                return None, "Age must be a whole number."
            if not 0 <= age <= 120:
                return None, "Age must be between 0 and 120."
            out["age_years"] = age
    if "sex" in data:
        sex = str(data.get("sex") or "").upper()[:1]
        out["sex"] = sex if sex in ("F", "M") else None
    if "is_pregnant" in data:
        v = data.get("is_pregnant")
        out["is_pregnant"] = v if isinstance(v, bool) else None
    for key, limit in (("village_lga", 100), ("next_of_kin_name", 100), ("allergies", 500)):
        if key in data:
            out[key] = str(data.get(key) or "").strip()[:limit] or None
    if "hospital_id" in data:
        hid = str(data.get("hospital_id") or "").strip()[:40]
        if hid:
            owner = hospital_id_owner(facility_id, hid, existing.id if existing else None)
            if owner:
                return None, f"Hospital ID {hid} already belongs to {owner.full_name}."
        out["hospital_id"] = hid or None
    return out, None


def receive_referral(src):
    fid = session["facility_id"]
    local = Encounter.query.filter_by(referred_from_id=src.id, facility_id=fid).first()
    if local:
        return local

    sp = src.patient
    patient = None
    if sp:
        if sp.phone:
            patient = Patient.query.filter_by(facility_id=fid, phone=sp.phone, name_key=sp.name_key, is_active=True).first()
        if not patient:
            patient = Patient(
                facility_id=fid, full_name=sp.full_name, name_key=sp.name_key, phone=sp.phone, age_years=sp.age_years,
                sex=sp.sex, village_lga=sp.village_lga, next_of_kin_name=sp.next_of_kin_name,
                next_of_kin_phone=sp.next_of_kin_phone, is_pregnant=sp.is_pregnant, allergies=sp.allergies,
                source_patient_id=sp.id, created_by_worker_id=session["worker_id"],
            )
            db.session.add(patient)
            db.session.flush()
    else:
        patient = Patient(facility_id=fid, full_name=src.patient_name, name_key=normalize_name(src.patient_name),
                          created_by_worker_id=session["worker_id"])
        db.session.add(patient)
        db.session.flush()

    local = new_encounter(
        patient, referred_from_id=src.id, statement=src.statement, translation=src.translation,
        vitals_json=src.vitals_json, triage_tier=src.triage_tier, triage_notes=src.triage_notes,
        diagnosis_notes=src.diagnosis_notes, diagnosis_medications=src.diagnosis_medications,
        current_stage="triage", status="active",
    )
    db.session.flush()
    return local


def fail(message, status=500, exc=None):
    if exc is not None:
        logger.error("%s\n%s", message, traceback.format_exc())
    return jsonify({"error": message}), status


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "worker_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Your session expired. Please log in again."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.before_request
def csrf_protect():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
        if not hmac.compare_digest(token, session["csrf"]):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Security check failed. Reload page."}), 400
            return redirect(url_for("login"))


@app.context_processor
def inject_globals():
    worker = None
    if "worker_id" in session:
        worker = {"name": session.get("worker_name"), "facility": session.get("facility_name")}
    # home_url: where logos and "home" links should go. Signed-in workers land on the CHW app.
    return {"csrf_token": session.get("csrf", ""), "current_worker": worker,
            "home_url": url_for("chw_app") if worker else url_for("home")}


@app.template_filter("fromjson")
def fromjson(value):
    try:
        return json.loads(value) if value else {}
    except (TypeError, ValueError):
        return {}


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Permissions-Policy", "microphone=(self)")
    if request.path.startswith("/api/"):
        # Patient data and tokens must never be served from a stale browser or service-worker cache.
        resp.headers["Cache-Control"] = "no-store"
    return resp


# Clean JSON errors for the apps, so a too-large upload or a bad URL shows a readable message, not an HTML page.
@app.errorhandler(413)
def too_large(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "That recording is too large. Keep recordings under 2 minutes."}), 413
    return "File too large", 413


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found."}), 404
    return e


@app.errorhandler(405)
def bad_method(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "That action is not allowed here."}), 405
    return e


@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500
    return "Something went wrong. Please try again.", 500


_attempts = {}


def login_throttled(ip):
    now = time.time()
    recent = [t for t in _attempts.get(ip, []) if now - t < 600]
    _attempts[ip] = recent
    return len(recent) >= 8


# ==========================================
# 5. TRIAGE RULES (deterministic)
# ==========================================
VITAL_LIMITS = {"temp": (25, 45), "hr": (20, 300), "sys_bp": (40, 300), "dia_bp": (20, 200), "spo2": (30, 100)}


def compute_triage(v):
    red, yellow = [], []
    temp, hr, sys_bp, dia_bp, spo2 = (v.get(k) for k in ("temp", "hr", "sys_bp", "dia_bp", "spo2"))

    if spo2 is not None:
        if spo2 < 90: red.append(f"SpO2 {spo2:g}% (below 90)")
        elif spo2 < 95: yellow.append(f"SpO2 {spo2:g}% (below 95)")
    if sys_bp is not None:
        if sys_bp < 90 or sys_bp >= 180: red.append(f"Systolic BP {sys_bp:g} mmHg")
        elif sys_bp < 100 or sys_bp >= 140: yellow.append(f"Systolic BP {sys_bp:g} mmHg")
    if dia_bp is not None:
        if dia_bp >= 120: red.append(f"Diastolic BP {dia_bp:g} mmHg")
        elif dia_bp >= 90: yellow.append(f"Diastolic BP {dia_bp:g} mmHg")
    if hr is not None:
        if hr < 40 or hr >= 130: red.append(f"Heart rate {hr:g} bpm")
        elif hr < 50 or hr >= 100: yellow.append(f"Heart rate {hr:g} bpm")
    if temp is not None:
        if temp >= 40 or temp < 35: red.append(f"Temperature {temp:g} °C")
        elif temp >= 38 or temp < 36: yellow.append(f"Temperature {temp:g} °C")

    if red:
        return "RED", red, "Immediate attention needed. Stabilise and refer to the nearest facility."
    if yellow:
        return "YELLOW", yellow, "Prompt assessment needed. Recheck in 15-30 min."
    return "GREEN", ["All vitals within expected range"], "Routine care. Explain danger signs to patient."


def apply_clinical(enc, data, stage="translation"):
    """Write whatever clinical fields the worker has so far onto an encounter. Used by /api/save_encounter and by
    /api/referral, so a referral always carries the visit's translation, vitals, triage and diagnosis.

    The triage tier is worked out HERE from the vitals whenever there are any, so a stale tier sent by the phone can
    never disagree with the readings. With no vitals at all, the tier the worker already saw is kept."""
    translation = str(data.get("translation") or "").strip()[:5000]
    statement = str(data.get("statement") or "").strip()[:5000]
    symptoms = str(data.get("symptoms") or "").strip()[:2000]
    if translation:
        enc.translation = translation
    if statement:
        enc.statement = statement
    elif symptoms and not enc.statement:
        enc.statement = symptoms
    if data.get("dialect"):
        enc.dialect = str(data["dialect"])[:50]
    if data.get("conclusion") or data.get("tc_actions"):
        enc.tc_steps_translation = json.dumps({
            "conclusion": data.get("conclusion"),
            "summary": data.get("tc_summary") or "Unconfirmed",
            "actions": data.get("tc_actions") or [],
        })

    vitals = {}
    for key, (low, high) in VITAL_LIMITS.items():
        v = to_float((data.get("vitals") or {}).get(key))
        if v is not None and low <= v <= high:
            vitals[key] = v
    if vitals and data.get("vitals_replace"):
        enc.vitals_json = json.dumps(vitals)      # the worker removed or changed readings: the screen is the truth
    elif vitals:
        merge_vitals(enc, vitals)

    client_tier = str(data.get("tier") or "").upper()
    current = json.loads(enc.vitals_json) if enc.vitals_json else {}
    if current and (vitals or client_tier):
        enc.triage_tier, _, enc.triage_notes = compute_triage(current)
        advance_stage(enc, "triage")
    elif client_tier in ("RED", "YELLOW", "GREEN"):
        enc.triage_tier = client_tier
        advance_stage(enc, "triage")

    diagnosis_text = str(data.get("diagnosis") or "").strip()[:500]
    if diagnosis_text:
        enc.diagnosis_notes = diagnosis_text
        enc.diagnosis_medications = str(data.get("medications") or "").strip()[:1000] or enc.diagnosis_medications
        advance_stage(enc, "diagnosis")
    if stage == "treatment" and enc.status != "referred":
        enc.status = "treated"
    return enc


# ==========================================
# 6. PAGE ROUTES
# ==========================================
@app.route("/")
def home():
    if "worker_id" in session:
        return redirect(url_for("chw_app"))
    return render_template("start.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # ProxyFix has already applied the trusted X-Forwarded-For hop to remote_addr. Reading the raw header here
        # would let anyone dodge the throttle by sending a made-up one.
        ip = request.remote_addr or ""
        if login_throttled(ip):
            return render_template("login.html", error="Too many attempts. Wait 10 minutes."), 429
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        worker = Worker.query.filter((Worker.username == username) | (Worker.worker_code == username.upper())).first()
        if worker and worker.check_password(password):
            session.clear()
            session.permanent = True
            session.update(
                worker_id=worker.id, facility_id=worker.facility_id, worker_code=worker.worker_code,
                worker_name=worker.full_name, facility_name=worker.facility.name, csrf=secrets.token_urlsafe(32),
            )
            _attempts.pop(ip, None)
            return redirect(url_for("chw_app"))
        _attempts.setdefault(ip, []).append(time.time())
        return render_template("login.html", error="Invalid worker code or password."), 401
    if "worker_id" in session:
        return redirect(url_for("chw_app"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/translator")
@login_required
def translator():
    return render_template("translator.html")


@app.route("/triage")
@login_required
def triage():
    return render_template("triage.html")


@app.route("/diagnosis")
@login_required
def diagnosis():
    return render_template("diagnosis.html")


@app.route("/referral")
@login_required
def referral():
    return render_template("referral.html")


@app.route("/lookup")
@login_required
def lookup_page():
    return render_template("lookup.html")


@app.route("/chw-app")
@login_required
def chw_app():
    """The v5 single-page CHW app (templates/chw_app.html). It reads its CSRF token from a <meta> tag."""
    return render_template("chw_app.html")


@app.route("/patient")
def patient_portal():
    return render_template("patient.html")


@app.route("/offline")
def offline():
    return render_template("offline.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/sw.js")
def service_worker():
    resp = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/csrf")
def api_csrf():
    """A fresh CSRF token for the current session. The CHW app calls this once if a request is refused because the
    page it is running was served from an old cache with an old token, then retries, so the worker never has to
    reload and lose what they typed."""
    return jsonify({"token": session.get("csrf", "")})


@app.route("/dashboard")
@login_required
def dashboard():
    facility_id = session["facility_id"]
    scope = db.and_(Encounter.facility_id == facility_id,
                    db.or_(Encounter.referral_status.is_(None), Encounter.referral_status != "revoked"))

    tier = request.args.get("tier", "all").upper()
    stage = request.args.get("stage", "all").lower()
    search = (request.args.get("search") or "").strip()[:60]
    page = request.args.get("page", 1, type=int)

    query = Encounter.query.filter(scope)
    if tier in ("RED", "YELLOW", "GREEN"):
        query = query.filter(Encounter.triage_tier == tier)
    else:
        tier = "all"
    if stage in ("triage", "diagnosis", "referred"):
        query = query.filter(Encounter.current_stage == ("referral" if stage == "referred" else stage))
    else:
        stage = "all"
    if search:
        like = f"%{search}%"
        query = query.outerjoin(Patient, Encounter.patient_id == Patient.id).filter(db.or_(
            Encounter.patient_name.ilike(like), Patient.hospital_id.ilike(like), Patient.phone.like(like)))

    pagination = query.order_by(Encounter.created_at.desc()).paginate(page=page, per_page=20, error_out=False)
    counts = dict(db.session.query(Encounter.triage_tier, func.count()).filter(scope).group_by(Encounter.triage_tier).all())

    incoming_q = Encounter.query.filter_by(destination_facility_id=facility_id, referral_status="pending")
    incoming = incoming_q.order_by(Encounter.updated_at.desc()).limit(15).all()

    sent_referrals = Encounter.query.filter(
        Encounter.facility_id == facility_id,
        Encounter.referral_code.isnot(None)
    ).order_by(Encounter.created_at.desc()).limit(20).all()

    return render_template(
        "dashboard.html",
        pagination=pagination,
        encounters=pagination.items,
        incoming=incoming,
        sent_referrals=sent_referrals,
        facility_centers={c.id: c.name for c in MedicalCenter.query.all()},
        current_tier=tier, current_stage=stage, current_search=search,
        stats={"red": counts.get("RED", 0), "yellow": counts.get("YELLOW", 0),
               "green": counts.get("GREEN", 0), "incoming": incoming_q.count()},
    )


# ==========================================
# 7. CLINICAL API ROUTES
# ==========================================
@app.route("/api/dashboard")
@login_required
def api_dashboard():
    """JSON twin of /dashboard for the v5 home screen: tier counts, referrals coming in, patients needing attention."""
    facility_id = session["facility_id"]
    scope = db.and_(Encounter.facility_id == facility_id,
                    db.or_(Encounter.referral_status.is_(None), Encounter.referral_status != "revoked"))
    counts = dict(db.session.query(Encounter.triage_tier, func.count()).filter(scope).group_by(Encounter.triage_tier).all())
    incoming_q = Encounter.query.filter_by(destination_facility_id=facility_id, referral_status="pending")

    incoming = []
    for e in incoming_q.order_by(Encounter.updated_at.desc()).limit(15).all():
        origin = db.session.get(MedicalCenter, e.origin_facility_id or e.facility_id) if (e.origin_facility_id or e.facility_id) else None
        incoming.append({"referral_code": e.referral_code, "patient_name": e.patient_name,
                         "from_facility": origin.name if origin else "Patient Self-Referral (Companion)",
                         "triage_tier": e.triage_tier or "YELLOW"})

    attention, seen = [], set()
    rows = (Encounter.query.filter(scope, Encounter.triage_tier.in_(("RED", "YELLOW")), Encounter.patient_id.isnot(None))
            .order_by(Encounter.created_at.desc()).limit(40).all())
    for e in rows:
        if e.patient_id in seen or not e.patient or not e.patient.is_active:
            continue
        seen.add(e.patient_id)
        attention.append({"patient": e.patient.to_dict(), "triage_tier": e.triage_tier,
                          "stage": "treatment" if e.status == "treated" else e.current_stage})
        if len(attention) >= 6:
            break

    return jsonify({
        "worker": {"name": session.get("worker_name"), "facility": session.get("facility_name"),
                   "facility_id": facility_id},
        "stats": {"RED": counts.get("RED", 0), "YELLOW": counts.get("YELLOW", 0), "GREEN": counts.get("GREEN", 0),
                  "incoming": incoming_q.count()},
        "incoming": incoming,
        "attention": attention,
    })


@app.route("/api/referrals")
@login_required
def api_referrals():
    """Referrals this facility has SENT, newest first, with their status, so a worker can see whether the patient
    has been received and can cancel one made by mistake. Companion self-referral passes are not listed."""
    rows = (Encounter.query.filter(Encounter.facility_id == session["facility_id"],
                                   Encounter.referral_code.isnot(None),
                                   Encounter.companion_user_id.is_(None),
                                   db.or_(Encounter.referral_status.is_(None), Encounter.referral_status != "revoked"))
            .order_by(Encounter.created_at.desc()).limit(15).all())
    return jsonify([{
        "id": e.id, "referral_code": e.referral_code, "patient_name": e.patient_name,
        "to_facility": e.referral_facility, "status": e.referral_status or "pending",
        "triage_tier": e.triage_tier, "reason": e.referral_reason,
        "date": e.created_at.strftime("%d %b %Y, %I:%M %p") if e.created_at else "",
    } for e in rows])


@app.route("/api/phrases")
@login_required
def api_phrases():
    """Quick-ask questions, treatment instructions and referral reasons, all from the database."""
    names = {"ask": "ask", "instruction": "instructions", "reason": "reasons"}
    out = {"ask": [], "instructions": [], "reasons": []}
    for r in Phrase.query.order_by(Phrase.sort_order, Phrase.id).all():
        if r.kind in names:
            out[names[r.kind]].append({"en": r.text_en, "ha": r.text_ha, "pcm": r.text_pcm})
    return jsonify(out)


@app.route("/api/centers")
@login_required
def get_centers():
    here = session.get("facility_id")
    return jsonify([
        {"id": c.id, "name": c.name, "location": c.location, "short_code": c.short_code, "current": c.id == here}
        for c in MedicalCenter.query.order_by(MedicalCenter.name).all()
    ])


@app.route("/api/centers/search")
def search_centers():
    q = (request.args.get("q") or "").strip()
    query = MedicalCenter.query
    if q:
        tokens = [t.strip() for t in q.split() if len(t.strip()) >= 2]
        for token in tokens:
            like = f"%{token}%"
            query = query.filter(db.or_(MedicalCenter.name.ilike(like), MedicalCenter.location.ilike(like)))
    centers = query.limit(10).all()
    if not centers:
        centers = MedicalCenter.query.limit(5).all()
    return jsonify([
        {"id": c.id, "name": c.name, "location": c.location or "", "short_code": c.short_code}
        for c in centers
    ])


@app.route("/api/encounter/<int:encounter_id>")
@login_required
def get_encounter(encounter_id):
    enc = get_facility_encounter(encounter_id)
    if not enc:
        return fail("Encounter not found.", 404)
    return jsonify({
        "id": enc.id,
        "patient_name": enc.patient_name,
        "patient": enc.patient.to_dict() if enc.patient else None,
        "vitals": json.loads(enc.vitals_json) if enc.vitals_json else {},
        "translation": enc.translation,
        "conclusion": fromjson(enc.tc_steps_translation).get("conclusion"),
        "triage_tier": enc.triage_tier,
        "diagnosis": enc.diagnosis_notes,
        "current_stage": enc.current_stage,
    })


@app.route("/api/encounter/<int:encounter_id>/patient", methods=["POST"])
@login_required
def link_encounter_patient(encounter_id):
    enc = get_facility_encounter(encounter_id)
    patient = get_facility_patient((request.get_json(silent=True) or {}).get("patient_id"))
    if not enc or not patient:
        return fail("Encounter or patient not found.", 404)
    attach_patient(enc, patient)
    db.session.commit()
    return jsonify({"status": "linked", "patient": patient.to_dict()})


@app.route("/api/patients")
@login_required
def search_patients():
    q = (request.args.get("q") or "").strip()[:60]
    query = Patient.query.filter_by(facility_id=session["facility_id"], is_active=True)
    tokens = normalize_name(q).split()
    for tok in tokens:
        like = f"%{tok}%"
        query = query.filter(db.or_(Patient.name_key.like(like), Patient.phone.like(like),
                                    func.lower(Patient.hospital_id).like(like)))
    rows = query.order_by(Patient.full_name if tokens else Patient.updated_at.desc()).limit(15 if tokens else 8).all()
    return jsonify([p.to_dict() for p in rows])


@app.route("/api/patients", methods=["POST"])
@login_required
def create_patient():
    data = request.get_json(silent=True) or {}
    fields, err = clean_patient_fields(data, session["facility_id"])
    if err:
        return fail(err, 409 if "already belongs" in err else 400)
    patient = Patient(facility_id=session["facility_id"], created_by_worker_id=session["worker_id"], **fields)
    db.session.add(patient)
    db.session.commit()
    return jsonify(patient.to_dict()), 201


@app.route("/api/patients/<int:patient_id>")
@login_required
def get_patient(patient_id):
    patient = get_facility_patient(patient_id)
    if not patient:
        return fail("Patient not found.", 404)
    history = (Encounter.query.filter_by(patient_id=patient.id, facility_id=session["facility_id"])
               .order_by(Encounter.created_at.desc()).limit(20).all())
    return jsonify({**patient.to_dict(), "encounters": [{
        "id": e.id, "date": e.created_at.strftime("%d %b %Y, %I:%M %p"), "triage_tier": e.triage_tier,
        "stage": "treatment" if e.status == "treated" else e.current_stage,
        "diagnosis": e.diagnosis_notes, "referral_code": e.referral_code, "referral_status": e.referral_status,
        "referral_facility": e.referral_facility, "medications": e.diagnosis_medications,
    } for e in history]})


@app.route("/api/patients/<int:patient_id>", methods=["PATCH"])
@login_required
def update_patient(patient_id):
    patient = get_facility_patient(patient_id)
    if not patient:
        return fail("Patient not found.", 404)
    fields, err = clean_patient_fields(request.get_json(silent=True) or {}, session["facility_id"], existing=patient)
    if err:
        return fail(err, 409 if "already belongs" in err else 400)
    for key, value in fields.items():
        setattr(patient, key, value)
    if "full_name" in fields:
        Encounter.query.filter_by(patient_id=patient.id).update({"patient_name": patient.full_name})
    db.session.commit()
    return jsonify(patient.to_dict())


# ==========================================
# 8. TRANSLATOR & TC ACTIONS  (3-STAGE PIPELINE)
# ==========================================
def ambiguous_intake(raw_text, readings):
    """Two likely readings of a statement the router could not settle. One analysis call gives each reading its own
    clinical summary, vitals and TC actions, so the worker can tell them apart with the patient in front of them.

    Returns the usual /api/translate payload for the most likely reading, plus mode="ambiguous" and
    `alternatives` (both readings: translation, vitals, conclusion, tc_actions). Raises if the AI reply is unusable."""
    reading_a, reading_b = readings
    prompt = f"""
    You are a medical data extraction assistant. Analyze this patient statement recorded by a health worker in Nigeria.
    ORIGINAL: "{raw_text}"
    Speech recognition could not settle what the patient meant, so there are TWO likely English readings:
    READING A: "{reading_a}"
    READING B: "{reading_b}"
    Treat each reading separately and extract ONLY what is in that reading. The ORIGINAL is raw speech-to-text and may
    contain phonetic errors, so use it only to detect the dialect and any patient name.
    Each reading object has exactly these fields:
    {{
      "vitals": {{
        "temp": "temperature in Celsius or 'N/A'",
        "bp_sys": "systolic BP or 'N/A'",
        "bp_dia": "diastolic BP or 'N/A'",
        "spo2": "oxygen saturation % or 'N/A'",
        "hr": "heart rate or 'N/A'"
      }},
      "conclusion": "One-sentence clinical summary of the main complaint in THIS reading",
      "tc_actions": [
        {{
          "claim": "Symptom extracted (e.g. Severe Headache)",
          "instruction": "Physical non-invasive instruction for the worker to give the patient (e.g. Ask patient to point with one finger to where it hurts)",
          "expected": "Expected physical response (e.g. Points to forehead or crown of head)"
        }}
      ]
    }}
    Return ONLY valid JSON (no markdown, no extra text):
    {{
      "patient_name_mentioned": "Patient name if explicitly mentioned in statement, otherwise null",
      "dialect": "Detected language of the original: Hausa, Pidgin, or English",
      "readings": [ <object for READING A>, <object for READING B> ]
    }}
    Rules for tc_actions: Provide 1 to 3 non-invasive physical checks per reading (pointing, showing fingers for days). Choose checks that would confirm THAT reading and, where possible, tell it apart from the other reading. No sensitive touching. No simple Yes/No questions. Do NOT invent symptoms.
    """
    data = llm_json(prompt, 0.2)

    rows = data.get("readings")
    rows = rows if isinstance(rows, list) else []
    alternatives = []
    for i, translation in enumerate(readings):
        row = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
        vitals = row.get("vitals") or {}
        stored = {
            "temp": to_float(vitals.get("temp")),
            "sys_bp": to_float(vitals.get("bp_sys")),
            "dia_bp": to_float(vitals.get("bp_dia")),
            "spo2": to_float(vitals.get("spo2")),
            "hr": to_float(vitals.get("hr")),
        }
        alternatives.append({
            "translation": translation,
            "vitals": {k: v for k, v in stored.items() if v is not None},
            "conclusion": row.get("conclusion"),
            "tc_actions": row.get("tc_actions") or [
                {
                    "claim": "Reported symptoms",
                    "instruction": "Ask the patient to point with their hand to where they feel discomfort.",
                    "expected": "Points to location of pain"
                }
            ],
        })

    # Top level mirrors the most likely reading, so the payload keeps the shape of a normal reply.
    return {
        "mode": "ambiguous",
        "patient_name_mentioned": data.get("patient_name_mentioned"),
        "dialect": data.get("dialect"),
        **alternatives[0],
        "alternatives": alternatives,
    }


@app.route("/api/translate", methods=["POST"])
@login_required
def translate_audio():
    audio_file = request.files.get("audio")
    if not audio_file:
        return fail("No audio file provided.", 400)

    t0 = time.time()
    # Optional hint from the "Patient speaks" selector. None means auto-detect.
    lang = STT_LANGS.get((request.form.get("language") or "").lower())

    # STAGE 1: Whisper transcribes
    try:
        raw_text = transcribe(audio_file, lang, STT_PROMPT_CLINIC)
    except ValueError as e:
        return fail(str(e), 400)
    except Exception as e:
        if getattr(e, "status_code", None) == 429:
            return fail("The speech service is busy. Wait a moment and try again.", 429, e)
        return fail("Could not transcribe the audio. Check connection and try again.", 502, e)

    # STAGE 2: Gemini translates. None means Gemini is unavailable, and LiteLLM translates in stage 3.
    # With meta-reasoning on, Gemini offers 3 ranked readings and decide_route() picks what to do with them:
    # confident -> the top reading, exactly as before; ambiguous -> two readings, each with its own TC checks;
    # unintelligible -> ask for a new recording. No usable options -> the original single translation.
    t1 = time.time()
    route = None
    if INTAKE_META_REASONING:
        options, _ = gemini_translation_options(raw_text, lang)
        route = decide_route(options) if options else None
    if route:
        logger.info("translate meta: mode=%s margin=%s", route["mode"], route["margin"])
        if route["mode"] == "unintelligible":
            return fail("I could not understand that recording. Ask the patient to repeat slowly and clearly, then record again.", 422)
    translation = None
    if route is None:
        translation = gemini_translate(raw_text, "English", lang)
    elif route["mode"] == "confident":
        translation = route["translations"][0]
    t2 = time.time()

    if route and route["mode"] == "ambiguous":
        try:
            data = ambiguous_intake(raw_text, route["translations"])
        except Exception as e:
            return fail("The AI returned unreadable data. Please try again.", 502, e)
        logger.info("translate timing (ambiguous): whisper=%.1fs gemini=%.1fs analysis=%.1fs total=%.1fs",
                    t1 - t0, t2 - t1, time.time() - t2, time.time() - t0)
        data["statement"] = raw_text   # the record must hold exactly what was heard
        return jsonify(data)

    # STAGE 3: LiteLLM analyses. It no longer has to translate: it reads the clean English and extracts from it.
    if translation:
        source_block = f"""
    ORIGINAL: "{raw_text}"
    TRANSLATION: "{translation}"
    The TRANSLATION is the authoritative English meaning. The ORIGINAL is raw speech-to-text and may contain
    phonetic errors, so use it only to detect the dialect. Extract ONLY what is in the TRANSLATION."""
        translation_field = ""
    else:
        source_block = f"""
    STATEMENT: "{raw_text}"
    The statement may be in Hausa, Nigerian Pidgin, or English. Speech recognition can mishear Hausa and Pidgin,
    so if a word looks odd, infer the most likely medical meaning instead of copying it blindly."""
        translation_field = '\n      "translation": "The statement translated into clear English",'

    prompt = f"""
    You are a medical data extraction assistant. Analyze this patient statement recorded by a health worker in Nigeria.
    {source_block}
    Return ONLY valid JSON (no markdown, no extra text):
    {{
      "patient_name_mentioned": "Patient name if explicitly mentioned in statement, otherwise null",
      "dialect": "Detected language of the original: Hausa, Pidgin, or English",{translation_field}
      "vitals": {{
        "temp": "temperature in Celsius or 'N/A'",
        "bp_sys": "systolic BP or 'N/A'",
        "bp_dia": "diastolic BP or 'N/A'",
        "spo2": "oxygen saturation % or 'N/A'",
        "hr": "heart rate or 'N/A'"
      }},
      "conclusion": "One-sentence clinical summary of the main complaint",
      "tc_actions": [
        {{
          "claim": "Symptom extracted (e.g. Severe Headache)",
          "instruction": "Physical non-invasive instruction for the worker to give the patient (e.g. Ask patient to point with one finger to where it hurts)",
          "expected": "Expected physical response (e.g. Points to forehead or crown of head)"
        }}
      ]
    }}
    Rules for tc_actions: Provide 1 to 3 non-invasive physical checks (pointing, showing fingers for days). No sensitive touching. No simple Yes/No questions. Do NOT invent symptoms.
    """
    try:
        data = llm_json(prompt, 0.2)
    except Exception as e:
        return fail("The AI returned unreadable data. Please try again.", 502, e)

    logger.info("translate timing: whisper=%.1fs gemini=%.1fs analysis=%.1fs total=%.1fs",
                t1 - t0, t2 - t1, time.time() - t2, time.time() - t0)

    # Never let the model rewrite the transcript: the record must hold exactly what was heard.
    data["statement"] = raw_text
    # Gemini's translation wins; the LiteLLM translation is only used when Gemini was unavailable.
    data["translation"] = translation or data.get("translation") or raw_text

    if not data.get("tc_actions"):
        data["tc_actions"] = [
            {
                "claim": "Reported symptoms",
                "instruction": "Ask the patient to point with their hand to where they feel discomfort.",
                "expected": "Points to location of pain"
            }
        ]

    vitals = data.get("vitals") or {}
    stored = {
        "temp": to_float(vitals.get("temp")),
        "sys_bp": to_float(vitals.get("bp_sys")),
        "dia_bp": to_float(vitals.get("bp_dia")),
        "spo2": to_float(vitals.get("spo2")),
        "hr": to_float(vitals.get("hr")),
    }
    data["vitals"] = {k: v for k, v in stored.items() if v is not None}
    return jsonify(data)


@app.route("/api/intake/save", methods=["POST"])
@login_required
def save_intake():
    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id")
    if not patient_id:
        return fail("Choose or create a patient first.", 400)

    patient = get_facility_patient(patient_id)
    if not patient:
        return fail("Patient not found in your facility.", 404)

    vitals = data.get("vitals") or {}
    tc_summary = data.get("tc_summary") or "Unconfirmed"
    tc_actions = data.get("tc_actions") or []

    enc = new_encounter(
        patient=patient,
        dialect=data.get("dialect"),
        statement=data.get("statement"),
        translation=data.get("translation"),
        vitals_json=json.dumps(vitals),
        tc_steps_translation=json.dumps({
            "conclusion": data.get("conclusion"),
            "summary": tc_summary,
            "actions": tc_actions
        }),
        current_stage="translation",
        status="active"
    )
    db.session.commit()

    return jsonify({
        "status": "saved",
        "encounter_id": enc.id,
        "patient_id": patient.id,
        "patient_name": patient.full_name
    })


@app.route("/api/followups", methods=["POST"])
@login_required
def api_followups():
    """2 to 4 short follow-up questions that dig deeper into what the patient just said (v4 Listen page).
    Each comes with a Hausa and a Pidgin version so the worker can play it to the patient. Fails soft: an empty
    list just hides the follow-up section."""
    data = request.get_json(silent=True) or {}
    translation = (data.get("translation") or "").strip()[:1500]
    if not translation:
        return jsonify({"questions": []})

    prompt = f"""You are an experienced clinical interviewer helping a community health worker in Nigeria.
    The patient said (already translated to English): "{translation}"
    Write exactly 3 short, simple follow-up questions that ask for MORE DETAIL about what the patient already
    reported (for example since when, where exactly, how bad, what makes it better or worse). Do not ask about
    symptoms the patient did not mention. Do not ask for vitals. Plain everyday words, one sentence each.
    Give each question in English, in simple spoken Hausa, and in Nigerian Pidgin.
    Return ONLY valid JSON, no markdown:
    {{"questions": [{{"en": "...", "ha": "...", "pcm": "..."}}]}}"""
    try:
        result = llm_json(prompt, 0.3)
        questions = []
        for q in result.get("questions") or []:
            if isinstance(q, dict) and str(q.get("en") or "").strip():
                questions.append({k: str(q.get(k) or "").strip()[:300] for k in ("en", "ha", "pcm")})
        return jsonify({"questions": questions[:4]})
    except Exception:
        logger.exception("Follow-up question generation failed")
        return jsonify({"questions": []})


@app.route("/api/followup_answer", methods=["POST"])
@login_required
def api_followup_answer():
    """Transcribe and translate the patient's spoken answer to ONE follow-up question (multipart: audio, language,
    question). Nothing is stored here; the page adds the English line to the translation and /api/save_encounter
    writes the combined text when the worker saves."""
    audio_file = request.files.get("audio")
    if not audio_file:
        return fail("No audio file provided.", 400)
    lang = STT_LANGS.get((request.form.get("language") or "").lower())

    try:
        raw_text = transcribe(audio_file, lang, STT_PROMPT_CLINIC)
    except ValueError as e:
        return fail(str(e), 400)
    except Exception as e:
        if getattr(e, "status_code", None) == 429:
            return fail("The speech service is busy. Wait a moment and try again.", 429, e)
        return fail("Could not transcribe the audio. Check connection and try again.", 502, e)

    translation = gemini_translate(raw_text, "English", lang)
    if not translation:
        try:
            translation = llm_text([
                {"role": "system", "content": "Translate the patient's short answer to clear English. It may be Hausa, Nigerian Pidgin or English. Return ONLY the translation, no quotes. Add nothing that was not said."},
                {"role": "user", "content": raw_text},
            ], 0.1, max_tokens=300)
        except Exception:
            logger.exception("Follow-up answer translation failed; using the transcript")
    return jsonify({"statement": raw_text, "translation": translation or raw_text})


@app.route("/api/save_encounter", methods=["POST"])
@login_required
def save_encounter():
    """Save whatever the worker has done so far (any stage, or the whole visit) to ONE patient's record.

    Creates the encounter if there is none yet, otherwise updates it, so calling it twice never duplicates a visit.
    stage: translation | diagnosis | treatment ('treatment' = treated here, no referral)."""
    data = request.get_json(silent=True) or {}
    patient = get_facility_patient(data.get("patient_id"))
    if not patient:
        return fail("Choose or create a patient first.", 400)
    enc = get_facility_encounter(data.get("encounter_id")) if data.get("encounter_id") else None
    if data.get("encounter_id") and not enc:
        return fail("Encounter not found.", 404)
    stage = str(data.get("stage") or "translation").lower()

    enc = enc or new_encounter(patient)
    attach_patient(enc, patient)
    apply_clinical(enc, data, stage)

    db.session.commit()
    return jsonify({"status": "saved", "encounter_id": enc.id, "patient_id": patient.id, "patient_name": patient.full_name})


@app.route("/api/tts_instruction", methods=["POST"])
@login_required
def tts_instruction():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()[:500]
    language = data.get("language", "en")
    if not text:
        return fail("No text provided.", 400)

    target_text = text
    if language != "en":
        lang_name = {"ha": "Hausa", "pcm": "Nigerian Pidgin"}.get(language, "Hausa")
        # A stored, checked translation beats a live one; Gemini next; the LiteLLM model is the fallback.
        row = Phrase.query.filter_by(text_en=text).first()
        stored = (row.text_ha if language == "ha" else row.text_pcm) if row else None
        translated = stored or _cache_get(_tr_cache, (lang_name, text)) or gemini_translate(text, lang_name)
        if translated:
            target_text = translated
        else:
            try:
                target_text = llm_text([
                    {"role": "system", "content": f"Translate the following medical instruction to {lang_name}. Return ONLY the translated text, no quotes."},
                    {"role": "user", "content": text},
                ], 0.3)
            except Exception:
                logger.exception("Instruction translation failed; using original text")
        if target_text != text:
            _cache_put(_tr_cache, (lang_name, text), target_text)

    audio = speak(target_text, "ha" if language == "ha" else "en")
    if not audio:
        return fail("Could not generate audio. Try again.", 502)
    return jsonify({"audio_url": audio, "translated_text": target_text})


# ==========================================
# 9. TRIAGE, DIAGNOSIS, REFERRAL & LOOKUP
# ==========================================
@app.route("/api/triage", methods=["POST"])
@login_required
def api_triage():
    data = request.get_json(silent=True) or {}
    enc = get_facility_encounter(data.get("encounter_id")) if data.get("encounter_id") else None
    if data.get("encounter_id") and not enc:
        return fail("Encounter not found.", 404)
    patient = resolve_patient(data, enc)   # optional: with no patient the tier is worked out but nothing is stored

    vitals = {k: to_float(data.get(k)) for k in VITAL_LIMITS}
    if all(v is None for v in vitals.values()):
        return fail("Enter at least one vital sign.", 400)
    for key, value in vitals.items():
        low, high = VITAL_LIMITS[key]
        if value is not None and not low <= value <= high:
            return fail(f"{key.replace('_', ' ').upper()} {value:g} looks like a typing error. Check the reading.", 400)
    if vitals["sys_bp"] and vitals["dia_bp"] and vitals["sys_bp"] <= vitals["dia_bp"]:
        return fail("Systolic BP must be higher than diastolic BP.", 400)

    tier, triggers, notes = compute_triage(vitals)
    triage_data = {"tier": tier, "trigger": "; ".join(triggers), "notes": notes}

    if not patient and not enc:
        return jsonify({"encounter_id": None, "patient": None, "triage_data": triage_data})

    enc = enc or new_encounter(patient)
    if patient:
        attach_patient(enc, patient)
    merge_vitals(enc, vitals)
    enc.triage_tier, enc.triage_notes = tier, notes
    enc.tc_steps_triage = json.dumps({"triggers": triggers})
    advance_stage(enc, "triage")
    db.session.commit()

    return jsonify({"encounter_id": enc.id, "patient": patient.to_dict() if patient else None,
                    "triage_data": triage_data})


@app.route("/api/diagnosis", methods=["POST"])
@login_required
def api_diagnosis():
    data = request.get_json(silent=True) or {}

    if "selected_diagnosis" in data:
        enc = get_facility_encounter(data.get("encounter_id"))
        if not enc:
            return fail("Encounter not found.", 404)
        selected = (data.get("selected_diagnosis") or "").strip()[:500]
        if not selected:
            return fail("Choose a diagnosis first.", 400)
        enc.diagnosis_notes = selected
        enc.diagnosis_medications = (data.get("medications") or "").strip()[:1000] or None
        advance_stage(enc, "diagnosis")
        db.session.commit()
        return jsonify({"status": "saved", "encounter_id": enc.id})

    enc = get_facility_encounter(data.get("encounter_id")) if data.get("encounter_id") else None
    if data.get("encounter_id") and not enc:
        return fail("Encounter not found.", 404)
    patient = resolve_patient(data, enc)   # optional: with no patient the causes are suggested but nothing is stored
    symptoms = (data.get("symptoms") or "").strip()[:2000]
    if not symptoms:
        return fail("Describe the symptoms first.", 400)

    vitals = {k: to_float(data.get(k)) for k in VITAL_LIMITS}
    vitals_text = ", ".join(f"{k}: {v:g}" for k, v in vitals.items() if v is not None) or "not recorded"
    extra = []
    if patient:
        if patient.age_years is not None: extra.append(f"age {patient.age_years}")
        if patient.sex: extra.append("female" if patient.sex == "F" else "male")
        if patient.is_pregnant: extra.append("pregnant")
        if patient.allergies: extra.append(f"allergies: {patient.allergies}")
    context = ("Patient: " + ", ".join(extra)) if extra else "Patient details: not recorded"
    prompt = f"""
    You are a clinical decision-support assistant for a community health worker in Nigeria.
    {context}
    Patient symptoms: "{symptoms}"
    Vitals: {vitals_text}
    Suggest the 3 to 5 most likely diagnoses, most likely first. Consider diseases common in Nigeria.
    For each one give: a short reason matching the symptoms, simple first-line care a community health worker can
    give or arrange (no drug doses; respect any allergy above), and whether hospital admission or referral is needed.
    Return ONLY valid JSON:
    {{
      "reasoning": "2 to 4 plain sentences explaining the pattern",
      "options": [
        {{"name": "Diagnosis 1", "why": "Brief reason", "meds": "Simple treatment", "adm": "Hospital needed? When?"}}
      ],
      "source_reference": "Clinical source or guideline"
    }}
    """
    try:
        result = llm_json(prompt, 0.2)
        details = []
        for o in result.get("options", []):
            if isinstance(o, dict):
                name = str(o.get("name") or "").strip()[:120]
                entry = {"name": name, "why": str(o.get("why") or "").strip()[:300],
                         "meds": str(o.get("meds") or "").strip()[:300], "adm": str(o.get("adm") or "").strip()[:300]}
            else:
                name = str(o).strip()[:120]
                entry = {"name": name, "why": "", "meds": "", "adm": ""}
            if name:
                details.append(entry)
        details = details[:5]
        options = [d["name"] for d in details]
        if not options:
            raise ValueError("No options returned")
    except Exception as e:
        return fail("Could not analyse the symptoms. Try again.", 502, e)

    reasoning = str(result.get("reasoning") or "")
    source = str(result.get("source_reference") or "Unknown")
    enc = enc or (new_encounter(patient) if patient else None)
    if enc:
        if patient:
            attach_patient(enc, patient)
        enc.statement = enc.statement or symptoms
        merge_vitals(enc, vitals)
        enc.tc_steps_diagnosis = json.dumps({"options": options, "details": details, "reasoning": reasoning, "source": source})
        advance_stage(enc, "diagnosis")
        db.session.commit()

    return jsonify({"encounter_id": enc.id if enc else None, "options": options, "details": details,
                    "reasoning": reasoning, "source_reference": source})


@app.route("/api/referral", methods=["POST"])
@login_required
def generate_referral():
    """Create a referral code. Besides the destination and reason it accepts the same clinical fields as
    /api/save_encounter (translation, symptoms, vitals, tier, diagnosis, medications), so the hospital that opens the
    code sees the whole visit even if the worker never pressed Save. With no patient_id it will create the patient
    from `patient_name`, which is what an emergency referral and a referral queued offline need."""
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip()[:1000]
    enc = get_facility_encounter(data.get("encounter_id")) if data.get("encounter_id") else None
    if data.get("encounter_id") and not enc:
        return fail("Encounter not found.", 404)
    patient = resolve_patient(data, enc)
    new_name = re.sub(r"\s+", " ", str(data.get("patient_name") or "")).strip()[:100]
    if not patient and len(new_name) < 2:
        return fail("Choose or create a patient first.", 400)
    try:
        center_id = int(data.get("center_id"))
    except (TypeError, ValueError):
        center_id = None
    if not center_id or not reason:
        return fail("Destination center and reason are required.", 400)

    dest = db.session.get(MedicalCenter, center_id)
    if not dest:
        return fail("Invalid medical center.", 404)
    if dest.id == session["facility_id"]:
        return fail("Choose a different center. You are already at this one.", 400)

    try:
        if not patient:
            patient = Patient(facility_id=session["facility_id"], full_name=new_name, name_key=normalize_name(new_name),
                              created_by_worker_id=session["worker_id"])
            db.session.add(patient)
            db.session.flush()
        elif not enc:
            # A retried request (for example one queued offline that reached the server but lost its reply) must not
            # create a second referral for the same patient, hospital and reason.
            recent = Encounter.query.filter(
                Encounter.patient_id == patient.id, Encounter.facility_id == session["facility_id"],
                Encounter.destination_facility_id == dest.id, Encounter.referral_status == "pending",
                Encounter.referral_reason == reason, Encounter.created_at > utcnow() - timedelta(minutes=10),
            ).first()
            if recent:
                return jsonify({"encounter_id": recent.id, "referral_code": recent.referral_code,
                                "referral_qr": recent.referral_qr, "facility": dest.name,
                                "patient": patient.to_dict(), "patient_name": patient.full_name, "reason": reason})

        enc = enc or new_encounter(patient)
        attach_patient(enc, patient)
        apply_clinical(enc, data)
        code = enc.referral_code if enc.referral_code and enc.destination_facility_id == dest.id \
            else generate_referral_code(dest.short_code)
        enc.referral_reason = reason
        enc.referral_facility = dest.name
        enc.referral_code = code
        enc.referral_qr = make_qr_data_uri(code)
        enc.destination_facility_id = dest.id
        enc.referral_status = "pending"
        enc.current_stage = "referral"
        enc.status = "referred"
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return fail("Could not create the referral. Try again.", 500, e)

    return jsonify({"encounter_id": enc.id, "referral_code": code, "referral_qr": enc.referral_qr,
                    "facility": dest.name, "patient": patient.to_dict(), "patient_name": patient.full_name, "reason": reason})


@app.route("/api/referral/<int:encounter_id>/cancel", methods=["POST"])
@login_required
def cancel_referral(encounter_id):
    """The sending worker withdraws a referral made by mistake. The visit stays on the patient's record; only the
    code stops working. A referral a hospital has already received cannot be cancelled."""
    enc = get_facility_encounter(encounter_id)
    if not enc or not enc.referral_code or enc.companion_user_id:
        return fail("Referral not found.", 404)
    if enc.referral_status == "fulfilled":
        return fail("A hospital has already received this patient, so the referral cannot be cancelled.", 409)
    if enc.referral_status in ("revoked", "cancelled"):
        return jsonify({"status": enc.referral_status})
    enc.referral_status = "cancelled"
    enc.status = "active"
    enc.current_stage = "diagnosis" if enc.diagnosis_notes else "triage" if enc.triage_tier else "translation"
    db.session.commit()
    return jsonify({"status": "cancelled"})


@app.route("/api/lookup", methods=["POST"])
@login_required
def lookup_patient():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        return fail("Please enter a referral code.", 400)

    src = Encounter.query.filter_by(referral_code=code).first()
    if not src:
        return fail("No patient found with this code.", 404)

    if src.referral_status == "revoked":
        return fail("This referral code was cancelled by the patient and is no longer valid.", 410)
    if src.referral_status == "cancelled":
        return fail("This referral code was cancelled by the health worker who sent it and is no longer valid.", 410)

    current_fid = session["facility_id"]
    current_fname = session.get("facility_name")

    if src.destination_facility_id and src.destination_facility_id != current_fid:
        if src.triage_tier == "RED" and src.referral_status == "pending":
            src.destination_facility_id = current_fid
            if current_fname:
                src.referral_facility = current_fname
        elif src.referral_status == "fulfilled":
            dest_center = db.session.get(MedicalCenter, src.destination_facility_id)
            dest_name = dest_center.name if dest_center else "another facility"
            return fail(f"This referral code was already checked in at {dest_name}.", 403)
        else:
            return fail("Access denied: this referral is addressed to a different facility.", 403)
    elif not src.destination_facility_id:
        src.destination_facility_id = current_fid
        if current_fname:
            src.referral_facility = current_fname

    try:
        local = receive_referral(src)
        if src.referral_status == "pending":
            src.referral_status = "fulfilled"
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return fail("Could not open this referral. Try again.", 500, e)

    from_center = None
    origin_id = src.origin_facility_id or src.facility_id
    if origin_id:
        mc = db.session.get(MedicalCenter, origin_id)
        if mc:
            from_center = mc.name
    if not from_center:
        from_center = "Patient Self-Referral (Companion)"

    vitals = {}
    if src.vitals_json:
        try:
            vitals = json.loads(src.vitals_json)
        except (TypeError, ValueError):
            vitals = {}

    return jsonify({
        "encounter_id": local.id,
        "patient": local.patient.to_dict() if local.patient else None,
        "patient_name": src.patient_name,
        "referral_code": src.referral_code,
        "facility": src.referral_facility or current_fname,
        "from_facility": from_center,
        "reason": src.referral_reason,
        "referral_status": src.referral_status,
        "triage_tier": src.triage_tier,
        "triage_notes": src.triage_notes,
        "diagnosis": src.diagnosis_notes,
        "medications": src.diagnosis_medications,
        "translation": src.translation,
        "vitals": vitals,
        "created_at": src.created_at.strftime("%Y-%m-%d %H:%M") if src.created_at else "",
    })


# ==========================================
# 10. VOICE COMPANION & IN-MEMORY EMERGENCY
# ==========================================
EMERGENCY_PATTERNS = re.compile(
    r"\b("
    r"ruwa ya zuba|jini yana zuba|jini na fita|farfadiya|numfashi da kyar|zawo mai tsanani|"
    r"water don break|blood de rush|blood dey rush|pikin de shake|breath no de catch|belly de pain well well|"
    r"water broke|water break|heavy bleeding|bleeding heavily|convulsion|convulsing|cannot breathe|can't breathe|unconscious"
    r")\b",
    re.IGNORECASE
)


# ---- Companion identity & passes: every phone number is its own private record ----
# Whisper language for the companion. "ha" suits a mostly-Hausa user base; set COMPANION_STT_LANG=auto
# to let Whisper detect the language instead (better when many users speak English or Pidgin).
COMPANION_STT_LANG = (os.getenv("COMPANION_STT_LANG", "ha") or "").strip().lower()
# Gemini translates the patient's Hausa/Pidgin to English BEFORE the LiteLLM model answers (on by default,
# because Gemini is far better at Hausa). Set COMPANION_PRETRANSLATE=0 only to skip it for speed tests.
COMPANION_PRETRANSLATE = os.getenv("COMPANION_PRETRANSLATE", "1") != "0"
# Meta-reasoning: Gemini offers 3 ranked readings of what the patient said and plain Python (decide_route)
# decides whether to answer, ask "did you mean X or Y?", or ask them to repeat. Margin = top confidence minus
# second confidence, in points out of 100. Set COMPANION_META_REASONING=0 to switch it off.
COMPANION_META_REASONING = os.getenv("COMPANION_META_REASONING", "1") != "0"
# The same router for the health worker's intake (/api/translate). Set INTAKE_META_REASONING=0 to switch it off.
INTAKE_META_REASONING = os.getenv("INTAKE_META_REASONING", "1") != "0"
CONFIDENT_MARGIN = int(os.getenv("CONFIDENT_MARGIN", "25"))
MIN_TOP_CONFIDENCE = int(os.getenv("MIN_TOP_CONFIDENCE", "40"))


def decide_route(options):
    """Margin-of-victory router. Pure Python, no AI call. `options` is [{"text", "confidence"}].

    confident      -> the top reading wins by CONFIDENT_MARGIN or more: pass 1 translation
    ambiguous      -> two readings are close: pass the top 2
    unintelligible -> nothing is trusted (top reading below MIN_TOP_CONFIDENCE, so the split is nearly flat): pass none
    """
    ranked = sorted(options or [], key=lambda o: o["confidence"], reverse=True)
    if not ranked:
        return {"mode": "unintelligible", "translations": [], "margin": 0}
    top = ranked[0]["confidence"]
    margin = top - (ranked[1]["confidence"] if len(ranked) > 1 else 0)
    if top < MIN_TOP_CONFIDENCE:
        mode, keep = "unintelligible", 0
    elif margin >= CONFIDENT_MARGIN:
        mode, keep = "confident", 1
    else:
        mode, keep = "ambiguous", 2
    return {"mode": mode, "translations": [o["text"] for o in ranked[:keep]], "margin": margin}


def emergency_reading(user_text, options):
    """The reading to answer if anything the patient may have said matches an emergency pattern, else None.
    Missing an emergency is worse than a false alarm, so the router never holds one back to ask a question."""
    if EMERGENCY_PATTERNS.search(user_text or ""):
        return options[0]["text"]
    for o in options:
        if EMERGENCY_PATTERNS.search(o["text"]):
            return o["text"]
    return None


def get_companion(phone, create=True):
    user = CompanionUser.query.filter_by(phone_number=phone).first()
    if user or not create:
        return user
    try:
        user = CompanionUser(phone_number=phone, preferred_language="ha")
        db.session.add(user)
        db.session.flush()
    except IntegrityError:  # two requests registered the same number at once
        db.session.rollback()
        user = CompanionUser.query.filter_by(phone_number=phone).first()
    return user


def pending_passes(user):
    """This user's unclaimed referral codes, newest first. Other users' codes can never appear here."""
    return (Encounter.query
            .filter(Encounter.companion_user_id == user.id,
                    Encounter.referral_status == "pending",
                    Encounter.referral_code.isnot(None))
            .order_by(Encounter.created_at.desc()))


def pass_payload(enc):
    return {"code": enc.referral_code, "qr": enc.referral_qr,
            "hospitals": enc.pass_hospitals or enc.referral_facility, "location": enc.pass_location}


def revoke_companion_referral(user):
    """Cancel the user's unclaimed referral. Returns 'revoked', 'claimed' (a hospital already took it) or 'none'.
    The caller commits."""
    pending = pending_passes(user).all()
    if pending:
        for enc in pending:
            enc.referral_status = "revoked"
            enc.status = "cancelled"
            enc.updated_at = utcnow()
        return "revoked"
    last = (Encounter.query.filter(Encounter.companion_user_id == user.id, Encounter.referral_code.isnot(None))
            .order_by(Encounter.created_at.desc()).first())
    if last and last.referral_status == "fulfilled" and last.updated_at \
            and (utcnow() - last.updated_at).total_seconds() < 12 * 3600:
        return "claimed"
    return "none"


# ---- "Cancel my referral" in the patient's own words (Hausa, Pidgin or English) ----
# Deterministic first, so cancelling is instant and works even when the AI service is slow or down.
# The AI model also reports `referral_action: "cancel"` as a fallback for phrasings these words do not cover.
_REVOKE_HA = ("soke", "sokar", "sokewa", "janye", "fasa", "goge", "cire", "kashe", "share")
_REVOKE_EN = {"cancel", "cancelled", "canceled", "revoke", "revoked", "withdraw", "delete", "remove", "erase", "annul"}
_REVOKE_PCM = {"comot", "scrap"}
_NOUN_HA = ("lamba", "tikiti", "turawa", "tura", "izini", "takardar")
_NOUN_EN = {"referral", "refer", "code", "pass", "ticket"}
_NO_NEED = re.compile(
    r"\b(?:ba ?(?:na|ni) (?:bukata|bukatar|so|son)|ban (?:bukata|bukatar|so|son)|bana (?:bukata|bukatar|so|son)|"
    r"no need|(?:do not|dont|don t|no longer|not) (?:need|want)|i no (?:need|want))\b")
# "don't cancel", "kar ka soke", "ba na so in soke": the person is saying the opposite
_NEGATED_VERB = re.compile(
    r"\b(?:do not|dont|don t|never|kar|karka|karki|kada|ba ?(?:na|ni) so in|bana so in|ban so in|ba na son)\b"
    r"(?: \w+){0,2} (?:soke|sokar|sokewa|janye|fasa|goge|cire|kashe|share|cancel|revoke|withdraw|delete|remove|erase|comot)")
_QUESTION_ONLY = re.compile(r"^(?:how|ta yaya|yaya)\b")

REVOKE_MESSAGES = {
    "ha": {
        "revoked": "An soke lambar turawarka. Ba za a iya amfani da ita a asibiti ba yanzu. Idan ka sake bukatar taimako, kawai ka gaya mini.",
        "claimed": "An riga an karbi lambar turawarka a asibiti, don haka ba za a iya soke ta ba.",
        "none": "Ban ga wata lambar turawa da ke jiran amfani a wajenka ba.",
    },
    "pcm": {
        "revoked": "I don cancel your referral code. Dem no fit use am for hospital again. If you need help later, just talk to me.",
        "claimed": "Hospital don already take your referral code, so I no fit cancel am again.",
        "none": "I no see any referral code wey dey wait for you.",
    },
    "en": {
        "revoked": "Your referral code has been cancelled and can no longer be used at a hospital. If you need help again, just tell me.",
        "claimed": "A hospital has already accepted your referral code, so it can't be cancelled any more.",
        "none": "I don't see any unclaimed referral code for you.",
    },
}


# Asked when the router cannot trust ANY reading of the audio. No AI call, so it is instant.
REPEAT_MESSAGES = {
    "ha": "Yi haƙuri, ban ji sosai ba. Don Allah a sake faɗa a hankali.",
    "pcm": "Sorry, I no hear you well. Abeg talk am again, slowly.",
    "en": "Sorry, I could not hear that clearly. Please say it again, slowly.",
}


def detect_revoke_request(raw):
    """(is_revoke, reply_language) for a transcript that asks to cancel a referral code."""
    t = re.sub(r"[’'`]", " ", (raw or "").lower())
    t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()
    if not t or _QUESTION_ONLY.match(t):
        return False, None
    words = t.split()
    wset = set(words)
    noun_ha = any(w.startswith(_NOUN_HA) for w in words)
    if not (noun_ha or wset & _NOUN_EN):
        return False, None
    verb_ha = any(w.startswith(_REVOKE_HA) for w in words)
    verb_other = bool(wset & (_REVOKE_EN | _REVOKE_PCM))
    if verb_ha or verb_other:
        if _NEGATED_VERB.search(t):
            return False, None
        hit = True
    else:
        hit = bool(_NO_NEED.search(t))  # "I don't need the code any more"
    if not hit:
        return False, None
    if verb_ha or noun_ha or re.search(r"\bba ?(?:na|ni)\b", t):
        return True, "ha"
    if wset & ({"comot", "scrap", "abeg", "dey", "wan", "wetin", "oga"}):
        return True, "pcm"
    return True, "en"


def reply_lang_code(detected):
    d = str(detected or "").lower()
    return "ha" if "haus" in d else "pcm" if "pidgin" in d else "en"


def save_exchange(user, user_text, translated, answer_local, answer_en, keywords=None):
    db.session.add_all([
        ChatMessage(user_id=user.id, role="user", original_text=user_text, translated_text=translated),
        ChatMessage(user_id=user.id, role="ai", original_text=answer_local, translated_text=answer_en),
    ])
    kws = [str(k).strip() for k in (keywords or []) if str(k).strip()][:10]
    if kws:
        memory = MemoryIndex(user_id=user.id, context_chunk=f"User: {translated}\nAI: {answer_en}")
        memory.keywords = [get_or_create_keyword(k) for k in kws]
        db.session.add(memory)


COMPANION_SYSTEM_PROMPT = """You are the SmartHealth Companion: a warm, calm, trustworthy community health worker talking with a person in Nigeria BY VOICE. Your reply is read aloud, so speak the way a kind nurse speaks to a neighbour.

LANGUAGE
- Detect the language of the ORIGINAL message (Hausa, Nigerian Pidgin or English) and reply ONLY in that language. Never mix languages.
- Use simple, everyday spoken words. Hausa: plain spoken Hausa, not formal or bookish words. Pidgin: natural, never exaggerated.
- The ORIGINAL comes from speech recognition. It can contain misheard, merged, split or misspelled words. Work out the most likely meaning from the whole sentence and the recent conversation. Never invent symptoms the person did not mention. If you truly cannot understand, say so kindly in their language and ask them to repeat. Do not guess about health.

HOW TO ANSWER
- Keep it short: 2 to 4 short sentences, about 60 words at most. No lists, numbering, headings, markdown, emojis or web links, because it will be spoken.
- Answer the actual question first. Then add at most ONE helpful next step or ONE follow-up question.
- Give general health information and practical self-care: rest, clean water, fluids and ORS for diarrhoea, handwashing, sleeping under a treated net, breastfeeding, when to go to a clinic.
- Never name prescription drugs, antibiotics or doses. Never say you are certain what illness someone has. Never promise a cure. Never tell anyone to stop medicine a health worker gave them.
- Take extra care with pregnancy, newborns, children under 5, older people, and long-term illness (HIV, diabetes, high blood pressure, sickle cell, TB). When unsure, send them to a clinic.
- Fever in Nigeria can be malaria: say it should be tested at a clinic rather than guessed at. A baby or small child with fever should be seen the same day.
- Never shame or lecture. Respect culture and religion.
- If someone talks about harming themselves, be gentle, tell them they are not alone, ask them to stay with someone they trust, and to go to the nearest hospital or call 112 now. Treat it as an emergency.

EMERGENCIES (is_emergency = true)
Heavy bleeding; bleeding or water breaking in pregnancy; fits or convulsions; hard or noisy breathing; not waking or not responding; chest pain; sudden severe headache with stiff neck or confusion; face drooping, one-sided weakness or slurred speech; snake bite; bad burns; poisoning or swallowing chemicals; a baby who is floppy, will not feed or is very hot; severe dehydration (diarrhoea and vomiting with weakness); thoughts of suicide. Mild or ordinary symptoms are NOT emergencies.
- In an emergency say plainly to go to the nearest hospital NOW. Then, if they have no code yet, offer to get them a hospital code right away.
- "spoken_location" is ONLY the place name (town, area or LGA) the person said. Otherwise null.

REFERRAL CODE (hospital pass)
- You can create and cancel the person's hospital referral code. The context tells you whether they already have an unclaimed code.
- CREATE RIGHT AWAY: when the person asks to go to a hospital, or asks for a referral or a code, and they have no code, set "referral_action" to "create" in this same reply. Do not ask which hospital and do not ask for their town just to make the code. If the person named a particular hospital, town or area, in this message or in PAST CONTEXT or RECENT CONVERSATION, put that name in "spoken_location". If they did not name one, leave "spoken_location" null: the code then works at any nearby hospital, which is fine.
- OFFER FIRST: if the person did NOT ask for a hospital but you advise one because it is serious, ask in one short sentence whether they want a hospital code, and set "create" only after they say yes.
- When you set "create", say in one short sentence that the code is on their screen and works at any nearby hospital (or at the place they named). Never say or invent the code itself.
- If you offered a code in this reply, say so in "answer_english" too, so the next turn knows an offer was made.
- If they already have a code, never create another. Tell them it is on their screen.
- CANCEL: set "referral_action" to "cancel" ONLY when the person clearly asks to cancel or no longer needs their code. If you are not sure, ask "Do you want me to cancel your code?" and keep "none".
- In every other case "referral_action" is "none".

OUTPUT: return ONLY one valid JSON object. No markdown fences, no extra text."""


def answer_patient(user_text, translated_text, past_memories, recent_chat, pending_emergency=None, active_pass=None,
                   alt_translation=None):
    pending_active = bool(pending_emergency and pending_emergency.get("active"))
    context = []
    if pending_active:
        context.append(
            f'ACTIVE EMERGENCY: the person already reported an emergency ("{pending_emergency.get("reason")}"). '
            'Check RECENT CONVERSATION. If you already offered a hospital code and they say yes or name a place, set '
            '"referral_action" to "create" (put the place in "spoken_location" if they named one; a place is not required). '
            'If you have not offered yet, offer it. If they refuse a code, accept kindly and tell them again to go to the nearest hospital.')
    if active_pass:
        context.append(f'ACTIVE REFERRAL CODE: the person already has an unclaimed code, shown on their screen, valid at: '
                       f'{active_pass.get("hospitals") or "nearby hospitals"}. Do not create another. Do not say the code aloud.')
    else:
        context.append("The person has no unclaimed referral code.")
    context.append(f"PAST CONTEXT (their history):\n{past_memories or 'No past history.'}")
    context.append(f"RECENT CONVERSATION:\n{recent_chat or 'This is the first message.'}")
    context.append(f'ORIGINAL MESSAGE (speech-to-text; decides the reply language):\n"{user_text}"')
    if translated_text and alt_translation:
        context.append(
            "UNCERTAIN HEARING: speech recognition could not settle what the person meant. They most likely said "
            f'ONE of these:\n(A) "{translated_text}"\n(B) "{alt_translation}"\n'
            "Do NOT give health advice, reassurance or a diagnosis yet, and do not pick one for them. Instead ask "
            "ONE short, friendly question, in the language of the ORIGINAL message, that puts both meanings in "
            'simple words so they can say which one they meant. Set "is_emergency" to true only if the ORIGINAL '
            'message itself clearly describes an emergency. Leave "spoken_location" null and "referral_action" "none".')
    elif translated_text:
        context.append(f'ENGLISH TRANSLATION (authoritative meaning, already checked; rely on it):\n"{translated_text}"')

    user_english_line = "" if translated_text else \
        '"user_english": "The ORIGINAL message in clear natural English. Add nothing that was not said.",\n  '
    schema = """Return exactly this JSON:
{
  {user_english_line}"answer_local": "Your spoken answer in the detected language.",
  "answer_english": "ONE short English sentence summarising your advice, for the record.",
  "detected_language": "Hausa" or "Pidgin" or "English",
  "keywords": ["up to 6 lowercase words: the key symptom or condition words in English, plus the same words as the person said them"],
  "is_emergency": true or false,
  "emergency_reason": "One short sentence, or null",
  "spoken_location": "Place name only, or null",
  "referral_action": "none" or "create" or "cancel"
}""".replace("{user_english_line}", user_english_line)
    messages = [
        {"role": "system", "content": COMPANION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(context) + "\n\n" + schema},
    ]
    try:
        res = parse_json_reply(llm_text(messages, 0.2, max_tokens=600))
    except Exception:
        logger.exception("Patient answer failed")
        res = {
            "user_english": translated_text or user_text,
            "answer_english": "Please visit the nearest health clinic immediately.",
            "answer_local": "Please visit the nearest health clinic immediately.",
            "detected_language": "English",
            "keywords": [],
            "is_emergency": pending_active,
            "emergency_reason": pending_emergency.get("reason") if pending_active else None,
            "spoken_location": None,
            "referral_action": "none",
        }

    # Deterministic keyword safety net
    text_corpus = f"{user_text} {translated_text or ''} {res.get('user_english') or ''}"
    if EMERGENCY_PATTERNS.search(text_corpus):
        res["is_emergency"] = True
        if not res.get("emergency_reason"):
            res["emergency_reason"] = "Urgent clinical danger signs detected in speech."

    return res


def revoke_response(user_text, reply, lang, outcome):
    return jsonify({
        "user_text": user_text,
        "ai_text_local": reply,
        "audio_url": None,                       # fetched separately by /api/patient_tts so the text appears at once
        "tts_lang": "ha" if lang == "ha" else "en",
        "emergency_alert": {"is_emergency": False, "reason": None},
        "auto_referral": None,
        "referral_revoked": True,                # the page clears its ticket and any pending-emergency state
        "revoke_outcome": outcome,
    })


def repeat_response(user_text, reply, lang):
    return jsonify({
        "user_text": user_text,
        "ai_text_local": reply,
        "audio_url": None,                       # fetched separately by /api/patient_tts, as for every reply
        "tts_lang": "ha" if lang == "ha" else "en",
        "emergency_alert": {"is_emergency": False, "reason": None},
        "auto_referral": None,
        "referral_revoked": False,
    })


@app.route("/api/patient_voice", methods=["POST"])
def patient_voice():
    t0 = time.time()
    phone = clean_phone(request.form.get("phone_number"))
    audio_file = request.files.get("audio")
    if not phone or not audio_file:
        return fail("Missing a valid phone number or audio.", 400)

    try:
        user_text = transcribe(audio_file, language=COMPANION_STT_LANG if COMPANION_STT_LANG in ("en", "ha") else None,
                               prompt=STT_PROMPT_COMPANION)
    except ValueError as e:
        return fail(str(e), 400)
    except Exception as e:
        if getattr(e, "status_code", None) == 429:
            return fail("Too many people are talking right now. Please try again in a minute.", 429, e)
        return fail("Could not understand the audio. Please speak clearly.", 400, e)
    t1 = time.time()

    try:
        patient = get_companion(phone)

        # In-memory pending referral recovery (sent by this phone's own page)
        raw_pending = request.form.get("pending_emergency")
        pending_emergency = None
        if raw_pending:
            try:
                pending_emergency = json.loads(raw_pending)
            except Exception:
                pending_emergency = None
        pending_active = bool(pending_emergency and pending_emergency.get("active"))

        # Fast path: "I want to cancel my referral" is handled instantly, with no AI call.
        is_revoke, rlang = detect_revoke_request(user_text)
        if is_revoke:
            outcome = revoke_companion_referral(patient)
            save_exchange(patient, user_text, user_text, REVOKE_MESSAGES[rlang][outcome], REVOKE_MESSAGES["en"][outcome])
            db.session.commit()
            logger.info("patient_voice revoke (%s): stt=%.1fs total=%.1fs", outcome, t1 - t0, time.time() - t0)
            return revoke_response(user_text, REVOKE_MESSAGES[rlang][outcome], rlang, outcome)

        # Gemini translates first (it is the one that handles Hausa well). Passing the language hint
        # helps it; on failure it returns None and the LiteLLM model translates inside its own answer.
        translated = alt_translation = None
        mode = "confident"
        heard_lang = None
        if COMPANION_PRETRANSLATE:
            stt_lang = COMPANION_STT_LANG if COMPANION_STT_LANG in ("en", "ha") else None
            options = None
            # Not while a location is awaited: a one-word town name must never be bounced as "unclear".
            if COMPANION_META_REASONING and not pending_active:
                options, heard_lang = gemini_translation_options(user_text, stt_lang)
            if options:
                route = decide_route(options)
                alarming = emergency_reading(user_text, options)
                if alarming:
                    translated = alarming
                else:
                    mode = route["mode"]
                    translated = route["translations"][0] if route["translations"] else options[0]["text"]
                    if mode == "ambiguous":
                        alt_translation = route["translations"][1]
                logger.info("patient_voice meta: mode=%s margin=%s emergency_override=%s", route["mode"], route["margin"], bool(alarming))
            else:
                translated = gemini_translate(user_text, "English", stt_lang)   # Gemini off or no usable options

        # Nothing is trustworthy: ask them to repeat, with no AI call, no advice and nothing remembered.
        if mode == "unintelligible":
            rlang = heard_lang or (patient.preferred_language if patient.preferred_language in REPEAT_MESSAGES else "ha")
            save_exchange(patient, user_text, "(unclear speech)", REPEAT_MESSAGES[rlang], REPEAT_MESSAGES["en"])
            db.session.commit()
            logger.info("patient_voice repeat-request: stt=%.1fs total=%.1fs", t1 - t0, time.time() - t0)
            return repeat_response(user_text, REPEAT_MESSAGES[rlang], rlang)

        recent = (ChatMessage.query.filter_by(user_id=patient.id)
                  .order_by(ChatMessage.created_at.desc()).limit(8).all())[::-1]
        recent_text = "\n".join(f"{m.role}: {m.translated_text or m.original_text}" for m in recent)

        # Long-term memory: matches on this person's own words, plus their two latest memories.
        memories = []
        keywords = extract_search_keywords(translated or user_text)
        mem_q = MemoryIndex.query.filter(MemoryIndex.user_id == patient.id)
        if keywords:
            memories = (mem_q.filter(MemoryIndex.keywords.any(Keyword.word.in_(keywords)))
                        .order_by(MemoryIndex.created_at.desc()).limit(3).all())
        seen = {m.id for m in memories}
        memories += [m for m in mem_q.order_by(MemoryIndex.created_at.desc()).limit(2).all() if m.id not in seen]
        long_term = "\n".join(f"Past Context: {m.context_chunk}" for m in memories)

        # The AI is told about the person's current code, so it knows whether to offer one, point to it, or cancel it.
        active = pending_passes(patient).first()
        has_pass = active is not None
        t2 = time.time()
        ai = answer_patient(user_text, translated, long_term, recent_text,
                            pending_emergency=pending_emergency,
                            active_pass=pass_payload(active) if active else None,
                            alt_translation=alt_translation)
        t3 = time.time()
        translated = str(ai.get("user_english") or translated or user_text)
        if alt_translation:   # keep the doubt on the record so the next turn knows the reading was a guess
            translated = f"(unclear) {translated} OR {alt_translation}"
        detected = str(ai.get("detected_language", "English"))
        lang = reply_lang_code(detected)
        action = str(ai.get("referral_action") or "none").strip().lower()

        # The AI decided the person wants to cancel (a phrasing the word rules did not catch).
        if action == "cancel" and not alt_translation and (has_pass or pending_active):
            outcome = revoke_companion_referral(patient)
            save_exchange(patient, user_text, translated, REVOKE_MESSAGES[lang][outcome], REVOKE_MESSAGES["en"][outcome])
            db.session.commit()
            logger.info("patient_voice revoke-by-model (%s): total=%.1fs", outcome, time.time() - t0)
            return revoke_response(user_text, REVOKE_MESSAGES[lang][outcome], lang, outcome)

        answer_en = str(ai.get("answer_english") or "")
        answer_local = str(ai.get("answer_local") or answer_en)
        # An uncertain reading is never indexed into long-term memory.
        save_exchange(patient, user_text, translated, answer_local, answer_en, None if alt_translation else ai.get("keywords"))
        db.session.commit()

        # A pass is created ONLY when the AI asks for one (referral_action = "create"). A place is optional: with
        # none, the pass is an open pass valid at any nearby hospital. Still only ONE unclaimed pass per person,
        # and never from an uncertain reading.
        spoken_loc = "" if alt_translation else str(ai.get("spoken_location") or "").strip()[:100]
        loc_label = spoken_loc or "any nearby hospital"
        auto_referral = None
        if action == "create" and not alt_translation:
            try:
                existing = pending_passes(patient).first()
                if existing:
                    auto_referral = pass_payload(existing)
                else:
                    matched_centers = []
                    if spoken_loc:
                        matched_centers = MedicalCenter.query.filter(
                            db.or_(MedicalCenter.name.ilike(f"%{spoken_loc}%"), MedicalCenter.location.ilike(f"%{spoken_loc}%"))
                        ).limit(5).all()
                    if not matched_centers:
                        matched_centers = MedicalCenter.query.limit(3).all()

                    hosp_names = ", ".join([c.name + (f" ({c.location})" if c.location else "") for c in matched_centers])
                    first_center = matched_centers[0] if matched_centers else None
                    code = generate_referral_code(first_center.short_code if first_center else "EM")
                    qr_uri = make_qr_data_uri(code)

                    enc = Encounter(
                        facility_id=first_center.id if first_center else 1,
                        origin_facility_id=first_center.id if first_center else 1,
                        destination_facility_id=None,
                        companion_user_id=patient.id,
                        pass_hospitals=hosp_names,
                        pass_location=spoken_loc or None,
                        patient_name="Emergency Voice Patient",
                        dialect=detected,
                        statement=f"Location: {loc_label} | Audio: {user_text}",
                        translation=f"Location: {loc_label} | Audio: {translated}",
                        triage_tier="RED",
                        triage_notes=f"Self-referral via voice companion ({loc_label}).",
                        referral_facility="Open Emergency Pass (Nearby Centers)",
                        referral_code=code,
                        referral_qr=qr_uri,
                        referral_reason=f"Hospital referral requested via voice companion ({loc_label})",
                        referral_status="pending",
                        current_stage="referral",
                        status="active",
                        created_at=utcnow(),
                        updated_at=utcnow()
                    )
                    db.session.add(enc)
                    db.session.commit()
                    auto_referral = pass_payload(enc)
            except Exception:
                db.session.rollback()
                logger.exception("Failed to auto-generate pass from spoken location")

    except Exception as e:
        db.session.rollback()
        return fail("Something went wrong. Please try again.", 500, e)

    logger.info("patient_voice timing: stt=%.1fs db=%.1fs llm=%.1fs total=%.1fs", t1 - t0, t2 - t1, t3 - t2, time.time() - t0)
    return jsonify({
        "user_text": user_text,
        "ai_text_local": answer_local,
        "audio_url": None,                       # fetched separately by /api/patient_tts so the text appears at once
        "tts_lang": "ha" if lang == "ha" else "en",
        "emergency_alert": {
            "is_emergency": bool(ai.get("is_emergency")),
            "reason": ai.get("emergency_reason")
        },
        "auto_referral": auto_referral,
        "referral_revoked": False,
    })


@app.route("/api/patient_tts", methods=["POST"])
def patient_tts():
    """Voice for a reply the page already shows. Split from /api/patient_voice so text never waits for gTTS."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()[:700]
    if not text:
        return fail("No text provided.", 400)
    audio = speak(text, "ha" if data.get("lang") == "ha" else "en")
    if not audio:
        return fail("Could not generate audio.", 502)
    return jsonify({"audio_url": audio})


@app.route("/api/patient_emergency_referral", methods=["POST"])
def patient_emergency_referral():
    """Unified handler: direct hospital referral and open emergency passes."""
    data = request.get_json(silent=True) or {}
    phone = clean_phone(data.get("phone_number"))
    name = (data.get("full_name") or "").strip()[:100] or "Emergency Patient"
    reason = (data.get("reason") or "Emergency referral via Voice Companion").strip()[:500]
    location = (data.get("location") or "").strip()[:100]
    center_id = data.get("facility_id")

    if not phone:
        return fail("Valid phone number required.", 400)

    hospital = db.session.get(MedicalCenter, center_id) if center_id else None
    is_open_pass = bool(location and not center_id)

    if not hospital:
        hospital = MedicalCenter.query.filter_by(short_code="MU").first() or MedicalCenter.query.first()

    short_code = hospital.short_code if hospital else "EM"
    dest_id = None if is_open_pass else (hospital.id if hospital else None)
    facility_name = "Open Emergency Pass (Nearby Centers)" if is_open_pass else (hospital.name if hospital else "Emergency Center")
    stmt = f"Location: {location} | Symptoms: {reason}" if location else reason

    try:
        companion = get_companion(phone)
        revoke_companion_referral(companion)  # one unclaimed referral per person: a new one replaces the old
        code = generate_referral_code(short_code)
        qr_uri = make_qr_data_uri(code)

        enc = Encounter(
            facility_id=hospital.id if hospital else 1,
            origin_facility_id=hospital.id if hospital else 1,
            destination_facility_id=dest_id,
            companion_user_id=companion.id,
            pass_hospitals=facility_name,
            pass_location=location or None,
            patient_name=name,
            dialect="Hausa/English",
            statement=stmt,
            translation=stmt,
            triage_tier="RED",
            triage_notes=f"Emergency self-referral from {location}." if location else "Emergency direct referral via Voice Companion.",
            referral_facility=facility_name,
            referral_code=code,
            referral_qr=qr_uri,
            referral_reason=reason,
            referral_status="pending",
            current_stage="referral",
            status="active",
            created_at=utcnow(),
            updated_at=utcnow()
        )
        db.session.add(enc)
        db.session.commit()

        return jsonify({
            "success": True,
            "referral_code": code,
            "referral_qr": qr_uri,
            "hospital_name": facility_name,
            "location": location or None,
            "patient_name": name,
            "reason": reason,
            "is_open_pass": is_open_pass
        })
    except Exception as e:
        db.session.rollback()
        return fail("Could not generate emergency referral. Please go directly to the hospital.", 500, e)


@app.route("/api/patient/active_referral")
def patient_active_referral():
    """The page asks the SERVER which code (if any) belongs to this phone number, so it never trusts what
    another number left behind in the same browser."""
    phone = clean_phone(request.args.get("phone"))
    user = get_companion(phone, create=False) if phone else None
    enc = pending_passes(user).first() if user else None
    return jsonify({"ticket": pass_payload(enc) if enc else None})


@app.route("/api/patient/revoke_referral", methods=["POST"])
def patient_revoke_referral():
    """Same as saying 'cancel my referral' in the chat, for the Cancel button on the ticket."""
    phone = clean_phone((request.get_json(silent=True) or {}).get("phone_number"))
    if not phone:
        return fail("Valid phone number required.", 400)
    try:
        user = get_companion(phone, create=False)
        outcome = revoke_companion_referral(user) if user else "none"
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return fail("Could not cancel the referral. Please try again.", 500, e)
    return jsonify({"status": outcome})


@app.route("/api/patient/referral_status")
def patient_referral_status():
    code = (request.args.get("code") or "").strip().upper()
    if not code:
        return jsonify({"status": "unknown"})
    enc = Encounter.query.filter_by(referral_code=code).first()
    if not enc:
        return jsonify({"status": "not_found"})
    return jsonify({"status": enc.referral_status})


# ==========================================
# DUPLICATE DETECTION & MERGE ENGINE
# ==========================================

@app.route("/api/patients/check-duplicates", methods=["POST"])
@login_required
def check_patient_duplicates():
    """Checks for likely existing matches in the same facility before saving a new patient."""
    data = request.get_json(silent=True) or {}
    facility_id = session["facility_id"]
    name = (data.get("full_name") or "").strip()
    phone = clean_phone(data.get("phone"))
    hospital_id = (data.get("hospital_id") or "").strip()
    age = to_float(data.get("age_years"))
    sex = str(data.get("sex") or "").upper()[:1]

    matches = []
    seen_ids = set()

    # 1. Exact match on Facility Hospital ID
    if hospital_id:
        p_hid = Patient.query.filter(
            Patient.facility_id == facility_id,
            Patient.is_active.is_(True),
            func.lower(Patient.hospital_id) == hospital_id.lower()
        ).first()
        if p_hid:
            matches.append({**p_hid.to_dict(), "match_reason": f"Exact Hospital ID match: {hospital_id}"})
            seen_ids.add(p_hid.id)

    # 2. Match on identical phone number
    if phone:
        p_phones = Patient.query.filter(
            Patient.facility_id == facility_id,
            Patient.is_active.is_(True),
            Patient.phone == phone,
            Patient.id.notin_(seen_ids)
        ).all()
        for p in p_phones:
            matches.append({**p.to_dict(), "match_reason": f"Same phone number: {phone}"})
            seen_ids.add(p.id)

    # 3. Fuzzy name match. The same name always counts; a partly matching name needs a close age or the same sex.
    if name:
        key = normalize_name(name)
        tokens = key.split()
        if tokens:
            # Narrow in the database first, so a big facility is not loaded into memory for every new patient.
            candidates = Patient.query.filter(
                Patient.facility_id == facility_id,
                Patient.is_active.is_(True),
                Patient.id.notin_(seen_ids),
                db.or_(*[Patient.name_key.like(f"%{t}%") for t in tokens]),
            ).limit(200).all()

            for c in candidates:
                c_tokens = set((c.name_key or "").split())
                overlap = len(set(tokens) & c_tokens)
                exact = c.name_key == key
                # If at least 2 tokens match or name is single word and identical
                if not (exact or overlap >= 2 or (len(tokens) == 1 and tokens[0] in c_tokens)):
                    continue
                is_close_age = age is not None and c.age_years is not None and abs(c.age_years - age) <= 3
                is_same_sex = bool(sex and c.sex and sex == c.sex)
                if exact or is_close_age or is_same_sex:
                    matches.append({**c.to_dict(),
                                    "match_reason": "Same name" if exact else "Similar name and matching age/gender"})
                    seen_ids.add(c.id)

    return jsonify({"matches": matches[:5]})


@app.route("/api/patients/merge", methods=["POST"])
@login_required
def merge_patients():
    """Combines secondary patient into primary. Reassigns encounters and soft-deletes secondary.
    What moved is written to secondary.merge_log, so /api/patients/undo-merge can put it all back."""
    data = request.get_json(silent=True) or {}
    facility_id = session["facility_id"]
    try:
        primary_id = int(data.get("primary_id"))
        secondary_id = int(data.get("secondary_id"))
    except (TypeError, ValueError):
        return fail("Both primary and secondary patient IDs are required.", 400)
    if primary_id == secondary_id:
        return fail("Cannot merge a patient into themselves.", 400)

    primary = db.session.get(Patient, primary_id)
    secondary = db.session.get(Patient, secondary_id)

    if not primary or primary.facility_id != facility_id or not primary.is_active:
        return fail("Primary patient not found or inactive.", 404)
    if not secondary or secondary.facility_id != facility_id or not secondary.is_active:
        return fail("Secondary patient not found or already merged.", 404)

    try:
        log = {"encounter_ids": [e.id for e in Encounter.query.filter_by(patient_id=secondary.id).all()]}

        # Reassign all encounters to primary
        Encounter.query.filter_by(patient_id=secondary.id).update({
            "patient_id": primary.id,
            "patient_name": primary.full_name
        })

        # Copy any missing demographic info from secondary to primary
        if not primary.phone and secondary.phone: primary.phone = secondary.phone
        if not primary.age_years and secondary.age_years: primary.age_years = secondary.age_years
        if not primary.sex and secondary.sex: primary.sex = secondary.sex
        if not primary.village_lga and secondary.village_lga: primary.village_lga = secondary.village_lga
        if not primary.allergies and secondary.allergies: primary.allergies = secondary.allergies
        # A hospital ID is unique per facility, so it must LEAVE the secondary record before the primary can hold it.
        if not primary.hospital_id and secondary.hospital_id:
            log["hospital_id"] = secondary.hospital_id
            secondary.hospital_id = None
            db.session.flush()          # free the unique ID first, then hand it to the primary
            primary.hospital_id = log["hospital_id"]

        # Soft-delete secondary record and point to primary
        secondary.is_active = False
        secondary.merged_into_id = primary.id
        secondary.merge_log = json.dumps(log)

        db.session.commit()
        return jsonify({
            "success": True,
            "primary": primary.to_dict(),
            "merged_secondary_id": secondary.id
        })
    except Exception as e:
        db.session.rollback()
        return fail("Failed to merge patient records.", 500, e)


@app.route("/api/patients/undo-merge/<int:secondary_id>", methods=["POST"])
@login_required
def undo_merge(secondary_id):
    """Reverses an accidental merge: the visits that moved go back, and so does the hospital ID."""
    facility_id = session["facility_id"]
    secondary = db.session.get(Patient, secondary_id)

    if not secondary or secondary.facility_id != facility_id:
        return fail("Patient record not found.", 404)
    if secondary.is_active or not secondary.merged_into_id:
        return fail("This patient record is not in a merged state.", 400)

    try:
        primary = db.session.get(Patient, secondary.merged_into_id)
        try:
            log = json.loads(secondary.merge_log) if secondary.merge_log else {}
        except (TypeError, ValueError):
            log = {}
        ids = [i for i in log.get("encounter_ids", []) if isinstance(i, int)]
        if ids and primary:
            Encounter.query.filter(Encounter.id.in_(ids), Encounter.patient_id == primary.id).update(
                {"patient_id": secondary.id, "patient_name": secondary.full_name}, synchronize_session=False)
        hid = log.get("hospital_id")
        if hid and primary and primary.hospital_id == hid:
            primary.hospital_id = None
            db.session.flush()          # free the unique ID first, then hand it back
            secondary.hospital_id = hid
        secondary.is_active = True
        secondary.merged_into_id = None
        secondary.merge_log = None
        db.session.commit()
        return jsonify({"success": True, "restored_patient": secondary.to_dict()})
    except Exception as e:
        db.session.rollback()
        return fail("Failed to undo merge.", 500, e)


if __name__ == "__main__":
    app.run(debug=DEBUG, port=int(os.getenv("PORT", 5000)))
