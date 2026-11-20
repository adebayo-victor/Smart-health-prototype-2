"""SmartHealth: offline-first clinical support for Nigerian communities.

Flow of a patient encounter:
    translation -> triage -> diagnosis -> referral
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
from datetime import datetime, timezone
from functools import wraps

import qrcode
from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from flask_sqlalchemy import SQLAlchemy
from groq import Groq
from gtts import gTTS
from openai import OpenAI
from sqlalchemy import func
from flask.sessions import SecureCookieSessionInterface
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

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

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
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
    origin_worker_id = db.Column(db.Integer, db.ForeignKey("worker.id"), nullable=True)
    origin_facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=True)
    destination_facility_id = db.Column(db.Integer, db.ForeignKey("medical_center.id"), nullable=True)
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


# ==========================================
# 3. INITIALIZATION & DEMO DATA
# ==========================================
def seed_demo_data():
    if MedicalCenter.query.count():
        return
    password = os.getenv("SEED_PASSWORD", "password123")
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


with app.app_context():
    db.create_all()
    seed_demo_data()


# ==========================================
# 4. HELPERS
# ==========================================
_groq = None
_llm = None

# ---- Speech-to-text settings ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
MIN_AUDIO_BYTES = 2000   # about 0.5 s; one-word answers such as a town name must still pass
STT_PROMPT_CLINIC = "Medical consultation in Nigeria. Focus only on clear speech. Ignore background noise, coughing, or pauses. Terms: malaria, zazzabi, BP, temperature, headache, stomach."
# Phrases Whisper tends to invent when it hears silence or noise.
WHISPER_GHOSTS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "you", "bye", "bye bye",
    "okay", "ok", "so", "uh", "um", "hmm", "mm", "mhm", "subtitles by the amaraorg community",
}
# What the browser may send as "language". Pidgin has no Whisper code; English works best for it.
STT_LANGS = {"en": "en", "pcm": "en", "ha": "ha"}


def get_groq():
    global _groq
    if _groq is None:
        # timeout + retries: Groq occasionally returns 429/5xx or stalls on poor links
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=45.0, max_retries=2)
    return _groq


def get_llm():
    global _llm
    if _llm is None:
        _llm = OpenAI(api_key=os.getenv("LITELLM_API_KEY"), base_url=os.getenv("LITELLM_BASE_URL"))
    return _llm


def llm_text(messages, temperature=0.2):
    resp = get_llm().chat.completions.create(model=LLM_MODEL, messages=messages, temperature=temperature)
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
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        return "data:audio/mp3;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
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
    return {"csrf_token": session.get("csrf", ""), "current_worker": worker}


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
    return resp


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


# ==========================================
# 6. PAGE ROUTES
# ==========================================
@app.route("/")
def home():
    return render_template("start.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
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
            return redirect(url_for("dashboard"))
        _attempts.setdefault(ip, []).append(time.time())
        return render_template("login.html", error="Invalid worker code or password."), 401
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


@app.route("/dashboard")
@login_required
def dashboard():
    facility_id = session["facility_id"]
    scope = Encounter.facility_id == facility_id

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
@app.route("/api/centers")
@login_required
def get_centers():
    return jsonify([
        {"id": c.id, "name": c.name, "location": c.location, "short_code": c.short_code}
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
        "stage": e.current_stage, "diagnosis": e.diagnosis_notes, "referral_code": e.referral_code,
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
# 8. TRANSLATOR & TC ACTIONS
# ==========================================
@app.route("/api/translate", methods=["POST"])
@login_required
def translate_audio():
    audio_file = request.files.get("audio")
    if not audio_file:
        return fail("No audio file provided.", 400)

    # Optional hint from the "Patient speaks" selector. None means auto-detect.
    lang = STT_LANGS.get((request.form.get("language") or "").lower())
    try:
        raw_text = transcribe(audio_file, lang, STT_PROMPT_CLINIC)
    except ValueError as e:
        return fail(str(e), 400)
    except Exception as e:
        if getattr(e, "status_code", None) == 429:
            return fail("The speech service is busy. Wait a moment and try again.", 429, e)
        return fail("Could not transcribe the audio. Check connection and try again.", 502, e)

    prompt = f"""
    You are a medical data extraction assistant. Analyze this patient statement recorded by a health worker in Nigeria.
    The statement may be in Hausa, Nigerian Pidgin, or English. Speech recognition can mishear Hausa and Pidgin,
    so if a word looks odd, infer the most likely medical meaning instead of copying it blindly.
    STATEMENT: "{raw_text}"
    Return ONLY valid JSON (no markdown, no extra text):
    {{
      "patient_name_mentioned": "Patient name if explicitly mentioned in statement, otherwise null",
      "dialect": "Detected language: Hausa, Pidgin, or English",
      "statement": "The original statement exactly as transcribed",
      "translation": "The statement translated into clear English",
      "vitals": {{
        "temp": "temperature in Celsius or 'N/A'",
        "bp_sys": "systolic BP or 'N/A'",
        "bp_dia": "diastolic BP or 'N/A'",
        "spo2": "oxygen saturation % or 'N/A'"
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
    Rules for tc_actions: Provide 1 to 3 non-invasive physical checks (pointing, showing fingers for days). No sensitive touching. No simple Yes/No questions.
    """
    try:
        data = llm_json(prompt, 0.2)
    except Exception as e:
        return fail("The AI returned unreadable data. Please try again.", 502, e)

    # Never let the model rewrite the transcript: the record must hold exactly what was heard.
    data["statement"] = raw_text

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
        "spo2": to_float(vitals.get("spo2"))
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
        try:
            target_text = llm_text([
                {"role": "system", "content": f"Translate the following medical instruction to {lang_name}. Return ONLY the translated text, no quotes."},
                {"role": "user", "content": text},
            ], 0.3)
        except Exception:
            logger.exception("Instruction translation failed; using original text")

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
    patient = resolve_patient(data, enc)
    if not patient:
        return fail("Choose or create a patient first.", 400)

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

    enc = enc or new_encounter(patient)
    attach_patient(enc, patient)
    merge_vitals(enc, vitals)
    enc.triage_tier, enc.triage_notes = tier, notes
    enc.tc_steps_triage = json.dumps({"triggers": triggers})
    advance_stage(enc, "triage")
    db.session.commit()

    return jsonify({"encounter_id": enc.id, "patient": patient.to_dict(),
                    "triage_data": {"tier": tier, "trigger": "; ".join(triggers), "notes": notes}})


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
    patient = resolve_patient(data, enc)
    if not patient:
        return fail("Choose or create a patient first.", 400)
    symptoms = (data.get("symptoms") or "").strip()[:2000]
    if not symptoms:
        return fail("Describe the symptoms first.", 400)

    vitals = {k: to_float(data.get(k)) for k in VITAL_LIMITS}
    vitals_text = ", ".join(f"{k}: {v:g}" for k, v in vitals.items() if v is not None) or "not recorded"
    extra = []
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
    Return ONLY valid JSON:
    {{
      "reasoning": "2 to 4 plain sentences explaining the pattern",
      "options": ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3"],
      "source_reference": "Clinical source or guideline"
    }}
    """
    try:
        result = llm_json(prompt, 0.2)
        options = [str(o).strip()[:120] for o in result.get("options", []) if str(o).strip()][:5]
        if not options:
            raise ValueError("No options returned")
    except Exception as e:
        return fail("Could not analyse the symptoms. Try again.", 502, e)

    enc = enc or new_encounter(patient)
    attach_patient(enc, patient)
    enc.statement = enc.statement or symptoms
    merge_vitals(enc, vitals)
    reasoning = str(result.get("reasoning") or "")
    source = str(result.get("source_reference") or "Unknown")
    enc.tc_steps_diagnosis = json.dumps({"options": options, "reasoning": reasoning, "source": source})
    advance_stage(enc, "diagnosis")
    db.session.commit()

    return jsonify({"encounter_id": enc.id, "options": options, "reasoning": reasoning, "source_reference": source})


@app.route("/api/referral", methods=["POST"])
@login_required
def generate_referral():
    data = request.get_json(silent=True) or {}
    center_id = data.get("center_id")
    reason = (data.get("reason") or "").strip()[:1000]
    enc = get_facility_encounter(data.get("encounter_id")) if data.get("encounter_id") else None
    if data.get("encounter_id") and not enc:
        return fail("Encounter not found.", 404)
    patient = resolve_patient(data, enc)
    if not patient:
        return fail("Choose or create a patient first.", 400)
    if not center_id or not reason:
        return fail("Destination center and reason are required.", 400)

    dest = db.session.get(MedicalCenter, center_id)
    if not dest:
        return fail("Invalid medical center.", 404)
    if dest.id == session["facility_id"]:
        return fail("Choose a different center. You are already at this one.", 400)

    try:
        enc = enc or new_encounter(patient)
        attach_patient(enc, patient)
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


def answer_patient(user_text, translated_text, past_memories, recent_chat, pending_emergency=None):
    pending_context = ""
    if pending_emergency and pending_emergency.get("active"):
        pending_context = f"""
    ACTIVE EMERGENCY CONTEXT:
    The patient reported an emergency: "{pending_emergency.get('reason')}".
    We need their town/area to prepare a hospital referral.
    - If the patient mentions where they are (town, city, area), extract it into "spoken_location".
    - If not, answer their concern calmly, remind them this is an emergency, and ask what town/area they are in.
    """

    prompt = f"""
    You are a trusted, empathetic health worker speaking to a Nigerian patient.
    Answer their question directly, simply, and calmly. If there are red-flag symptoms, advise them to see a doctor immediately.
    Do not name prescription drugs or doses. For emergencies, tell them to go to the nearest hospital now.

    {pending_context}

    PAST CONTEXT (Their history):
    {past_memories or 'No past history.'}

    RECENT CONVERSATION:
    {recent_chat or 'This is the first message.'}

    CURRENT QUESTION (Translated to English for your understanding):
    "{translated_text}"

    ORIGINAL QUESTION (Use this ONLY to detect the language they are speaking):
    "{user_text}"

    INSTRUCTIONS:
    1. Detect the language of the ORIGINAL QUESTION (Hausa, Nigerian Pidgin, or English).
    2. Formulate your medical answer in THAT EXACT SAME DETECTED LANGUAGE.
       - If the user spoke English, reply ONLY in English.
       - If the user spoke Nigerian Pidgin, reply ONLY in Pidgin.
       - If the user spoke Hausa, reply ONLY in Hausa.
       - NEVER mix languages in your reply.
    3. If there is an emergency (or pending emergency) and the patient has not given their location yet, your answer in that same language must conclude by asking them to speak what town or area they are in right now so you can find the nearest hospital.
    4. If the user is speaking their location/town/city, return it in "spoken_location".

    Return ONLY valid JSON in this exact format:
    {{
      "answer_english": "The medical answer in simple English (for our records).",
      "answer_local": "Your medical answer in the DETECTED language (Hausa, Pidgin, or English).",
      "detected_language": "Hausa" or "Pidgin" or "English",
      "keywords": ["word1", "word2"],
      "is_emergency": true or false,
      "emergency_reason": "One short sentence why this is an emergency (or null)",
      "spoken_location": "Extracted city/town/LGA if mentioned, otherwise null"
    }}
    """
    try:
        res = llm_json(prompt, 0.2)
    except Exception:
        logger.exception("Patient answer failed")
        res = {
            "answer_english": "Please visit the nearest health clinic immediately.",
            "answer_local": "Please visit the nearest health clinic immediately.",
            "detected_language": "English",
            "keywords": [],
            "is_emergency": bool(pending_emergency and pending_emergency.get("active")),
            "emergency_reason": pending_emergency.get("reason") if pending_emergency else None,
            "spoken_location": None
        }

    # Deterministic keyword safety net
    text_corpus = f"{user_text} {translated_text}"
    if EMERGENCY_PATTERNS.search(text_corpus):
        res["is_emergency"] = True
        if not res.get("emergency_reason"):
            res["emergency_reason"] = "Urgent clinical danger signs detected in speech."

    return res


@app.route("/api/patient_voice", methods=["POST"])
def patient_voice():
    phone = clean_phone(request.form.get("phone_number"))
    audio_file = request.files.get("audio")
    if not phone or not audio_file:
        return fail("Missing a valid phone number or audio.", 400)

    try:
        # Auto-detect language here: companion users speak Hausa, Pidgin or English.
        user_text = transcribe(audio_file, None, STT_PROMPT_COMPANION)
    except ValueError as e:
        return fail(str(e), 400)
    except Exception as e:
        if getattr(e, "status_code", None) == 429:
            return fail("Too many people are talking right now. Please try again in a minute.", 429, e)
        return fail("Could not understand the audio. Please speak clearly.", 400, e)

    try:
        patient = CompanionUser.query.filter_by(phone_number=phone).first()
        if not patient:
            patient = CompanionUser(phone_number=phone, preferred_language="ha")
            db.session.add(patient)
            db.session.flush()

        translated = llm_text([
            {"role": "system", "content": "Translate to English. Return ONLY the translated text."},
            {"role": "user", "content": user_text},
        ], 0.3)

        recent = (ChatMessage.query.filter_by(user_id=patient.id)
                  .order_by(ChatMessage.created_at.desc()).limit(10).all())[::-1]
        recent_text = "\n".join(f"{m.role}: {m.translated_text or m.original_text}" for m in recent)

        long_term = ""
        keywords = extract_search_keywords(translated)
        if keywords:
            memories = (MemoryIndex.query
                        .filter(MemoryIndex.user_id == patient.id, MemoryIndex.keywords.any(Keyword.word.in_(keywords)))
                        .order_by(MemoryIndex.created_at.desc()).limit(5).all())
            long_term = "\n".join(f"Past Context: {m.context_chunk}" for m in memories)

        # In-memory pending referral recovery
        raw_pending = request.form.get("pending_emergency")
        pending_emergency = None
        if raw_pending:
            try:
                pending_emergency = json.loads(raw_pending)
            except Exception:
                pending_emergency = None

        ai = answer_patient(user_text, translated, long_term, recent_text, pending_emergency=pending_emergency)
        answer_en = ai.get("answer_english", "")
        answer_local = ai.get("answer_local") or answer_en
        detected = str(ai.get("detected_language", "English")).lower()

        db.session.add_all([
            ChatMessage(user_id=patient.id, role="user", original_text=user_text, translated_text=translated),
            ChatMessage(user_id=patient.id, role="ai", original_text=answer_local, translated_text=answer_en),
        ])
        kws = [str(k) for k in ai.get("keywords", []) if str(k).strip()][:10]
        if kws:
            memory = MemoryIndex(user_id=patient.id, context_chunk=f"User: {translated}\nAI: {answer_en}")
            memory.keywords = [get_or_create_keyword(k) for k in kws]
            db.session.add(memory)
        db.session.commit()

        # If user spoke their location, auto-generate the emergency open pass
        spoken_loc = ai.get("spoken_location")
        auto_referral = None

        if spoken_loc:
            try:
                q = spoken_loc.strip()
                matched_centers = MedicalCenter.query.filter(
                    db.or_(MedicalCenter.name.ilike(f"%{q}%"), MedicalCenter.location.ilike(f"%{q}%"))
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
                    patient_name="Emergency Voice Patient",
                    dialect=detected,
                    statement=f"Location: {spoken_loc} | Audio: {user_text}",
                    translation=f"Location: {spoken_loc} | Audio: {translated}",
                    triage_tier="RED",
                    triage_notes=f"Emergency self-referral from {spoken_loc}.",
                    referral_facility="Open Emergency Pass (Nearby Centers)",
                    referral_code=code,
                    referral_qr=qr_uri,
                    referral_reason=f"Emergency near {spoken_loc}",
                    referral_status="pending",
                    current_stage="referral",
                    status="active",
                    created_at=utcnow(),
                    updated_at=utcnow()
                )
                db.session.add(enc)
                db.session.commit()

                auto_referral = {
                    "code": code,
                    "qr": qr_uri,
                    "location": spoken_loc,
                    "hospitals": hosp_names
                }
            except Exception:
                logger.exception("Failed to auto-generate pass from spoken location")

    except Exception as e:
        db.session.rollback()
        return fail("Something went wrong. Please try again.", 500, e)

    return jsonify({
        "user_text": user_text,
        "ai_text_local": answer_local,
        "audio_url": speak(answer_local, "ha" if "haus" in detected else "en"),
        "emergency_alert": {
            "is_emergency": bool(ai.get("is_emergency")),
            "reason": ai.get("emergency_reason")
        },
        "auto_referral": auto_referral
    })


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
        code = generate_referral_code(short_code)
        qr_uri = make_qr_data_uri(code)

        enc = Encounter(
            facility_id=hospital.id if hospital else 1,
            origin_facility_id=hospital.id if hospital else 1,
            destination_facility_id=dest_id,
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

    # 3. Fuzzy name match (sorted tokens + close age or sex)
    if name:
        tokens = normalize_name(name).split()
        if tokens:
            candidates = Patient.query.filter(
                Patient.facility_id == facility_id,
                Patient.is_active.is_(True),
                Patient.id.notin_(seen_ids)
            ).all()

            for c in candidates:
                c_tokens = set((c.name_key or "").split())
                overlap = len(set(tokens) & c_tokens)
                # If at least 2 tokens match or name is single word and identical
                if (overlap >= 2) or (len(tokens) == 1 and tokens[0] in c_tokens):
                    is_close_age = age is not None and c.age_years is not None and abs(c.age_years - age) <= 3
                    is_same_sex = sex and c.sex and sex == c.sex
                    
                    if is_close_age or is_same_sex:
                        matches.append({**c.to_dict(), "match_reason": f"Similar name and matching age/gender"})
                        seen_ids.add(c.id)

    return jsonify({"matches": matches[:5]})


@app.route("/api/patients/merge", methods=["POST"])
@login_required
def merge_patients():
    """Combines secondary patient into primary. Reassigns encounters and soft-deletes secondary."""
    data = request.get_json(silent=True) or {}
    primary_id = data.get("primary_id")
    secondary_id = data.get("secondary_id")
    facility_id = session["facility_id"]

    if not primary_id or not secondary_id:
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
        # Reassign all encounters to primary
        Encounter.query.filter_by(patient_id=secondary.id).update({
            "patient_id": primary.id,
            "patient_name": primary.full_name
        })

        # Copy any missing demographic info from secondary to primary
        if not primary.phone and secondary.phone: primary.phone = secondary.phone
        if not primary.hospital_id and secondary.hospital_id: primary.hospital_id = secondary.hospital_id
        if not primary.age_years and secondary.age_years: primary.age_years = secondary.age_years
        if not primary.sex and secondary.sex: primary.sex = secondary.sex
        if not primary.village_lga and secondary.village_lga: primary.village_lga = secondary.village_lga
        if not primary.allergies and secondary.allergies: primary.allergies = secondary.allergies

        # Soft-delete secondary record and point to primary
        secondary.is_active = False
        secondary.merged_into_id = primary.id

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
    """Reverses an accidental merge."""
    facility_id = session["facility_id"]
    secondary = db.session.get(Patient, secondary_id)

    if not secondary or secondary.facility_id != facility_id:
        return fail("Patient record not found.", 404)
    if secondary.is_active or not secondary.merged_into_id:
        return fail("This patient record is not in a merged state.", 400)

    try:
        primary_id = secondary.merged_into_id
        secondary.is_active = True
        secondary.merged_into_id = None
        db.session.commit()
        return jsonify({"success": True, "restored_patient": secondary.to_dict()})
    except Exception as e:
        db.session.rollback()
        return fail("Failed to undo merge.", 500, e)

if __name__ == "__main__":
    app.run(debug=DEBUG, port=int(os.getenv("PORT", 5000)))
