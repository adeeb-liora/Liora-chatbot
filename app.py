"""
Liora — Full-featured Streamlit Chatbot (final)

Features:
- Romantic + relationship + emotional AI assistant
- Admin panel with password: generate weekly/monthly codes, multi-use, expiry
- Local JSON persistence for codes and learning (works without DB)
- Optional OpenAI integration if OPENAI_API_KEY is set in environment or Streamlit secrets
- UPI payment instructions; manual redemption workflow
- Export codes to CSV, revoke/remove codes, view used/unused/expired
- Feedback logging for iterative improvement

Usage:
- Save as app.py
- Provide requirements.txt:
    streamlit
    openai
    requests
- Deploy to Streamlit Cloud or run locally: `streamlit run app.py`
"""

import streamlit as st
import os
import json
import random
import string
import csv
from datetime import datetime, timedelta
from io import StringIO

# Optional OpenAI import
try:
    import openai
except Exception:
    openai = None

# ---------------- CONFIG ----------------
APP_NAME = "Liora"
CODES_FILE = "lgs_codes.json"
LEARNING_FILE = "lgs_learning.json"
CONTEXT_WINDOW = 6
OWNER_UPI = "7428660149@ptsbi"   # your UPI shown to users
DEFAULT_WEEKLY_PRICE = 79
DEFAULT_MONTHLY_PRICE = 249

# ADMIN PASSWORD: set in Streamlit Secrets or env var named ADMIN_PASSWORD
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# ---------------- UTILITIES ----------------
def now_iso():
    return datetime.now().isoformat()

def parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed saving {path}: {e}")

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

# ---------------- PERSISTENCE ----------------
class Store:
    def __init__(self):
        self.codes = load_json(CODES_FILE, {"codes": []})
        self.learning = load_json(LEARNING_FILE, {"queries": [], "feedback": []})

    # codes
    def add_code(self, entry):
        self.codes.setdefault("codes", []).append(entry)
        save_json(CODES_FILE, self.codes)

    def list_codes(self):
        return self.codes.get("codes", [])

    def find_code(self, code_str):
        for c in self.codes.get("codes", []):
            if c.get("code") == code_str:
                return c
        return None

    def remove_code(self, code_str):
        before = len(self.codes.get("codes", []))
        self.codes["codes"] = [c for c in self.codes.get("codes", []) if c.get("code") != code_str]
        save_json(CODES_FILE, self.codes)
        return before != len(self.codes.get("codes", []))

    def redeem_code(self, code_str, redeemer="unknown"):
        c = self.find_code(code_str)
        if not c:
            return "invalid"
        # expiry
        exp = parse_iso(c.get("expires_at")) if c.get("expires_at") else None
        if exp and datetime.now() > exp:
            return "expired"
        # multi-use handling
        if c.get("single_use", True):
            if c.get("used"):
                return "used"
            c["used"] = True
            c["redeemed_by"] = redeemer
            c["redeemed_at"] = now_iso()
            save_json(CODES_FILE, self.codes)
            return "success"
        else:
            # allow limited uses if uses_allowed set
            uses_allowed = c.get("uses_allowed", 1)
            uses = c.get("uses", 0)
            if uses >= uses_allowed:
                return "used"
            c["uses"] = uses + 1
            c.setdefault("redeemers", []).append({"by": redeemer, "at": now_iso()})
            save_json(CODES_FILE, self.codes)
            return "success"

    # learning
    def log_query(self, q, cat):
        self.learning.setdefault("queries", []).append({"query": q, "category": cat, "date": now_iso()})
        save_json(LEARNING_FILE, self.learning)

    def save_feedback(self, response_text, rating):
        self.learning.setdefault("feedback", []).append({"response": response_text, "rating": rating, "date": now_iso()})
        save_json(LEARNING_FILE, self.learning)

store = Store()

# ---------------- TEMPLATES & DETECTION ----------------
TEMPLATES = {
    "general": [
        "Ask about a small joyful moment in their day — it opens the door to deeper conversation.",
        "Be genuinely curious: name one thing you liked from their previous message and ask about it."
    ],
    "romantic": [
        "You could say: 'You were on my mind today — what's been making you smile lately?'",
        "Try a soft invite: 'Would you like to watch a short movie together this evening?'"
    ],
    "relationship": [
        "When emotions are high, say: 'I want to understand — can you tell me more about how you felt?'",
        "Share a need, not a blame: 'I feel disconnected sometimes; can we try a short daily check-in?'"
    ],
    "emotional": [
        "If you're overwhelmed, say: 'I'm feeling a bit heavy right now. I need a small break.'",
        "Validate first: 'I hear you — that sounds really hard.'"
    ],
    "safety": [
        "If you feel unsafe, prioritize leaving the situation and contacting local help or a trusted person."
    ],
}

CATEGORY_KEYWORDS = {
    "romantic": ["romance", "flirt", "love", "date", "crush", "flirty"],
    "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "couple", "commit"],
    "emotional": ["sad", "anxious", "depressed", "lonely", "overwhelmed", "upset"],
    "safety": ["unsafe", "threat", "abuse", "harass", "harassment", "danger"],
}

def detect_category(text):
    t = text.lower()
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for k in kws:
            if k in t:
                scores[cat] = scores.get(cat, 0) + 1
    if scores:
        return max(scores, key=scores.get)
    if len(t.split()) <= 3 and t.endswith("?"):
        return "general"
    return "general"

# ---------------- OPENAI CALL ----------------
def build_messages(history, user_input):
    sys = ("You are Liora, an empathetic, romantic and practical dating assistant. "
           "Be warm, respectful, and concise (under ~120 words). If the user requests anything harmful, refuse and provide safe alternatives.")
    msgs = [{"role": "system", "content": sys}]
    # recent context
    for m in history[-CONTEXT_WINDOW*2:]:
        msgs.append({"role": m["role"], "content": m["text"]})
    msgs.append({"role": "user", "content": user_input})
    return msgs

def call_openai(messages, api_key, model="gpt-4o-mini"):
    if not openai or not api_key:
        return None
    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.8, max_tokens=300)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning("AI call failed — using templates. (" + str(e) + ")")
        return None

# ---------------- UI ----------------
st.set_page_config(page_title=f"{APP_NAME} — Romantic Chatbot", layout="centered")
st.markdown("""
<style>
body { background: linear-gradient(180deg,#fff8fb 0%, #fff2f6 100%); font-family: Inter, sans-serif; }
.header { font-weight:700; }
.chat-box { background: rgba(255,255,255,0.9); border-radius:12px; padding:10px; margin:6px 0; }
.user { text-align:right; color:#ad1457; }
.bot { text-align:left; color:#7b1fa2; }
.stButton>button { background-color:#ff6f91; color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.title(f"💞 {APP_NAME}")
st.write("Romantic, relationship, and emotional AI support — powered by Liora.")

# Sidebar admin panel controls
st.sidebar.header("⚙ Admin")
admin_toggle = st.sidebar.checkbox("Admin login")

# show stats
codes_count = len(store.list_codes())
used_count = sum(1 for c in store.list_codes() if c.get("used"))
st.sidebar.markdown(f"**Codes total:** {codes_count}  \n**Used:** {used_count}")

# Admin login block
is_admin = False
if admin_toggle:
    pw = st.sidebar.text_input("Enter admin password", type="password")
    # Also allow reading from Streamlit secrets if set
    secrets_pw = ""
    try:
        secrets_pw = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        secrets_pw = os.environ.get("ADMIN_PASSWORD", "")
    if pw and (pw == secrets_pw or pw == ADMIN_PASSWORD):
        is_admin = True
        st.sidebar.success("Admin access granted ✅")
    else:
        if pw:
            st.sidebar.error("Wrong password")

# ADMIN FEATURES
if is_admin:
    st.sidebar.subheader("Generate Access Code")
    dur = st.sidebar.selectbox("Duration", ["weekly", "monthly"])
    prefix = st.sidebar.text_input("Prefix (short)", value="LIO")
    single_use = st.sidebar.checkbox("Single-use code?", value=True)
    uses_allowed = 1
    if not single_use:
        uses_allowed = st.sidebar.number_input("Uses allowed", min_value=1, value=3)
    days_valid = 7 if dur == "weekly" else 30
    days_valid = st.sidebar.number_input("Custom validity days (0 = no expiry)", min_value=0, value=days_valid)
    note = st.sidebar.text_input("Note (plan name etc.)")

    if st.sidebar.button("Generate code"):
        code = prefix.strip().upper() + "-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if days_valid > 0:
            exp = datetime.now() + timedelta(days=int(days_valid))
            exp_iso = exp.isoformat()
        else:
            exp_iso = ""
        entry = {
            "code": code,
            "type": dur,
            "single_use": bool(single_use),
            "uses_allowed": int(uses_allowed) if not single_use else 1,
            "uses": 0,
            "note": note,
            "created_at": now_iso(),
            "expires_at": exp_iso,
            "used": False
        }
        store.add_code(entry)
        st.sidebar.success(f"Created code: {code}  (expires: {exp_iso or 'no expiry'})")
        st.experimental_rerun()

    st.sidebar.subheader("All codes")
    codes = store.list_codes()
    if codes:
        for c in codes:
            used_flag = "✅ Used" if c.get("used") else "❌ Not used"
            st.sidebar.write(f"{c['code']} | {c.get('type')} | Expires: {c.get('expires_at') or '—'} | {used_flag}")
        if st.sidebar.button("Export codes CSV"):
            si = StringIO()
            cw = csv.writer(si)
            cw.writerow(["code","type","single_use","uses_allowed","uses","expires_at","used","note","created_at","redeemed_at"])
            for c in codes:
                cw.writerow([c.get("code"),c.get("type"),c.get("single_use"),c.get("uses_allowed"),c.get("uses"),c.get("expires_at"),c.get("used"),c.get("note"),c.get("created_at"),c.get("redeemed_at","")])
            st.sidebar.download_button("Download CSV", si.getvalue(), file_name="lgs_codes.csv", mime="text/csv")
    else:
        st.sidebar.info("No codes yet. Generate one above.")

    st.sidebar.subheader("Remove / Revoke code")
    rem = st.sidebar.text_input("Enter code to remove/revoke")
    if st.sidebar.button("Remove code"):
        ok = store.remove_code(rem.strip().upper())
        if ok:
            st.sidebar.success("Removed")
        else:
            st.sidebar.error("Not found")
        st.experimental_rerun()

# ---------- Main UI: Access gating ----------
if "authorized" not in st.session_state:
    st.session_state.authorized = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.authorized:
    st.header("🔐 Access Required")
    st.write("To use Liora, please pay the owner via UPI and get an access code.")
    st.info(f"Pay to UPI: `{OWNER_UPI}` (amount: Weekly ₹{DEFAULT_WEEKLY_PRICE} / Monthly ₹{DEFAULT_MONTHLY_PRICE})")
    st.write("After payment message the owner transaction details and they'll provide a code.")
    name = st.text_input("Your name (for redemption)", key="redeemer_name")
    code_in = st.text_input("Enter access code", key="redeem_input")
    if st.button("Redeem"):
        if not code_in:
            st.error("Please enter a code.")
        else:
            res = store.redeem_code(code_in.strip().upper(), redeemer=name or "unknown")
            if res == "success":
                st.success("Code accepted — you're authorized. Enjoy Liora 💖")
                st.session_state.authorized = True
                st.session_state.username = name or "Guest"
            elif res == "expired":
                st.error("This code has expired. Please renew your subscription.")
            elif res == "used":
                st.error("This code has already been used.")
            else:
                st.error("Invalid code. Contact the owner after payment.")
    st.write("---")
    st.write("Have questions? Contact the owner after payment to get your code.")
else:
    # Authorized chat screen
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authorized = False
        st.session_state.username = ""
        st.session_state.history = []
        st.experimental_rerun()

    st.header("Chat with Liora")
    st.write("Ask anything about romance, dating, relationships, or emotional support.")

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Write your message to Liora...")
        submitted = st.form_submit_button("Send")
    if submitted and user_msg:
        st.session_state.history.append({"role":"user", "text": user_msg})
        # category detection and logging
        cat = detect_category(user_msg)
        store.log_query(user_msg, cat)
        # optional LLM
        api_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
        reply = None
        if api_key and openai:
            msgs = build_messages(st.session_state.history, user_msg)
            reply = call_openai(msgs, api_key)
        if not reply:
            reply = random.choice(TEMPLATES.get(cat, TEMPLATES["general"]))
        st.session_state.history.append({"role":"assistant", "text": reply})
    # Display chat
    for m in st.session_state.history:
        if m["role"] == "user":
            st.markdown(f"<div class='chat-box user'><b>You:</b> {m['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-box bot'><b>Liora:</b> {m['text']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.success("Chat cleared.")
    # feedback
    last_bot = next((x for x in reversed(st.session_state.history) if x["role"]=="assistant"), None)
    if last_bot:
        with st.expander("Give feedback for the last response"):
            rating = st.slider("How helpful was it?", 1, 5, 4, key="fb_slider")
            if st.button("Submit feedback", key="fb_submit"):
                store.save_feedback(last_bot["text"], rating)
                st.success("Thanks — feedback saved!")

st.markdown("---")
st.caption("Liora — not a replacement for professional therapy. For emergency/harm, contact local services.")
