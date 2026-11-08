import streamlit as st
import os
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import re
import json
import time
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings from legacy genai SDK
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)

# ---------------------------------------------------------
# CONSTANTS & PRODUCTION CONFIG
# ---------------------------------------------------------
DEFAULT_PRODUCTION_MODEL = "gemini-1.5-flash"
PRODUCTION_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

# ---------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------
@st.cache_resource
def get_db():
    conn = sqlite3.connect('period_app.db', check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    tables = [
        '''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_name_hash TEXT UNIQUE,
            created_at DATE
        )''',
        '''CREATE TABLE IF NOT EXISTS period_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            start_date DATE,
            end_date DATE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''',
        '''CREATE TABLE IF NOT EXISTS daily_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            log_date DATE,
            mood TEXT,
            symptoms TEXT,
            activities TEXT,
            food_intake TEXT,
            protein_grams INTEGER,
            ai_analysis TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''',
        '''CREATE TABLE IF NOT EXISTS emergency_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            phone TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''',
        '''CREATE TABLE IF NOT EXISTS preferences (
            user_id INTEGER PRIMARY KEY,
            fav_foods TEXT,
            preferred_language TEXT DEFAULT 'Hindi'
        )'''
    ]
    for table in tables:
        c.execute(table)

    try:
        c.execute("ALTER TABLE daily_logs ADD COLUMN ai_analysis TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()

# ---------------------------------------------------------
# GEMINI AI HELPER (ROBUST RETRY & PRODUCTION FALLBACK)
# ---------------------------------------------------------
def clean_json_response(raw_text: str) -> str:
    """Robustly strips markdown fences and extracts the JSON payload."""
    text = raw_text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    else:
        brace_match = re.search(r'(\{[\s\S]*\})', text)
        if brace_match:
            text = brace_match.group(1).strip()
    return text

def execute_gemini_prompt(prompt: str, is_json: bool = False, system_instruction: Optional[str] = None) -> str:
    """
    Executes a prompt against Google Gemini using the production model,
    supporting exponential backoff retries and graceful model fallback.
    """
    api_key = st.session_state.get("gemini_api_key") if hasattr(st, "session_state") and "gemini_api_key" in st.session_state else os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing Gemini API Key. Please provide one in the sidebar or via .env file.")

    genai.configure(api_key=api_key)
    chosen_model = st.session_state.get("selected_model", DEFAULT_PRODUCTION_MODEL) if hasattr(st, "session_state") and "selected_model" in st.session_state else DEFAULT_PRODUCTION_MODEL
    
    # Model candidate order starting with the chosen model, followed by fallbacks
    candidate_models = [chosen_model] + [m for m in PRODUCTION_MODELS if m != chosen_model]

    last_error = None
    for model_name in candidate_models:
        for attempt in range(3):
            try:
                generation_config = {"temperature": 0.7}
                if is_json:
                    generation_config["response_mime_type"] = "application/json"

                kwargs = {"model_name": model_name}
                if system_instruction:
                    kwargs["system_instruction"] = system_instruction

                model = genai.GenerativeModel(**kwargs)
                response = model.generate_content(prompt, generation_config=generation_config)
                
                if response and response.text:
                    return response.text.strip()
                raise ValueError("Empty response received from Gemini.")

            except Exception as e:
                last_error = e
                err_lower = str(e).lower()
                # Rate limit / Quota exceeded -> exponential backoff
                if "429" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
                    time.sleep(2 * (attempt + 1))
                    continue
                # If model is not found (404), break to try next model candidate
                if "404" in err_lower or "not found" in err_lower:
                    break
                # Other transient network errors
                if attempt < 2:
                    time.sleep(1.5)
                    continue
                break

    raise RuntimeError(f"Gemini API call failed after retries: {str(last_error)}")

# ---------------------------------------------------------
# AI CORE FUNCTIONS
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def ai_analyze_food(food_text: str, phase: str) -> Dict[str, Any]:
    prompt = f"""
    Analyze this food intake for a woman in her {phase} menstrual cycle phase:
    "{food_text}"
    Context: Indian food, nutrition values, Ayurvedic balance, hormones.
    Return pure JSON with EXACTLY this structure:
    {{
        "total_protein_g": <integer estimated total grams of protein>,
        "calories": <integer estimated total calories>,
        "phase_rating": "<excellent/good/average/poor>",
        "suggestions": "<2-3 actionable nutrition tips tailored to her phase>",
        "breakdown": "<simple bulleted food and macronutrient breakdown>"
    }}
    """
    try:
        raw_text = execute_gemini_prompt(prompt, is_json=True)
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        return {
            "total_protein_g": 0,
            "calories": 0,
            "phase_rating": "Notice",
            "suggestions": f"Could not analyze meal: {str(e)}",
            "breakdown": "Please verify your Gemini API key in the sidebar."
        }

@st.cache_data(ttl=300, show_spinner=False)
def ai_day_plan(user_id: int, phase: str) -> Dict[str, Any]:
    prompt = f"""
    Create a personalized 1-day wellness plan for a woman in her {phase} menstrual phase.
    Focus on authentic Indian foods (warm spices, lentils, curd, rotis/rice, nuts/seeds).
    Return pure JSON with this EXACT structure:
    {{
        "meals": [
            "Breakfast: description + key nutrients",
            "Mid-Morning Snack: light snack description",
            "Lunch: nourishing meal description",
            "Evening Snack: seed cycling/tea snack",
            "Dinner: easy-to-digest dinner description"
        ],
        "activity": "30-min recommended movement for this phase (e.g. yoga, brisk walk, pilates)",
        "hydration": "Hydration goal and herbal drink suggestion (e.g. 2.5L water + warm cumin tea)"
    }}
    """
    try:
        raw_text = execute_gemini_prompt(prompt, is_json=True)
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        return {
            "meals": [
                f"Notice: {str(e)}",
                "Warm ginger tea + almonds",
                "Moong dal khichdi + spinach curry",
                "Roasted makhana",
                "Steamed rice + light rasam"
            ],
            "activity": "Gentle restorative yoga or light stretching",
            "hydration": "2.5 Liters of warm water with lemon or cumin seeds"
        }

@st.cache_data(ttl=60, show_spinner=False)
def ai_insights(user_id: int) -> str:
    conn = get_db()
    logs = pd.read_sql_query(
        "SELECT * FROM daily_logs WHERE user_id=? ORDER BY log_date DESC LIMIT 30",
        conn,
        params=(user_id,)
    )

    if logs.empty:
        return "Log your daily symptoms and meals for at least a few days to unlock your bespoke AI health patterns! 📊🌸"

    protein_mean = logs['protein_grams'].dropna().mean()
    avg_protein = f"{protein_mean:.1f}g" if pd.notnull(protein_mean) else "N/A"
    mood_counts = logs['mood'].dropna().value_counts().to_dict()
    
    symptom_series = logs['symptoms'].dropna().astype(str).str.split(';').explode()
    symptom_series = symptom_series[symptom_series.str.strip() != '']
    symptom_counts = symptom_series[symptom_series != 'None'].value_counts()
    top_symptoms = '; '.join(symptom_counts.head(4).index.tolist()) or "None reported"

    data_summary = f"""
    Summary of recent {len(logs)} log entries:
    - Average Daily Protein: {avg_protein}
    - Mood Distribution: {mood_counts}
    - Common Symptoms: {top_symptoms}
    """

    prompt = f"""
    You are an expert women's hormonal health coach.
    Analyze these menstrual and lifestyle trends for an Indian woman:
    {data_summary}

    Provide:
    1. 🌸 **Key Patterns Spotted**: 2 clear correlations between her symptoms, mood, and nutrition.
    2. 💡 **Actionable Glow Up Tips**: 3 practical diet, lifestyle, and Ayurvedic tweaks.
    3. 🔮 **Cycle Prediction**: What to expect in the coming week and how to prep.
    
    Keep the tone warm, empowering, empathetic, and scientific yet friendly.
    """
    try:
        return execute_gemini_prompt(prompt, is_json=False)
    except Exception as e:
        return f"⚠️ AI Pattern Analysis Error: {str(e)}\n\nPlease ensure your Gemini API key is valid."

def ai_chat_response(messages: List[Dict[str, str]], phase: str) -> str:
    system_instruction = f"""
    You are 'Glow Bestie', an ultra-caring, empathetic, knowledgeable women's health coach.
    The user is currently in the {phase.upper()} phase of her cycle.
    Provide warm, culturally-aware health, lifestyle, and nutrition advice (Indian context).
    Always be loving, supportive, concise, and gentle.
    If the user asks in Hindi, Malayalam, Tamil, or Hinglish, answer naturally in that style.
    Remind her gently that you provide wellness support, not medical diagnoses.
    """
    conversation = ""
    for msg in messages[-6:]:
        role = "User" if msg["role"] == "user" else "Bestie"
        conversation += f"{role}: {msg['content']}\n"
    conversation += f"User: {messages[-1]['content']}\nBestie:"

    try:
        return execute_gemini_prompt(conversation, is_json=False, system_instruction=system_instruction)
    except Exception as e:
        return f"Oh darling, I couldn't reach the AI cloud right now! ({str(e)}) 🌸 Please check your API key in the sidebar."

# ---------------------------------------------------------
# DATABASE USER & CYCLE HELPERS
# ---------------------------------------------------------
def get_or_create_user(code_name: str) -> int:
    conn = get_db()
    c = conn.cursor()
    hash_name = hashlib.sha256(code_name.strip().encode()).hexdigest()[:16]
    c.execute("SELECT id FROM users WHERE code_name_hash=?", (hash_name,))
    user = c.fetchone()
    if not user:
        c.execute(
            "INSERT INTO users (code_name_hash, created_at) VALUES (?, ?)",
            (hash_name, date.today())
        )
        user_id = c.lastrowid
        conn.commit()
        return user_id
    return user[0]

def get_current_phase_details(user_id: int) -> Dict[str, Any]:
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT start_date FROM period_logs WHERE user_id=? ORDER BY start_date DESC, id DESC LIMIT 1",
        (user_id,)
    )
    result = c.fetchone()

    if not result or not result[0]:
        return {
            "name": "Unknown",
            "cycle_day": None,
            "description": "Please log the start date of your last period to reveal your current cycle phase.",
            "color": "#ff69b4"
        }

    try:
        last_period_start = date.fromisoformat(str(result[0]))
    except Exception:
        return {
            "name": "Unknown",
            "cycle_day": None,
            "description": "Invalid period date recorded. Please log your period start date again.",
            "color": "#ff69b4"
        }

    days_since = (date.today() - last_period_start).days
    if days_since < 0:
        days_since = 0

    cycle_day = (days_since % 28) + 1

    if cycle_day <= 5:
        return {
            "name": "Menstrual",
            "cycle_day": cycle_day,
            "description": "Rest, replenish iron, sip warm ginger-turmeric teas, and take things slow 🍵",
            "color": "#e91e63"
        }
    elif cycle_day <= 13:
        return {
            "name": "Follicular",
            "cycle_day": cycle_day,
            "description": "Rising estrogen! Energy is rebounding. Great time for fresh salads, light seeds & new projects 🌸",
            "color": "#9c27b0"
        }
    elif cycle_day <= 16:
        return {
            "name": "Ovulatory",
            "cycle_day": cycle_day,
            "description": "Peak energy and radiance! Enjoy antioxidant-rich berries, colorful veggies, and workouts ✨",
            "color": "#ff4081"
        }
    else:
        return {
            "name": "Luteal",
            "cycle_day": cycle_day,
            "description": "Progesterone peak. Craving warmth? Opt for magnesium-rich foods (dark chocolate, pumpkin seeds) 🍫",
            "color": "#673ab7"
        }

# ---------------------------------------------------------
# STREAMLIT APPLICATION RUNNER
# ---------------------------------------------------------
def render_app():
    st.set_page_config(
        page_title="Flow Glow ✨ | Cycle Care & AI Bestie",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stApp {
            background-color: #fffafc;
        }
        h1, h2, h3 {
            color: #d81b60 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-weight: 700;
        }
        .glow-card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(255, 105, 180, 0.12);
            border: 1px solid #ffe4ed;
            margin-bottom: 16px;
        }
        .glow-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            background: linear-gradient(135deg, #ff4081, #ff80ab);
            color: white;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
        }
        [data-testid="stPopover"] {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 999999;
        }
        [data-testid="stPopover"] > button {
            border-radius: 50% !important;
            width: 65px !important;
            height: 65px !important;
            background: linear-gradient(135deg, #ff1493, #ff69b4) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 6px 18px rgba(255, 20, 147, 0.45) !important;
            font-size: 30px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            transition: transform 0.2s ease-in-out;
        }
        [data-testid="stPopover"] > button:hover {
            transform: scale(1.08);
        }
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: none;
        }
        [data-testid="stMetricValue"] {
            color: #ff1493 !important;
            font-weight: 700 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    init_db()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hey gorgeous! 🌸 I'm your Flow Glow bestie. Ask me anything about your cycle, food cravings, energy tips, or hormonal harmony! ✨"}
        ]

    # Sidebar
    with st.sidebar:
        st.title("🌸 Flow Glow 💖")
        st.caption("Your Private Hormonal Wellness & AI Sanctuary")

        if "user_id" not in st.session_state:
            st.subheader("🔐 Secret Diary Access")
            code_name = st.text_input("Enter your secret nickname", type="password", help="Your data is privately tied to this unique pseudonym.")
            if st.button("✨ Unlock Diary", type="primary", use_container_width=True):
                if code_name.strip():
                    st.session_state.user_id = get_or_create_user(code_name)
                    st.session_state.code_name = code_name.strip()
                    st.rerun()
                else:
                    st.error("Please enter a nickname to proceed!")
        else:
            st.success(f"🌸 Hey, {st.session_state.code_name}! ✨")
            if st.button("🚪 Lock Diary (Logout)", use_container_width=True):
                for key in ["user_id", "code_name", "nutrition", "current_day_plan", "latest_insights"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

            st.markdown("---")
            st.subheader("🩸 Cycle Tracker")
            phase_info = get_current_phase_details(st.session_state.user_id)
            
            if phase_info["name"] == "Unknown":
                st.warning("⚠️ Log your last period to calculate your current cycle phase.")
            else:
                st.info(f"**Day {phase_info['cycle_day']}** • **{phase_info['name']} Phase**\n\n_{phase_info['description']}_")

            new_period_date = st.date_input("Log Period Start Date", value=date.today())
            if st.button("💾 Save Period Date", use_container_width=True):
                conn = get_db()
                c = conn.cursor()
                c.execute(
                    "INSERT INTO period_logs (user_id, start_date) VALUES (?, ?)",
                    (st.session_state.user_id, new_period_date)
                )
                conn.commit()
                st.cache_data.clear()
                st.success("✅ Cycle updated!")
                st.rerun()

            st.markdown("---")
            st.subheader("⚙️ AI Model Settings")
            env_key = os.getenv("GEMINI_API_KEY", "")
            
            user_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.get("gemini_api_key", env_key),
                help="Free key available from Google AI Studio (aistudio.google.com)"
            )
            if user_key:
                st.session_state.gemini_api_key = user_key
                st.caption("🟢 API Key Active")
            else:
                st.caption("🔴 API Key Required for AI Features")

            st.session_state.selected_model = st.selectbox(
                "Production Model",
                options=PRODUCTION_MODELS,
                index=0,
                help="gemini-1.5-flash is the Google recommended production model with fast inference and high rate limits."
            )

    # Main Screen Content
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.markdown("""
        <div class="glow-card" style="text-align: center; padding: 40px;">
            <h1 style="font-size: 2.5rem; margin-bottom: 10px;">🌸 Welcome to Flow Glow ✨</h1>
            <p style="font-size: 1.1rem; color: #666; max-width: 650px; margin: 0 auto 24px auto;">
                Your private, safe, and intelligent menstrual cycle sanctuary. Track your symptoms, get AI-powered phase-specific Indian nutrition plans, uncover hormonal patterns, and chat with your empathetic AI Bestie.
            </p>
            <div style="background: #fff0f5; padding: 15px 25px; border-radius: 12px; display: inline-block; border: 1px dashed #ff69b4;">
                👉 <b>Enter your secret nickname in the sidebar</b> on the left to unlock your private space!
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    current_phase_info = get_current_phase_details(user_id)
    phase_name = current_phase_info["name"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗓️ My Glow Plan 🌸",
        "📝 Dear Diary 🎀",
        "✨ My Sparkle Report 📊",
        "🆘 Safe Space 🚨"
    ])

    # Tab 1: Glow Plan
    with tab1:
        st.header("🎀 My Bespoke Glow Up Plan ✨")
        st.markdown(f"""
        <div class="glow-card">
            <span class="glow-badge">{phase_name.upper()} PHASE</span>
            <h3 style="margin-top: 5px; color: {current_phase_info['color']};">Cycle Phase Status</h3>
            <p style="color: #555; font-size: 1.05rem;">{current_phase_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ Generate Fresh AI Plan", type="primary"):
            with st.spinner(f"Curating your personalized {phase_name} day plan with Indian superfoods..."):
                plan = ai_day_plan(user_id, phase_name)
                st.session_state.current_day_plan = plan

        if "current_day_plan" in st.session_state:
            plan = st.session_state.current_day_plan
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("🍲 Nourishing Meals")
                for i, meal in enumerate(plan.get("meals", []), 1):
                    st.markdown(f"""
                    <div style="background: white; border-left: 4px solid #ff4081; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                        <b>Meal {i}:</b> {meal}
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.subheader("⚡ Movement & Hydration")
                st.markdown(f"""
                <div class="glow-card">
                    <h4>🏃 Recommended Movement</h4>
                    <p style="color: #444;">{plan.get('activity', 'Gentle walking or restorative yoga')}</p>
                    <hr style="border: 0; border-top: 1px solid #ffe4ed; margin: 15px 0;">
                    <h4>💧 Hydration Target</h4>
                    <p style="color: #444;">{plan.get('hydration', '2.5L water with warm herbal infusions')}</p>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: Dear Diary
    with tab2:
        st.header("💖 How are we feeling today, bestie? 🌸")
        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            log_date = st.date_input("📅 Date", value=date.today())
            mood = st.selectbox(
                "😊 Mood",
                ["Very Low 🌧️", "Low ☁️", "Neutral 🌤️", "High 🌸", "Very High ✨"],
                index=2
            )

        with col2:
            symptoms = st.multiselect(
                "🤕 Symptoms",
                ["Cramps", "Bloating", "Headache", "Fatigue", "Back pain", "Leg pain", "Acne", "Breast tenderness", "None"],
                default=[]
            )
            activities = st.multiselect(
                "⚡ Today's Movement",
                ["Yoga", "Walk", "Work", "Gym", "Dance", "Rest", "Pilates"],
                default=["Rest"]
            )

        with col3:
            st.subheader("🍲 What did you eat today?")
            food_text = st.text_area(
                "Log your meals & snacks",
                placeholder="e.g. 2 idlis with sambar, 1 boiled egg, palak paneer with 2 rotis...",
                height=90
            )

            if "nutrition" in st.session_state and st.session_state.nutrition:
                nut = st.session_state.nutrition
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("🥩 Est. Protein", f"{nut.get('total_protein_g', 0)}g")
                metric_col2.metric("🔥 Est. Calories", f"{nut.get('calories', 0)}")
                
                if nut.get("suggestions"):
                    st.caption(f"💡 **Tip:** {nut.get('suggestions')}")

            if st.button("🔍 Get Nutrition Analysis", key="analyze_btn", use_container_width=True):
                if food_text.strip():
                    with st.spinner("🫕 Analyzing Indian nutritional balance..."):
                        st.session_state.nutrition = ai_analyze_food(food_text.strip(), phase_name)
                        st.rerun()
                else:
                    st.warning("Please type your meals before analyzing!")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Dear Diary Entry", type="primary", use_container_width=True):
            nutrition = st.session_state.get("nutrition", {})
            if not nutrition and food_text.strip():
                with st.spinner("Analyzing nutrition before saving..."):
                    nutrition = ai_analyze_food(food_text.strip(), phase_name)

            conn = get_db()
            c = conn.cursor()
            c.execute(
                """INSERT INTO daily_logs (
                    user_id, log_date, mood, symptoms, activities, food_intake, protein_grams, ai_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    log_date,
                    mood,
                    ";".join(symptoms) if symptoms else "None",
                    ";".join(activities) if activities else "Rest",
                    food_text.strip(),
                    nutrition.get("total_protein_g", 0),
                    json.dumps(nutrition)
                )
            )
            conn.commit()
            st.cache_data.clear()
            
            if "nutrition" in st.session_state:
                del st.session_state["nutrition"]

            st.success("✅ Your diary entry has been safely tucked away in your secret journal! 💖")
            st.rerun()

    # Tab 3: Sparkle Report
    with tab3:
        st.header("✨ Your Magical Health Insights 🔮")
        conn = get_db()
        logs_df = pd.read_sql_query(
            "SELECT * FROM daily_logs WHERE user_id=? ORDER BY log_date ASC",
            conn,
            params=(user_id,)
        )

        if logs_df.empty:
            st.info("📊 No diary logs found yet. Start logging in the 'Dear Diary' tab to see your sparkle charts!")
        else:
            # Mood Trend Line Chart
            mood_map = {
                "Very Low": 1, "Very Low 🌧️": 1,
                "Low": 2, "Low ☁️": 2,
                "Neutral": 3, "Neutral 🌤️": 3,
                "High": 4, "High 🌸": 4,
                "Very High": 5, "Very High ✨": 5
            }
            logs_df['mood_score'] = logs_df['mood'].map(mood_map)

            st.subheader("💖 Mood & Emotional Energy Flow")
            fig_mood = go.Figure()
            fig_mood.add_trace(go.Scatter(
                x=logs_df['log_date'],
                y=logs_df['mood_score'],
                mode='lines+markers',
                line=dict(color='#ff1493', width=3, shape='spline'),
                marker=dict(size=10, color='#d81b60', symbol='circle'),
                text=logs_df['mood'],
                hoverinfo='x+text',
                name='Mood'
            ))
            fig_mood.update_yaxes(
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Very Low 🌧️", "Low ☁️", "Neutral 🌤️", "High 🌸", "Very High ✨"],
                range=[0.5, 5.5]
            )
            fig_mood.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255, 240, 245, 0.4)',
                margin=dict(l=20, r=20, t=20, b=20),
                height=260
            )
            st.plotly_chart(fig_mood, use_container_width=True)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("📈 Daily Protein Intake (vs 50g Goal)")
                fig_protein = go.Figure()
                fig_protein.add_trace(go.Bar(
                    x=logs_df['log_date'],
                    y=logs_df['protein_grams'],
                    marker_color='#ff4081',
                    name='Protein (g)'
                ))
                fig_protein.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="#7b1fa2",
                    annotation_text="Target (50g)",
                    annotation_position="top left"
                )
                fig_protein.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255, 240, 245, 0.4)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=280
                )
                st.plotly_chart(fig_protein, use_container_width=True)

            with col_chart2:
                st.subheader("🤕 Top Reported Symptoms")
                symptoms_all = logs_df['symptoms'].dropna().str.split(';').explode()
                symptoms_all = symptoms_all[~symptoms_all.isin(["", "None"])]
                
                if not symptoms_all.empty:
                    symp_counts = symptoms_all.value_counts().reset_index()
                    symp_counts.columns = ['Symptom', 'Count']
                    fig_symp = px.bar(
                        symp_counts,
                        x='Count',
                        y='Symptom',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Purples'
                    )
                    fig_symp.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255, 240, 245, 0.4)',
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=280
                    )
                    st.plotly_chart(fig_symp, use_container_width=True)
                else:
                    st.write("No major symptoms reported! Feeling fabulous! 🌸")

            with st.expander("📜 View Past Diary Logs"):
                display_cols = ['log_date', 'mood', 'symptoms', 'activities', 'food_intake', 'protein_grams']
                available_cols = [c for c in display_cols if c in logs_df.columns]
                sorted_logs = logs_df[available_cols].sort_values(by='log_date', ascending=False)
                st.dataframe(sorted_logs, use_container_width=True)
                
                csv_data = sorted_logs.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Health Diary (CSV)",
                    data=csv_data,
                    file_name=f"flow_glow_diary_{date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("---")
        st.subheader("🔮 Deep Pattern & Hormonal Analysis")
        if st.button("🔍 Generate AI Sparkle Analysis", type="primary"):
            with st.spinner("🤖 Bestie is analyzing your 30-day patterns..."):
                insights = ai_insights(user_id)
                st.session_state.latest_insights = insights

        if "latest_insights" in st.session_state:
            st.markdown(f"""
            <div class="glow-card" style="border-left: 5px solid #d81b60;">
                {st.session_state.latest_insights}
            </div>
            """, unsafe_allow_html=True)

    # Tab 4: Safe Space
    with tab4:
        st.header("🚨 Girl Code: Safe Space 🛡️")
        st.write("Fast access to national emergency assistance and your private trusted contacts.")

        st.subheader("📞 Instant National Helplines (India)")
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        with hcol1:
            st.link_button("📞 Women Helpline 1091", "tel:1091", use_container_width=True)
        with hcol2:
            st.link_button("🚨 Police 100", "tel:100", use_container_width=True)
        with hcol3:
            st.link_button("🚑 Ambulance 108", "tel:108", use_container_width=True)
        with hcol4:
            st.link_button("🧠 Mental Health 14416", "tel:14416", use_container_width=True)

        st.markdown("---")
        st.subheader("💖 My Trusted Contacts")
        
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id, name, phone FROM emergency_contacts WHERE user_id=?", (user_id,))
        contacts = c.fetchall()

        if contacts:
            for cid, cname, cphone in contacts:
                ccol1, ccol2, ccol3 = st.columns([3, 2, 1])
                ccol1.markdown(f"**{cname}**\n`{cphone}`")
                with ccol2:
                    clean_phone = re.sub(r'[^0-9+]', '', cphone)
                    st.link_button(f"📞 Call", f"tel:{clean_phone}", use_container_width=True)
                with ccol3:
                    if st.button("🗑️", key=f"del_contact_{cid}", help="Remove contact"):
                        c.execute("DELETE FROM emergency_contacts WHERE id=? AND user_id=?", (cid, user_id))
                        conn.commit()
                        st.rerun()
        else:
            st.caption("No personal emergency contacts added yet. Add your mom, partner, bestie, or doctor below.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button(
            "📲 Send Instant SOS Message via WhatsApp",
            "https://api.whatsapp.com/send?text=🚨%20EMERGENCY%20ALERT!%20I%20am%20in%20an%20urgent%20situation%20and%20need%20help.%20Please%20check%20on%20me%20immediately!",
            type="primary",
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("➕ Add Trusted Emergency Contact"):
            with st.form("add_contact_form", clear_on_submit=True):
                contact_name = st.text_input("Contact Name (e.g. Mom, Bestie, Dr. Anita)")
                contact_phone = st.text_input("Phone Number")
                submitted = st.form_submit_button("Save Contact")
                if submitted:
                    if contact_name.strip() and contact_phone.strip():
                        c.execute(
                            "INSERT INTO emergency_contacts (user_id, name, phone) VALUES (?, ?, ?)",
                            (user_id, contact_name.strip(), contact_phone.strip())
                        )
                        conn.commit()
                        st.success(f"Added {contact_name} to your trusted circle!")
                        st.rerun()
                    else:
                        st.error("Please provide both name and phone number.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>✨ <i>Powered by Google Gemini AI & Girl Math 💖</i></div>", unsafe_allow_html=True)

    # Floating Chatbot
    with st.popover("💬"):
        st.markdown("### 🌸 Flow Glow AI Bestie")
        st.caption(f"Currently attuned to your **{phase_name} Phase** ✨")

        chat_container = st.container(height=350)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        user_prompt = st.chat_input("Spill the tea or ask advice...", key="floating_chat_input")
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_prompt)

            with st.spinner("✨ Bestie is thinking..."):
                reply = ai_chat_response(st.session_state.chat_history, phase_name)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

        if st.button("🧹 Clear Chat", key="clear_chat_btn"):
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Chat refreshed! How can I support your glow today, bestie? 🌸"}
            ]
            st.rerun()

if st.runtime.exists():
    render_app()
