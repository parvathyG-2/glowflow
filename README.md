Flow Glow
Your Private Hormonal Wellness Diary and AI Bestie

Flow Glow is an empathetic, intelligent women's health sanctuary built with Streamlit and powered by Google Gemini AI. It tracks menstrual phases, generates bespoke phase-attuned Indian nutrition plans, reveals symptom correlations via interactive Plotly analytics, provides emergency support, and features a floating 24/7 AI health companion.

Key Features

Cycle Diary and Phase Tracking: Automatically calculates current cycle day and predicts menstrual phases (Menstrual, Follicular, Ovulatory, Luteal).

Bespoke Glow Up Plan: Generates personalized 5-meal daily nutrition plans infused with Indian superfoods, seed cycling recommendations, hydration targets, and phase-friendly movement.

Dear Diary: Log daily mood, symptoms, physical activities, and meals with instant AI macronutrient (protein and calorie) breakdown.

Sparkle Analytics: Interactive Plotly visualizations including Mood and Emotional Energy Flow trend analysis over time, Daily Protein Intake vs. 50g Goal, Top Reported Symptoms frequency, and deep pattern analysis and cycle predictions.

Girl Code (Safe Space): Quick-dial access to Indian national emergency helplines (1091, 100, 108, 14416) alongside a custom trusted contacts directory.

24/7 AI Bestie: Floating conversational companion equipped with chat memory for culturally attuned, supportive wellness advice.

Tech Stack

Frontend / Framework: Streamlit
Language: Python 3.10+
AI / LLM: Google Gemini API (gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro)
Database: SQLite (configured with WAL mode for fast concurrent operations)
Data Visualization: Plotly Express and Plotly Graph Objects

Getting Started

Prerequisites
Python 3.10 or higher installed
A Google Gemini API key from Google AI Studio

Installation

Clone the repository
git clone https://github.com/parvathyg260/glowflow.git
cd glowflow

Create and activate a virtual environment (recommended)
For Windows:
python -m venv venv
venv\Scripts\activate

For macOS / Linux:
python3 -m venv venv
source venv/bin/activate

Install project dependencies
pip install -r requirements.txt

Set up environment variables
Create a .env file in the root directory:
GEMINI_API_KEY=your_actual_gemini_api_key_here
(Alternatively, provide the API key securely via the application sidebar interface).

Running the Application

Launch the Streamlit dashboard locally:
python -m streamlit run main.py

Open your browser and navigate to http://localhost:8501

Project Architecture

glowflow/

main.py: Main Streamlit application and page routing

requirements.txt: Python packages and runtime dependencies

.env.example: Example environment configuration

database/: SQLite schema initialization and CRUD queries

components/: UI modules, analytics widgets, and chat sidebar

README.md: Project documentation

Privacy and Disclaimer

Flow Glow is an assistive self-tracking and lifestyle wellness tool. It is not a replacement for professional medical advice, clinical diagnosis, or gynecological treatment. All personal log data and diary entries remain stored locally in your private database.
