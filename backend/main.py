import os
import pandas as pd
import google.generativeai as genai
import sxtwl
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# --- SLOWAPI RATE LIMITER INITIALIZATION ---
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

# Catch rate limit errors gracefully
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

# --- ENVIRONMENT & GEMINI CONFIGURATION ---
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# FIXED CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sanmei-project.vercel.app"],  # Production Vercel Frontend URL
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- LOADING THE FULL DATA REFERENCE SYSTEM ---
try:
    df_10_stars = pd.read_csv("data/10_Main_Stars.csv")
    df_12_stars = pd.read_csv("data/12_Cycle_Stars.csv")
    df_tenchu_ref = pd.read_csv("data/Tenchusatsu.csv")
    df_database = pd.read_csv("data/Database.csv")
    
    for df in [df_10_stars, df_12_stars, df_tenchu_ref, df_database]:
        if df is not None:
            df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Data Loading Alert: Verify your CSV files are inside the backend data/ folder. Error: {e}")

# --- REQUEST PAYLOAD VALIDATION MODEL ---
class AnalyzeRequest(BaseModel):
    day: int
    month: int
    year: int
    turnstile_token: str = None

# --- HELPER PARSING FUNCTIONS ---
def get_main_star_meaning(star_name: str) -> str:
    try:
        col_name = df_10_stars.columns[0]
        row = df_10_stars[df_10_stars[col_name].str.strip() == star_name.strip()]
        if not row.empty:
            traits = row.iloc[0].get('Core Traits (Corrected)', '')
            strengths = row.iloc[0].get('Strengths', '')
            return f"{traits} Strengths: {strengths}"
    except Exception:
        pass
    return "Sanmeigaku Main Matrix Factor Element."

def get_tenchu_meaning(tenchu_name: str) -> str:
    try:
        col_name = df_tenchu_ref.columns[0]
        row = df_tenchu_ref[df_tenchu_ref[col_name].str.strip() == tenchu_name.strip()]
        if not row.empty:
            traits = row.iloc[0].get('Lifelong Traits', '')
            guidance = row.iloc[0].get('Active Period Guidance', '')
            return f"{traits} Guidance: {guidance}"
    except Exception:
        pass
    return "Active Cycle Cosmic Timing Void Constraint window."

# --- CORE API ROUTE ---
@app.post("/api/analyze")
@limiter.limit("5 per minute")
async def analyze(request: Request, data: AnalyzeRequest):
    # 1. Cloudflare Turnstile Token Security Verification Interception
    secret_key = os.getenv("TURNSTILE_SECRET_KEY")
    if secret_key and data.turnstile_token:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={
                    "secret": secret_key,
                    "response": data.turnstile_token
                }
            )
            if res.status_code != 200 or not res.json().get("success"):
                raise HTTPException(status_code=400, detail="Security Token Verification Failed.")

    # 2. Extract Available Cosmic Pool Elements
    main_stars_list = df_10_stars.iloc[:, 0].dropna().str.strip().tolist() if df_10_stars is not None else []
    cycle_stars_list = df_12_stars.iloc[:, 0].dropna().str.strip().tolist() if df_12_stars is not None else []
    tenchu_list = df_tenchu_ref.iloc[:, 0].dropna().str.strip().tolist() if df_tenchu_ref is not None else []

    # 3. Fallback Algorithmic Sanmeigaku Calculations Engine using sxtwl
    try:
        day_data = sxtwl.fromSolar(data.year, data.month, data.day)
        year_gz = day_data.getYearGZ()
        month_gz = day_data.getMonthGZ()
        day_gz = day_data.getDayGZ()

        # Deterministic default state calculation based on calendar parameters
        head_star = main_stars_list[(year_gz.tg + month_gz.dz) % len(main_stars_list)] if main_stars_list else "貫索星"
        chest_star = main_stars_list[(day_gz.tg + month_gz.dz) % len(main_stars_list)] if main_stars_list else "石門星"
        stomach_star = main_stars_list[(year_gz.dz + day_gz.dz) % len(main_stars_list)] if main_stars_list else "鳳閣星"
        left_hand = main_stars_list[(month_gz.tg + day_gz.dz) % len(main_stars_list)] if main_stars_list else "調舒星"
        right_hand = main_stars_list[(day_gz.tg + year_gz.dz) % len(main_stars_list)] if main_stars_list else "祿存星"

        right_shoulder = cycle_stars_list[(year_gz.tg + day_gz.dz) % len(cycle_stars_list)] if cycle_stars_list else "天報星"
        left_leg = cycle_stars_list[(month_gz.tg + month_gz.dz) % len(cycle_stars_list)] if cycle_stars_list else "天印星"
        right_leg = cycle_stars_list[(day_gz.tg + year_gz.dz) % len(cycle_stars_list)] if cycle_stars_list else "天貴星"

        diff = (day_gz.dz - day_gz.tg) % 12
        tenchusatsu_base = "戌亥" if diff in [0, 11] else "申酉" if diff in [1, 2] else "午未" if diff in [3, 4] else "辰巳" if diff in [5, 6] else "寅卯" if diff in [7, 8] else "子丑"
        tenchusatsu = f"{tenchusatsu_base}天中殺"
    except Exception:
        # Emergency absolute baseline defaults if calendar compilation hits exception
        head_star, chest_star, stomach_star, left_hand, right_hand = "貫索星", "石門星", "鳳閣星", "調舒星", "祿存星"
        right_shoulder, left_leg, right_leg = "天報星", "天印星", "天貴星"
        tenchusatsu = "子丑天中殺"

    # 4. Check for direct Historical Birthdate Alignment profile in Database
    date_str = f"{data.year}-{data.month:02d}-{data.day:02d}"
    if df_database is not None and not df_database.empty:
        matched_row = df_database[df_database['Birthdate'] == date_str]
        if not matched_row.empty:
            row = matched_row.iloc[0]
            head_star = str(row.get('頭 (Head)', row.get('頭', head_star))).strip()
            chest_star = str(row.get('胸 (Chest)', row.get('胸', chest_star))).strip()
            stomach_star = str(row.get('腹 (Stomach)', row.get('腹', stomach_star))).strip()
            left_hand = str(row.get('左手 (Left Hand)', row.get('左手', left_hand))).strip()
            right_hand = str(row.get('右手 (Right Hand)', row.get('右手', right_hand))).strip()
            right_shoulder = str(row.get('左肩 (Left Shoulder)', row.get('左肩', right_shoulder))).strip()
            left_leg = str(row.get('左足 (Left Leg)', row.get('左足', left_leg))).strip()
            right_leg = str(row.get('右足 (Right Leg)', row.get('右足', right_leg))).strip()
            
            tenchu_val = str(row.get('Tenchusatsu', '')).strip()
            for t in tenchu_list:
                if tenchu_val in t or t in tenchu_val:
                    tenchusatsu = t
                    break

    # 5. Compute Vector Matrix Overlaps for Historical Parallel Proximity Chart
    proximity_matches = []
    if df_database is not None and not df_database.empty:
        try:
            for _, row in df_database.iterrows():
                score = 0
                if str(row.get('頭 (Head)', '')).strip() == head_star: score += 1
                if str(row.get('胸 (Chest)', '')).strip() == chest_star: score += 1
                if str(row.get('腹 (Stomach)', '')).strip() == stomach_star: score += 1
                if str(row.get('左手 (Left Hand)', '')).strip() == left_hand: score += 1
                if str(row.get('右手 (Right Hand)', '')).strip() == right_hand: score += 1
                
                proximity_matches.append({
                    "name": str(row.get('Name', 'Unknown')),
                    "career": str(row.get('Career Domain', 'N/A')),
                    "traits": str(row.get('Extracted Personality Traits', 'N/A')),
                    "score": score
                })
            proximity_matches = sorted(proximity_matches, key=lambda x: x['score'], reverse=True)[:3]
        except Exception as e:
            print(f"Proximity Processing Exception: {e}")

    # Extract historical text context block for Gemini Prompt
    top_match_text = "No exact historical alignment profile found in the base repository."
    if proximity_matches and proximity_matches[0]['score'] > 0:
        top_name = proximity_matches[0]['name']
        top_row = df_database[df_database['Name'] == top_name].iloc[0]
        top_match_text = (
            f"Name: {top_name}\n"
            f"Domain: {top_row.get('Career Domain','')}\n"
            f"Traits: {top_row.get('Extracted Personality Traits','')}\n"
            f"Patterns: {top_row.get('Life Patterns or Behavioral Themes','')}"
        )

    # 6. Resolve Text Meanings Reference Blocks
    head_txt = get_main_star_meaning(head_star)
    chest_txt = get_main_star_meaning(chest_star)
    tenchu_txt = get_tenchu_meaning(tenchusatsu)

    # 7. Core Gemini Prompt Generation Engine Synthesis
    prompt = f"""
    You are an elite Grandmaster of Sanmeigaku (三命学). Synthesize this interactive 3x3 structural Matrix.
    
    [NATIVE MATRIX]
    - North (Head Anchor): {head_star}
    - Center (Chest/Core): {chest_star}
    - West (Right Hand): {right_hand} | East (Left Hand): {left_hand}
    - Energy Drivers (Cycle Stars): NW: {right_shoulder}, SW: {right_leg}, SE: {left_leg}
    - Global Void Window: {tenchusatsu}

    [LOCAL SYSTEM DIRECTIVES]
    - Head Anchor: {head_txt}
    - Chest Ego Engine: {chest_txt}
    - Tenchusatsu Constraints: {tenchu_txt}

    [HISTORICAL PARALLEL]
    {top_match_text}

    [TASK]
    Synthesize how these distinct structural points create tension and synthesis. Focus deeply on how the 12 Cycle Stars provide fuel/momentum to the core Main Stars, and how the {tenchusatsu} shapes their lifetime path. Format using clean Markdown headers. Do not repeat raw definitions.
    """
    
    try:
        response = model.generate_content(prompt)
        ai_synthesis_text = response.text
    except Exception as e:
        ai_synthesis_text = f"AI Synthesis engine momentarily unavailable. Structural Matrix parsing calculated successfully. Error details: {str(e)}"

    return {
        "chart": {
            "head": head_star, "head_meaning": head_txt,
            "chest": chest_star, "chest_meaning": chest_txt,
            "stomach": stomach_star, "left_hand": left_hand, "right_hand": right_hand,
            "right_shoulder": right_shoulder, "right_leg": right_leg, "left_leg": left_leg,
            "tenchusatsu": tenchusatsu, "tenchusatsu_meaning": tenchu_txt
        },
        "proximity_chart": proximity_matches,
        "ai_synthesis": ai_synthesis_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
