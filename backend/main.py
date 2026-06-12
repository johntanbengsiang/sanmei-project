import os
import pandas as pd
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sxtwl
import httpx

from dotenv import load_dotenv
load_dotenv()

# --- RELIABLE PATH RESOLUTION ---
# Resolves data/ relative to this file's location, not the working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- GEMINI CLIENT SETUP ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)

app = FastAPI()

# --- CORS CONFIGURATION ---
# Add your exact Vercel frontend URL(s) below. Include both with and without
# trailing slash variants if needed. Also supports local dev on port 5500/3000.
ALLOWED_ORIGINS = [
    "https://sanmei-project.vercel.app",   # <-- update to your actual Vercel domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],  # OPTIONS required for CORS preflight
    allow_headers=["*"],
)

# --- LOAD REFERENCE CSV DATA ---
df_10_stars = None
df_12_stars = None
df_tenchu_ref = None
df_database = None

try:
    df_10_stars = pd.read_csv(os.path.join(DATA_DIR, "10_Main_Stars.csv"))
    df_12_stars = pd.read_csv(os.path.join(DATA_DIR, "12_Cycle_Stars.csv"))
    df_tenchu_ref = pd.read_csv(os.path.join(DATA_DIR, "Tenchusatsu.csv"))
    df_database = pd.read_csv(os.path.join(DATA_DIR, "Database.csv"))

    for df in [df_10_stars, df_12_stars, df_tenchu_ref, df_database]:
        if df is not None:
            df.columns = df.columns.str.strip()

    print(f"[OK] All CSVs loaded successfully from: {DATA_DIR}")

except Exception as e:
    print(f"[ERROR] CSV loading failed. DATA_DIR resolved to: {DATA_DIR}. Error: {e}")

# --- CONSTANTS ---
STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

HIDDEN_STEMS_TAKAO = {
    "子": "癸", "丑": "己", "寅": "甲", "卯": "乙", "辰": "乙", "巳": "丙",
    "午": "丁", "未": "己", "申": "庚", "酉": "辛", "戌": "辛", "亥": "壬"
}

# --- HEALTH CHECK ENDPOINT ---
@app.get("/")
def read_root():
    csv_status = {
        "10_Main_Stars": df_10_stars is not None,
        "12_Cycle_Stars": df_12_stars is not None,
        "Tenchusatsu": df_tenchu_ref is not None,
        "Database": df_database is not None,
    }
    return {"message": "Sanmeigaku Engine is online", "data_loaded": csv_status}

# --- 10 MAIN STARS ENGINE ---
def calculate_10_star(day_stem: str, target_stem: str) -> str:
    elements   = {"甲":0, "乙":0, "丙":1, "丁":1, "戊":2, "己":2, "庚":3, "辛":3, "壬":4, "癸":4}
    polarities = {"甲":0, "乙":1, "丙":0, "丁":1, "戊":0, "己":1, "庚":0, "辛":1, "壬":0, "癸":1}

    d_el, d_pol = elements[day_stem], polarities[day_stem]
    t_el, t_pol = elements[target_stem], polarities[target_stem]

    rel = (t_el - d_el) % 5
    same_polarity = (d_pol == t_pol)

    matrix = {
        0: {True: "貫索星", False: "石門星"},
        1: {True: "鳳閣星", False: "調舒星"},
        2: {True: "禄存星", False: "司禄星"},
        3: {True: "車騎星", False: "牽牛星"},
        4: {True: "龍高星", False: "玉堂星"}
    }
    return matrix[rel][same_polarity]

# --- 12 CYCLE STARS ENGINE ---
def calculate_12_cycle_star(day_stem: str, target_branch: str) -> str:
    stars_sequence = ["天貴星", "天恍星", "天南星", "天祿星", "天将星", "天堂星",
                      "天胡星", "天極星", "天庫星", "天馳星", "天報星", "天印星"]

    start_positions = {"甲":11, "丙":2, "戊":2, "庚":5, "壬":8,
                       "乙":6,  "丁":9, "己":9, "辛":0, "癸":3}
    is_yang = day_stem in ["甲", "丙", "戊", "庚", "壬"]

    b_idx = BRANCHES.index(target_branch)
    base  = start_positions[day_stem]

    offset = (b_idx - base) % 12 if is_yang else (base - b_idx) % 12
    return stars_sequence[offset]

# --- TENCHUSATSU ENGINE ---
def calculate_tenchusatsu(day_stem: str, day_branch: str) -> str:
    s_idx = STEMS.index(day_stem)
    b_idx = BRANCHES.index(day_branch)
    void_offset = (b_idx - s_idx - 2) % 12
    mapping = {
        10: "戌亥天中殺", 8: "申酉天中殺", 6: "午未天中殺",
        4:  "辰巳天中殺", 2: "寅卯天中殺", 0: "子丑天中殺"
    }
    return mapping[void_offset]

# --- HISTORICAL PROFILE MATCHER ---
def find_closest_profiles(user_head: str, user_chest: str, user_tenchu: str):
    matches = []
    if df_database is None or df_database.empty:
        return matches

    for _, row in df_database.iterrows():
        score = 0
        db_head   = str(row.get("頭 (Head)", "")).strip()
        db_chest  = str(row.get("胸 (Chest)", "")).strip()
        db_tenchu = str(row.get("Tenchusatsu", "")).strip()

        if db_chest  and db_chest  in user_chest:  score += 3
        if db_head   and db_head   in user_head:   score += 2
        if db_tenchu and db_tenchu in user_tenchu: score += 1

        if score > 0:
            matches.append({
                "name":      row.get("Name", "Unknown"),
                "domain":    row.get("Career Domain", "Unknown"),
                "themes":    row.get("Life Patterns or Behavioral Themes", "No context available"),
                "proximity": int((score / 6) * 100)
            })

    return sorted(matches, key=lambda x: x["proximity"], reverse=True)[:3]

# --- CSV LOOKUP HELPER ---
def get_meta_text(df, col: str, val: str, target_col: str) -> str:
    try:
        if df is None or df.empty or col not in df.columns or target_col not in df.columns:
            return "System baseline blueprint details."
        clean_val = val.replace("星", "").replace("天中殺", "").strip()
        res = df[df[col].astype(str).str.contains(clean_val, na=False, case=False)]
        return res[target_col].values[0] if len(res) > 0 else "System baseline blueprint details."
    except Exception:
        return "System baseline blueprint details."

# --- REQUEST MODEL ---
class BirthDate(BaseModel):
    day:     int
    month:   int
    year:    int
    cf_token: str

# --- CLOUDFLARE TURNSTILE VERIFICATION ---
async def verify_turnstile(token: str) -> bool:
    secret_key = os.getenv("TURNSTILE_SECRET_KEY")
    if not secret_key:
        # If key is not configured, log a warning but allow through in dev
        # Change to `return False` in production once key is confirmed set
        print("WARNING: TURNSTILE_SECRET_KEY not set. Bypassing verification.")
        return True

    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data={"secret": secret_key, "response": token})
        result = response.json()
        print(f"[Turnstile] Verification result: {result}")
        return result.get("success", False)

# --- MAIN ANALYSIS ENDPOINT ---
@app.post("/api/analyze")
async def analyze(dob: BirthDate):

    # 1. Bot protection
    is_human = await verify_turnstile(dob.cf_token)
    if not is_human:
        raise HTTPException(status_code=403, detail="Security check failed. Please refresh and try again.")

    try:
        # 2. Solar → Lunar conversion
        lunar_day   = sxtwl.fromSolar(dob.year, dob.month, dob.day)
        y_gz = lunar_day.getYearGZ(True)
        m_gz = lunar_day.getMonthGZ()
        d_gz = lunar_day.getDayGZ()

        day_stem,   day_branch   = STEMS[d_gz.tg], BRANCHES[d_gz.dz]
        year_stem,  year_branch  = STEMS[y_gz.tg], BRANCHES[y_gz.dz]
        month_stem, month_branch = STEMS[m_gz.tg], BRANCHES[m_gz.dz]

        # 3. Hidden stems (Zoukan / Takao School)
        year_hidden  = HIDDEN_STEMS_TAKAO[year_branch]
        month_hidden = HIDDEN_STEMS_TAKAO[month_branch]
        day_hidden   = HIDDEN_STEMS_TAKAO[day_branch]

        # 4. Main Star Matrix (十大主星)
        head_star    = calculate_10_star(day_stem, year_stem)      # North
        chest_star   = calculate_10_star(day_stem, month_hidden)   # Center
        stomach_star = calculate_10_star(day_stem, month_stem)     # South
        right_hand   = calculate_10_star(day_stem, day_hidden)     # West
        left_hand    = calculate_10_star(day_stem, year_hidden)    # East

        # 5. 12 Cycle Star Matrix (十二大従星)
        right_shoulder = calculate_12_cycle_star(day_stem, day_branch)    # NW
        right_leg      = calculate_12_cycle_star(day_stem, month_branch)  # SW
        left_leg       = calculate_12_cycle_star(day_stem, year_branch)   # SE

        # 6. Tenchusatsu (Global Void)
        tenchusatsu = calculate_tenchusatsu(day_stem, day_branch)

        # 7. Reference text lookups
        head_txt   = get_meta_text(df_10_stars,  "星",  head_star,   "Core Traits (Corrected)")
        chest_txt  = get_meta_text(df_10_stars,  "星",  chest_star,  "Core Traits (Corrected)")
        tenchu_txt = get_meta_text(df_tenchu_ref, "Type", tenchusatsu, "Lifelong Traits")

        # 8. Historical proximity match
        proximity_matches = find_closest_profiles(head_star, chest_star, tenchusatsu)
        top_match_text = (
            f"Aligned life markers with {proximity_matches[0]['name']}: {proximity_matches[0]['themes']}"
            if proximity_matches else "Standalone trajectory."
        )

        # 9. AI synthesis prompt
        prompt = f"""
        You are an elite Sanmeigaku grandmaster practitioner. Synthesize a comprehensive structural destiny reading based on this 3x3 Yin Chart Grid.

        [YIN GRID CONFIGURATION]
        - North (Head): {head_star} | South (Belly): {stomach_star}
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

        response = model.generate_content(prompt)

        return {
            "chart": {
                "head":             head_star,
                "head_meaning":     head_txt,
                "chest":            chest_star,
                "chest_meaning":    chest_txt,
                "stomach":          stomach_star,
                "left_hand":        left_hand,
                "right_hand":       right_hand,
                "right_shoulder":   right_shoulder,
                "right_leg":        right_leg,
                "left_leg":         left_leg,
                "tenchusatsu":      tenchusatsu,
                "tenchusatsu_meaning": tenchu_txt
            },
            "proximity_chart": proximity_matches,
            "ai_synthesis":    response.text
        }

    except Exception as e:
        # Surface the real error message to help debugging
        raise HTTPException(status_code=500, detail=str(e))
