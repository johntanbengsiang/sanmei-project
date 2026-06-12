import os
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sxtwl

# --- 1. SETUP & CONFIGURATION ---
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD DATA IN MEMORY ---
# (Ensure your CSVs are exported and placed in a 'data' folder)
try:
    df_10_stars = pd.read_csv("data/10_Main_Stars.csv")
    df_database = pd.read_csv("data/Database.csv")
    # Clean up whitespace in column names if necessary
    df_10_stars.columns = df_10_stars.columns.str.strip()
    df_database.columns = df_database.columns.str.strip()
except Exception as e:
    print(f"Warning: Could not load CSV files. Ensure they are in the 'data' folder. Error: {e}")

# --- 3. SANMEIGAKU MATH ENGINE ---
STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

# Takao Gakukan Hidden Stems (蔵干 - 本元 only)
HIDDEN_STEMS_TAKAO = {
    "子": "癸", "丑": "己", "寅": "甲", "卯": "乙", "辰": "乙", "巳": "丙",
    "午": "丁", "未": "己", "申": "庚", "酉": "辛", "戌": "辛", "亥": "壬"
}

# 10 Main Stars Calculation Matrix
# Formula: Rel = (Target_Element - Day_Element) % 5
def get_10_star(day_stem, target_stem):
    elements = {"甲":0, "乙":0, "丙":1, "丁":1, "戊":2, "己":2, "庚":3, "辛":3, "壬":4, "癸":4}
    polarities = {"甲":0, "乙":1, "丙":0, "丁":1, "戊":0, "己":1, "庚":0, "辛":1, "壬":0, "癸":1}
    
    d_el, d_pol = elements[day_stem], polarities[day_stem]
    t_el, t_pol = elements[target_stem], polarities[target_stem]
    
    rel = (t_el - d_el) % 5
    same_pol = (d_pol == t_pol)
    
    matrix = {
        0: {True: "貫索星", False: "石門星"}, # Same element
        1: {True: "鳳閣星", False: "調舒星"}, # Day generates Target
        2: {True: "禄存星", False: "司禄星"}, # Day controls Target
        3: {True: "車騎星", False: "牽牛星"}, # Target controls Day
        4: {True: "龍高星", False: "玉堂星"}  # Target generates Day
    }
    return matrix[rel][same_pol]

# Tenchusatsu (Void Period) Calculation
def get_tenchusatsu(day_stem, day_branch):
    stem_idx = STEMS.index(day_stem)
    branch_idx = BRANCHES.index(day_branch)
    # The offset math to find the void branches
    void_idx = (branch_idx - stem_idx - 2) % 12
    
    t_map = {
        10: "戌亥天中殺", 8: "申酉天中殺", 6: "午未天中殺",
        4: "辰巳天中殺", 2: "寅卯天中殺", 0: "子丑天中殺"
    }
    return t_map[void_idx]

class BirthDate(BaseModel):
    day: int
    month: int
    year: int

# --- 4. API ROUTE ---
@app.post("/api/analyze")
async def analyze_sanmeigaku(dob: BirthDate):
    try:
        # A. CALCULATE PILLARS & STARS (Solar Term accurate)
        lunar_day = sxtwl.fromSolar(dob.year, dob.month, dob.day)
        
        y_gz = lunar_day.getYearGZ(True) 
        m_gz = lunar_day.getMonthGZ()
        d_gz = lunar_day.getDayGZ()
        
        day_stem = STEMS[d_gz.tg]
        year_stem = STEMS[y_gz.tg]
        month_branch = BRANCHES[m_gz.dz]
        day_branch = BRANCHES[d_gz.dz]
        
        month_hidden = HIDDEN_STEMS_TAKAO[month_branch]
        
        # Calculate specific chart points
        head_star = get_10_star(day_stem, year_stem)
        chest_star = get_10_star(day_stem, month_hidden)
        tenchusatsu = get_tenchusatsu(day_stem, day_branch)

        calculated_chart = {
            "head": head_star,
            "chest": chest_star,
            "tenchusatsu": tenchusatsu,
            "day_pillar": f"{day_stem}{day_branch}"
        }

        # B. LOCAL DATA LOOKUP (Filtering your Excel data)
        try:
            head_data = df_10_stars[df_10_stars['星'].str.contains(head_star, na=False)]['Core Traits (Corrected)'].values[0]
            chest_data = df_10_stars[df_10_stars['星'].str.contains(chest_star, na=False)]['Core Traits (Corrected)'].values[0]
            
            # Simple Proximity Example (Grabbing a verified profile to compare)
            # In a real scenario, you'd match by similar stars in the DB
            closest_match_name = "Barack Obama"
            closest_match_traits = df_database[df_database['Name'] == closest_match_name]['Life Patterns or Behavioral Themes'].values[0]
        except Exception as e:
            # Fallbacks if CSV data is missing/mismatched
            head_data, chest_data, closest_match_traits = "Inherited traits.", "Core Ego traits.", "A dynamic life."
            closest_match_name = "a similar profile"

        # C. CONSTRUCT THE LEAN AI PROMPT
        prompt = f"""
        You are an expert Sanmeigaku practitioner. Provide a cohesive, highly detailed psychological and destiny profile for a user with the following energetic blueprint.

        [USER CHART]
        - Head Star (SuperEgo/Inherited): {head_star}
        - Chest Star (Core Ego): {chest_star}
        - Tenchusatsu: {tenchusatsu}

        [SYSTEM RULES - USE THESE DEFINITIONS STRICTLY]
        - {head_star} Traits: {head_data}
        - {chest_star} Traits: {chest_data}

        [CELEBRITY PROXIMITY]
        This user shares a highly similar energetic structure to {closest_match_name}. 
        {closest_match_name}'s themes: "{closest_match_traits}".
        Briefly draw a parallel to how the user might experience similar life patterns.

        [TASK]
        Do not list the definitions. Weave them into a fluid, empathetic, and deep narrative. Focus heavily on the tension between their inherited traits (Head) and their core ego (Chest), and explain how their Tenchusatsu affects their timing. Format the response beautifully using Markdown headings and bullet points.
        """

        # D. CALL GEMINI
        response = model.generate_content(prompt)

        return {
            "chart": calculated_chart,
            "proximity_match": closest_match_name,
            "ai_interpretation": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
