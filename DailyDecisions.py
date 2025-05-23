"""
This script implements the daily decision making loop. It takes in a given day, loads agents of each type, and makes decisions for each agent's context.
It outputs three CSVs that track the (pid, time, action), a CSV for each agent type. For example: Mood2024-07-01Decisions.csv
"""

from datetime import date
import pandas as pd
import numpy as np
from priors import build_step_priors
from AgentManager import AgentManager
from DataManager import DataManager
from pathlib import Path
import os
from config import *
from utils import *

run_date = pd.Timestamp(date(2024, 7, 9), tz="UTC")

base_dir  = Path(r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\Data\decisions")
date_str  = run_date.strftime("%Y-%m-%d")               # "2024-07-09"
daily_dir = base_dir / date_str                         # ".../decisions/2024-07-09"
daily_dir.mkdir(parents=True, exist_ok=True)            # make it (and parents) if needed

#prepare each day's slice
steps_slice = prepare_day_slice(run_date, STEP_MAIN_COLS, STEP_INTERACTION_COLS, action_col = ACTION_COL)
sleep_slice = prepare_day_slice(run_date, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, action_col = ACTION_COL)
mood_slice  = prepare_day_slice(run_date, MOOD_MAIN_COLS, MOOD_INTERACTION_COLS, action_col = ACTION_COL)


#load pretrained managers
stepsMgr = AgentManager.load(STEPS_MANAGER_PATH)
sleepMgr = AgentManager.load(SLEEP_MANAGER_PATH)
moodMgr  = AgentManager.load(MOOD_MANAGER_PATH)

#for each setting:
for setting, mgr, df_slice in [
    ("steps", stepsMgr, steps_slice),
    ("sleep", sleepMgr, sleep_slice),
    ("mood",  moodMgr,  mood_slice),
]:
    mgr.setEventsDf(df_slice)
    df_decisions = mgr.make_decisions(df_slice)

    fname = f"{setting}_decisions.csv"  
    df_decisions.to_csv(daily_dir / fname, index=False)
    
