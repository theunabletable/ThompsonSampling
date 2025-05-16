"""
Implements a single weekly update using a simple ThompsonSampler.
"""


import os
import pandas as pd
from datetime import date
from AgentManager import AgentManager
from utils import prepare_features, prepare_week_slice
from config import *

#    ----CONFIGURATION----
run_date    = pd.Timestamp(date(2024, 7, 8), tz="UTC")  # your “week end” date
window      = 7                                        # days back from run_date
SET_CURRENT = True                                      # flip to False to skip overwriting current

#this will prepare a slice of the main dataframe from [run_date - window, run_date)

#    ----prepare output folder----
date_str    = run_date.strftime("%Y-%m-%d")             # e.g. "2024-07-08"
weekly_dir  = MODELS_WEEKLY_DIR / date_str
weekly_dir.mkdir(parents=True, exist_ok=True)


#    ----load existing AgentManagers----
stepsMgr = AgentManager.load(STEPS_MANAGER_PATH)
sleepMgr = AgentManager.load(SLEEP_MANAGER_PATH)
moodMgr  = AgentManager.load(MOOD_MANAGER_PATH)

#    ----full events table----
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

# 4) For each domain, slice last week via helper, update, and save
for kind, mgr, main_cols, inter_cols, feat_cols, reward, save_path in [
    ("steps", stepsMgr, STEP_MAIN_COLS, STEP_INTERACTION_COLS, STEP_FEATURE_COLS, STEPS_REWARD,   STEPS_MANAGER_PATH),
    ("sleep", sleepMgr, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, SLEEP_FEATURE_COLS, SLEEP_REWARD,   SLEEP_MANAGER_PATH),
    ("mood", moodMgr,  MOOD_MAIN_COLS,  MOOD_INTERACTION_COLS,  MOOD_FEATURE_COLS,  MOOD_REWARD,    MOOD_MANAGER_PATH),
]:
    # a) slice & feature‐engineer the past week in one shot
    df_week = prepare_week_slice(
        end_day         = run_date,
        window_days     = window,
        main_cols       = main_cols,
        interaction_cols= inter_cols,
        action_col      = ACTION_COL,
        reward_col      = reward
    )

    #build training DataFrame with exactly the columns AgentManager expects
    keep = ["PARTICIPANTIDENTIFIER","time", ACTION_COL, reward] + feat_cols
    # make sure keep has no duplicates
    keep = list(dict.fromkeys(keep))
    df_train = df_week[keep].reset_index(drop=True)
    mgr.update_posteriors(df_train)

    #always save into weekly archive folder
    weekly_path = weekly_dir / f"{kind}Manager.pkl"
    mgr.save(weekly_path)
    print(f"[ARCHIVED] Saved weekly {kind} manager to {weekly_path}")

    #optionally overwrite the “current” models
    if SET_CURRENT:
        current_path = MODELS_CURRENT_DIR / f"{kind}Manager.pkl"
        mgr.save(current_path)
        print(f"[CURRENT]  Overwrote current {kind} manager at {current_path}")