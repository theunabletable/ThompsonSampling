# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:35:30 2025

@author: Drew
"""

#!/usr/bin/env python3
# weekly_update.py

import os
import pandas as pd
from datetime import date
from AgentManager import AgentManager
from utils import prepare_features, prepare_week_slice
from config import (
    DF_EVENTS_PATH,
    ACTION_COL,
    STEP_MAIN_COLS, STEP_INTERACTION_COLS, STEP_FEATURE_COLS, STEPS_REWARD, STEPS_MANAGER_PATH,
    SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, SLEEP_FEATURE_COLS, SLEEP_REWARD, SLEEP_MANAGER_PATH,
    MOOD_MAIN_COLS, MOOD_INTERACTION_COLS, MOOD_FEATURE_COLS, MOOD_REWARD, MOOD_MANAGER_PATH,
)

run_date = pd.Timestamp(date(2024, 7, 8), tz="UTC")
# 1) Anchor date and window
#run_date  = pd.Timestamp(date.today(), tz="UTC")
window    = 7  # days

# 2) Load existing AgentManagers
stepsMgr = AgentManager.load(STEPS_MANAGER_PATH)
sleepMgr = AgentManager.load(SLEEP_MANAGER_PATH)
moodMgr  = AgentManager.load(MOOD_MANAGER_PATH)

# 3) Read full events table (with actions & rewards already populated)
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

# 4) For each domain, slice last week via helper, update, and save
for mgr, main_cols, inter_cols, feat_cols, reward, save_path in [
    (stepsMgr, STEP_MAIN_COLS, STEP_INTERACTION_COLS, STEP_FEATURE_COLS, STEPS_REWARD,   STEPS_MANAGER_PATH),
    (sleepMgr, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, SLEEP_FEATURE_COLS, SLEEP_REWARD,   SLEEP_MANAGER_PATH),
    (moodMgr,  MOOD_MAIN_COLS,  MOOD_INTERACTION_COLS,  MOOD_FEATURE_COLS,  MOOD_REWARD,    MOOD_MANAGER_PATH),
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

    # d) re‐save the updated manager
    mgr.save(save_path)
    print(f"Updated and saved manager to {save_path}")
