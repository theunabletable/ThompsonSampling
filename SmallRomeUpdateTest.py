# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:03:58 2025

@author: Drew
"""

# weekly_update_rome.py
import pandas as pd
from scipy import sparse
import numpy as np
from datetime import timedelta
from utils import *
from config import *
from RoMEAgentManager import RoMEAgentManager   

# ------------------------------------------------------------------
# 1) Load master events  & feature‑engineer once
# ------------------------------------------------------------------
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

df = prepare_features(df, STEP_MAIN_COLS,  STEP_INTERACTION_COLS,  ACTION_COL)
df = prepare_features(df, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, ACTION_COL)
df = prepare_features(df, MOOD_MAIN_COLS,  MOOD_INTERACTION_COLS,  ACTION_COL)

participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()

# ------------------------------------------------------------------
# 2) Build user Laplacians  (k‑NN graph on survey + outcomes)
# ------------------------------------------------------------------
L_mood, L_steps, L_sleep = compute_user_laplacians(df, k=20)

# ------------------------------------------------------------------
# 3) Common prior pieces
# ------------------------------------------------------------------
p_steps  = len(STEP_FEATURE_COLS)
p_sleep  = len(SLEEP_FEATURE_COLS)
p_mood   = len(MOOD_FEATURE_COLS)

user_cov_steps  = np.eye(p_steps)  * 25.0
user_cov_sleep  = np.eye(p_sleep) * 25.0
user_cov_mood   = np.eye(p_mood)  * 25.0

# ------------------------------------------------------------------
# 4) Instantiate managers
# ------------------------------------------------------------------
stepsMgr = RoMEAgentManager(
    participants           = participants,
    L_user                 = L_steps,
    user_cov               = user_cov_steps,
    gamma_ridge            = 0.5,
    lambda_penalty         = 1.0,
    baseFeatureCols        = STEP_MAIN_COLS,
    interactionFeatureCols = STEP_INTERACTION_COLS,
    featureCols            = STEP_FEATURE_COLS,
    rewardName             = STEPS_REWARD,
    actionName             = ACTION_COL,
)

sleepMgr = RoMEAgentManager(
    participants           = participants,
    L_user                 = L_sleep,
    user_cov               = user_cov_sleep,
    gamma_ridge            = 0.5,
    lambda_penalty         = 1.0,
    baseFeatureCols        = SLEEP_MAIN_COLS,
    interactionFeatureCols = SLEEP_INTERACTION_COLS,
    featureCols            = SLEEP_FEATURE_COLS,
    rewardName             = SLEEP_REWARD,
    actionName             = ACTION_COL,
)

moodMgr = RoMEAgentManager(
    participants           = participants,
    L_user                 = L_mood,
    user_cov               = user_cov_mood,
    gamma_ridge            = 0.5,
    lambda_penalty         = 1.0,
    baseFeatureCols        = MOOD_MAIN_COLS,
    interactionFeatureCols = MOOD_INTERACTION_COLS,
    featureCols            = MOOD_FEATURE_COLS,
    rewardName             = MOOD_REWARD,
    actionName             = ACTION_COL,
)

# ------------------------------------------------------------------
# 5) Slice the last 7 days
# ------------------------------------------------------------------
end_day = df["time"].max().normalize() + pd.Timedelta(days=1)
df_week_steps = prepare_week_slice(
        end_day=end_day,
        window_days=7,
        main_cols=STEP_MAIN_COLS,
        interaction_cols=STEP_INTERACTION_COLS,
        action_col=ACTION_COL,
        reward_col=STEPS_REWARD,
)

df_week_sleep = prepare_week_slice(
        end_day=end_day,
        window_days=7,
        main_cols=SLEEP_MAIN_COLS,
        interaction_cols=SLEEP_INTERACTION_COLS,
        action_col=ACTION_COL,
        reward_col=SLEEP_REWARD,
)

df_week_mood = prepare_week_slice(
        end_day=end_day,
        window_days=7,
        main_cols=MOOD_MAIN_COLS,
        interaction_cols=MOOD_INTERACTION_COLS,
        action_col=ACTION_COL,
        reward_col=MOOD_REWARD,
)

# ------------------------------------------------------------------
# 6) Run pure‑IPW updates  & quick diagnostics
# ------------------------------------------------------------------
for mgr, df_week, name in [
        (stepsMgr, df_week_steps,  "Steps"),
        (sleepMgr, df_week_sleep,  "Sleep"),
        (moodMgr,  df_week_mood,   "Mood"),
]:
    theta_before = mgr.V_chol(mgr.b).copy()
    mgr.update_posteriors(df_week)
    theta_after  = mgr.V_chol(mgr.b)
    print(f"{name}: ||Δθ̂||₂ = {np.linalg.norm(theta_after-theta_before):.3f}")

# ------------------------------------------------------------------
# 7) Save managers so tomorrow's decision script can load them
# ------------------------------------------------------------------
stepsMgr.save(STEPS_MANAGER_PATH)
sleepMgr.save(SLEEP_MANAGER_PATH)
moodMgr.save(MOOD_MANAGER_PATH)
print("Saved updated RoME managers.")
