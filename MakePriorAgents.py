# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:28:37 2025

@author: Drew
"""

import pandas as pd
from priors import build_step_priors
from AgentManager import AgentManager
from utils import prepare_features
from config import *


# 1) Load full historical events
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df['time'] = pd.to_datetime(df['time'], utc=True)

# 2) Initialize flags if missing
for flag in ['processedAction', 'processedReward']:
    if flag not in df.columns:
        df[flag] = False

# 3) Feature engineering: add intercept and sent_* interactions for all domains
df = prepare_features(df, STEP_MAIN_COLS, STEP_INTERACTION_COLS, action_col=ACTION_COL)
df = prepare_features(df, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, action_col=ACTION_COL)
df = prepare_features(df, MOOD_MAIN_COLS, MOOD_INTERACTION_COLS, action_col=ACTION_COL)

# Shared participant list
participants = df['PARTICIPANTIDENTIFIER'].unique().tolist()

# --- Steps AgentManager ---
mu_s, Sigma_s, sigma2_s = build_step_priors(
    df, STEP_FEATURE_COLS, STEPS_REWARD
)
stepsMgr = AgentManager(
    participants=participants,
    mu0=mu_s,
    Sigma0=Sigma_s,
    noiseVariance=sigma2_s,
    baseFeatureCols = STEP_MAIN_COLS,
    interactionFeatureCols = STEP_INTERACTION_COLS,
    featureCols=STEP_FEATURE_COLS,
    rewardName=STEPS_REWARD,
    dfEvents=df,
    actionName=ACTION_COL
)
stepsMgr.save(STEPS_MANAGER_PATH)
print(f"Saved Steps AgentManager to {STEPS_MANAGER_PATH}")

# --- Sleep AgentManager ---
mu_sl, Sigma_sl, sigma2_sl = build_step_priors(
    df, SLEEP_FEATURE_COLS, SLEEP_REWARD
)
sleepMgr = AgentManager(
    participants=participants,
    mu0=mu_sl,
    Sigma0=Sigma_sl,
    noiseVariance=sigma2_sl,
    baseFeatureCols = SLEEP_MAIN_COLS,
    interactionFeatureCols = SLEEP_INTERACTION_COLS,
    featureCols=SLEEP_FEATURE_COLS,
    rewardName=SLEEP_REWARD,
    dfEvents=df,
    actionName=ACTION_COL
)
sleepMgr.save(SLEEP_MANAGER_PATH)
print(f"Saved Sleep AgentManager to {SLEEP_MANAGER_PATH}")

# --- Mood AgentManager ---
mu_m, Sigma_m, sigma2_m = build_step_priors(
    df, MOOD_FEATURE_COLS, MOOD_REWARD
)
moodMgr = AgentManager(
    participants=participants,
    mu0=mu_m,
    Sigma0=Sigma_m,
    noiseVariance=sigma2_m,
    baseFeatureCols = MOOD_MAIN_COLS,
    interactionFeatureCols= MOOD_INTERACTION_COLS,
    featureCols=MOOD_FEATURE_COLS,
    rewardName=MOOD_REWARD,
    dfEvents=df,
    actionName=ACTION_COL
)
moodMgr.save(MOOD_MANAGER_PATH)
print(f"Saved Mood AgentManager to {MOOD_MANAGER_PATH}")

