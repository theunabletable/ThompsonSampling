
# ======================================================================
# RoMEWeeklyUpdate.py
# ======================================================================
"""
Roll a one-week batch of logged data into each RoMEAgentManager
(steps / sleep / mood).  Called once per week.
"""

import pandas as pd
from datetime import date
from pathlib import Path

import config as cfg
from utils import prepare_week_slice
from RoMEAgentManager import RoMEAgentManager         

# --------------------------- configuration ---------------------------
run_date    = pd.Timestamp(date(2024, 7, 8), tz="UTC")   # week-end date
window      = 7                                          # days back
SET_CURRENT = True                                       # overwrite “current”?

# --------------------------- output dirs -----------------------------
date_str    = run_date.strftime("%Y-%m-%d")              # e.g. "2024-07-08"
weekly_dir  = cfg.MODELS_WEEKLY_DIR / date_str
weekly_dir.mkdir(parents=True, exist_ok=True)

# --------------------------- load RoME managers ----------------------
stepsMgr = RoMEAgentManager.load(cfg.MODELS_CURRENT_DIR / "stepsRoMEManager.pkl")
sleepMgr = RoMEAgentManager.load(cfg.MODELS_CURRENT_DIR / "sleepRoMEManager.pkl")
moodMgr  = RoMEAgentManager.load(cfg.MODELS_CURRENT_DIR / "moodRoMEManager.pkl")

# --------------------------- full events table -----------------------
df_events = pd.read_csv(cfg.DF_EVENTS_PATH, parse_dates=["time"])
df_events["time"] = pd.to_datetime(df_events["time"], utc=True)

# --------------------------- per-domain loop -------------------------
for kind, mgr, main_cols, inter_cols, feat_cols, reward, current_name in [
    ("steps", stepsMgr, cfg.STEP_MAIN_COLS,  cfg.STEP_INTERACTION_COLS,  cfg.STEP_FEATURE_COLS, cfg.STEPS_REWARD, "stepsRoMEManager.pkl"),
    ("sleep", sleepMgr, cfg.SLEEP_MAIN_COLS, cfg.SLEEP_INTERACTION_COLS, cfg.SLEEP_FEATURE_COLS, cfg.SLEEP_REWARD, "sleepRoMEManager.pkl"),
    ("mood",  moodMgr,  cfg.MOOD_MAIN_COLS,  cfg.MOOD_INTERACTION_COLS,  cfg.MOOD_FEATURE_COLS,  cfg.MOOD_REWARD,  "moodRoMEManager.pkl"),
]:
    # ---- 1) slice & feature-engineer past week -----------------------
    df_week = prepare_week_slice(
        end_day         = run_date,
        window_days     = window,
        main_cols       = main_cols,
        interaction_cols= inter_cols,
        action_col      = cfg.ACTION_COL,
        reward_col      = reward,
    )

    # keep exactly what RoME expects
    keep = ["PARTICIPANTIDENTIFIER", "time", cfg.ACTION_COL, reward] + feat_cols
    keep = list(dict.fromkeys(keep))          # dedupe
    df_train = df_week[keep].reset_index(drop=True)

    # ---- 2) update posterior ----------------------------------------
    mgr.update_posteriors(df_train)

    # ---- 3) save to weekly archive ----------------------------------
    weekly_path = weekly_dir / current_name
    mgr.save(weekly_path)
    print(f"[ARCHIVED] Saved weekly {kind} manager → {weekly_path}")

    # ---- 4) optionally overwrite current ----------------------------
    if SET_CURRENT:
        current_path = cfg.MODELS_CURRENT_DIR / current_name
        mgr.save(current_path)
        print(f"[CURRENT]  Overwrote current {kind} manager at {current_path}")
