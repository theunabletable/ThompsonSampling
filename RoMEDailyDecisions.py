# =======================================================================
# DailyDecisions_RoME.py
# =======================================================================
"""Load RoME managers, create today’s decision list, and persist it."""

import pandas as pd
from datetime import date
import pickle
from pathlib import Path
import config as cfg
from Decision import Decision

# --------------------------- load managers ------------------------------
with open(cfg.MODELS_CURRENT_DIR / "stepsRoMEManager.pkl", "rb") as f:
    stepsMgr = pickle.load(f)
with open(cfg.MODELS_CURRENT_DIR / "sleepRoMEManager.pkl", "rb") as f:
    sleepMgr = pickle.load(f)
with open(cfg.MODELS_CURRENT_DIR / "moodRoMEManager.pkl", "rb") as f:
    moodMgr  = pickle.load(f)

# --------------------------- prepare slice ------------------------------
today = pd.Timestamp(date.today())

run_date = pd.Timestamp(date(2024, 7, 9), tz="UTC")

base_dir  = Path(r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\Data\decisions")
date_str  = run_date.strftime("%Y-%m-%d")               # "2024-07-09"
daily_dir = base_dir / date_str                         # ".../decisions/2024-07-09"
daily_dir.mkdir(parents=True, exist_ok=True)            # make it (and parents) if needed

df_day = pd.read_csv(cfg.DF_EVENTS_PATH, parse_dates=["time"])  # or use pre‑sliced file
mask   = df_day["time"].dt.normalize() == run_date
if mask.any():
    df_day = df_day.loc[mask].copy()
else:
    raise RuntimeError("No rows for today in dfEvents – ETL step missing?")

# --------------------------- make decisions -----------------------------
records = []
for _, row in df_day.iterrows():
    pid = row["PARTICIPANTIDENTIFIER"]

    # --- STEPS agent ----------------------------------------------------
    action_steps = stepsMgr.agents[pid].decide(row)
    p_send_steps = stepsMgr.agents[pid].probabilityOfSend(row)

    # store pi_no_send so that WeeklyUpdate can skip recompute
    record = {
        **row,
        "decision_action": action_steps,
        "p_send": p_send_steps,
        "pi_no_send": 1.0 - p_send_steps,
    }
    records.append(record)

# save to CSV
out_dir  = Path(cfg.DECISIONS_SAVE_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"decisions_{today.date()}.csv"
pd.DataFrame(records).to_csv(out_path, index=False)
print("[DailyDecisions_RoME] Saved", len(records), "decisions →", out_path)