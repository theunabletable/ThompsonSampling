# =======================================================================
# DailyDecisions_RoME.py
# =======================================================================
"""Load RoME managers, create today’s decision lists for *steps*, *sleep*,
*and mood*, then save three CSVs under:

    Data/decisions/YYYY-MM-DD/{steps,sleep,mood}_decisions.csv

Each CSV includes the taken action, π(send), and π(no_send) so the weekly
update can reuse the logged behaviour policy.
"""

from datetime import date
from pathlib import Path
import pickle
import pandas as pd
import config as cfg
from utils import prepare_day_slice  # same helper you use elsewhere

# --------------------------- I/O paths ----------------------------------
run_date  = pd.Timestamp(date(2024, 7, 9), tz="UTC")  # ← set programmatically
base_dir  = Path(r"C:/Users/Drew/Desktop/JOB/ThompsonSampling/Data/decisions")
# (raw string for Windows slashes; Path normalises)

daily_dir = base_dir / run_date.strftime("%Y-%m-%d")
daily_dir.mkdir(parents=True, exist_ok=True)           # ensure folder exists

# --------------------------- load managers ------------------------------
with open(cfg.MODELS_CURRENT_DIR / "stepsRoMEManager.pkl", "rb") as f:
    stepsMgr = pickle.load(f)
with open(cfg.MODELS_CURRENT_DIR / "sleepRoMEManager.pkl", "rb") as f:
    sleepMgr = pickle.load(f)
with open(cfg.MODELS_CURRENT_DIR / "moodRoMEManager.pkl",  "rb") as f:
    moodMgr  = pickle.load(f)

# --------------------------- prepare day slices -------------------------
steps_slice = prepare_day_slice(run_date,
                                cfg.STEP_MAIN_COLS,
                                cfg.STEP_INTERACTION_COLS,
                                action_col=cfg.ACTION_COL)

sleep_slice = prepare_day_slice(run_date,
                                cfg.SLEEP_MAIN_COLS,
                                cfg.SLEEP_INTERACTION_COLS,
                                action_col=cfg.ACTION_COL)

mood_slice  = prepare_day_slice(run_date,
                                cfg.MOOD_MAIN_COLS,
                                cfg.MOOD_INTERACTION_COLS,
                                action_col=cfg.ACTION_COL)

# Map for iteration ------------------------------------------------------
settings = [
    ("steps", stepsMgr, steps_slice),
    ("sleep", sleepMgr, sleep_slice),
    ("mood",  moodMgr,  mood_slice),
]

for setting, mgr, df_slice in settings:
    records = []
    for _, row in df_slice.iterrows():
        pid        = row["PARTICIPANTIDENTIFIER"]
        action     = mgr.agents[pid].decide(row)
        p_send     = mgr.agents[pid].probabilityOfSend(row)

        records.append({
            **row,
            "decision_action": action,
            "p_send": p_send,
            "pi_no_send": 1.0 - p_send,
        })

    out_path = daily_dir / f"{setting}_decisions.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[DailyDecisions_RoME] {setting}: saved {len(records):,} decisions → {out_path}")
