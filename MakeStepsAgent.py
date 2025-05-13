# make_steps_manager.py
# Build and pickle the initial Steps AgentManager from historical data

import pandas as pd
from priors import build_step_priors
from AgentManager import AgentManager
from utils import prepare_features
from config import (
    DF_EVENTS_PATH,
    STEP_MAIN_COLS,
    STEP_FEATURE_COLS,
    STEPS_REWARD,
    ACTION_COL,
)

# 1) Load full events history
#    Parse 'time' as datetime
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])

# 2) Initialize flags if they don't exist
if "processedAction" not in df:
    df["processedAction"] = False
if "processedReward" not in df:
    df["processedReward"] = False

# 3) Feature engineering: add intercept and sent_* interactions
#    using STEP_MAIN_COLS and ACTION_COL
df = prepare_features(df, STEP_MAIN_COLS, action_col=ACTION_COL)

# 4) Compute pooled priors on the entire historical data
mu0, Sigma0, sigma2 = build_step_priors(
    df, STEP_FEATURE_COLS, STEPS_REWARD
)

# 5) Instantiate a Steps AgentManager
participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()
stepsMgr = AgentManager(
    participants=participants,
    mu0=mu0,
    Sigma0=Sigma0,
    noiseVariance=sigma2,
    featureCols=STEP_FEATURE_COLS,
    rewardName=STEPS_REWARD,
    dfEvents=df,
    actionName=ACTION_COL
)

# 6) Serialize to disk for daily use
stepsMgr.save(r"C:\Users\Drew\Desktop\JOB\ThompsonSampling\AgentSaves\StepsManagerPriors.pkl")
print(f"Saved Steps AgentManager with {len(participants)} participants to 'steps_manager.pkl'.")
