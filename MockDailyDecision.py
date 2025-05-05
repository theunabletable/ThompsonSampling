# mock_daily_decisions.py
# Quick demo of daily decisions (steps only) — no rewards or training

import pandas as pd
import numpy as np
from priors import build_step_priors
from AgentManager import AgentManager
from DataManager import DataManager

# Helper to add interaction columns
def add_sent_interactions(df, base_cols, sent_var='sent'):
    for col in base_cols:
        if col == sent_var:
            continue
        inter_name = f"{sent_var}_{col}"
        df[inter_name] = df[col] * df[sent_var]
    return df

# --- Configuration & Mock Data Setup ---
CSV_PATH = "C:/Users/Drew/Desktop/JOB/2024 data/new dataframes/dfEventsThreeMonthsTimezones.csv"
REWARD_COL = "LogStepsReward"
stepsColumns = [
    'StepsPastDay', 'Steps1HourBefore', 'is_weekend',
    'StepsPastWeek', 'StepsPast3Days',
    'maxHRPast24Hours', 'RHR', 'HoursSinceLastMood'
]

# Load full events and initialize flags
dfEvents = pd.read_csv(CSV_PATH)
dfEvents['time'] = pd.to_datetime(dfEvents['time'], utc=True)
dfEvents['processedAction'] = False
dfEvents['processedReward'] = False
# Add interactions & intercept for features
df = dfEvents.copy()
df = add_sent_interactions(df, stepsColumns, sent_var='sent')
df['intercept'] = 1.0

#build featureCols
baseCols = ['sent', 'intercept']

# 3) The interaction columns you already injected
interCols = [f"sent_{col}" for col in stepsColumns]

# 4) Put ’em together in the order you like
featureCols = stepsColumns + baseCols + interCols

# Compute priors on pooled historical data (using all rows)
mu0, Sigma0, sigma2 = build_step_priors(
    df, featureCols, REWARD_COL
)




dfEvents = df
dfToday = dfEvents[dfEvents['time'].dt.date == pd.Timestamp("2024-07-01")]

start = pd.Timestamp("2024-07-01T00:00:00Z")
end   = pd.Timestamp("2024-07-02T00:00:00Z")
dfToday = dfEvents[(dfEvents['time'] >= start) & (dfEvents['time'] < end)]
# Instantiate DataManager and AgentManager for steps
dm = DataManager(
    dfHourly=None, dfEvents=dfToday,
    stepsFeatureCols=featureCols, stepsRewardName=REWARD_COL
)


stepsMgr = AgentManager(
    participants=df['PARTICIPANTIDENTIFIER'].unique().tolist(),
    mu0=mu0, Sigma0=Sigma0, noiseVariance=sigma2,
    featureCols=featureCols, rewardName=REWARD_COL,
    dfEvents=df, actionName='sent'
)

# --- Daily Decision Pipeline (only decisions) ---
# 1) Load latest slice into Managers
dfSteps = dm.getStepsDf()
stepsMgr.setEventsDf(dfSteps)

# 2) Find unprocessed events
keys = stepsMgr.findDecisions()

# 3) Make decisions
decisions = stepsMgr.makeDecisions()

# 4) Apply action flags back to master dfEvents
dm.applyActionFlags(decisions)

# 5) Write send list for messaging service
send_list = [
    {'pid': d.pid, 'time': d.time, 'action': d.action}
    for d in decisions
]
pd.DataFrame(send_list).to_csv("decisions_today.csv", index=False)

print(f"Daily decisions complete: {len(decisions)} total, {len(send_list)} sends recorded.")
