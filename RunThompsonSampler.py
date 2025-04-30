# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:38:01 2025

@author: Drew
"""

from priors import *
import pandas as pd
import numpy as np
from ThompsonSampling import *

dfEvents = pd.read_csv("C:/Users/Drew/Desktop/JOB/2024 data/new dataframes/dfEventsThreeMonthsTimezones.csv")

#REWARD_COL = "NextMoodScore"
#REWARD_COL = "SleepReward"
REWARD_COL = "LogStepsReward"

stepsColumns = ['StepsPastDay', 'Steps1HourBefore', 'is_weekend', 'StepsPastWeek',
                'StepsPast3Days', 'maxHRPast24Hours', 'RHR', 'HoursSinceLastMood']


#mainEffects = moodColumns
#mainEffect = sleepColumns
mainEffects = stepsColumns

#choose columns to use in df
df = dfEvents[mainEffects + [REWARD_COL, 'sent', 'PARTICIPANTIDENTIFIER', 'time']]

#adds interactions to all mainEffects
df = add_sent_interactions(df, mainEffects, sent_var = 'sent')
df['intercept'] = 1
featureCols = df.drop(columns = [REWARD_COL, 'PARTICIPANTIDENTIFIER', 'time']).columns

#reorder columns
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('PARTICIPANTIDENTIFIER')))
cols.insert(1, cols.pop(cols.index('time')))
cols.insert(2, cols.pop(cols.index(REWARD_COL)))
cols.insert(3, cols.pop(cols.index('sent')))

df = df[cols]
#df prepared

mu0, Sigma0, sigma2 = build_step_priors(df, featureCols, REWARD_COL)

agents = {}
for pid, grp in df.groupby("PARTICIPANTIDENTIFIER"):
    #initialize fresh TS agent with pooled prior
    agent = ThompsonSampling(priorMean = mu0, priorCov = Sigma0, featureNames = featureCols, noiseVariance = 0.1*sigma2, actionType = 'sent')
    for _, row in grp.sort_values("time").iterrows():
        print(row.index.tolist())
        context = row[featureCols].astype(float)
        action = row["sent"] #action taken
        reward = row["LogStepsReward"] #reward observed
        agent.updatePosterior(context, action, reward)
    agents[pid] = agent
    
# 1. extract prior variances (diagonal of Sigma0)
prior_vars = np.diag(Sigma0)

# 2. build a summary table
rows = []
for pid, agent in agents.items():
    post_vars = np.diag(agent.currentCov)
    rows.append({
        'PARTICIPANTIDENTIFIER': pid,
        'mean_prior_var':    prior_vars.mean(),
        'mean_post_var':     post_vars.mean(),
        'var_ratio':         post_vars.mean() / prior_vars.mean()
    })

df_summary = pd.DataFrame(rows)

# 3. inspect the results, sorted by how much variance has shrunk
print(df_summary.sort_values('var_ratio').head(20))


rows = []
for pid, agent in agents.items():
    grp = df[df["PARTICIPANTIDENTIFIER"] == pid]
    
    # 1) actual fraction of sends in the data
    actual_frac = grp["sent"].mean()
    
    # 2) fraction of sends the TS agent would choose
    #    we pass only the featureCols slice into decide()
    sim_sends = grp.apply(
        lambda row: agent.decide(row[featureCols]),
        axis=1
    )
    sim_frac = sim_sends.mean()
    
    rows.append({
        "PARTICIPANTIDENTIFIER": pid,
        "actual_frac_send": actual_frac,
        "simulated_frac_send": sim_frac
    })

df_compare = pd.DataFrame(rows)
print(df_compare.sort_values("simulated_frac_send").head(20))

print(df_compare['simulated_frac_send'].mean())
