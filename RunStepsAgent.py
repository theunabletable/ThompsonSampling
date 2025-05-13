# -*- coding: utf-8 -*-
"""
Created on Sun May  4 17:12:08 2025

@author: Drew
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:38:01 2025

@author: Drew
"""

from priors import *
import pandas as pd
import numpy as np
from ThompsonSampling import *
from AgentManager import *

CSV_PATH = "C:/Users/Drew/Desktop/JOB/2024 data/new dataframes/dfEventsThreeMonthsTimezones.csv"
dfEvents = pd.read_csv(CSV_PATH)

dfEvents['processedAction'] = False
dfEvents['processedReward'] = False


REWARD_COL = "LogStepsReward"

stepsColumns = ['StepsPastDay', 'Steps1HourBefore', 'is_weekend', 'StepsPastWeek',
                'StepsPast3Days', 'maxHRPast24Hours', 'RHR', 'HoursSinceLastMood']



mainEffects = stepsColumns

#choose columns to use in df
df = dfEvents[mainEffects + [REWARD_COL, 'sent', 'PARTICIPANTIDENTIFIER', 'time', 'processedAction', 'processedReward']]

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

dfEvents = df[cols]
#df prepared

#build the priors
mu0, Sigma0, sigma2 = build_step_priors(df, featureCols, REWARD_COL)

participants = dfEvents["PARTICIPANTIDENTIFIER"].unique().tolist()

agMgr = AgentManager(
    participants = participants,
    mu0 = mu0,
    Sigma0 = Sigma0,
    noiseVariance = sigma2,
    featureCols = featureCols,
    rewardName = REWARD_COL,
    dfEvents = df,
    actionName = 'sent'
)
agents = agMgr.agents

agMgr.findDecisions()
#finds events for which decisions haven't been made yet
decisions = agMgr.makeDecisions()
#makes decisions for each of those events, leaving a list of Decision objects
dfMgr.processDecisions(decisions)
#need to update the ground-truth dataframe to say that these events have had their decisions processed

#convert the decisions to a dataframe and csv, to be saved on desk, and sent elsewhere for sending
df_decisions = pd.DataFrame([{
    "PARTICIPANTIDENTIFIER": d.pid,
    "time":                  d.time,
    "action":                d.action,
    "p_send":                d.p_send,
} for d in decisions])
dfDecisions.to_csv(DECISION_OUT_PATH, index = False)
print(f"Wrote {len(dfDecisions)} decisions to {DECISIONS_OUT_PATH}")



#train agents
agents = {}
for pid, grp in df.groupby("PARTICIPANTIDENTIFIER"):
    #initialize fresh TS agent with global prior
    agent = ThompsonSampling(priorMean = mu0, priorCov = Sigma0, featureNames = featureCols, noiseVariance = 0.1*sigma2, actionType = 'sent')
    for _, row in grp.sort_values("time").iterrows():
        print(row.index.tolist())
        context = row[featureCols].astype(float)
        action = row["sent"] #action taken
        reward = row["LogStepsReward"] #reward observed
        agent.updatePosterior(context, action, reward)
    agents[pid] = agent
    

#check fraction of sends
rows = []
for pid, agent in agents.items():
    grp = df[df["PARTICIPANTIDENTIFIER"] == pid]
    
    #actual fraction of sends in the data
    actual_frac = grp["sent"].mean()
    
    #fraction of sends the TS agent would choose
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
