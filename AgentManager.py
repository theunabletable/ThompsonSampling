# -*- coding: utf-8 -*-
"""
This class implements an "AgentManager" object. The AgentManager for a particular reward-type is responsible for keeping the dictionary of agents,
calling on those agents to perform actions, and assigning rewards to the agents which performed actions.

The AgentManager is not responsible for the updates to the dfEvents. dfEvents is the ground truth most up-to-date dataframe of events. It's given 
an updated dfEvents each day, and it finds the new events (rows from dfEvents which haven't been processed yet [could be new context, or new reward]), 
and forms "Decision" objects. The dataframe manager is responsible for finding decisions in decisionsMade to update processedAction,
and for finding decisions in decisionsUpdated to update processedReward in the underlying dataframe.

This class essentially handles a conveyor belt: it finds unprocessed rows and constructs Decisions, and fills decisionsPending with them.
Then makeDecisions moves each decision to decisionsMade, updating their action attribute. Then assignRewards moves them to
decisionsRewarded, updating their reward attribute. Finally, updatePosterior uses all of the decisionsRewarded and updates the agents.

findDecisions       ->        makeDecisions      ->       assignRewards        ->        updatePosterior -> clear
             (decisionsPending)~~~~~~~~~~~~(decisionsMade)~~~~~~~~~~~~~(decisionsRewarded)~~~~~~~~~~~~~~(empty)

"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ThompsonSampling import ThompsonSampling
import numpy as np
import pandas as pd
from Decision import *
import pickle
# global constants
SEND    = 1
NO_SEND = 0

class AgentManager:
    def __init__(
        self,
        participants: List[str],
        mu0: np.ndarray,
        Sigma0: np.ndarray,
        noiseVariance: float,
        baseFeatureCols: List[str],
        interactionFeatureCols: List[str],
        featureCols: List[str],
        rewardName: str,
        dfEvents: pd.DataFrame,
        actionName: str = "sent"
        
    ):
        """
        participants: list of participant IDs
        mu0, Sigma0: prior mean and covariance
        noiseVariance: observation noise σ²
        baseFeatureCols: raw features + intercept
        interactionFeatureCols: list of sent_* interaction columns
        featureCols: full design (base + action + interaction)
        rewardName: name of reward column
        dfEvents: full events slice (with pid, time, features, action, reward)
        actionName: column name for action flag (default 'sent')
        """
        self.baseFeatureCols        = baseFeatureCols
        self.interactionFeatureCols = interactionFeatureCols
        self.featureCols            = featureCols
        self.rewardName             = rewardName
        self.actionName             = actionName
        self.noiseVariance          = noiseVariance
        # events indexed by (pid, time)
        self.dfEvents = dfEvents.set_index(["PARTICIPANTIDENTIFIER","time"])

        # build samplers
        self.agents: Dict[str, ThompsonSampling] = self._buildAgents(
            participants, mu0, Sigma0
        )

    def _buildAgents(
        self,
        participants: List[str],
        mu0: np.ndarray,
        Sigma0: np.ndarray
    ) -> Dict[str, ThompsonSampling]:
        agents = {}
        for pid in participants:
            agents[pid] = ThompsonSampling(
                priorMean               = mu0,
                priorCov                = Sigma0,
                baseFeatureNames        = self.baseFeatureCols,
                interactionFeatureNames = self.interactionFeatureCols,
                featureNames            = self.featureCols,
                noiseVariance           = self.noiseVariance,
                actionType              = self.actionName
            )
        return agents


        
    def findDecisions(self) -> List[tuple]:
        """
        Scan self.dfEvents for rows where processedAction == False,
        create Decision objects, append them to decisionsPending,
        and return (pid, time) keys.
        """
        mask    = ~self.dfEvents["processedAction"]
        newRows = self.dfEvents.loc[mask] #rows which haven't been processed yet
        
        keys = []
        for (pid, time), row in newRows.iterrows():
            context = row[self.featureCols].astype(float)
            d = Decision(pid=pid, time=time, context=context)
            self.decisionsPending.append(d)
            keys.append((d.pid, d.time))
        return keys


    
    
    def rewardCalculator(self, pid, time, context, action) -> float:
        """
        look up the true reward in self.dfEvents.
        """
        try:
            return float(self.dfEvents.loc[(pid, time), self.rewardName])
        except KeyError:
            raise RuntimeError(f"Reward not found for {(pid, time)}")
            
            
    
    def assignRewards(self) -> List[Decision]:
        """
        Consume decisionsMade: compute rewards and move to decisionsRewarded.
        Return the list of Decisions rewarded.
        """
        rewarded: List[Decision] = []
        pending: List[Decision] = []
        for d in self.decisionsMade:
            try:
                d.reward = self.rewardCalculator(d.pid, d.time, d.context, d.action)
                self.decisionsRewarded.append(d)
                rewarded.append(d)
            except RuntimeError:
                pending.append(d)
        # update the decisionsMade to only those still pending
        self.decisionsMade = pending
        # warn if some decisions couldn't be rewarded
        if pending:
            print(f"Warning: {len(pending)} decisions could not be rewarded and remain pending.")
        return rewarded
        
        

        
    def clearUpdatedDecisions(self):
        self.decisionsUpdated.clear()
        
        
    def make_decisions(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df_slice.iterrows():
            pid    = row['PARTICIPANTIDENTIFIER']
            time   = row['time']
            ctx    = row[self.featureCols]
            agent  = self.agents[pid]
            action = agent.decide(ctx)
            p_send = agent.probabilityOfSend(ctx)
            records.append({'PARTICIPANTIDENTIFIER': pid,
                            'time': time,
                            'action': action,
                            'p_send': p_send})
        return pd.DataFrame(records)

    def update_posteriors(self, df_train: pd.DataFrame) -> None:
        """
        Expects df_train with columns:
           - 'PARTICIPANTIDENTIFIER'
           - self.actionName  (e.g. 'sent')
           - self.rewardName
           - *all* of self.featureCols (which includes intercept, mains, interactions)
        """
        for _, row in df_train.iterrows():
            pid    = row['PARTICIPANTIDENTIFIER']
            # *convert* the Series slice into a dict of floats
            ctx    = {f: float(row[f]) for f in self.featureCols}
            action = int(row[self.actionName])
            reward = float(row[self.rewardName])
            agent  = self.agents[pid]
            agent.updatePosterior(ctx, action, reward)   
            
    def setEventsDf(self, newdfEvents: pd.DataFrame) -> None:
        """
        Swap in a fresh slice of the master events table.
        AgentManager will then operate on this new dfEvents
        for its find/make/assign/update pipeline.
        """
        self.dfEvents = newdfEvents.set_index(["PARTICIPANTIDENTIFIER","time"])
        

    def save(self, path: str) -> None:
        """
        Serialize this AgentManager (and all its agents) to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
            
    @classmethod
    def load(cls, path: str) -> "AgentManager":
        """
        Reconstruct an AgentManager from a pickle file.
        """
        with open(path, "rb") as f:
            return pickle.load(f)