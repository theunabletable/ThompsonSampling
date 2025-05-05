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
        featureCols: List[str],
        rewardName: str,
        dfEvents: pd.DataFrame,
        actionName: str = "sent" #Parameter without a default follows a parameter with a default
        
    ):
        """
        Args:
          participants   : list of PARTICIPANTIDENTIFIER strings
          mu0            : prior mean vector (length d)
          Sigma0         : prior covariance matrix (d × d)
          noiseVariance  : σ², the noise variance
          featureCols    : list of feature‐column names (len d)
          rewardName     : name of the reward column
          actionName     : name of the “sent” column (default 'sent')
        """
        self.featureCols   = featureCols
        self.rewardName    = rewardName
        self.actionName    = actionName
        self.noiseVariance = noiseVariance
        
        #ground truth most up-to-date dataframe slice for agents. Is updated externally. Indexed for fast lookup
        self.dfEvents = dfEvents.set_index(["PARTICIPANTIDENTIFIER","time"])

        # build TS agents for each participant
        self.agents: Dict[str, ThompsonSampling] = self._buildAgents(
            participants, mu0, Sigma0
        )

        #buffers for storing decisions & rewards
        self.decisionsPending :  List[Decision] = []  #awaiting makeDecisions
        self.decisionsMade :     List[Decision] = []  #action and p_send assign
        self.decisionsRewarded : List[Decision] = []  #reward filled in
        self.decisionsUpdated :  List[Decision] = []  #decisions have been used in updates, and can be flagged as processed
        
    def _buildAgents(
            self,
            participants: List[str],
            mu0: np.ndarray,
            Sigma0: np.ndarray
        ) -> Dict[str, ThompsonSampling]:
            """
            Create one ThompsonSampling agent per participant,
            all sharing the same prior (mu0, Sigma0).
            """
            agents = {}
            for pid in participants:
                agents[pid] = ThompsonSampling(
                    priorMean     = mu0,
                    priorCov      = Sigma0,
                    featureNames  = self.featureCols,
                    noiseVariance = self.noiseVariance,
                    actionType    = self.actionName,
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


    def makeDecisions(self) -> List[Decision]:
        """
        Consume decisionsPending: sample actions for each Decision,
        move them to decisionsMade, and return the list of Decisions.
        """
        new_decisions: List[Decision] = []
        for d in self.decisionsPending:
            agent = self.agents[d.pid]
            d.action = agent.decide(d.context)
            d.p_send = agent.probabilityOfSend(d.context)
            self.decisionsMade.append(d)
            new_decisions.append(d)
        self.decisionsPending.clear()
        return new_decisions
    
    
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
        
        
    def updatePosteriors(self):
        """
        For each rewarded decision, find its agent and call updatePosterior.
        Afterwards, clear the decisionsRewarded list.
        """
        for d in self.decisionsRewarded:
            agent = self.agents[d.pid]
            # d.context: np.ndarray, d.action: int, d.reward: float
            agent.updatePosterior(d.context, d.action, d.reward)
            self.decisionsUpdated.append(d)

        # clear out for next week
        self.decisionsRewarded.clear()
        
    def clearUpdatedDecisions(self):
        self.decisionsUpdated.clear()
        
        
    def setEventsDf(self, newdfEvents: pd.DataFrame) -> None:
        """
        Swap in a fresh slice of the master events table.
        AgentManager will then operate on this new dfEvents
        for its find/make/assign/update pipeline.
        """
        self.dfEvents = newdfEvents.set_index(["PARTICIPANTIDENTIFIER","time"])