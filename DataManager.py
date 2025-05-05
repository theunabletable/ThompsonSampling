# -*- coding: utf-8 -*-
"""
This program implements the DataManager class. The DataManager is responsible for handling the raw data and dataframes.
It takes the raw CSVs for minute-to-minute data across all devices as well as the surveys, and the mood scores, and constructs (or updates) dfHourly, and saves a backup on disk.
It constructs and updates the dfEvents from the dfHourly.
It takes slices of dfEvents and provides them for the respective AgentManagers.

----------(processedAction and processedReward explanation)----------
It also takes data back from the AgentManagers and updates dfEvents accordingly. Specifically, a context could be new with no action decided on yet (processedAction == False).
After an action is decided, that decision is read out of decisionsPending, processedAction is set to true for that event, and it's moved to decisionsMade. This happens on a daily level.
On the weekly level, we acquire all of the rewards for the events and update dfHourly and dfEvents. We update the dataframe slices for the agent managers. The agent managers take care
of the agent updates, and moves the decisions to decisionsUpdated. This DataManager reads from either decisionsRewarded or decisionsUpdated and sets the respective processedReward == True,
and calls to clear out decisionsUpdated after the dataframe is flagged.
"""

import pandas as pd
from typing import List, Tuple
from AgentManager import *  # assumes Decision dataclass is in Decision.py or similar
from Decision import *

class DataManager:
    """
    Owns master tables dfHourly and dfEvents, and provides slices and flag updates
    for different bandit settings (steps, sleep, mood).

    Raw ingestion and feature construction methods are omitted. Instead, use
    setters to supply dfHourly and dfEvents when available.
    """
    def __init__(
        self,
        # optional initial tables
        dfHourly: pd.DataFrame = None,
        dfEvents: pd.DataFrame = None,
        #feature and reward specs for steps
        stepsFeatureCols: List[str] = None,
        stepsRewardName: str = None,
        #specs for sleep
        sleepFeatureCols: List[str] = None,
        sleepRewardName: str = None,
        #specs for mood
        moodFeatureCols: List[str] = None,
        moodRewardName: str = None,
        actionName: str = 'sent'
    ):
        # master tables
        self.dfHourly = dfHourly
        self.dfEvents = dfEvents
        if dfEvents is not None:
            self.setEventsDf(dfEvents)
        self.metaCols = [
            "PARTICIPANTIDENTIFIER",
            "time",
            "processedAction",
            "processedReward",
        ]
                
        #bandit settings specs
        self.stepsFeatureCols = stepsFeatureCols
        self.stepsRewardName  = stepsRewardName
        self.sleepFeatureCols = sleepFeatureCols
        self.sleepRewardName  = sleepRewardName
        self.moodFeatureCols  = moodFeatureCols
        self.moodRewardName   = moodRewardName
        self.actionName = actionName

    def setHourlyDf(self, df: pd.DataFrame) -> None:
        """Supply a fresh dfHourly."""
        self.dfHourly = df

    def setEventsDf(self, df: pd.DataFrame) -> None:
        """Supply or reload dfEvents; enforce MultiIndex on (pid, time)."""
        self.dfEvents = df.set_index(['PARTICIPANTIDENTIFIER','time'])

    def getStepsDf(self) -> pd.DataFrame:
        """Return the slice of dfEvents needed by the steps agent."""
        df = self.dfEvents.reset_index()
        cols = self.metaCols + self.stepsFeatureCols + [self.stepsRewardName]
        return df[cols]

    def getSleepDf(self) -> pd.DataFrame:
        """Return the slice of dfEvents needed by the sleep agent."""
        df = self.dfEvents.reset_index()
        cols = self.metaCols + self.sleepFeatureCols + [self.sleepRewardName]
        return df[cols]

    def getMoodDf(self) -> pd.DataFrame:
        """Return the slice of dfEvents needed by the mood agent."""
        df = self.dfEvents.reset_index()
        cols = self.metaCols + self.moodFeatureCols + [self.moodRewardName]
        return df[cols]

    def applyActionFlags(self, decisions: List[Decision]) -> None:
        """
        Given a list of Decisions (with pid and time), set processedAction=True
        in the master dfEvents.
        """
        for d in decisions:
            idx = (d.pid, d.time)
            self.dfEvents.at[idx, self.actionName]  = d.action
            self.dfEvents.at[idx, 'processedAction'] = True

    def applyRewardFlags(self, decisions: List[Decision]) -> None:
        """
        Given a list of Decisions (with pid and time), set processedReward=True
        in the master dfEvents.
        """
        for d in decisions:
            idx = (d.pid, d.time)
            self.dfEvents.at[idx, 'processedReward'] = True

    def refresh(self) -> None:
        """
        Placeholder for reloading or rebuilding dfHourly and dfEvents from source.
        """
        pass

    def backup_hourly(self, path: str) -> None:
        """Write dfHourly to disk as a backup."""
        if self.dfHourly is not None:
            self.dfHourly.to_csv(path, index=False)


    def backup_events(self, path: str) -> None:
        """Write dfEvents to disk as a backup."""
        if self.dfEvents is not None:
            self.dfEvents.reset_index().to_csv(path, index=False)