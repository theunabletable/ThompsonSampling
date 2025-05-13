'''
utils contains helper functions and helper classes
'''


#utils.py
import pandas as pd
from typing import Dict, Any, List
from config import *
from priors import *
#Sets up the features for the daily decision script with whatever they need.
#Currently, they need interaction terms.
def prepare_features(df, main_cols, interaction_cols, action_col="sent"):
    df = df.copy()
    df["intercept"] = 1.0
    # leave main_cols and action_col in place
    # create only the specified interactions
    for col in interaction_cols:
        # col is like "sent_StepsPastDay"
        base = col.split(f"{action_col}_",1)[1]
        df[col] = df[base] * df[action_col]
    return df

#grabs the events from dfEvents for the specified date.
#def slice_by_date(df, date):
#    start = pd.Timestamp(f"{date}T00:00:00Z")
#    end   = pd.Timestamp(f"{date + pd.Timedelta(1,'D')}T00:00:00Z")
#    return df[(df["time"] >= start) & (df["time"] < end)]

def slice_by_date(df, day):
    # ensure day is a date
    target = pd.to_datetime(day).date()
    return df[df["time"].dt.date == target]

#a pipeline that builds the dataframe ready for the steps agent
def prepare_steps_day(day: pd.Timestamp) -> Dict[str, Any]:
    """
    1) Load the full events table (with raw features & sent flag).
    2) Parse time, init flags if missing.
    3) Build interactions + intercept.
    4) Compute pooled priors on the entire history.
    5) Slice out exactly today’s rows.
    
    Returns a dict with:
      - 'dfToday'    : DataFrame for today's decision slice
      - 'mu0'        : prior mean vector
      - 'Sigma0'     : prior covariance matrix
      - 'sigma2'     : noise variance
      - 'featureCols': list of columns to use as context
    """
    # (1) load & (2) parse
    df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "processedAction" not in df:
        df["processedAction"] = False
    if "processedReward" not in df:
        df["processedReward"] = False

    # (3) feature engineering
    df = prepare_features(df, STEP_MAIN_COLS, action_col=ACTION_COL)

    # (4) priors from *all* available history
    mu0, Sigma0, sigma2 = build_step_priors(
        df, STEP_FEATURE_COLS, STEPS_REWARD
    )

    # (5) slice out just this day
    df_today = slice_by_date(df, day)

    return {
        "dfDate":     df_today,
        "mu0":         mu0,
        "Sigma0":      Sigma0,
        "sigma2":      sigma2,
        "featureCols": STEP_FEATURE_COLS,
        "rewardName":  STEPS_REWARD,
        "actionName":  ACTION_COL
    }


# in utils.py

def prepare_day_slice(
    day: pd.Timestamp,
    main_cols: List[str],
    interaction_cols: List[str],
    action_col: str = "sent"
) -> pd.DataFrame:
    """
    1) Load the full events CSV (with raw features & sent flag).
    2) Parse time.
    3) Add intercept and *only* the given interaction columns.
    4) Return only the rows for `day`.
    """
    df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # 2) feature‐engineering
    df = prepare_features(df, main_cols, interaction_cols, action_col=action_col)

    # 3) subset only the columns you actually need
    keep = (
        ["PARTICIPANTIDENTIFIER", "time", action_col]
        + ["intercept"]
        + main_cols
        + interaction_cols
    )
    df = df[keep]

    # 4) slice out just this day
    return slice_by_date(df, day)

def prepare_week_slice(
    end_day: pd.Timestamp,
    window_days: int,
    main_cols: List[str],
    interaction_cols: List[str],
    action_col: str,
    reward_col: str
) -> pd.DataFrame:
    """
    1) Load the full events CSV.
    2) Parse & normalize the `time` column to UTC.
    3) Add intercept, main effects, and only the given interaction columns.
    4) Subset to [pid, time, action, reward, intercept, mains, interactions].
    5) Filter rows where `time` is in [end_day - window_days, end_day).
    """
    # 1 & 2
    df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # 3) feature‐engineering
    df = prepare_features(df, main_cols, interaction_cols, action_col=action_col)

    # 4) subset
    keep = (
        ["PARTICIPANTIDENTIFIER", "time", action_col, reward_col]
        + ["intercept"]
        + main_cols
        + interaction_cols
    )
    df = df[keep]

    # 5) time‐window filter
    start = (end_day - pd.Timedelta(days=window_days)).normalize()
    end   = end_day.normalize()
    return df[(df["time"] >= start) & (df["time"] < end)]
