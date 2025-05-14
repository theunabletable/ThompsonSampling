'''
utils contains helper functions and helper classes
'''


#utils.py
import pandas as pd
from typing import Dict, Any, List
from config import *
from priors import *
import numpy as np
from sklearn.metrics import pairwise_distances
import statsmodels.formula.api as smf
from scipy import sparse

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



#helper function for building a Laplacian matrix.
#takes in an aggregated user-level dataframe, an outcome variable, predictor columns, and k for k-nearest-neighbors
#and performs an OLS on outcome ~ predictors, does variable selection to pick influential coefficients, then uses those to
#compute the Laplacian matrix
def build_laplacian_graph(user_df, outcome, predictor_cols, k=5, sigma=None):
    """
    Parameters:
    - user_df: DataFrame with one row per user, containing 'outcome' and predictor_cols.
    - outcome: str, name of the outcome column (e.g., 'avg_steps_per_day').
    - predictor_cols: list of str, names of features to use.
    - k: int, number of nearest neighbors to keep.
    - sigma: float or None, RBF length-scale; if None, use median of distances.
    
    Returns:
    - A: adjacency matrix (n_users x n_users)
    - L: combinatorial Laplacian (D - A)
    - L_sym: symmetric normalized Laplacian (I - D^{-1/2} A D^{-1/2})
    - betas: Series of OLS coefficients (absolute values)
    """
    # 1. Drop missing data
    df = user_df.dropna(subset=[outcome] + predictor_cols).copy()
    
    #standardize predictors and outcome
    for col in predictor_cols + [outcome]:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    
    #OLS and get absolute betas
    formula = f"{outcome} ~ " + " + ".join(predictor_cols)
    model = smf.ols(formula, data=df).fit()
    betas = model.params.drop('Intercept').abs()
    
    #weighted feature matrix
    X = df[predictor_cols].values
    W_sqrt = np.sqrt(betas.values)[None, :]  # shape (1, p)
    Xw = X * W_sqrt
    
    #compute pairwise distances
    D = pairwise_distances(Xw, metric='euclidean')
    
    #determine sigma if not given
    if sigma is None:
        sigma = np.median(D)
    
    #compute RBF adjacency
    A_full = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(A_full, 0)
    
    #kNN sparsification
    n = D.shape[0]
    A = np.zeros_like(A_full)
    neighbors = np.argsort(D, axis=1)[:, 1:k+1]  # skip self at index 0
    for i in range(n):
        A[i, neighbors[i]] = A_full[i, neighbors[i]]
    #compute symmetric symmetric
    A = np.maximum(A, A.T)
    
    #compute degree matrix
    degrees = A.sum(axis=1)
    Dg = np.diag(degrees)
    
    #combinatorial Laplacian
    L = Dg - A
    
    #symmetric normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-12))
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    return A, L, L_sym, betas

#Takes in a dfEvents for a time period, a threshold_frac, and a k for k-nearest-neighbors and using the above helper, 
#computes the Laplacian matrices for mood, steps, and sleep
#"threshold_frac" used in variable selection, 0.25 means that the smallest beta we'll select is 1/4 the absolute value of the largest beta, 
#for predicting aggregate outcomes from user-level data and baseline survey variables.
def compute_user_laplacians(dfEvents: pd.DataFrame, k: int, threshold_frac: float = 0.25):
    """
    
    Parameters:
    - dfEvents: DataFrame with columns ['PARTICIPANTIDENTIFIER', 'time',
        survey columns..., 'StepsPastDay', 'MinutesSleepLast24Hours',
        'MoodLast24Hours'].
    - k: number of nearest neighbors for graph sparsification.
    - threshold_frac: fraction of max |beta| to select predictors.
    
    Returns:
    - L_steps, L_sleep, L_mood: combinatorial Laplacian matrices.
    """
    #preprocess
    df = dfEvents.copy()
    df['time'] = pd.to_datetime(df['time'])
    participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()
    surveyCols = [
        "Age","Sex","SigOther0","ChildNumber0","Ethnicity_1","Ethnicity_2",
        "Ethnicity_3","Ethnicity_4","Ethnicity_9","Ethnicity_5","Ethnicity_6",
        "Ethnicity_7","EFE0","Neu0","deprRelat","depr0","deprTreat0","therapy0",
        "medication0_None","PHQtot0","PHQ10above0","GADtot0","GAD10above0",
        "hours0","sleep24h0","sleepAve0","SLE0","caffeine0_None","USBorn",
        "LateCaffeine","CaffeineDose","has_child","isHeterosexual"
    ]
    stepsCol = "StepsPastDay"
    sleepCol = "MinutesSleepLast24Hours"
    moodCol  = "MoodLast24Hours"
    
    #aggregate overall user means
    user_df = (
        df
        .groupby('PARTICIPANTIDENTIFIER')
        .agg(
            avg_steps_per_day = (stepsCol, 'mean'),
            avg_sleep_minutes = (sleepCol, 'mean'),
            avg_mood          = (moodCol,  'mean'),
            days_recorded     = ('time',    'nunique')
        )
        .reset_index()
    )
    #merge in baseline survey
    survey_info = (
        df
        .groupby('PARTICIPANTIDENTIFIER')[surveyCols]
        .first()
        .reset_index()
    )
    user_df = user_df.merge(survey_info, on='PARTICIPANTIDENTIFIER', how='left')
    
    #select predictors by OLS + beta threshold
    outcomes = ['avg_steps_per_day', 'avg_sleep_minutes', 'avg_mood']
    selected = {}
    for out in outcomes:
        work = user_df.dropna(subset=[out] + surveyCols).copy()
        # standardize
        for col in surveyCols + [out]:
            work[col] = (work[col] - work[col].mean()) / work[col].std(ddof=0)
        # fit
        formula = f"{out} ~ " + " + ".join(surveyCols)
        m = smf.ols(formula, data=work).fit()
        # choose predictors with |beta| >= threshold_frac * max|beta|
        betas_abs = m.params.drop('Intercept').abs()
        thresh = betas_abs.max() * threshold_frac
        selected[out] = betas_abs[betas_abs >= thresh].index.tolist()
    
    def make_full_L(out_col, sel_cols):
        # rows that have both the outcome and the selected predictors
        valid_ids = (
            user_df
            .dropna(subset=[out_col] + sel_cols)["PARTICIPANTIDENTIFIER"]
            .tolist()
        )
    
        _, L_dense, *_ = build_laplacian_graph(user_df, out_col, sel_cols, k)
        L_small = sparse.csr_matrix(L_dense)            # convert to sparse
        return pad_laplacian_pool_missing(
            L_small,
            valid_ids,                                  # <-- now defined
            participants                                # full list from outer scope
        )

    L_steps = make_full_L("avg_steps_per_day", selected["avg_steps_per_day"])
    L_sleep = make_full_L("avg_sleep_minutes", selected["avg_sleep_minutes"])
    L_mood  = make_full_L("avg_mood",          selected["avg_mood"])

    return L_mood, L_steps, L_sleep


#the above construction loses some users who don't have enough data. This helper function reconstructs the laplacian by ensuring that
#the users with missing data are pooled together
def pad_laplacian_pool_missing(L_small, valid_ids, all_ids):
    """
    Expand an n_valid×n_valid Laplacian to N×N so that
    • valid users keep their entries
    • missing users form a complete graph (weight 1)
    Returned matrix is CSR aligned with all_ids.
    """
    idx_full = {pid: i for i, pid in enumerate(all_ids)}
    N        = len(all_ids)
    L_full   = sparse.lil_matrix((N, N), dtype=float)

    # ----- copy existing Laplacian entries (use COO to avoid fancy‑index) -----
    L_coo = L_small.tocoo()
    for r_s, c_s, val in zip(L_coo.row, L_coo.col, L_coo.data):
        i_full = idx_full[valid_ids[r_s]]
        j_full = idx_full[valid_ids[c_s]]
        L_full[i_full, j_full] = val

    # ----- complete graph on missing users -----
    missing = [pid for pid in all_ids if pid not in valid_ids]
    m = len(missing)
    if m > 1:
        for a in range(m):
            ia = idx_full[missing[a]]
            L_full[ia, ia] = m - 1        # degree
            for b in range(a + 1, m):
                ib = idx_full[missing[b]]
                L_full[ia, ib] = L_full[ib, ia] = -1.0
    # (m == 0 or 1 needs no action)

    return L_full.tocsr()

