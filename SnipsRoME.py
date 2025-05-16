
"""
Created on Thu May 15 14:12:39 2025

@author: Drew
"""

# rome_snips_steps_single.py
# --------------------------------------------------------------
# 0) Imports & config  -----------------------------------------
# --------------------------------------------------------------
import numpy as np, pandas as pd
from   scipy import sparse
from   utils            import prepare_features, compute_user_laplacians
from   config           import *
from   RoMEAgentManager import RoMEAgentManager
from   ThompsonSampling import NO_SEND
from datetime import timedelta
# --------------------------------------------------------------
# 1) Load log & STEP features  ---------------------------------
# --------------------------------------------------------------
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

df = prepare_features(df, STEP_MAIN_COLS, STEP_INTERACTION_COLS, ACTION_COL)
participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()

# --------------------------------------------------------------
# 2) Build Laplacian (k-NN graph)  ------------------------------
# --------------------------------------------------------------
L_mood, L_steps, L_sleep = compute_user_laplacians(df, k=20)
L_user = L_steps                          

# --------------------------------------------------------------
# 3) Choose ONE γ,λ you want to test  ---------------------------
# --------------------------------------------------------------
gamma_ridge     = 0.5
lambda_penalty  = 1.0

# --------------------------------------------------------------
# 4) Instantiate a *fresh* RoME manager  ------------------------
# --------------------------------------------------------------
p_steps   = len(STEP_FEATURE_COLS)

romeMgr = RoMEAgentManager(
    participants           = participants,
    L_user                 = L_user,
    gamma_ridge            = gamma_ridge,
    lambda_penalty         = lambda_penalty,
    baseFeatureCols        = STEP_MAIN_COLS,
    interactionFeatureCols = STEP_INTERACTION_COLS,
    featureCols            = STEP_FEATURE_COLS,
    rewardName             = STEPS_REWARD,
    actionName             = ACTION_COL,
)

# --------------------------------------------------------------
# 5) SNIPS for RoME  -------------------------------------------
# -------------------------------------------------------------- 

def snips(df_log: pd.DataFrame,
          mgr,
          batch_days: int = 1,
          verbose: bool = True) -> tuple[dict, float, float]:
    """
    Sequential SNIPS with batched RoME updates.

    ------------------------------------------------------------------
    • df_log must contain
        'PARTICIPANTIDENTIFIER', mgr.actionName, mgr.rewardName
        + all mgr.featureCols
      and be in **time-ascending** order.
    • mgr is a *fresh* RoMEAgentManager.
    • batch_days   size of the sliding training window (default 7).
    ------------------------------------------------------------------

    Returns
    --------
    snips_user   : dict pid → estimate
    cohort_mean  : mean of user-level estimates (NaN-safe)
    overall_snips: single estimate pooling all rows
    """
    act_c, rew_c, fcols = mgr.actionName, mgr.rewardName, mgr.featureCols

    participants = df_log["PARTICIPANTIDENTIFIER"].unique().tolist()
    numer   = {pid: 0.0 for pid in participants}
    denom   = {pid: 0.0 for pid in participants}
    numer_g = 0.0          # “global” numerator
    denom_g = 0.0

    # -------- iterate in weekly batches -------------------
    df_log = df_log.sort_values("time")
    t0 = df_log["time"].min().normalize()
    t_end = df_log["time"].max()

    batch_idx = 0
    while t0 <= t_end:
        t1 = t0 + timedelta(days=batch_days)
        mask = (df_log["time"] >= t0) & (df_log["time"] < t1)
        df_batch = df_log.loc[mask]
        if df_batch.empty:
            t0 = t1
            continue

        # ---- compute propensities before learning from the batch ---
        for j, row in df_batch.iterrows():
            pid   = row["PARTICIPANTIDENTIFIER"]
            a_log = int(row[act_c])
            r_log = float(row[rew_c])
            ctx   = row[fcols]

            ag     = mgr.agents[pid]
            pi_e   = ag.probabilityOfSend(ctx)
            if a_log == NO_SEND:
                pi_e = 1.0 - pi_e        # P(A=a_log | π_e)

            w = 2.0 * pi_e               # behaviour policy is 0.5 / 0.5
            numer[pid]   += w * r_log
            denom[pid]   += w
            numer_g      += w * r_log
            denom_g      += w

        # ---- *one* posterior update for the whole batch --------------
        mgr.update_posteriors(df_batch)

        batch_idx += 1
        if verbose:
            print(f"[batch {batch_idx}] rows processed: {len(df_batch)}")

        # advance window
        t0 = t1

    # -------- final SNIPS estimates ----------------------------------
    snips_user = {
        pid: (numer[pid] / denom[pid]) if denom[pid] > 0 else np.nan
        for pid in participants
    }
    cohort_mean  = np.nanmean(list(snips_user.values()))
    overall_snips = (numer_g / denom_g) if denom_g > 0 else np.nan
    return snips_user, cohort_mean, overall_snips

snips_dict, snips_mean, snips_global = snips(df, romeMgr)

print("\nRoME-Steps SNIPS results")
print("  pooled-SNIPS      :", round(snips_global, 4))
print("  mean-user SNIPS   :", round(snips_mean, 4))
print("  first 5 users ⇒", {k: round(v,4) for k,v in list(snips_dict.items())[:5]})



# --------------------------------------------------------------
# 5) SNIPS (timed first-batch profile)  -------------------------
# --------------------------------------------------------------
import time

def snips_profile(df_log: pd.DataFrame,
                  mgr,
                  batch_days: int = 1,
                  verbose: bool = True):
    """
    Same logic as your batched SNIPS but:
      • prints time spent in (i) propensity-loop, (ii) update_posteriors
      • exits after the *first* non-empty batch so you see timings fast
    """
    act_c, rew_c, fcols = mgr.actionName, mgr.rewardName, mgr.featureCols
    participants = df_log["PARTICIPANTIDENTIFIER"].unique().tolist()
    numer = {pid: 0.0 for pid in participants}
    denom = {pid: 0.0 for pid in participants}
    numer_g = denom_g = 0.0

    # --- 1) pull first batch -----------------------------------
    df_log = df_log.sort_values("time")
    t0      = df_log["time"].min().normalize()
    df_batch = pd.DataFrame()
    while df_batch.empty:
        t1        = t0 + timedelta(days=batch_days)
        mask      = (df_log["time"] >= t0) & (df_log["time"] < t1)
        df_batch  = df_log.loc[mask]
        t0        = t1                               # advance window

    print(f"\n[profile] batch size = {len(df_batch):,} rows")

    # --- 2) propensity + numerator/denominator ----------------
    t_loop0 = time.perf_counter()
    for _, row in df_batch.iterrows():
        pid   = row["PARTICIPANTIDENTIFIER"]
        a_log = int(row[act_c])
        r_log = float(row[rew_c])
        ctx   = row[fcols]

        ag   = mgr.agents[pid]
        pi_e = ag.probabilityOfSend(ctx)
        if a_log == NO_SEND:
            pi_e = 1.0 - pi_e

        w = 2.0 * pi_e
        numer[pid] += w * r_log
        denom[pid] += w
        numer_g    += w * r_log
        denom_g    += w
    t_loop1 = time.perf_counter()

    # --- 3) single posterior update ---------------------------
    t_up0 = time.perf_counter()
    mgr.update_posteriors(df_batch)
    t_up1 = time.perf_counter()

    # --- 4) report timings ------------------------------------
    loop_prop_time = t_loop1 - t_loop0
    update_time    = t_up1   - t_up0
    print(f"[timing] propensity loop : {loop_prop_time:6.2f} s")
    print(f"[timing] update_posteriors: {update_time:6.2f} s")

    # --- 5) quick SNIPS output for this batch -----------------
    snips_user = {
        pid: (numer[pid] / denom[pid]) if denom[pid] > 0 else np.nan
        for pid in participants
    }
    cohort_mean  = np.nanmean(list(snips_user.values()))
    overall_snips= numer_g / denom_g if denom_g>0 else np.nan
    return snips_user, cohort_mean, overall_snips


# ------ run once ----------------------------------------------
snips_dict, snips_mean, snips_global = snips_profile(df, romeMgr)

print("\nRoME-Steps  (one-batch profile)")
print("  pooled-SNIPS :", round(snips_global, 4))
print("  mean-user    :", round(snips_mean,   4))
print("  first 5 ⇒", {k: round(v,4) for k,v in list(snips_dict.items())[:5]})
