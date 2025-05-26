# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:09:15 2025

@author: Drew
"""

# rome_snips_steps_single.py
# --------------------------------------------------------------
# Fast SNIPS for the “steps” domain using the new dense-inverse
# RoME implementation.  Nothing in the rest of your pipeline
# needs to change.
# --------------------------------------------------------------
import time
from   datetime       import timedelta
from   pathlib        import Path

import numpy  as np
import pandas as pd
from   scipy import sparse
from   scipy.stats import norm

from utils              import prepare_features, compute_user_laplacians
from config             import *
from RoMEAgentManager   import RoMEAgentManager
from ThompsonSampling   import NO_SEND

# --------------------------------------------------------------
# 1)  Load full event log & engineer step features
# --------------------------------------------------------------
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

df  = prepare_features(df,
                       STEP_MAIN_COLS,
                       STEP_INTERACTION_COLS,
                       ACTION_COL)

participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()

# --------------------------------------------------------------
# 2)  k-NN Laplacian  (steps domain)
# --------------------------------------------------------------
L_mood, L_steps, L_sleep = compute_user_laplacians(df, k=20)
L_user = L_steps

# --------------------------------------------------------------
# 3)  Choose hyper-parameters you want to try
# --------------------------------------------------------------
gamma_ridge    = 0.5
lambda_penalty = 1.0

# --------------------------------------------------------------
# 4)  Instantiate a *fresh* RoME manager  (dense inverse built once)
# --------------------------------------------------------------
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
romeMgr._rebuild_dense_inverse()           # << key speed-up: builds V_inv, θ̂, β

# --------------------------------------------------------------
# 5)  Super-fast propensity helper  (no sparse solves)
# --------------------------------------------------------------
def fast_p_send(row, mgr=romeMgr) -> float:
    """Deterministic replica of agent.probabilityOfSend, but ~100× faster."""
    pid = row["PARTICIPANTIDENTIFIER"]

    phi_s = mgr.make_phi(pid, row, 1)
    phi_n = mgr.make_phi(pid, row, 0)
    phi   = (phi_s - phi_n).tocsc()               # 1×d, ≈32 non-zeros
    idx   = phi.indices
    vec   = phi.data

    mu  = float(vec @ mgr._theta_cache[idx])

    Vinv_sub = mgr.V_inv[np.ix_(idx, idx)]        # 32×32 dense slice
    base_var = float(vec @ (Vinv_sub @ vec))

    beta = mgr._beta_cache[pid]
    var  = base_var * (beta / mgr.first_beta) ** 2
    sd   = np.sqrt(max(var, 1e-12))

    p_no = norm.cdf(-mu / sd)                     # Φ(−μ / σ)
    p_no = min(mgr.agents[pid].pi_max,
               max(mgr.agents[pid].pi_min, p_no))
    return 1.0 - p_no                             # P(send)

# --------------------------------------------------------------
# 6)  Batched SNIPS estimator (unchanged logic, faster backend)
# --------------------------------------------------------------
def snips(df_log: pd.DataFrame,
          mgr: RoMEAgentManager,
          batch_days: int = 7,
          verbose: bool = True) -> tuple[dict, float, float]:

    act_c, rew_c, fcols = mgr.actionName, mgr.rewardName, mgr.featureCols
    participants        = mgr.participants

    numer   = {pid: 0.0 for pid in participants}
    denom   = {pid: 0.0 for pid in participants}
    numer_g = denom_g = 0.0

    # --- iterate through sliding windows -----------------------
    df_log = df_log.sort_values("time")
    t0     = df_log["time"].min().normalize()
    t_end  = df_log["time"].max()

    batch_idx = 0
    while t0 <= t_end:
        t1        = t0 + timedelta(days=batch_days)
        mask      = (df_log["time"] >= t0) & (df_log["time"] < t1)
        df_batch  = df_log.loc[mask]

        if df_batch.empty:
            t0 = t1
            continue
        # ---------- propensities (fast) ------------------------
        pi_no_list = []                      # ← NEW
        
        for _, row in df_batch.iterrows():
            pid   = row["PARTICIPANTIDENTIFIER"]
            a_log = int(row[act_c])
            r_log = float(row[rew_c])
        
            p_send = fast_p_send(row)
            p_no   = 1.0 - p_send
            pi_e   = p_send if a_log == 1 else p_no        # prob. of the logged arm
        
            w = 2.0 * pi_e                                 # behaviour policy ½ / ½
            numer[pid] += w * r_log
            denom[pid] += w
            numer_g    += w * r_log
            denom_g    += w
        
            pi_no_list.append(p_no)        # ← keep it for update_posteriors
        
        # add the logged propensity *once*; update_posteriors will re-use it
        df_batch = df_batch.copy()         # (avoid SettingWithCopy warnings)
        df_batch["pi_no_send"] = pi_no_list
        
        # ---------- one posterior update ----------------------
        mgr.update_posteriors(df_batch)
       
        batch_idx += 1
        if verbose:
            print(f"[batch {batch_idx}] processed {len(df_batch):,} rows")

        t0 = t1                                   # slide window

    # ---------- SNIPS aggregations ----------------------------
    snips_user = {pid: numer[pid]/denom[pid] if denom[pid] > 0 else np.nan
                  for pid in participants}
    cohort_mean   = np.nanmean(list(snips_user.values()))
    overall_snips = numer_g/denom_g if denom_g > 0 else np.nan
    return snips_user, cohort_mean, overall_snips

# --------------------------------------------------------------
# 7)  Run full SNIPS (timed)
# --------------------------------------------------------------
t0 = time.perf_counter()
snips_dict, snips_mean, snips_global = snips(df, romeMgr)
print(f"\n⏱  full SNIPS finished in {time.perf_counter()-t0:,.2f} s")

print("\nRoME-Steps SNIPS summary")
print("  pooled-SNIPS        :", round(snips_global, 4))
print("  mean user SNIPS     :", round(snips_mean,   4))
print("  first 5 users       :", {k: round(v,4)
                                  for k,v in list(snips_dict.items())[:5]})