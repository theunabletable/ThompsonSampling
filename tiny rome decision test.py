# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:41:02 2025

@author: Drew
"""

import numpy as np
import pandas as pd
from scipy import sparse
from config import *
from utils import prepare_features
from priors import build_step_priors         
from RoMEAgentManager import RoMEAgentManager



# ------------------------------------------------------------
# 1)  Load events & engineer features  (same as before)
# ------------------------------------------------------------
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df['time'] = pd.to_datetime(df['time'], utc=True)

df = prepare_features(df, STEP_MAIN_COLS,  STEP_INTERACTION_COLS,  ACTION_COL)
# (sleep & mood omitted—only testing steps)

participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()

# ------------------------------------------------------------
# 2)  Very simple Laplacian  (identity ⇒ no pooling yet)
#     You can replace with a K‑NN graph later.
# ------------------------------------------------------------
N = len(participants)
L_user = sparse.identity(N, format="csr", dtype=float)

# ------------------------------------------------------------
# 3)  Baseline user covariance  (diagonal λ·Iₚ).
#     Use priors builder just to get Σ prior on θ_i; here we
#     only need the *covariance* for one user so we build by hand.
# ------------------------------------------------------------
p = len(STEP_FEATURE_COLS)         # low‑dimensional x_d dimension
user_cov = np.eye(p) * 25.0        # same scale as shared block

# ------------------------------------------------------------
# 4)  Instantiate RoME manager (Steps only)
# ------------------------------------------------------------
romeMgr = RoMEAgentManager(
    participants            = participants,
    L_user                  = L_user,
    user_cov                = user_cov,
    gamma_ridge             = 0.0,          # keep but unused for now
    lambda_penalty          = 1.0,          # Laplacian weight
    baseFeatureCols         = STEP_MAIN_COLS,
    interactionFeatureCols  = STEP_INTERACTION_COLS,
    featureCols             = STEP_FEATURE_COLS,
    rewardName              = STEPS_REWARD,
    actionName              = ACTION_COL,
    v                       = 1.0,
    delta                   = 0.01,
    zeta                    = 10.0,
)

# ------------------------------------------------------------
# 5)  Pick one row for a smoke test
# ------------------------------------------------------------
row = df.iloc[0]
pid = row["PARTICIPANTIDENTIFIER"]

agent = romeMgr.agents[pid]

action      = agent.decide(row)
p_send      = agent.probabilityOfSend(row)

print(f"PID {pid}  action={action}  p_send={p_send:.3f}")
