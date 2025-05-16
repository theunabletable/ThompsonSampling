import pandas as pd
import numpy as np
from ThompsonSampling import SEND, NO_SEND
from AgentManager      import AgentManager
from priors            import build_step_prior
from utils             import prepare_features
from config            import *

# ------------------------------------------------------------------
# Load events & add engineered features
# ------------------------------------------------------------------
df = pd.read_csv(DF_EVENTS_PATH, parse_dates=["time"])
df["time"] = pd.to_datetime(df["time"], utc=True)

df = prepare_features(df, STEP_MAIN_COLS,  STEP_INTERACTION_COLS,  ACTION_COL)
df = prepare_features(df, SLEEP_MAIN_COLS, SLEEP_INTERACTION_COLS, ACTION_COL)
df = prepare_features(df, MOOD_MAIN_COLS,  MOOD_INTERACTION_COLS,  ACTION_COL)

participants = df["PARTICIPANTIDENTIFIER"].unique().tolist()

# ------------------------------------------------------------------
#Build priors + AgentManagers
# ------------------------------------------------------------------
# -------- Steps --------
mu_s, Sig_s, s2_s = build_step_priors(df, STEP_FEATURE_COLS, STEPS_REWARD)
stepsMgr = AgentManager(
    participants            = participants,
    mu0                     = mu_s,
    Sigma0                  = Sig_s,
    noiseVariance           = s2_s,
    baseFeatureCols         = STEP_MAIN_COLS,
    interactionFeatureCols  = STEP_INTERACTION_COLS,
    featureCols             = STEP_FEATURE_COLS,
    rewardName              = STEPS_REWARD,
    dfEvents                = df,
    actionName              = ACTION_COL
)

# -------- Sleep --------
mu_sl, Sig_sl, s2_sl = build_step_priors(df, SLEEP_FEATURE_COLS, SLEEP_REWARD)
sleepMgr = AgentManager(
    participants            = participants,
    mu0                     = mu_sl,
    Sigma0                  = Sig_sl,
    noiseVariance           = s2_sl,
    baseFeatureCols         = SLEEP_MAIN_COLS,
    interactionFeatureCols  = SLEEP_INTERACTION_COLS,
    featureCols             = SLEEP_FEATURE_COLS,
    rewardName              = SLEEP_REWARD,
    dfEvents                = df,
    actionName              = ACTION_COL
)

# -------- Mood --------
mu_m, Sig_m, s2_m = build_step_priors(df, MOOD_FEATURE_COLS, MOOD_REWARD)
moodMgr = AgentManager(
    participants            = participants,
    mu0                     = mu_m,
    Sigma0                  = Sig_m,
    noiseVariance           = s2_m,
    baseFeatureCols         = MOOD_MAIN_COLS,
    interactionFeatureCols  = MOOD_INTERACTION_COLS,
    featureCols             = MOOD_FEATURE_COLS,
    rewardName              = MOOD_REWARD,
    dfEvents                = df,
    actionName              = ACTION_COL
)

# ------------------------------------------------------------------
#SNIPS evaluation
# ------------------------------------------------------------------
def snips(df_log: pd.DataFrame, mgr) -> tuple[dict, float, float]:
    ''' 
    Takes in a slice of the main dataframe and a simple agent manager
    and outputs
        • a dictionary of SNIPS scores for each user,
        • the mean of those user-level SNIPS,
        • the overall population SNIPS (single estimate using all rows).
    '''
    act_c, rew_c, fcols = mgr.actionName, mgr.rewardName, mgr.featureCols

    numer = {pid: 0.0 for pid in mgr.agents}
    denom = {pid: 0.0 for pid in mgr.agents}

    # --- global accumulators for “all-rows” SNIPS ---
    numer_all, denom_all = 0.0, 0.0

    for _, row in df_log.iterrows():
        pid   = row["PARTICIPANTIDENTIFIER"]
        a_log = int(row[act_c])
        r_log = float(row[rew_c])
        ctx   = row[fcols]

        ag = mgr.agents[pid]

        # probability under agent's policy
        pi_e = ag.probabilityOfSend(ctx)
        if a_log == NO_SEND:
            pi_e = 1.0 - pi_e

        # w = agentProbability / probabilityOfAction = π_e / 0.5 = 2·π_e
        w = 2.0 * pi_e

        # ---- per-user accumulators ----
        numer[pid] += w * r_log
        denom[pid] += w

        # ---- global accumulators ----
        numer_all += w * r_log
        denom_all += w

        # Bayesian update with (s, a_log, r_log)
        ag.updatePosterior(ctx, a_log, r_log)

    snips_user = {
        pid: (numer[pid] / denom[pid]) if denom[pid] > 0 else np.nan
        for pid in mgr.agents
    }
    cohort_mean   = np.nanmean(list(snips_user.values()))
    cohort_snips  = (numer_all / denom_all) if denom_all > 0 else np.nan

    return snips_user, cohort_mean, cohort_snips
# ------------------------------------------------------------------
# 4.  Run SNIPS for each outcome
# ------------------------------------------------------------------
snipsSteps, meanSteps, populationSnipsSteps  = snips(df, stepsMgr)
snipsSleep, meanSleep, populationSnipsSleep = snips(df, sleepMgr)
snipsMood,  meanMood, populationSnipsMood = snips(df, moodMgr)

print("\n--- SNIPS Evaluation (Cohort Means) ---")
print(f"{'Outcome':<10} | {'SNIPS Mean':>12} | {'Population Mean':>18}")
print("-" * 45)
print(f"{'Steps':<10} | {meanSteps:>12.4f} | {populationSnipsSteps:>18.4f}")
print(f"{'Sleep':<10} | {meanSleep:>12.4f} | {populationSnipsSleep:>18.4f}")
print(f"{'Mood':<10}  | {meanMood:>12.4f} | {populationSnipsMood:>18.4f}")