
"""

"""

import numpy as np
import statsmodels.api as sm
import pandas as pd

def build_step_priors(df_hist, feature_cols, reward_col, shrink = 1):
    """
    Empirical-Bayes priors from pooled OLS with c=1:
      μ₀ = betahat_OLS,    Σ₀ = σ̂² (XᵀX)⁻¹,    σ² = σ̂²
    """
    # ensure intercept and float dtype
    X = df_hist[feature_cols].astype(float)
    y = df_hist[reward_col].astype(float)

    ols     = sm.OLS(y, X).fit()
    mu0     = ols.params.values          # β̂_OLS
    sigma2  = ols.mse_resid              # σ̂²
    Sigma0  = sigma2 * np.linalg.inv(X.T @ X)

    return mu0, shrink*Sigma0, sigma2

def add_sent_interactions(df, base_cols, sent_var="sent"):
    """
    Returns a *view* of df with extra columns named f"{sent_var}_{col}"
    equal to df[col] * df[sent_var].
    """
    for col in base_cols:
        if col == sent_var:          # skip "sent*sent"
            continue
        inter_name = f"{sent_var}_{col}"
        if inter_name not in df.columns:
            df[inter_name] = df[col] * df[sent_var]
    return df


def build_priors(df_hist, feature_cols, reward_col, shrink = 1):
    """
    Empirical-Bayes priors from pooled OLS with c=1:
      μ₀ = betahat_OLS,    Σ₀ = σ̂² (XᵀX)⁻¹,    σ² = σ̂²
    """
    # ensure intercept and float dtype
    X = df_hist[feature_cols].astype(float)
    y = df_hist[reward_col].astype(float)

    ols     = sm.OLS(y, X).fit()
    mu0     = ols.params.values          # β̂_OLS
    sigma2  = ols.mse_resid              # σ̂²
    Sigma0  = sigma2 * np.linalg.inv(X.T @ X)

    return mu0, shrink*Sigma0, sigma2