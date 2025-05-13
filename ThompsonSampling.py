
r"""
Provides functionality for Thompson Sampling algorithm, under priors from an OLS Linear Regression.

The action-space is still currently just SEND or NO_SEND corresponding in our dataset to "sent" = 1 or "sent" = 0.

sampleCoefficients samples coefficients that represent its current "prediction function".

decide(context) is given a single context (a row), and uses its sampled coefficients to predict the reward
that would be given to action SEND and NO_SEND, then outputs the action with the highest predicted reward.

updatePosterior takes in the above context, the action chosen, and the actual reward received, and updates
the currentMean and currentCov according to the formulas for a conjugate-Gaussian update over a normal prior:

\Sigma^-1_{post} = \Sigma^-1_{prior} + (1/\sigma^2) X^T X,
\mu_post = \Sigma_post(\Sigma^-1_prior \mu_prior + (1/\sigma^2) X^T r)
where \Sigma_pre is the prior covariant matrix, \sigma is the standard error of the regression model,
X is the data matrix, and r is the reward.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
SEND = 1
NO_SEND = 0

class ThompsonSampling:
    def __init__(self, priorMean, priorCov, baseFeatureNames, interactionFeatureNames, featureNames, noiseVariance, actionType = 'sent'):
        self.currentMean = priorMean
        self.currentCov = priorCov
        self.baseFeatureNames = baseFeatureNames
        self.interactionFeatureNames = interactionFeatureNames
        self.featureNames = featureNames
        self.noiseVariance = noiseVariance
        self.actionType = actionType
        self.lastPredictedReward = 0

    #samples one set of coefficients from current posterior distribution
    def sampleCoefficients(self):
        return np.random.multivariate_normal(self.currentMean, self.currentCov)
    
    def _build_x(self, context, action) -> np.ndarray:
        # build your x dict…
        base = {c: float(context[c]) for c in self.baseFeatureNames}
        x = {
          **base,
          self.actionType: action,
          "intercept": 1.0,
        }
        for col in self.interactionFeatureNames:
            base_var = col.split(f"{self.actionType}_",1)[1]
            x[col] = action * base[base_var]
        
        # --- debug: find any non‐scalar entries ---
        for f, val in x.items():
            # numpy scalars count as scalar, but lists/arrays do not
            if not np.isscalar(val):
                print("⚠️ _build_x got non‐scalar for", f, ":", val, "\n")
        
        return np.array([ x[f] for f in self.featureNames ], dtype=float)

    def updatePosterior(self, context, action, reward):
        """
        Given a context, action, and reward, perform a Bayesian update on currentMean and currentCov.
        """
        x_vec = self._build_x(context, action).reshape(1, -1)
        # Σ⁻¹_prior + (1/σ²) XᵀX
        priorPrec = np.linalg.inv(self.currentCov)
        postPrec = priorPrec + (x_vec.T @ x_vec) / self.noiseVariance
        # Σ_post (Σ⁻¹_prior μ_prior + (1/σ²) Xᵀ r)
        first_term = priorPrec @ self.currentMean
        second_term = (x_vec.T * reward).reshape(-1)
        inter = first_term + second_term / self.noiseVariance
        postCov = np.linalg.inv(postPrec)
        postMean = postCov @ inter
        self.currentMean = postMean
        self.currentCov = postCov

    def decide(self, context):
        """
        Sample coefficients, predict reward for SEND and NO_SEND,
        and return the action with the higher prediction.
        """
        theta = self.sampleCoefficients()
        x_send = self._build_x(context, 1)
        x_no = self._build_x(context, 0)
        rewardSend = theta.dot(x_send)
        rewardNoSend = theta.dot(x_no)
        if rewardSend > rewardNoSend:
            self.lastPredictedReward = rewardSend
            return 1
        else:
            self.lastPredictedReward = rewardNoSend
            return 0

    def predictReward(self, context, action, coefficients):
        """
        Return dot(coefficients, x) for a given action in context.
        """
        x_vec = self._build_x(context, action)
        return np.dot(coefficients, x_vec)

    def probabilityOfSend(self, context):
        """
        Returns P(reward_send > reward_no_send) under the current posterior.
        """
        x_send = self._build_x(context, 1)
        x_no   = self._build_x(context, 0)
        Delta = x_send - x_no
        meanDelta = Delta.dot(self.currentMean)
        varDelta = Delta.dot(self.currentCov.dot(Delta))
        if varDelta <= 1e-12:
            return 1.0 if meanDelta > 0 else 0.0
        z = meanDelta / np.sqrt(varDelta)
        return norm.cdf(z)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        