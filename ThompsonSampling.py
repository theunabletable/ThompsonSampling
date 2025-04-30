
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
    def __init__(self, priorMean, priorCov, featureNames, noiseVariance, actionType = 'sent'):
        self.currentMean = priorMean
        self.currentCov = priorCov
        self.featureNames = featureNames
        self.noiseVariance = noiseVariance
        self.actionType = actionType
        self.lastPredictedReward = 0

    #samples one set of coefficients from current posterior distribution
    def sampleCoefficients(self):
        return np.random.multivariate_normal(self.currentMean, self.currentCov)
    
    #given a context, action chosen, and a reward, performs a Bayesian update on currentMean and currentCov.
    def updatePosterior(self, context, action, reward):
        contextCopy = context.copy()
        contextCopy[self.actionType] = action
        #fix numpy being annoying
        contextCopy = contextCopy.astype(float)
        X = contextCopy[self.featureNames].to_numpy().reshape(1, -1)
       # print(contextCopy[self.featureNames].dtypes)
        
        priorPrecision = np.linalg.inv(self.currentCov)
        posteriorPrecision = priorPrecision + np.matmul(X.T, X)/self.noiseVariance
        
        first_term = np.matmul(priorPrecision, self.currentMean)
        second_term = (X.T * reward).reshape(-1)
        intermediateUpdate = first_term + second_term/self.noiseVariance
        
        posteriorCov = np.linalg.inv(posteriorPrecision)
        posteriorMean = np.matmul(posteriorCov, intermediateUpdate)
        
        self.currentMean = posteriorMean
        self.currentCov = posteriorCov
        
    #make a decision given a context vector
    def decide(self, context):
        sampledCoefficients = self.sampleCoefficients()
        
        rewardSend = self.predictReward(context, SEND, sampledCoefficients)
        rewardNoSend = self.predictReward(context, NO_SEND, sampledCoefficients)
        
        if rewardSend > rewardNoSend:
            self.lastPredictedReward = rewardSend
            return SEND
        else:
            self.lastPredictedReward = rewardNoSend
            return NO_SEND
    
    #outputs a predicted reward for action in context
    def predictReward(self, context, action, coefficients):
        contextCopy = context.copy()
        contextCopy[self.actionType] = action
        features = contextCopy[self.featureNames].to_numpy()
        return np.dot(coefficients, features)
    
    
    
    def probabilityOfSend(self, context):
        """
        Returns the probability that Thompson Sampler picks SEND under current posterior (self.currentMean, self.currentCov)
        
        With two actions:
            Delta = theta^T(x_send) - theta^T(x_nosend)
            = theta^T(x_send - x_nosend).
            
        So theta ~ N(mu, Sigma) with Delta univariate N(m, v)
        with m = (x_send - x_nosend)^T mu
             v = (x_send - x_nosend)^T Sigma (x_send - x_nosend)
        
        where we output P(Delta > 0) = F(m/sqrt(v)) where F is the standard normal cdf.
        """
        xSendCopy = context.copy()
        xNoSendCopy = context.copy()
        
        xSendCopy[self.actionType] = 1
        xSend = xSendCopy[self.featureNames].values
        xNoSendCopy[self.actionType] = 0
        xNoSend = xNoSendCopy[self.featureNames].values
        
        Delta = xSend - xNoSend
        
        meanDelta = np.matmul(Delta, self.currentMean)
        varDelta = np.matmul(Delta, np.matmul(self.currentCov, Delta))
        
        #potential edgecase
        if varDelta <= 1e-12:
            return 1.0 if meanDelta > 0 else 0.0
        
        #probability that Delta > 0
        z = meanDelta/np.sqrt(varDelta)
        return norm.cdf(z)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        