# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:15:40 2025

@author: Drew
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from ThompsonSampling import ThompsonSampling

NO_SEND = 0
SEND = 1
class RoMEThompsonSampling(ThompsonSampling):
    """
    A thin wrapper over ThompsonSampling that draws its posterior from a shared RoMEAgentManager.
    Each instance holds only a pid and reference to the manager; all heavy lifting stays in the manager.
    """
    def __init__(
        self,
        pid: str,
        manager,
    ):
        """
        Args:
            pid: participant identifier
            manager: instance of RoMEAgentManager providing global posterior
        """
        #dimension of feature vector x(s, a)
        p = manager.p
        #dummy priors
        dummy_mean = np.zeros(p)
        dummy_cov = np.eye(p)
        
        base_features = manager.baseFeatureCols_d
        interaction_features = manager.interactionFeatureNames_d
        feature_names = manager.featureNames_x_d
        action_type = manager.actionName
        
        self.pi_min = 0.2 #minimum no_send probability
        self.pi_max = 0.8 #maximum no_send probability

        super().__init__(
            priorMean              = dummy_mean,
            priorCov               = dummy_cov,
            baseFeatureNames       = base_features,
            interactionFeatureNames= interaction_features,
            featureNames           = feature_names,
            noiseVariance          = 1.0,
            actionType             = action_type
        )
        self.pid = pid
        self.manager = manager

    def _refresh_posterior(self):
        """
        Fetch the current posterior mean & covariance for this pid from the manager.
        """
        mu_i, Sigma_i = self.manager.get_parameters_for_pid(self.pid)
        # override ThompsonSampling's fields
        self.currentMean = mu_i
        self.currentCov  = Sigma_i



    def updatePosterior(self, context, action, reward):
        """
        RoME updates happen globally in the manager—prevent individual updates.
        """
        raise NotImplementedError("RoMEThompsonSampling updates via its manager, not individually.")
        
    #advantage features are phi(s, i) = phi(s, SEND, i) - phi(s, NO_SEND, i) and is the set of features which are different under send vs no_send
    #advantage mean is the expected treatment effect of sending, and ths variance is needed for the advantage's distribution
    def _advantage_stats(self, context: pd.Series):
        """returns (mean, var) of advantage (send - no_send) """
        m = self.manager
        theta_hat = m.V_chol(m.b) #(1 + N)p vector
        phi_send = m.make_phi(self.pid, context, SEND)
        phi_no = m.make_phi(self.pid, context, NO_SEND)
        phi_adv = phi_send - phi_no #sparse 1xd
        
        #advantage mean, the treatment effect
        adv_mean = float(phi_adv @ theta_hat)
        
        #beta inflation factor
        C_i = m._build_C_i(self.pid)
        Vi_Ct = m.V_chol(C_i.T)
        V_bar   = (C_i @ Vi_Ct).toarray()
        Lambda  = (Vi_Ct.T @ m.V0 @ Vi_Ct).toarray()
        beta = m.v * np.sqrt(
            2 * np.log(2*m.K * (m.K + 1) / m.delta)
            + np.linalg.slogdet(V_bar)[1]   #why slogdet again??
            - np.linalg.slogdet(Lambda)[1]
            ) + m.beta_const
        
        #variance phi V^-1 phi^T
        base_var = (phi_adv @ m.V_chol(phi_adv.T)).A[0, 0]
        adv_var = base_var * (beta/m.first_beta) ** 2
        return adv_mean, adv_var

    def _clip(self, p):
        '''clipping helper'''
        return min(self.pi_max, max(self.pi_min, p))
    
    
    #advantage has a Gaussian A ~ N(adv_mean, adv_var)
    #P(adv > 0) = cdf(adv_mean / sqrt(adv_var))
    def probabilityOfSend(self, context: pd.Series) -> float:
        mean, var = self._advantage_stats(context)
        sd = np.sqrt(max(var, 1e-12))
        
        prob_no_send = self._clip(norm.cdf(-mean / sd))
        return 1.0 - prob_no_send
    
    def decide(self, context: pd.Series) -> int:
        self._refresh_posterior()
        mean, var = self._advantage_stats(context)
        sd = np.sqrt(max(var, 1e-12))
        
        prob_no_send = self._clip(norm.cdf(-mean / sd)) #P(adv < 0)
        prob_no_send = min(self.pi_max, max(self.pi_min, prob_no_send))
        prob_send = 1.0 - prob_no_send
        action = NO_SEND if np.random.uniform() < prob_no_send else SEND
        return action