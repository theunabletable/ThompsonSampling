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
    A wrapper over ThompsonSampling that draws its posterior from a shared RoMEAgentManager.
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
        feature_names = manager.featureCols
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



    def _advantage_stats(self, context: pd.Series) -> tuple[float, float]:
        """
        Mean and variance of the treatment advantage SEND − NO_SEND.

        Notes
        -----
        Uses the manager's cached dense inverse to fetch only the
        32×32 sub-matrix required for this user & feature set.
        """
        mgr          = self.manager

        # --------- build sparse (1 × d) difference row
        phi_send_vec = mgr.make_phi(self.pid, context, SEND)
        phi_no_vec   = mgr.make_phi(self.pid, context, NO_SEND)
        phi_vec      = (phi_send_vec - phi_no_vec).tocsc()

        nz_idx       = phi_vec.indices           # 2p indices
        phi_values   = phi_vec.data

        # --------- mean  μ = φ θ_est
        mu_adv  = float(phi_values @ mgr._theta_cache[nz_idx])

        # --------- variance  σ² = φ V_inv φᵀ
        Vinv_sub  = mgr.V_inv[np.ix_(nz_idx, nz_idx)]
        base_var  = float(phi_values @ (Vinv_sub @ phi_values))

        beta_i    = mgr._beta_cache[self.pid]
        var_adv   = base_var * (beta_i / mgr.first_beta) ** 2
        return mu_adv, var_adv

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
    
"""    
    #advantage features are phi(s, i) = phi(s, SEND, i) - phi(s, NO_SEND, i) and is the set of features which are different under send vs no_send
    #advantage mean is the expected treatment effect of sending, and ths variance is needed for the advantage's distribution
    def _advantage_stats(self, context: pd.Series):
#        returns (mean, var) of advantage (send - no_send) 
        m              = self.manager
        theta_hat      = m.V_chol(m.b)                            #(1 + N)p vector
        #to be clear, V_chol(m.b) does -not- multiply V_chol by m.b. It uses the cholmod factor object's solve method, to solve Vx = b for x, that is, getting x = V^-1 b as desired.
        phi_send       = m.make_phi(self.pid, context, SEND)
        phi_no         = m.make_phi(self.pid, context, NO_SEND)
        phi_adv        = (phi_send - phi_no).tocsc()              #sparse 1xd
        phi_adv_T      = phi_adv.T.tocsc()
        #advantage mean, the treatment effect
        adv_mean = float(phi_adv @ theta_hat)
        
        #beta inflation factor
        C_i, C_i_T    = m._build_C_i(self.pid)
        Vi_Ct         = m.V_chol(C_i_T)
        V_bar         = (C_i @ Vi_Ct).toarray()
        Lambda        = (Vi_Ct.T @ m.V0 @ Vi_Ct).toarray()
        
        #caching beta saves lots of time in testing
        beta = m._beta_cache[self.pid]
        
        #variance phi V^-1 phi^T
        base_var = (phi_adv @ m.V_chol(phi_adv_T)).A[0, 0]
        adv_var = base_var * (beta/m.first_beta) ** 2
        return adv_mean, adv_var
"""