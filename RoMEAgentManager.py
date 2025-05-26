# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:56:00 2025

@author: Drew
"""

from AgentManager import AgentManager
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky
import scipy.sparse
from RoMEThompsonSampling import RoMEThompsonSampling
import pandas as pd


NO_SEND = 0
SEND = 1
class RoMEAgentManager(AgentManager):
    def __init__(self,
                 participants: list[str],
                 L_user: sparse.csr_matrix,
                 gamma_ridge: float,
                 lambda_penalty: float,
                 baseFeatureCols: list[str],          # ← main effects
                 interactionFeatureCols: list[str],   # ← those that get action interaction
                 featureCols: list[str],              # ← full x_d order (len = p)
                 rewardName: str,
                 actionName: str = "sent",
                 v: float = 1.0,
                 delta: float = 0.01,
                 zeta: float = 10.0):
        
        #features
        self.baseFeatureCols_d         = baseFeatureCols
        self.interactionFeatureNames_d = interactionFeatureCols
        self.featureCols               = featureCols   # this defines p
        self.actionName                = actionName
        self.rewardName                = rewardName
        
        #core dimensions
        self.participants = participants      #list of participants
        self.pid_to_idx   = {pid: i for i, pid in enumerate(self.participants)}  #have to make sure this targets the participants correctly.
        self.N            = len(participants) #number of participants
        self.K            = 1 + self.N        #number of blocks in V matrix
        self.p            = len(featureCols)
        
        #constants
        self.v          = v          #mysterious constants abound
        self.delta      = delta
        self.zeta       = zeta
        self.beta_const = self.zeta * max(1.0, np.log(self.K)**0.75)
        
        #user graph and ridge
        self.L_user        = L_user       
        self.gamma_ridge = gamma_ridge
        self.lambda_penalty = lambda_penalty
        

        #build priors
        self.V0 = self._initialize_V0()  #shape ((1+N)*p) x ((1 + N)*p)
        self.V = self.V0.copy()
        self.b = np.zeros((1 + self.N)*self.p, dtype = float)

        #cholesky factor
        self.V_chol = cholesky(self.V)
        
        self.first_beta = (
            self.v * np.sqrt(2 * np.log(2 * self.K * (self.K + 1) / self.delta))
            + self.beta_const
        )
        #cache for selector matrices C_i
        self._C_cache = {}
        self._beta_cache = {}
        self._refresh_beta_cache()
        self.agents = self._buildAgents()
        self.V_inv            : np.ndarray | None = None   # set by _rebuild_dense
        self._theta_cache : np.ndarray | None = None
        
    
    def _rebuild_dense_inverse(self) -> None:
        """Refresh `V_inv`, `_theta_cache` and `_beta_cache`."""
        dense_V        = self.V.toarray()
        self.V_inv     = np.linalg.inv(dense_V)
        self._theta_cache = self.V_inv @ self.b
        self._refresh_beta_cache_dense()

    # 3) dense β-cache (all ascii)
    def _refresh_beta_cache_dense(self) -> None:
        """Cheap closed-form β_i using blocks of `V_inv`."""
        p          = self.p
        log_term   = 2 * np.log(2*self.K*(self.K+1) / self.delta)
        V_shared   = self.V_inv[:p, :p]
        beta_dict  = {}
        for i, pid in enumerate(self.participants):
            blk   = p + i*p
            V_user= self.V_inv[blk:blk+p, blk:blk+p]
            V_su  = self.V_inv[:p, blk:blk+p]
            V_bar = V_shared + V_su + V_su.T + V_user
            V_iCt = np.vstack((V_shared+V_su, V_su.T+V_user))
            lambda_mat = self.gamma_ridge * (V_iCt.T @ V_iCt)
            beta_val = ( self.v *
                         np.sqrt(log_term
                                 + np.linalg.slogdet(V_bar)[1]
                                 - np.linalg.slogdet(lambda_mat)[1])
                         + self.beta_const )
            beta_dict[pid] = float(beta_val)
        self._beta_cache = beta_dict

    def _initialize_V0(self) -> sparse.csc_matrix:
        """
        V0 = block_diag([V_shared, V_user]) where:

          V_shared = shared_precision * I_p
          V_user   = kron(I_N, user_precision) + laplacian_strength * kron(L_user, I_p)
        """
        N = self.N
        p = self.p
        I_p = sparse.identity(p, format = "csc", dtype = float)
        
        #shared block shape pxp
        V_shared = self.gamma_ridge * I_p # γ · I_p
        
        
        ridge_block = self.gamma_ridge * sparse.kron(
        sparse.identity(N, format="csc"), I_p      # γ · I_{N·p}
    )
        ##ADD GAMMA IN HERE
        
        
        laplace_block = sparse.kron(self.L_user, I_p) * self.lambda_penalty # λ · (L_user ⊗ I_p)
        
        V_user = ridge_block + laplace_block
        
        return sparse.block_diag([V_shared, V_user], format="csc")
    
    def _build_C_i(self, pid:str) -> csc_matrix:
        """
        Construct the selector matrix C_i for a participant:
        C_i is p x ((1+N)*p) with ones selecting shared and user-i blocks.
        """        
        if pid in self._C_cache:
            return self._C_cache[pid]
        
        idx = self.pid_to_idx[pid]
        p = self.p
        N = self.N
        total_dim = (1 + N)*p
        
        
        #each of the p rows will have two ones
        C_i = scipy.sparse.lil_matrix((p, total_dim), dtype=float)
        
        #shared block: columns 0 to p-1
        for row in range(p):
            C_i[row, row] = 1.0
        
        #user block: columns p + idx*p to p + (idx+1)*p - 1
        offset = p + idx * p
        for row in range(p):
            C_i[row, offset + row] = 1.0
        
        #convert to CSC for efficient arithmetic and cache it
        C_i = C_i.tocsc()
        C_i_T = C_i.T.tocsc()
        self._C_cache[pid] = (C_i, C_i_T)
        return self._C_cache[pid]
    
    def compute_covariance_block(self, pid: str) -> np.ndarray:
        """
        Compute Sigma_i = C_i V^{-1} C_i^T for participant pid.
        Uses the CHOLMOD factor's call operator directly.
        """
        #selector matrix for this user
        C_i, C_i_T = self._build_C_i(pid)  # shape: (p, total_dim)

        #solve V * Vi_Ct = C_i^T  =>  Vi_Ct = V^{-1} * C_i^T
        Vi_Ct = self.V_chol(C_i_T)  # shape: (total_dim, p)

        #form the p×p covariance block: C_i @ (V^{-1} C_i^T)
        Sigma_i = (C_i @ Vi_Ct).toarray()
        return Sigma_i

    def get_parameters_for_pid(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mu_i, Sigma_i) for a given pid:
        mu_i := shared + user block from V^{-1}b
        Sigma_i from compute_covariance_block
        """
        #global posterior mean
        theta_hat = self.V_chol(self.b)  # length (1+N)*p

        #extract per-user effect
        p   = self.p
        idx = self.pid_to_idx[pid]
        shared_part = theta_hat[0 : p]
        user_part   = theta_hat[p + idx*p : p + (idx+1)*p]
        mu_i        = shared_part + user_part

        #full covariance for user i
        Sigma_i = self.compute_covariance_block(pid)
        return mu_i, Sigma_i
    
    def _buildAgents(self) -> dict[str, RoMEThompsonSampling]:
        agents = {}
        for pid in self.participants:
            agents[pid] = RoMEThompsonSampling(pid, manager=self)
            
        return agents
    

    #This is needed to ensure that interaction terms function correctly
    def _transform_context_to_x_d(
            self,
            context: pd.Series,
            action: int,
            base_cols: list[str],
            inter_cols: list[str],
            full_cols: list[str],
            action_col: str
        ) -> np.ndarray:
        """
        Build the p‑dim. feature vector x_d(s,a).
    
        Rules:
          • Every base column       → its raw value                               (main effect)
          • 'intercept'            → 1.0
          • action column          → float(action)
          • Each interaction col   →   value(base) * action   if it is
                                      of the form  f"{action_col}_{base}"
                                      and 'base' is in base_cols.
        """
        x_dict = {}
    
        # main effects
        for col in base_cols:
            x_dict[col] = float(context[col])
    
        # intercept & action indicator
        x_dict["intercept"]   = 1.0
        x_dict[action_col]    = float(action)
    
        # interactions
        for inter in inter_cols:
            base_var = inter.split(f"{action_col}_", 1)[1]
            x_dict[inter] = float(action) * x_dict.get(base_var, 0.0)
    
        # assemble in prescribed order
        return np.array([x_dict.get(col, 0.0) for col in full_cols], dtype=float)
    
    #this phi_i = [x_1, ... x_p (shared block) | 0, ..., 0 | ... |x_1, ..., x_p (user i block) | 0, ..., 0]
    #it's set up so that when we do phi_i^T \theta_global we get [x_1, ..., x_p]^T \theta_shared + [x_1, ..., x_p]^T \theta_i
    def _construct_phi_vector(self, x_d: np.ndarray, pid: str) -> csc_matrix:
        """
        Place x_d into the shared and user‑pid blocks.
        """
        idx       = self.pid_to_idx[pid]
        total_dim = (1 + self.N) * self.p
    
        rows = np.zeros(2 * self.p, dtype=int)                       # repeated rows
        cols_shared = np.arange(self.p)
        cols_user   = self.p + idx * self.p + np.arange(self.p)
        cols = np.concatenate([cols_shared, cols_user])
    
        data = np.concatenate([x_d, x_d])
    
        return csc_matrix((data, (rows, cols)), shape=(1, total_dim), dtype=float)
    
    #this helper useful for the decision function, builds phi from raw context + action
    def make_phi(self, pid: str, context: pd.Series, action: int) -> csc_matrix:
        """
        Convenience wrapper: context,row  ➜  φ.
        """
        x_d = self._transform_context_to_x_d(
            context,
            action,
            self.baseFeatureCols_d,
            self.interactionFeatureNames_d,
            self.featureCols,
            self.actionName,
        )
        return self._construct_phi_vector(x_d, pid)

    
    def update_posteriors(self, batch: pd.DataFrame) -> None:
        """
        Roll an IPW batch into the global posterior.
    
        Parameters
        ----------
        batch : pd.DataFrame
            Must contain columns
    
            * PARTICIPANTIDENTIFIER
            * ``self.actionName``  (0 ≡ NO_SEND, 1 ≡ SEND)
            * ``self.rewardName``  (numeric reward)
            * feature columns required by :py:meth:`make_phi`
            * optionally ``pi_no_send`` logged at decision time.
    
        Notes
        -----
        Steps
        1.  Build design matrix ``Phi`` (n × d), weight vector ``w`` and
            pseudo-rewards ``y`` in one pass over the rows.
        2.  Low-rank update::
    
                V_new = V + Phi.T @ diag(w) @ Phi
                b_new = b + Phi.T @ y
    
            Both ``V`` and its sparse Cholesky factor are refreshed,
            then a dense inverse is rebuilt for fast agent queries.
        3.  Per-agent caches (`currentMean`, `currentCov`) are invalidated.
        """
        # --- refresh dense inverse & β-cache for current V -------------
        self._rebuild_dense_inverse()
        if batch.empty:
            return
    
        # --- 1)  build Phi, w, y ---------------------------------------
        row_idx, col_idx, values = [], [], []
        w_list, y_list = [], []
    
        for k, (_, row) in enumerate(batch.iterrows()):
            pid   = row["PARTICIPANTIDENTIFIER"]
            act   = int(row[self.actionName])
            rew   = float(row[self.rewardName])
    
            # behaviour policy  π(i,t)  (probability of NO_SEND)
            pi_no = (row.get("pi_no_send")
                     or 1.0 - self.agents[pid].probabilityOfSend(row))
    
            weight      = pi_no * (1.0 - pi_no)                # σ̃²
            denom       = (1.0 - pi_no) if act == SEND else -pi_no
            reward_tilt = rew / denom                          # r̃
    
            phi_row = self.make_phi(pid, row, act).tocoo()
            row_idx.extend(np.full_like(phi_row.row, k))
            col_idx.extend(phi_row.col)
            values.extend(phi_row.data)
    
            w_list.append(weight)
            y_list.append(weight * reward_tilt)
    
        n_rows = len(batch)
        dim    = (1 + self.N) * self.p
    
        Phi = sparse.coo_matrix(
            (values, (row_idx, col_idx)),
            shape=(n_rows, dim)
        ).tocsr()
        W_diag = sparse.diags(w_list)
        y_vec  = np.asarray(y_list)
    
        # --- 2)  precision & mean updates ------------------------------
        V_delta = Phi.T @ (W_diag @ Phi)
        b_delta = (Phi.T @ y_vec).ravel()
    
        self.V += V_delta.tocsc()
        self.b += b_delta
        self.V_chol = cholesky(self.V)          # ~6 s on 14k × 14k
    
        # rebuild dense inverse + β-cache (<< 1 s)
        #self._rebuild_dense_inverse()
    
        # --- 3)  clear per-agent caches --------------------------------
        self._C_cache.clear()
        for agent in self.agents.values():
            agent.currentMean = None
            agent.currentCov  = None



    
    #these have to be added because CHOLMOD can't be pickled
    def __getstate__(self):
        """
        On pickling: drop the CHOLMOD factor (not pickleable).
        It is cheap to recompute from V on reload.
        """
        state = self.__dict__.copy()
        state["V_chol"] = None          # remove factor
        return state
    
    def __setstate__(self, state):
        """
        On unpickling: restore instance dict and rebuild factor.
        """
        from sksparse.cholmod import cholesky   # local import avoids hard dep at top
        self.__dict__.update(state)
        # Rebuild sparse Cholesky factor in-place
        self.V_chol = cholesky(self.V)
        if "_beta_cache" not in self.__dict__:
            self._beta_cache = {}
        if not self._beta_cache:                # build if empty / stale
            self._refresh_beta_cache()
    
    # Optional convenience wrappers ------------------------------------------
    def save(self, path):
        """Pickle self → *path* (now safe)."""
        import pickle, gzip, pathlib
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") if path.suffix == ".gz" else open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Inverse of save()."""
        import pickle, gzip, pathlib
        path = pathlib.Path(path)
        with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    

    def _refresh_beta_cache(self) -> None:
        """Re-compute β_i for every participant given current V_chol."""
        self._beta_cache.clear()
        log_term = 2 * np.log(2 * self.K * (self.K + 1) / self.delta)
        for pid in self.participants:
            C_i, C_i_T = self._build_C_i(pid)
            Vi_Ct      = self.V_chol(C_i_T)           # solve   V X = C_iᵀ
            V_bar      = (C_i @ Vi_Ct).toarray()
            Lambda     = (Vi_Ct.T @ self.V0 @ Vi_Ct).toarray()
            beta_i = (
                self.v * np.sqrt(log_term
                                 + np.linalg.slogdet(V_bar)[1]
                                 - np.linalg.slogdet(Lambda)[1])
                + self.beta_const
            )
            self._beta_cache[pid] = float(beta_i)



'''
    def update_posteriors(self, df_batch: pd.DataFrame) -> None:
        if df_batch.empty:
            return
    
        # ---------- 1) build Φ and weights in *one* pass -----------------
        rows, cols, data = [], [], []
        w_vec  = []          # σ̃²  (length n)
        y_vec  = []          # σ̃² · r̃
    
        for k, (_, row) in enumerate(df_batch.iterrows()):
            pid    = row['PARTICIPANTIDENTIFIER']
            action = int(row[self.actionName])
            reward = float(row[self.rewardName])
    
            # ----- π, σ̃², r̃ -------------------------------------------
            if 'pi_no_send' in row:
                pi_no = float(row['pi_no_send'])
            else:
                pi_no = self.agents[pid].probabilityOfSend(row)
                pi_no = 1.0 - pi_no
    
            sigma_tilde2 = pi_no * (1.0 - pi_no)
            denom        = (1.0 - pi_no) if action == SEND else -pi_no
            r_tilde      = reward / denom
    
            # ----- Φ row ------------------------------------------------
            phi          = self.make_phi(pid, row, action).tocoo()
            rows.extend( np.full_like(phi.row, k) )
            cols.extend( phi.col )
            data.extend( phi.data )
    
            w_vec.append( sigma_tilde2 )
            y_vec.append( sigma_tilde2 * r_tilde )
    
        n   = len(df_batch)
        d   = (1 + self.N) * self.p
        Phi = sparse.coo_matrix((data, (rows, cols)), shape=(n, d)).tocsr()
    
        W   = sparse.diags(w_vec)              # n×n
        y   = np.array(y_vec)                  # length n
    
        # ---------- 2) accumulate V and b  ------------------------------
        #   V ← V + Φᵀ W Φ
        #   b ← b + Φᵀ y
        # the first is low-rank (≤ n·p non-zeros), so use update_inplace
        V_delta = Phi.T @ (W @ Phi)
        b_delta = (Phi.T @ y).ravel()              # .A1 = 1-D ndarray
    
        # sparse.csc_matrix ensures CSC for CHOLMOD
        V_delta = V_delta.tocsc()
        self.V  = self.V + V_delta            # keep full Matrix for later
    
        # ---- 3) Cholesky rank-k update ---------------------------------
        # CHOLMOD’s update_inplace needs L *L^T  +=  A  (here V_delta)
        self.V_chol.update_inplace(V_delta)
    
        # ---- 4) mean term ----------------------------------------------
        self.b += b_delta
    
        # ---- 5) reset caches so agents refresh on next call ------------
        self._C_cache.clear()
        for ag in self.agents.values():
            ag.currentMean = None
            ag.currentCov  = None




    def update_posteriors(self, df_batch: pd.DataFrame) -> None:
#        """
#        One-pass IPW update using CHOLMOD's rank-1 update_inplace.
#    
#        df_batch must have columns
#            PARTICIPANTIDENTIFIER, self.actionName, self.rewardName
#            + all feature columns needed by _transform_context_to_x_d
#        and optionally 'pi_no_send' (logged π_{i,t} on the NO_SEND arm).
#        """
        for _, row in df_batch.iterrows():
            pid    = row['PARTICIPANTIDENTIFIER']
            action = int(row[self.actionName])          # 0 / 1
            reward = float(row[self.rewardName])
    
            # ---------- behaviour π_i,t (prob NO_SEND) ----------
            if 'pi_no_send' in row:
                pi_no = float(row['pi_no_send'])
            else:
                pi_no = 1.0 - self.agents[pid].probabilityOfSend(row)
    
            sigma_tilde2 = pi_no * (1.0 - pi_no)        # weight factor
            denom = (1.0 - pi_no) if action == SEND else -pi_no
            r_tilde = reward / denom                    # IPW pseudo-reward
    
            # ---------- feature vector --------------------------
            phi = self.make_phi(pid, row, action).tocsr()   # 1 × d
            w   = np.sqrt(sigma_tilde2)
            phi_w = (w * phi).tocsr()                      # scale *data* only
    
            # ---------- rank-1  V  ←  V + φ_wᵀ φ_w -------------
            #   update_inplace expects CSC & (d × 1)
            self.V_chol.update_inplace(phi_w.T.tocsc())
    
            # ---------- b  ←  b + σ̃² · r̃ · φᵀ -----------------
            self.b += sigma_tilde2 * r_tilde * phi.T.toarray().ravel()     #flat ndarray
    
        # Nothing else to do:  self.V_chol already current.
        # If you still keep an explicit V for inspection, update it here:
        #     self.V += phi_w.T @ phi_w   (inside the loop or accumulate)
        self._C_cache.clear()               # invalidate cached C_i blocks
        for ag in self.agents.values():     # force posterior refresh
            ag.currentMean = None
            ag.currentCov  = None

'''            

'''            
    def update_posteriors(self, df_batch: pd.DataFrame) -> None:
#        """
#        Update V and b using a batch of logged rows.  Assumes df_batch
#        has at least columns:
#            PARTICIPANTIDENTIFIER,  actionName,  rewardName,
#            and all feature columns needed by _transform_context_to_x_d.
#        Also assumes π (prob_no_send) was logged at decision time in
#            column 'pi_no_send'; if not present we recompute it.
#        """
        V_add = sparse.csc_matrix(( (1+self.N)*self.p, (1+self.N)*self.p ), dtype=float)
        b_add = np.zeros( (1+self.N)*self.p )
        for _, row in df_batch.iterrows():
            pid    = row['PARTICIPANTIDENTIFIER']
            action = int(row[self.actionName])           # 0 or 1
            reward = float(row[self.rewardName])
    
            # π_{i,t} (prob NO_SEND).  Prefer logged value, else recompute.
            if 'pi_no_send' in row:
                pi_no = float(row['pi_no_send'])
            else:
                pi_no = self.agents[pid].probabilityOfSend(row)
                pi_no = 1.0 - pi_no                      # convert to NO_SEND prob
    
            sigma_tilde2 = pi_no * (1.0 - pi_no)         # weight
            if action == SEND:
                denom = 1.0 - pi_no
            else:  # NO_SEND
                denom = -pi_no                           # (a - π)
    
            r_tilde = reward / denom                     # IPW pseudo‑reward
    
            #φ vector for the taken action
            phi = self.make_phi(pid, row, action)        # 1 × d  (CSR)
    
            #accumulate
            V_add += (sigma_tilde2 * (phi.T @ phi)).tocsc()
            b_add += sigma_tilde2 * r_tilde * phi.T.A.ravel()
    
        #update global posterior
        self.V += V_add
        self.b += b_add
        self.V_chol = cholesky(self.V)   # fresh factor
    
        self._C_cache.clear()
        for ag in self.agents.values():
            ag.currentMean = None        # force refresh
            ag.currentCov  = None
            
'''