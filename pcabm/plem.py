import numpy as np
import pcabm.commFunc as cf
from scipy.optimize import minimize
from sklearn.metrics.cluster import adjusted_rand_score


class PLEM ():

    def __init__(self, Ab, Z, k, gamma=0) :
        self.n = Ab. shape[0]
        self.zdNodes = np. sum(Ab, 1) == 0
        self.Ab = Ab[np.ix_(~self.zdNodes, ~self.zdNodes)]
        self.Z = Z[np.ix_(~self.zdNodes,~self.zdNodes)][:]
        self.k = k
        self.eff_n = self.Ab. shape[0]

        if len(self.Z.shape)==2:
            self.p = 1
        else:
            self.p = self.Z.shape [2]
        if np.array(gamma).all():
            self.gamma = gamma
        else:
            self.gamma = minimize(cf.nLLGamma, np.zeros(self.p), args=(np.random.randint(self.k, size=self.eff_n), self.Ab, self.Z,), method='BFGS', options={'disp': False}, tol=10e-5).x

    def valid_assignment(self, community_init):
        uni_cnt = np.unique(community_init, return_counts=True)
        if len(uni_cnt[0])<self.k:
            return False
        if min(uni_cnt[1])<10:
            return False
        return True

    def fit(self, community_init = np.array([0]), gt = np.array ([]), em_max = 100, itr_num= 20, pl_diff = 1e-2):
        if self.valid_assignment(community_init):
            community_est = community_init[~self.zdNodes]
        else:
            community_est = np.random.randint(self.k, size=self.eff_n)

        # Compute initial pi_hat : K array
        pi = cf.num(community_est)/self.eff_n
        # Compute initial B_hat : K * K matrix
        Bhat = np.divide(cf.O(community_est, self.Ab), cf.E(community_est, self.gamma, self.Z))

        for t in range(itr_num) :
            delVec = np.zeros(em_max)
            em_steps = 0
            CONVERGED = False

            if not self.valid_assignment(community_est):
                community_est = np.random.randint (self.k, size=self.eff_n)

            # Compute initial Xi and Bb: both n * K matrix
            Xi = cf.xi(community_est, self.gamma, self.Z)
            Bb = cf.b(community_est, self.Ab)

            # EM algo
            while em_steps<em_max and ~CONVERGED:
                # compute intermediate matrix
                Bhat = np.clip(Bhat,a_min = 1e-10, a_max = 1e300)
                interMat = -np.matmul(Bhat, Xi.T) + np.matmul(np.log(Bhat) , Bb.T) # K * n matrix
                interMat_mean = np.mean(interMat,0) # n array
                interMat -= interMat_mean # demean
                interMat = np.clip(interMat, a_min = -800, a_max = 700)

                # E step
                nom = np.multiply(np.tile(pi, (self.eff_n, 1)).T, np.exp(interMat)) # K * n matrix
                post_denom = np.sum(nom, 0) # n array
                nom = np.divide(nom, np.tile(post_denom, (self.k,1))) #pi_{li}hat

                # M step
                pi = np.mean(nom, 1)
                d = np.matmul(nom, Xi)
                Xi = np.clip(Xi, a_min = 1, a_max = np.inf)
                Bhat = np.divide (np.matmul(nom, Bb), np.matmul(nom, Xi))
                Bhat = np.clip(Bhat,a_min = 1e-10, a_max = 1e300)
                plVal = np.sum(np.multiply(nom, np.tile(pi, (self.eff_n, 1)).T) - np.matmul(Bhat, Xi.T) + np.matmul(np.log(Bhat) , Bb.T))

                # community estimation
                community_est = np.argmax(nom, axis = 0)

                if em_steps > 0:
                    delta = abs( (plVal-plValOld)/plValOld)
                    CONVERGED = delta < pl_diff
                    delVec[em_steps-1] = delta
                plValOld = plVal
                em_steps +=1

        # Label disconnected nodes as -1
        community_res = np.array([-1]*self.n)
        community_res[~self.zdNodes] = community_est
        return(community_res,plVal)