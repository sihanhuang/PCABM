import numpy as np
import pcabm.commFunc as cf
from scipy.optimize import minimize
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.sparse.linalg import svds
from numpy import linalg as LA
from sklearn.cluster import KMeans
import pcabm.sc as sc
import inspect


class chooseK():
    def __init__(self, Ab, Z, k, gamma = 0):
        self.n = Ab.shape[0]
        self.zdNodes = np.sum(Ab, 1) == 0

        self.Ab = Ab[np.ix_(~self.zdNodes, ~self.zdNodes)]
        self.Z = Z[np.ix_(~self.zdNodes,~self.zdNodes)][:]
        self.k = k
        self.eff_n = self.Ab.shape[0]

        if len(self.Z.shape)==2:
            self.p = 1
        else:
            self.p = self.Z. shape[2]
        if np.array(gamma).all():
            self.gamma = gamma
        else:
            self.gamma = minimize(cf.nLLGamma,np.zeros(self.p), args=(np.random.randint(self.k, size=self.eff_n), self.Ab, self.Z,), method='BFGS', options={'disp': False}, tol=10e-5).x

        self.expZ = np.exp(np.dot(self.Z,self.gamma))
        self.A1 = np.divide(self.Ab, self.expZ)

    def fit(self, Nrep = 3, Kmax = 6, p_subsam = 0.9) :
        self.LKm_lik = np.zeros((Kmax, Nrep));
        self.LKm_lik_scaled = np.zeros((Kmax, Nrep));
        self.LKm_se = np.zeros((Kmax, Nrep));

        for m in range(Nrep):
            subOmega = np.random.binomial(1,p_subsam, size= (self.eff_n,self.eff_n));
            test_size = self.eff_n**2 - np.sum(subOmega);

            subsam_A1 = np.multiply(self.A1, subOmega);
            subsam_Ab = np.multiply(self.Ab, subOmega);
            subsam_expZ = np.multiply(self.expZ, subOmega);

            U, S, V = svds (0.0+subsam_A1 / self.p, Kmax)

            for k in range(1, Kmax+1):
                Ahat_k = np.matmul(np.matmul(U[:, -k:], np.diag(S[-k:])), V[-k:,:]) # n * n matrix
                eKm, _ = sc.SCWA_r(Ahat_k, np.zeros(shape = (self.eff_n,self.eff_n,1)) , k) 
                comm_est = eKm.labels_
                Oll = np.clip(cf.O(comm_est, subsam_Ab), a_min = 1, a_max = 1e300 )
                Ell = np.clip(cf.O(comm_est, subsam_expZ), a_min = 1, a_max = 1e300 )
                Bll = np.divide(Oll, Ell)
                EA_hat = np.multiply(Bll[np.ix_(comm_est, comm_est)], self.expZ) # n * n matrix

                self.LKm_lik[k-1][m] = np.sum(np.multiply(self.Ab-subsam_Ab, np.log(EA_hat))-np.multiply(EA_hat, 1-subOmega))/test_size
                self.LKm_lik_scaled[k-1][m] = np.sum(np.multiply(self.A1-subsam_A1, np.log(EA_hat))-np.multiply(Bll[np.ix_(comm_est, comm_est) ], 1-subOmega))/test_size
                self.LKm_se[k-1][m] = np.sum(np.multiply(self.A1-Bll[np.ix_(comm_est, comm_est)],1-subOmega)**2)/test_size

        LK_lik_res = np.mean (self.LKm_lik, 1)
        LK_lik_scaled_res = np.mean(self.LKm_lik_scaled, 1)
        LK_se_res = np.mean(self.LKm_se, 1)
        return 1+np.argmax(LK_lik_res), 1+np.argmax(LK_lik_scaled_res), 1+np.argmin(LK_se_res)

