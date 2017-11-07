import numpy as np
import numpy.linalg as la
import scipy.stats as st
import scipy.special as sp
import numba
import copy


class VBGMM:
    def __init__(self, X, m_0, b_0, inv_W_0, mu_0, alpha_0, m_k, b_k, W_k, mu_k, alpha_k, R_k):
        self.X = copy.deepcopy(X)
        self.m_0 = copy.deepcopy(m_0)
        self.b_0 = copy.deepcopy(b_0)
        self.inv_W_0 = copy.deepcopy(inv_W_0)
        self.mu_0 = copy.deepcopy(mu_0)
        self.alpha_0 = copy.deepcopy(alpha_0)

        self.m_k = copy.deepcopy(m_k)
        self.b_k = copy.deepcopy(b_k)
        self.W_k = copy.deepcopy(W_k)
        self.inv_W_k = np.linalg.inv(W_k)
        self.mu_k = copy.deepcopy(mu_k)
        self.alpha_k = copy.deepcopy(alpha_k) 
        self.R_k = copy.deepcopy(R_k)

    def UpdateParams(self):
        self.b_k = self.b_0 + np.sum(self.R_k, axis=0)
        self.m_k = (self.b_0 * self.m_0 + self.R_k.T.dot(self.X)) / self.b_k[:,None]
        
        self.mu_k = self.mu_0 + np.sum(self.R_k, axis=0) + 1
        
        x_mean = self.R_k.T.dot(self.X) / np.sum(self.R_k, axis=0)[:,None]

        mix_cov = np.zeros_like(self.inv_W_k)
        for k in range(self.R_k.shape[1]):
            x_cov = np.einsum("Bi, Bj -> Bij", self.X - x_mean[k,:], self.X - x_mean[k,:])
            mix_cov[k,:,:] = np.sum(self.R_k[:,k].reshape(R_k.shape[0], 1, 1) * x_cov, axis=0) 

            mean_cov = np.outer(x_mean[k,:] - self.m_0, x_mean[k,:] - self.m_0,)
            mean_cov *= (self.b_0 * np.sum(self.R_k, axis=0)[k] / self.b_k[k]) 

            mix_cov[k,:,:] += mean_cov
            mix_cov[k,:,:] += self.inv_W_0
        self.inv_W_k = mix_cov
        self.W_k = la.inv(mix_cov)


    def UpdateAlpha(self):
        self.alpha_k = self.alpha_0 + np.sum(self.R_k, axis=0)

    def UpdateR(self):
        D = self.X.shape[1]
        K = self.R_k.shape[1]

        det = la.det(self.W_k)
        expect_log_W = sp.polygamma(D, self.mu_k * 0.5) + D * np.log(2) + np.log(det)


        for k in range(K):
            self.R_k[:,k] = st.multivariate_normal.logpdf(X, mean=self.m_k[k,:], cov=self.inv_W_k[k,:,:] / self.mu_k[k]) 
            self.R_k[:,k] -= 0.5 * D / self.b_k[k]
            self.R_k += 0.5 * (sp.polygamma(D, self.mu_k[k] * 0.5) + D * np.log(2))

        expect_log_alpha = sp.digamma(self.alpha_k) - sp.digamma(np.sum(self.alpha_k))
        self.R_k += expect_log_alpha

        self.R_k -= np.max(self.R_k, axis=1)[:,None]
        self.R_k = np.exp(self.R_k)
        self.R_k /= np.sum(self.R_k, axis=1)[:,None]

    def ELBO(self):
        pass

    def VariationalUpdates(self, max_iter):
        for i in range(max_iter):
            self.UpdateR()
            self.UpdateAlpha()
            self.UpdateParams()