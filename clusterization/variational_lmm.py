import numpy as np
import numpy.linalg as la
import scipy.stats as st
import scipy.special as sp
import numba
import copy

class VBLMM:
    def __init__(self, X, Y, R_k, beta_mu_0, beta_sigma_0, a_0, b_0, alpha_0):
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)

        self.R_k = copy.deepcopy(R_k)

        self.alpha_0 = copy.deepcopy(alpha_0)
        self.alpha_k = copy.deepcopy(alpha_0)
        
        self.beta_mu_0 = copy.deepcopy(beta_mu_0)
        self.beta_sigma_0 = copy.deepcopy(beta_sigma_0)
        self.beta_sigma_0_inv = la.inv(beta_sigma_0)
        self.a_0 = copy.deepcopy(a_0)
        self.b_0 = copy.deepcopy(b_0)

        self.beta_mu_k = \
        np.random.multivariate_normal(np.zeros(X.shape[1]), 5. * np.identity(X.shape[1]), alpha_0.shape[0])

        self.beta_sigma_k = np.repeat(np.identity(X.shape[1])[None,:,:], alpha_0.shape[0], axis=0)
        self.a_k = np.array([1.] * alpha_0.shape[0])
        self.b_k = np.array([1.] * alpha_0.shape[0])


        
    def UpdateBeta(self):
        E_inv_tau = self.a_k / self.b_k
        X = self.X
        Y = self.Y

        K = self.beta_sigma_k.shape[0]
        for k in range(K):
            W = np.diag(self.R_k[:,k] / E_inv_tau[k])
            sigma_k = X.T.dot(W).dot(X) + self.beta_sigma_0_inv
            self.beta_sigma_k[k,:,:] = la.inv(sigma_k)
            
            mu_k = X.T.dot(W).dot(Y) + self.beta_sigma_0_inv.dot(self.beta_mu_0)
            mu_k = self.beta_sigma_k[k,:,:].dot(mu_k)
            self.beta_mu_k[k,:] = mu_k

    def UpdateTau(self):
        N = self.X.shape[0]
        K = self.beta_sigma_k.shape[0]
        X = self.X
        Y = self.Y

        self.a_k = self.a_0 + 0.5 * N * np.sum(self.R_k, axis=0)
        for k in range(K):
            Z = np.diag(self.R_k[:,k])
            beta_mean = self.beta_mu_k[k,:]
            noise = Y - X.dot(beta_mean)
            b_new = self.b_0 + 0.5 * np.dot(noise.T, Z).dot(noise)
            b_new += 0.5 * np.trace(np.dot(X.T, Z).dot(X).dot(self.beta_sigma_k[k,:,:]))
            self.b_k[k] = b_new

    def UpdateR(self):
        K = self.beta_sigma_k.shape[0]
        E_ln_pi = sp.digamma(self.alpha_k) - sp.digamma(np.sum(self.alpha_k))
        E_ln_tau = np.log(self.b_k) - sp.digamma(self.a_k)
        E_inv_tau = self.a_k / self.b_k
        Y = self.Y
        X = self.X
        
        for k in range(K):
            beta_mean = self.beta_mu_k[k,:]
            quadratic_term = (Y - np.dot(X, beta_mean)) ** 2
            quadratic_term +=\
             np.einsum('...i,...i->...', X.dot(self.beta_sigma_k[k,:,:]), X)
            quadratic_term /= E_inv_tau[k]
            self.R_k[:,k] = -0.5 * (quadratic_term + E_ln_tau[k] + np.log(2 * np.pi)) + E_ln_pi[k]
        self.R_k -= np.max(self.R_k, axis=1)[:,None]
        self.R_k = np.exp(self.R_k)
        self.R_k /= np.sum(self.R_k, axis=1)[:,None]


    def UpdateAlpha(self):
        self.alpha_k = self.alpha_0 + np.sum(self.R_k, axis=0)

    def VariationalUpdate(self, max_iter):
        for i in range(max_iter):
            self.UpdateR()
            self.UpdateBeta()
            self.UpdateAlpha()
            self.UpdateTau()