class VLR:
    def __init__(self, X, Y, w_mu_0, w_sigma_0, a_0, b_0):
        self.X = X
        self.Y = Y
        
        self.w_mu_0 = w_mu_0
        self.w_sigma_0 = w_sigma_0
        self.w_sigma_0_inv = np.linalg.inv(w_sigma_0)
        self.a_0 = a_0
        self.b_0 = b_0
        
        self.w_mu_N = sp.stats.multivariate_normal.rvs(np.zeros(X.shape[1]), np.identity(X.shape[1]))
        self.w_sigma_N = np.identity(X.shape[1])
        self.a_N = sp.stats.invgamma.rvs(1., 1.)
        self.b_N = sp.stats.invgamma.rvs(1., 1.)
        
    
    def UpdateA(self):
        self.a_N = self.a_0 + self.X.shape[0] * 0.5
    
    def UpdateB(self):
        X = self.X
        Y = self.Y
        
        self.b_N = self.b_0 + 0.5 * np.inner(Y - X.dot(self.w_mu_N), Y - X.dot(self.w_mu_N))
        self.b_N += 0.5 * np.einsum('ii', X.dot(self.w_sigma_N).dot(X.T))
    
    def UpdateSigma(self):
        X = self.X
        self.w_sigma_N = self.w_sigma_0_inv + self.a_N / self.b_N * X.T.dot(X) 
        self.w_sigma_N = np.linalg.inv(self.w_sigma_N )
    
    def UpdateMu(self):
        X = self.X
        Y = self.Y
        self.w_mu_N = self.w_sigma_N.dot(self.a_N / self.b_N * X.T.dot(Y) + self.w_sigma_0_inv.dot(self.w_mu_0))
        
    def ELBO(self):
        elbo = -(self.w_mu_0 - self.w_mu_N).T.dot(self.w_sigma_0_inv).dot(self.w_mu_0 - self.w_mu_N)
        elbo -= np.einsum('ii', self.w_sigma_0_inv.dot(self.w_sigma_N))
        elbo += np.log(np.linalg.det(self.w_sigma_N))
        elbo -= 2. * self.a_N * np.log(self.b_N)
        return elbo
    
    def VariationalUpdate(self, max_iter, tol):
        self.UpdateA()
        elbo_values = []
        for i in range(max_iter):
            self.UpdateSigma()
            self.UpdateMu()
            self.UpdateB()
            elbo_values.append(self.ELBO())
            if i > 0 and np.abs(elbo_values[-1] - elbo_values[-2]) < tol:
                print('CONVERGE BY TOL')
                break
        plt.plot(elbo_values)
        plt.title('ELBO vs. iteration')P