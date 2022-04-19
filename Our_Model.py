import numpy as np
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        None
    
    def quantity_Theta_Beta_Eta(self, sample=True):
        # Store these quantities to avoid repetitive computation
        self.temp_diff_Mu_XBeta = self.Mu - self.X@self.Beta
        if sample:
            self.temp_diff_Mu_XBeta_sample = self.Mu_sample - self.X_sample@self.Beta
        
        if self.network:
            self.temp_Theta_Eta_ThetaT = self.Theta @ self.Eta @ self.Theta.T
            if sample:
                self.temp_Theta_Eta_ThetaT_sample = np.zeros([self.D,self.D])
                for i in range(self.n_sample):
                    self.temp_Theta_Eta_ThetaT_sample = self.temp_Theta_Eta_ThetaT_sample + self.Theta_sample[(i*self.D):(i*self.D+self.D),:] @ self.Eta @ self.Theta_sample[(i*self.D):(i*self.D+self.D),:].T
    
    # Model Fitting
    def fit(self, N, K, A, X, Y, step_length=0.5, n_iter=50, tol=5e-6, n_sample=10, n_batch=3, sigma_Beta_prior=1, sigma_Alpha_prior=1, plot_fitting=False):
        self.plot_fitting = plot_fitting
        self.step_length = step_length
        self.n_sample = n_sample
        self.n_batch = n_batch
        self.n_iter = n_iter
        self.tol = tol
        
        self.sigma_Beta_prior = sigma_Beta_prior
        self.sigma_Alpha_prior = sigma_Alpha_prior
        
        self.D, self.W = N.shape
        self.K = K
        self.N = N
        
        self.Phi = np.random.uniform(0, 1, [self.K, self.W])
        self.Phi = self.row_sum_one(self.Phi)
        
        self.Sigma_inv = np.identity(self.K)
        self.Sigma = np.linalg.inv(self.Sigma_inv)
        
        self.Mu = np.random.normal(0, 0.2, [self.D, self.K])
        self.Theta = self.row_logit(self.Mu)
        
        if A is None or np.max(A)==0:
            self.network = False
        else:
            self.network = True
            self.A = A
            self.Eta = np.diag( np.ones(self.K)/2 )
        
        if X is None:
            # self.assortative = False
            self.X = np.ones([self.D,1])
        else:
            self.X = X
        self.P = X.shape[1]
        self.Beta = np.random.normal(0, 0.1, [self.P, self.K])
        
        if Y is None:
            # self.generative = False
            self.Y = []
            self.Y.append(np.ones([self.D,1]))
        else:
            self.Y = Y
        self.L = len(self.Y)
        self.Alpha = []
        self.Alpha_temp = []
        self.R = []
        self.d_Alpha = []
        for l in range(self.L):
            self.d_Alpha.append( np.random.normal(0, 1, [self.K, self.Y[l].shape[1]]) )
            self.Alpha.append( np.random.normal(0, 1, [self.K, self.Y[l].shape[1]]) )
            self.Alpha_temp.append( np.random.normal(0, 1, [self.K, self.Y[l].shape[1]]) )
            self.R.append( np.tensordot( self.Theta, self.Alpha[l], axes=(1,0) ) )
        
        self.update_R()
        # some auxiliary variables
        self.d_Theta_d_Mu = np.zeros([self.D,self.K,self.K])
        self.d_Theta_partial = np.zeros([self.D,self.K])
        self.d_Mu = np.zeros([self.D,self.K])
        self.Hessian_Theta_partial = np.zeros([self.D, self.K, self.K])
        self.Hessian_Mu = np.zeros([self.D, self.K])
        self.dd_Theta_dd_Mu = np.zeros([self.K, self.K]) # not symmetric
        
        self.GA_Alpha_MAP_temp1 = []
        for l in range(self.L):
            self.GA_Alpha_MAP_temp1.append( np.zeros([self.D*self.n_sample,self.Y[l].shape[1]]) )
        
        self.Hessian_temp2 = [np.zeros([self.D, self.K]) for l in range(self.L)]
        
        # For stochastic EM
        self.Mu_sample = np.random.normal(0, 0.1, [n_sample*self.D, self.K])
        self.Theta_sample = self.row_logit(self.Mu_sample)
        self.R_sample = []
        for l in range(self.L):
            self.R_sample.append( np.tensordot( self.Theta_sample, self.Alpha[l], axes=(1,0) ) )
        
        # Stacked Parameters
        self.X_sample = np.zeros([n_sample*self.D, self.P])
        self.Y_sample = [np.zeros([n_sample*self.D, self.Y[l].shape[1]]) for l in range(self.L)]
        self.N_sample = np.zeros([n_sample*self.D, self.W])
        for i in range(n_sample):
            self.X_sample[(i*self.D):(i*self.D+self.D),:] = self.X
            self.N_sample[(i*self.D):(i*self.D+self.D),:] = self.N
            for l in range(self.L):
                self.Y_sample[l][(i*self.D):(i*self.D+self.D),:] = self.Y[l]
        self.temp_inv_X_sample_T_X_sample = np.linalg.inv( self.X_sample.transpose() @ self.X_sample )
        
        self.Cov_Mu = np.ones([self.D,self.K])
        
        
        self.update_R()
        self.update_R_sample()
        self.quantity_Theta_Beta_Eta()
        
        print('D = '+str(self.D))
        print('W = '+str(self.W))
        print('K = '+str(self.K))
        print('P = '+str(self.P))
        print('L = '+str(self.L))
        # Log-likelihood
        self.loglik = np.zeros([self.n_iter])
        for j in range(self.n_iter):
            print('Iteration '+str(j+1))
            self.loglik[j] = self.one_iteration()
            if (j+1)%5 == 0:
                if abs( (self.loglik[j]-self.loglik[j-1])/self.loglik[j-1] ) < self.tol:
                    self.n_iter = j+1
                    self.loglik = self.loglik[1:(j+1)]
                    break
        
        self.delete_auxiliary_variables()
        #end
    
    def one_iteration(self):
        # Get MAP of Mu
        self.GA_Mu_MAP(GA_times=50, n_try=10, plot=True)
        self.update_Hessian_Mu()
        
        self.Hessian_Mu = np.minimum( self.Hessian_Mu, -1e-100 ) # prevent division by 0
        temp_Hessian_Mu_range = np.median(self.Cov_Mu)*10
        
        self.Cov_Mu = np.minimum(-1/self.Hessian_Mu, temp_Hessian_Mu_range)
        #
        print('Range of Hessian_Mu: (' + str(np.min(self.Hessian_Mu)) + ', ' + str(np.max(self.Hessian_Mu)) + ')')
        
        loglik_temp = 0
        self.Scale_mat = np.zeros([self.K, self.K])
        for b in range(self.n_batch):
            print('Batch: '+str(b+1))
            
            # Generate random samples for SEM
            for d in range(self.D):
                idx = range(d, self.D*self.n_sample, self.D)
                self.Mu_sample[idx,:] = multivariate_normal.rvs(mean=self.Mu[d,:], cov=self.Cov_Mu[d,:], size=self.n_sample, random_state=None)
            self.Theta_sample = self.row_logit(self.Mu_sample)
            self.update_R_sample()
            
            self.quantity_Theta_Beta_Eta(sample=True)
            
            # log-likelihood
            loglik_temp = loglik_temp + ( self.l_all(sample=True, sigma=True) + self.prior_l_Alpha() + self.prior_l_Beta() + self.prior_l_Sigma_inv() )/self.n_batch
            
            # Get MLE of other parameters
            self.update_Eta()
            
            for i in range(3):
                self.update_Phi()
            
            self.GA_Alpha_MAP(GA_times=5, n_try=10)
            
            self.GA_Beta_MAP(GA_times=5, n_try=10)
            
            # For Sigma
            self.Scale_mat = self.Scale_mat + ( self.temp_diff_Mu_XBeta_sample.T @ self.temp_diff_Mu_XBeta_sample )/self.n_sample/self.n_batch
        
        # Update Sigma
        self.Sigma = ( self.Scale_mat + np.identity(self.K) )/(self.D+self.K+self.K+1)
        self.Sigma_inv = np.linalg.inv( self.Sigma )
        
        return loglik_temp
        # end of one_iteration
    
    
    # Helper functions
    def row_sum_one(self, M):
        return M / np.sum(M,1)[:,None]
    
    def row_logit(self, M):
        M = M - np.max(M,axis=1,keepdims=True) + 100 # Prevent overflow
        return self.row_sum_one( np.exp(M) )
    
    
    # Update parameters
    def update_R_sample(self):
        for l in range(self.L):
            self.R_sample[l] = self.Theta_sample @ self.Alpha[l]
            self.R_sample[l] = self.row_logit( self.R_sample[l] - np.max(self.R_sample[l], 1)[:,None] )
    
    def update_R(self):
        for l in range(self.L):
            self.R[l] = self.Theta @ self.Alpha[l]
            self.R[l] = self.row_logit( self.R[l] - np.max(self.R[l], 1)[:,None] )
    
    # Derivatives
    def update_d_Beta(self):
        self.d_Beta = self.X_sample.T @ self.temp_diff_Mu_XBeta_sample @ self.Sigma_inv / self.n_sample - self.Beta/self.sigma_Beta_prior
    
    def update_d_Alpha(self):
        for l in range(self.L):
            self.d_Alpha[l] = np.tensordot( self.Theta_sample, self.Y_sample[l]-self.R_sample[l], axes=(0,0) ) / self.n_sample - self.Alpha[l]/self.sigma_Alpha_prior
    
    def update_d_Theta_d_Mu(self):
        for d in range(self.D):
            self.d_Theta_d_Mu[d,:,:] = np.diag(self.Theta[d,:]) - self.Theta[d,None].T @ self.Theta[d,None]
    
    def update_d_Theta_partial(self):
        # Network
        if not self.network:
            self.d_Theta_partial = np.zeros([self.D, self.K])
        else:
            self.d_Theta_partial = ( self.A/self.temp_Theta_Eta_ThetaT - 1 ) @ self.Theta @ self.Eta
        # Text
        self.d_Theta_partial = self.d_Theta_partial + np.matmul( self.N/(self.Theta@self.Phi), self.Phi.T )
        # Generative
        for l in range(self.L):
            self.d_Theta_partial = self.d_Theta_partial + (self.Y[l]-self.R[l]) @ self.Alpha[l].T
    
    def update_d_Mu(self):
        self.update_d_Theta_partial()
        self.update_d_Theta_d_Mu()
        for d in range(self.D):
            self.d_Mu[d,:] = self.d_Theta_partial[d,:] @ self.d_Theta_d_Mu[d,:,:]
        self.d_Mu = self.d_Mu - self.temp_diff_Mu_XBeta @ self.Sigma_inv
    
    
    # Gradient Ascent (GA)
    def GA_Beta_MAP(self, GA_times=10, n_try=10):
        def temp_loglik(Beta_temp):
            retval = -0.5/(self.sigma_Beta_prior**2)*np.sum(Beta_temp**2)
            self.GA_Beta_MAP_temp1 = self.Mu_sample - self.X_sample@Beta_temp
            for d in range(self.D*self.n_sample):
                retval = retval - 0.5/self.n_sample*float( self.GA_Beta_MAP_temp1[d,None] @ self.Sigma_inv @ self.GA_Beta_MAP_temp1[d,None].T )
            return(retval)
        
        # Useful quantities
        loglik_current = temp_loglik(self.Beta)
        loglik_all = np.zeros(GA_times)
        
        for g in range(GA_times):
            step_length_temp = self.step_length
            loglik_all[g] = loglik_current
            
            self.update_d_Beta()
            
            for b in range(n_try):
                self.Beta_temp = self.Beta + step_length_temp*self.d_Beta
                loglik_temp = temp_loglik(self.Beta_temp)
                if loglik_temp > loglik_current:
                    loglik_current = loglik_temp
                    self.Beta = self.Beta_temp
                    self.temp_diff_Mu_XBeta_sample = self.Mu_sample - self.X_sample@self.Beta
                    break
                else:
                    step_length_temp = step_length_temp/5
            if b==n_try-1 and loglik_temp<=loglik_current:
                break
        if self.plot_fitting:
            plt.plot(loglik_all[:(g+1)])
            plt.title('Beta Optimization')
            plt.show()
        # Update related quantities
        self.temp_diff_Mu_XBeta = self.Mu - self.X@self.Beta
        # end of GA_Beta_MAP
    
    def GA_Alpha_MAP(self, GA_times=10, n_try=10):
        def temp_loglik(Alpha_temp):
            for l in range(self.L):
                self.GA_Alpha_MAP_temp1[l] = self.Theta_sample @ Alpha_temp[l]
                self.GA_Alpha_MAP_temp1[l] = self.row_logit( self.GA_Alpha_MAP_temp1[l] - np.max(self.GA_Alpha_MAP_temp1[l], 1)[:,None] )
            retval = 0
            for l in range(self.L):
                retval = retval -0.5/(self.sigma_Alpha_prior**2)*np.sum(Alpha_temp[l]**2)
                retval = retval + np.sum( self.Y_sample[l]*np.log(self.GA_Alpha_MAP_temp1[l]+np.exp(-200)) ) / self.n_sample
            return retval
        
        loglik_current = temp_loglik(self.Alpha)
        loglik_all = np.zeros(GA_times)
        
        for g in range(GA_times):
            step_length_temp = self.step_length
            loglik_all[g] = loglik_current
            self.update_d_Alpha()
            
            for b in range(n_try):
                for l in range(self.L):
                    self.Alpha_temp[l] = self.Alpha[l] + step_length_temp*self.d_Alpha[l]
                loglik_temp = temp_loglik(self.Alpha_temp)
                if loglik_temp > loglik_current:
                    loglik_current = loglik_temp
                    for l in range(self.L):
                        self.Alpha[l] = self.Alpha_temp[l]
                    self.update_R_sample()
                    break
                else:
                    step_length_temp = step_length_temp/5
            if b==n_try-1 and loglik_temp<=loglik_current:
                break
        if self.plot_fitting:
            plt.plot(loglik_all[:(g+1)])
            plt.title('Alpha Optimization')
            plt.show()
        # Update these quantities
        self.update_R()
        # end of GA_Alpha_MAP
    
    def GA_Mu_MAP(self, GA_times=10, n_try=10, plot=False):
        loglik_current = self.l_all(sample=False, sigma=False)
        loglik_all = np.zeros(GA_times)
        
        for g in range(GA_times):
            step_length_temp = self.step_length
            loglik_all[g] = loglik_current
            self.update_d_Mu()
            
            for b in range(n_try):
                self.Mu_temp, self.Mu = self.Mu, self.Mu + step_length_temp*self.d_Mu
                self.Theta_temp, self.Theta = self.Theta, self.row_logit(self.Mu)
                self.update_R()
                self.quantity_Theta_Beta_Eta(sample=False)
                loglik_temp = self.l_all(sample=False, sigma=False)
                if loglik_temp > loglik_current:
                    loglik_current = loglik_temp
                    break
                else:
                    self.Mu = self.Mu_temp
                    self.Theta = self.Theta_temp
                    self.update_R()
                    self.quantity_Theta_Beta_Eta(sample=False)
                    step_length_temp = step_length_temp/5
            if b==n_try-1 and loglik_temp<=loglik_current:
                break
        self.quantity_Theta_Beta_Eta(sample=True)
        
        if plot:
            plt.plot(loglik_all[:(g+1)])
            plt.title('Theta Optimization')
            plt.show()
        # end of GA_Mu_MAP
    
    # Local EM algorithm for $\phi$ and $\eta$
    def update_Eta(self):
        if not self.network:
            return
        
        self.q = np.zeros([self.K, self.D, self.D])
        for i in range(self.n_sample):
            self.denominator_q = self.Theta_sample[(i*self.D):(i*self.D+self.D),:] @ self.Eta @ self.Theta_sample[(i*self.D):(i*self.D+self.D),:].T + np.exp(-200)
            for k in range(self.K):
                self.q[k] = self.q[k] + ( self.Theta_sample[(i*self.D):(i*self.D+self.D),k:(k+1)] @ self.Theta_sample[(i*self.D):(i*self.D+self.D),k:(k+1)].T ) * self.Eta[k,k] / self.denominator_q
        
        self.denominator_Eta = np.zeros(self.K)
        for i in range(self.n_sample):
            self.denominator_Eta = self.denominator_Eta + np.sum( self.Theta_sample[(i*self.D):(i*self.D+self.D),:], 0)**2
        
        for k in range(self.K):
            self.Eta[k,k] = np.sum( self.A * self.q[k] ) / self.denominator_Eta[k]
        # Update related quantities
        self.temp_Theta_Eta_ThetaT = self.Theta @ self.Eta @ self.Theta.T
        self.temp_Theta_Eta_ThetaT_sample = np.zeros([self.D,self.D])
        for i in range(self.n_sample):
            self.temp_Theta_Eta_ThetaT_sample = self.temp_Theta_Eta_ThetaT_sample + self.Theta_sample[(i*self.D):(i*self.D+self.D),:] @ self.Eta @ self.Theta_sample[(i*self.D):(i*self.D+self.D),:].T
        # end of update_Eta
    
    
    def update_Phi(self):
        self.denominator_h = self.Theta_sample @ self.Phi + 0.01/self.W
        
        for k in range(self.K):
            self.Phi[k,:] = np.sum( self.N_sample * ( self.Theta_sample[:,k:(k+1)] @ self.Phi[k:(k+1),:] ) / self.denominator_h, 0 )
        self.Phi = self.Phi/np.sum(self.Phi, 1)[:,None]
        # end of update_Phi
    
    
    # Hessian Matrix
    def update_Hessian_Theta_partial(self):
        # from lt
        self.Hessian_temp1 = - self.N / (self.Theta @ self.Phi)**2
        for d in range(self.D):
            for k1 in range(self.K):
                for k2 in range(k1+1):
                    self.Hessian_Theta_partial[d,k1,k2] = self.Hessian_Theta_partial[d,k2,k1] = np.sum( self.Phi[k1,:]*self.Phi[k2,:]*self.Hessian_temp1[d,:] )
        
        # from lg
        for l in range(self.L):
            self.Hessian_temp2[l] = self.R[l] @ self.Alpha[l].T
        
        for d in range(self.D):
            for k1 in range(self.K):
                for k2 in range(k1+1):
                    for l in range(self.L):
                        self.Hessian_Theta_partial[d,k1,k2] = self.Hessian_Theta_partial[d,k1,k2] -np.sum( self.R[l][d,:]*self.Alpha[l][k1,:]*self.Alpha[l][k2,:] ) + self.Hessian_temp2[l][d,k1]*self.Hessian_temp2[l][d,k2]
                    self.Hessian_Theta_partial[d,k2,k1] = self.Hessian_Theta_partial[d,k1,k2]
        
        # from lc
        if not self.network:
            return
        
        self.Hessian_temp_A_P = self.A / self.temp_Theta_Eta_ThetaT
        self.Hessian_temp_A_PP = self.Hessian_temp_A_P / self.temp_Theta_Eta_ThetaT
        self.Hessian_temp_Theta_Eta = self.Theta @ self.Eta
        
        for d in range(self.D):
            self.Hessian_Theta_partial[d,:,:] = self.Hessian_Theta_partial[d,:,:] + (self.Hessian_temp_A_P[d,d] - 1)*self.Eta
            for k1 in range(self.K):
                for k2 in range(k1+1):
                    self.Hessian_Theta_partial[d,k1,k2] = self.Hessian_Theta_partial[d,k1,k2] - np.sum(self.Hessian_temp_A_PP[d,:] * self.Hessian_temp_Theta_Eta[:,k1] * self.Hessian_temp_Theta_Eta[:,k2]) - (self.Hessian_temp_A_PP[d,d] * self.Hessian_temp_Theta_Eta[d,k1]*self.Hessian_temp_Theta_Eta[d,k2])
                    self.Hessian_Theta_partial[d,k2,k1] = self.Hessian_Theta_partial[d,k1,k2]
        # end of update_Hessian_Theta_partial
    
    
    def update_Hessian_Mu(self):
        #######################
        self.update_d_Theta_partial()
        
        for d in range(self.D):
            for k1 in range(self.K):
                for k2 in range(self.K):
                    if k1==k2:
                        self.dd_Theta_dd_Mu[k1,k2] = self.Theta[d,k1] - 3*self.Theta[d,k1]**2 + 2*self.Theta[d,k1]**3
                    else:
                        self.dd_Theta_dd_Mu[k1,k2] = -self.Theta[d,k1]*self.Theta[d,k2] + 2*self.Theta[d,k1]*(self.Theta[d,k2]**2)
            self.Hessian_Mu[d,None] = self.d_Theta_partial[d,None] @ self.dd_Theta_dd_Mu
        
        #######################
        self.update_d_Theta_d_Mu()
        self.update_Hessian_Theta_partial()
        
        for d in range(self.D):
            self.Hessian_Mu[d,:] = self.Hessian_Mu[d,:] - np.diag(self.Sigma_inv)
            for k in range(self.K):
                self.Hessian_Mu[d,k] = self.Hessian_Mu[d,k] + self.d_Theta_d_Mu[d,k,:].T @ self.Hessian_Theta_partial[d,:,:] @ self.d_Theta_d_Mu[d,:,k]
        # end of update_Hessian_Mu
    
    
    
    # Log-Likelihood
    def l_a(self, sample=True, sigma=True):
        if sigma:
            retval = self.D/2*(np.log( np.linalg.det(self.Sigma_inv) ) - np.log(2*np.pi)*self.K)
        else:
            retval = 0
        if sample:
            for d in range(self.D*self.n_sample):
                retval = retval - 0.5/self.n_sample*float( self.temp_diff_Mu_XBeta_sample[d,None] @ self.Sigma_inv @ self.temp_diff_Mu_XBeta_sample[d,None].T )
        else:
            for d in range(self.D):
                retval = retval - 0.5*float( self.temp_diff_Mu_XBeta[d,None] @ self.Sigma_inv @ self.temp_diff_Mu_XBeta[d,None].T )
        return retval
    
    def l_t(self, sample=True):
        if sample:
            return np.sum( self.N_sample * np.log( self.Theta_sample @ self.Phi + np.exp(-200) ) ) / self.n_sample
        else:
            return np.sum( self.N * np.log( self.Theta @ self.Phi + np.exp(-200) ) )
    
    def l_g(self, sample=True):
        retval = 0
        if sample:
            for l in range(self.L):
                retval = retval + np.sum( self.Y_sample[l]*np.log(self.R_sample[l]+np.exp(-200)) ) / self.n_sample
        else:
            for l in range(self.L):
                retval = retval + np.sum( self.Y[l]*np.log(self.R[l]+np.exp(-200)) )
        return retval
    
    def l_c(self, sample=True):
        if not self.network:
            return 0.0
        if sample:
            return ( np.sum( self.A * np.log(self.temp_Theta_Eta_ThetaT_sample) ) - np.sum( self.temp_Theta_Eta_ThetaT_sample ) )/2/self.n_sample
        else:
            return ( np.sum( self.A * np.log(self.temp_Theta_Eta_ThetaT) ) - np.sum( self.temp_Theta_Eta_ThetaT ) )/2
    
    def l_all(self, sample=True, sigma=True):
        if sample:
            self.update_R_sample()
        else:
            self.update_R()
        retval = 0
        retval = retval + self.l_t(sample)
        retval = retval + self.l_a(sample, sigma)
        retval = retval + self.l_g(sample)
        retval = retval + self.l_c(sample)
        return retval
    
    # Log-Priors
    # Adding these two priors fixes the model identifiability issue
    def prior_l_Alpha(self):
        retval = 0
        for l in range(self.L):
            retval = retval + np.sum(-0.5/(self.sigma_Alpha_prior**2)*(self.Alpha[l]**2))
        return retval
    
    def prior_l_Beta(self):
        return np.sum(-0.5/(self.sigma_Beta_prior**2)*(self.Beta**2))
    
    def prior_l_Sigma_inv(self, v=-1):
        if v==-1:
            v = self.Sigma_inv.shape[0]
        return invwishart.logpdf( x=np.linalg.inv( self.Sigma_inv ), df=v, scale=np.identity(self.Sigma_inv.shape[0]) )
    
    
    def delete_auxiliary_variables(self):
        try:
            del self.denominator_q
            del self.denominator_h
            del self.denominator_Eta
            del self.q
        except:
            None
        
        del self.Scale_mat
        
        del self.Alpha_temp
        del self.Beta_temp
        del self.d_Alpha
        del self.d_Beta
        del self.d_Theta_d_Mu
        del self.d_Theta_partial
        del self.d_Mu
        del self.Hessian_Theta_partial
        del self.Hessian_Mu
        del self.dd_Theta_dd_Mu
        del self.GA_Alpha_MAP_temp1
        del self.GA_Beta_MAP_temp1
        del self.Hessian_temp1
        del self.Hessian_temp2
        
        del self.Mu_sample
        del self.Theta_sample
        del self.R_sample
        del self.X_sample
        del self.Y_sample
        del self.N_sample
        
        del self.temp_inv_X_sample_T_X_sample
        del self.Cov_Mu
        del self.temp_diff_Mu_XBeta
        del self.temp_diff_Mu_XBeta_sample
        del self.Mu_temp
        del self.Theta_temp
        
        try:
            del self.temp_Theta_Eta_ThetaT
            del self.temp_Theta_Eta_ThetaT_sample
            del self.Hessian_temp_A_P
            del self.Hessian_temp_A_PP
            del self.Hessian_temp_Theta_Eta
        except:
            None
    
    
        
    