import numpy as np
from numba import njit
from scipy.optimize import minimize

class StochasticProcess:
    '''A base class for stochastic diffusion process 
       dxt = b(x,t)dt + sigma(x,t)dWt

       Args:
        params: random process parameters. List
        T: time interval length. float
        Nt: number of time steps. Int
        Nx: number of trajectories. Int
        InitState: initial state. List of length Nx

       Implemented:
        1. Numerical schemes of integration (Euler, Milstein 1, Milstein 2, Predictor-corrector)
        2. Sampling from transition density
        3. Parameter estimation (Euler, Ozaki, ShojiOzaki, Kessler)
        4. Non parametric stationary density estimation
        5. Non parametric drift and diffusion coefficients estimation
       '''
    def __init__(self, params: np.array, T: float, Nx: int, Nt: int, InitState: np.array = None):
        self.params = np.array(params)
        self.T = T
        self.Nx = Nx # Number of trajectories
        self.Nt = Nt # Number of discretization steps
        self.InitState = InitState

    #use @staticmethod for numba compatibility with @njit
    @staticmethod
    @njit
    def bxt(x, t, params):
        '''Stochastic process parameter b(x,t)'''
        return np.zeros_like(x)

    @staticmethod
    @njit
    def sigmaxt(x, t, params):
        '''Stochastic process parameter sigma(x,t)'''
        return np.zeros_like(x)
        
    @staticmethod
    @njit
    def bxt_x(x, t, params):
        '''Stochastic process parameter derivative b_x(x,t)'''
        return np.zeros_like(x)
        
    @staticmethod
    @njit
    def bxt_xx(x, t, params):
        '''Stochastic process parameter derivative b_xx(x,t)'''
        return np.zeros_like(x)
        
    @staticmethod
    @njit
    def sigmaxt_x(x, t, params):
        '''Stochastic process parameter derivative sigma_x(x,t)'''
        return np.zeros_like(x)
        
    @staticmethod
    @njit
    def sigmaxt_xx(x, t, params):
        '''Stochastic process parameter derivative sigma_xx(x,t)'''
        return np.zeros_like(x)

    @staticmethod
    @njit
    def bxt_t(x, t, params):
        '''Stochastic process parameter derivative b_t(x,t)'''
        return np.zeros_like(x)

    def GenerateWienerProcess(self, dwt):
        '''Generation of Wiener process for numerical schemes'''
        dt = self.T / self.Nt
        if dwt is None:
            dwt = np.random.normal(0, 1, size = (self.Nt, self.Nx)) * np.sqrt(dt)
        return dwt
    
    def DefaultInitState(self):
        '''Default initial state of a random process'''
        return np.zeros(self.Nx)
    
    def StationaryState(self):
        '''Stationary state of a random process (if available)'''
        raise NotImplementedError("Not implemented for this class")

    def StationaryDensity(self, x):
        '''Stationary distribution of a random process (if available)'''
        raise NotImplementedError("Not implemented for this class")
    
    def TransitionDensity(self, x, t, x0, t0):
        '''Transition density of a random process (if available, time dependent in general).

        The deinsity is the solution of Fokker-Planck equation with initial conditions x(0) = const
        '''
        raise NotImplementedError("Not implemented for this class")

    def SampleFromDensity(self, x0, t, t0):
        '''Sample from transition density'''
        raise NotImplementedError("Not implemented for this class")

    def PathGenerator(self):
        '''Sampling trajectories of a random process using transition density'''
        dt = self.T / self.Nt
        t = np.linspace(0, self.T, self.Nt + 1)
        xt = np.zeros((self.Nt + 1, self.Nx))

        if self.InitState is None:
            xt[0] = self.DefaultInitState()
        else:
            xt[0] = self.InitState

        for i in range(1, self.Nt + 1):
            xt[i] = self.SampleFromDensity(xt[i - 1], t[i], t[i - 1])
        return t, xt  

    def SolutionExact(self, dwt = None):
        '''Numerical scheme for stochastic integrals in closed form'''
        raise NotImplementedError("Not implemented for this class")

    def SolutionEuler(self, dwt = None):
        '''Euler scheme for random process solution'''
        dt = self.T / self.Nt
        t = np.linspace(0, self.T, self.Nt + 1)
        xt = np.zeros((self.Nt + 1, self.Nx))
        
        dwt = self.GenerateWienerProcess(dwt)

        if self.InitState is None:
            xt[0] = self.DefaultInitState()
        else:
            xt[0] = self.InitState

        for i in range(1, self.Nt + 1):
            xt[i] = xt[i - 1] + self.bxt(xt[i - 1], t[i - 1], self.params) * dt +\
                        self.sigmaxt(xt[i - 1], t[i - 1], self.params) * dwt[i - 1]
        return t, xt            

    def SolutionMilstein1(self, dwt = None):
        '''Milstein 1 scheme for random process solution'''
        dt = self.T / self.Nt
        t = np.linspace(0, self.T, self.Nt + 1)
        xt = np.zeros((self.Nt + 1, self.Nx))

        dwt = self.GenerateWienerProcess(dwt)

        if self.InitState is None:
            xt[0] = self.DefaultInitState()
        else:
            xt[0] = self.InitState

        for i in range(1, self.Nt + 1):
            xt[i] = xt[i - 1] + self.bxt(xt[i - 1], t[i - 1], self.params) * dt\
                  + self.sigmaxt(xt[i - 1], t[i - 1], self.params) * dwt[i - 1]\
                  + 1/2 * self.sigmaxt(xt[i - 1], t[i - 1], self.params)\
                  * self.sigmaxt_x(xt[i - 1], t[i - 1], self.params) * (dwt[i - 1]**2 - dt)
        return t, xt

    def SolutionMilstein2(self, dwt = None):
        '''Milstein 2 scheme for random process solution'''
        dt = self.T / self.Nt
        t = np.linspace(0, self.T, self.Nt + 1)
        xt = np.zeros((self.Nt + 1, self.Nx))

        dwt = self.GenerateWienerProcess(dwt)

        if self.InitState is None:
            xt[0] = self.DefaultInitState()
        else:
            xt[0] = self.InitState

        for i in range(1, self.Nt + 1):
            xt[i] = xt[i - 1] + (self.bxt(xt[i - 1], t[i - 1], self.params) - 1/2 * self.sigmaxt(xt[i - 1], t[i - 1], self.params) * self.sigmaxt_x(xt[i - 1], t[i - 1], self.params)) * dt +\
                        self.sigmaxt(xt[i - 1], t[i - 1], self.params) * dwt[i - 1] + 1/2 * self.sigmaxt(xt[i - 1], t[i - 1], self.params) * self.sigmaxt_x(xt[i - 1], t[i - 1], self.params) * dwt[i - 1]**2 +\
                        (1/2 * self.bxt(xt[i - 1], t[i - 1], self.params) * self.sigmaxt_x(xt[i - 1], t[i - 1], self.params) + 1/2 * self.bxt_x(xt[i - 1], t[i - 1], self.params) * self.sigmaxt(xt[i - 1], t[i - 1], self.params) +\
                         1/4 * self.sigmaxt(xt[i - 1], t[i - 1], self.params)**2 * self.sigmaxt_xx(xt[i - 1], t[i - 1], self.params)) * dt * dwt[i - 1] +\
                        (1/2 * self.bxt(xt[i - 1], t[i - 1], self.params) * self.bxt_x(xt[i - 1], t[i - 1], self.params) + 1/4 * self.bxt_xx(xt[i - 1], t[i - 1], self.params) * self.sigmaxt(xt[i - 1], t[i - 1], self.params)**2) * dt**2   
        return t, xt
 
    def SolutionPredictorCorrector(self, dwt = None, eta = 1/2, alpha = 1/2):
        '''Predictor-corrector scheme for random process solution'''

        dt = self.T / self.Nt
        t = np.linspace(0, self.T, self.Nt + 1)
        xt = np.zeros((self.Nt + 1, self.Nx))

        dwt = self.GenerateWienerProcess(dwt)

        if self.InitState is None:
            xt[0] = self.DefaultInitState()
        else:
            xt[0] = self.InitState

        for i in range(1, self.Nt + 1):
            yw = xt[i - 1] + self.bxt(xt[i - 1], t[i - 1], self.params) * dt\
                  + self.sigmaxt(xt[i - 1], t[i - 1], self.params) * dwt[i - 1]
            bwim1 = self.bxt(xt[i - 1], t[i - 1], self.params)\
                  - eta * self.sigmaxt(xt[i - 1], t[i - 1], self.params)\
                  * self.sigmaxt_x(xt[i - 1], t[i - 1], self.params)
            bwi = self.bxt(yw, t[i], self.params)\
                  - eta * self.sigmaxt(yw, t[i], self.params)\
                  * self.sigmaxt_x(yw, t[i], self.params)
            xt[i] = xt[i - 1] + alpha * bwi * dt + (1 - alpha) * bwim1 * dt +\
                        eta * self.sigmaxt(yw, t[i], self.params) * dwt[i - 1]\
                        + (1 - eta) * self.sigmaxt(xt[i - 1], t[i - 1], self.params) * dwt[i - 1]
        return t, xt

    @staticmethod
    @njit
    def DensityEuler(x, t, x0, t0, params, bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t):
        '''Euler's estimation of transition density'''
        xs = x0 + bxt(x0, t0, params) * (t - t0)
        sigma2 = sigmaxt(x0, t0, params)**2 * (t - t0)

        return -1/2 * np.log(2 * np.pi * sigma2) - (x - xs)**2 / (2 * sigma2)

    @staticmethod
    @njit
    def DensityOzaki(x, t, x0, t0, params, bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t):
        '''Ozaki's estimation of transition density'''
        K = 1 / (t - t0) * np.log(1 + bxt(x0, t0, params) / (x0 * bxt_x(x0, t0, params)) * (np.exp(bxt_x(x0, t0, params) * (t - t0)) - 1))
        E = x0 + bxt(x0, t0, params) / bxt_x(x0, t0, params) * (np.exp(bxt_x(x0, t0, params) * (t - t0)) - 1)
        V = sigmaxt(x0, t0, params)**2 / (2 * K) * (np.exp(2 * K * (t - t0)) - 1)

        xs = E
        sigma2 = V

        return -1/2 * np.log(2 * np.pi * sigma2) - (x - xs)**2 / (2 * sigma2)     

    @staticmethod
    @njit
    def DensityShojiOzaki(x, t, x0, t0, params, bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t):
        '''Shoji-Ozaki's estimation of transition density'''        
        L = bxt_x(x0, t0, params)

        #safe division
        if L == 0:
            L = L + 0.001
            
        M = sigmaxt(x0, t0, params)**2 / 2 * bxt_xx(x0, t0, params) + bxt_t(x0, t0, params)
        A = 1 + bxt(x0, t0, params) / (x0 * L) * (np.exp(L * (t - t0)) - 1) + M / (x0 * L**2) * \
            (np.exp(L * (t - t0)) - 1 - L * (t - t0))
        B = sigmaxt(x0, t0, params)**2 * 1/(2 * L) * (np.exp(2 * L * (t - t0)) - 1)

        xs = A * x0
        sigma2 = B

        return -1/2 * np.log(2 * np.pi * sigma2) - (x - xs)**2 / (2 * sigma2)        

    @staticmethod
    @njit
    def DensityKessler(x, t, x0, t0, params, bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t):
        '''Kessler's estimation of transition density'''        
        E = x0 + bxt(x0, t0, params) * (t - t0) + (bxt(x0, t0, params) * bxt_x(x0, t0, params) + 1/2 * \
            sigmaxt(x0, t0, params)**2 * bxt_xx(x0, t0, params)) * 1/2 * (t - t0)**2
        V = x0**2 + (2 * bxt(x0, t0, params) * x0 + sigmaxt(x0, t0, params)**2 ) * (t - t0)\
            + (2 * bxt(x0, t0, params) * (bxt_x(x0, t0, params) * x0 + bxt(x0, t0, params) + \
                sigmaxt(x0, t0, params) * sigmaxt_x(x0, t0, params))\
            + sigmaxt(x0, t0, params)**2 * (bxt_xx(x0, t0, params) * x0 + 2 * bxt_x(x0, t0, params) +\
                sigmaxt_x(x0, t0, params)**2 + sigmaxt(x0, t0, params) * sigmaxt_xx(x0, t0, params))) * (t - t0)**2 / 2 - E**2

        xs = E

        #safe division
        if V == 0:
            V = V + 0.001
            
        sigma2 = V

        return -1/2 * np.log(2 * np.pi * sigma2) - (x - xs)**2 / (2 * sigma2)  

    @staticmethod
    @njit
    def MLogLik(params, t, xt, density, bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t):
        '''Objective function for numerical process parameter estimation'''
        Nt = len(xt)
        Nx = len(xt[0])
        log_data = np.zeros(Nx)
        for k in range(0, Nx):
            m_log_lik = 0
            for j in range(1, Nt):
                m_log_lik += density(xt[j][k], t[j], xt[j - 1][k], t[j - 1], params, 
                                     bxt, bxt_x, bxt_xx, sigmaxt, sigmaxt_x, sigmaxt_xx, bxt_t)
            log_data[k] = m_log_lik

        res = -np.mean(log_data)
        return res

    def EstimationEuler(self, t, xt):
        '''Parameter estimation using Euler density'''
        x0 = np.ones(len(self.params))

        ftol = 1e-8
        eps = 1e-8

        min_result = minimize(self.MLogLik, 
                              x0 = x0, 
                              args=(t, xt, self.DensityEuler, 
                              self.bxt, self.bxt_x, self.bxt_xx, self.sigmaxt, self.sigmaxt_x, self.sigmaxt_xx,
                              self.bxt_t), 
                              method = 'L-BFGS-B', 
                              options={'ftol': ftol, 'eps': eps})
        return min_result

    def EstimationOzaki(self, t, xt):        
        '''Parameter estimation using Ozaki density'''
        x0 = np.ones(len(self.params))

        ftol = 1e-8
        eps = 1e-8

        min_result = minimize(self.MLogLik, 
                              x0 = x0, 
                              args=(t, xt, self.DensityOzaki, 
                              self.bxt, self.bxt_x, self.bxt_xx, self.sigmaxt, self.sigmaxt_x, self.sigmaxt_xx,
                              self.bxt_t), 
                              method = 'L-BFGS-B', 
                              options={'ftol': ftol, 'eps': eps})
        return min_result

    def EstimationShojiOzaki(self, t, xt):
        '''Parameter estimation using Shoji-Ozaki density'''  
        x0 = np.ones(len(self.params))

        ftol = 1e-8
        eps = 1e-8
        
        min_result = minimize(self.MLogLik, 
                              x0 = x0, 
                              args=(t, xt, self.DensityShojiOzaki, 
                              self.bxt, self.bxt_x, self.bxt_xx, self.sigmaxt, self.sigmaxt_x, self.sigmaxt_xx,
                              self.bxt_t), 
                              method = 'L-BFGS-B', 
                              options={'ftol': ftol, 'eps': eps})
        return min_result

    def EstimationKessler(self, t, xt):   
        '''Parameter estimation using Kessler density'''       
        x0 = np.ones(len(self.params))

        ftol = 1e-8
        eps = 1e-8

        min_result = minimize(self.MLogLik, 
                              x0 = x0, 
                              args=(t, xt, self.DensityKessler, 
                              self.bxt, self.bxt_x, self.bxt_xx, self.sigmaxt, self.sigmaxt_x, self.sigmaxt_xx,
                              self.bxt_t), 
                              method = 'L-BFGS-B', 
                              options={'ftol': ftol, 'eps': eps})
        return min_result

    @staticmethod
    def StationaryDensityKernelEstimator(x, xt, delta = None):
        '''Non parametric estimation of stationary density
        
        Args:
            x - set of points where estimation is calculated
            xt - random process values
            delta - estimation parameter. If None - Scott's rule is used

        Returns:
            Density at point x
        '''
        n = len(xt)
        m = 1

        hn = 1

        if delta is None:
            hn = np.std(xt) * n**(-1 / (m + 4))
        else:
            hn = delta

        pi = np.zeros(len(x))
        
        for j in range(0, len(x)):
            K = 1/np.sqrt(2 * np.pi) * np.exp(-1/2 * (x[j] - xt)**2 / hn**2)

            pi[j] = 1 / (n * hn) * np.sum(K)
        
        return pi
    
    @staticmethod
    def NormalKernel(z):
        return 1/np.sqrt(2 * np.pi) * np.exp(-1/2 * z**2)

    def DiffusionKernelEstimator(self, x, xt, delta = None):
        '''Non parametric estimation of diffusion coefficient sigma(xt)
        of ergodic process (without time dependence)
        
        Args:
            x - set of points where estimation is calculated
            xt - random process values
            delta - estimation parameter

        Returns:
            diffusion coefficient sigma(x)
        '''
        n = len(xt)
        m = 1

        hn = 1

        hn = np.std(xt) * n**(-1 / (m + 4))

        s2 = np.zeros(len(x))

        for j in range(0, len(x)):
            K1 = 0
            K2 = 0
            z = (x[j] - xt)**2 / hn**2
            K = self.NormalKernel(z)
               
            K1 = np.sum(K[0:n-1] * (xt[1:] - xt[0:n-1])**2)
            K2 = np.sum(K)
            if K2 == 0:
                s2[j] = 0
            else:
                s2[j] = K1 / K2

        if delta is None:
            delta = 1
            
        return s2 / delta

    def DriftKernelEstimator(self, x, xt, delta = None):
        '''Non parametric estimation of drift coefficient b(xt)
        of ergodic process (without time dependence)
        
        Args:
            x - set of points where estimation is calculated
            xt - random process values
            delta - estimation parameter

        Returns:
            drift coefficient b(x)
        '''
        n = len(xt)
        m = 1

        hn = 1

        hn = np.std(xt) * n**(-1 / (m + 4))

        s2 = np.zeros(len(x))

        for j in range(0, len(x)):
            K1 = 0
            K2 = 0
            z = (x[j] - xt)**2 / hn**2
            K = self.NormalKernel(z)

            K1 = np.sum(K[0:n-1] * (xt[1:] - xt[0:n-1]))
            K2 = np.sum(K)
            if K2 == 0:
                s2[j] = 0
            else:
                s2[j] = K1 / K2

        if delta is None:
            delta = 1
            
        return s2 / delta