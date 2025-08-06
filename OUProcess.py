from StochasticProcess import StochasticProcess
import numpy as np
from numba import njit

class OUProcess(StochasticProcess):
    __slots__ = ()
    def __init__(self, params, T, Nx, Nt, InitState = None):
        super().__init__(params, T, Nx, Nt, InitState)

    # params = [theta, mu, nu]
    
    @staticmethod
    @njit
    def bxt(x, t, params):
        return params[0] * (params[1] - x)

    @staticmethod
    @njit
    def sigmaxt(x, t, params):
        return params[2] * np.ones_like(x)

    @staticmethod
    @njit
    def bxt_x(x, t, params):
        return -params[0] * np.ones_like(x)
    
    def DefaultInitState(self):
        return self.params[1] * np.ones(self.Nx)
    
    def SolutionExact(self, dwt = None):
        '''Numerical scheme for stochastic integrals in closed form'''
        dt = self.T / self.Nt
        t_data = np.linspace(0, self.T, self.Nt + 1)
        x_data = np.zeros((self.Nt + 1, self.Nx))

        dwt = self.GenerateWienerProcess(dwt)
       
        if self.InitState is None:
            x_data[0] = self.DefaultInitState()
        else:
            x_data[0] = self.InitState

        theta, mu, nu = self.params[0], self.params[1], self.params[2]
        D = nu**2 / 2

        Ito_integral_sum = np.zeros(self.Nx)

        Mx0 = np.mean(np.asarray(x_data[0]))

        for i in range(1, self.Nt + 1):
            t = i * dt

            x_data[i] = x_data[i - 1] + dwt[i - 1]
            xs = (Mx0 - mu) * np.exp(-theta * t) + mu

            st = np.exp(-theta * t)
            Determinated_part = xs + st * x_data[0] - st * Mx0
            Ito_integral_sum = (Ito_integral_sum + nu * dwt[i - 1]) * np.exp(-theta * dt)
            x_data[i] = Determinated_part + Ito_integral_sum

        return t_data, x_data
    
    def StationaryState(self):
        theta, mu, nu = self.params[0], self.params[1], self.params[2]
        D = nu**2 / 2

        xs = mu
        sigma2 = D/theta

        return np.random.normal(xs, np.sqrt(sigma2), size = self.Nx)
    
    def TransitionDensity(self, x, t, x0, t0):
        theta, mu, nu = self.params[0], self.params[1], self.params[2]
        
        D = nu**2/2

        if t == 0:
            t = t + 0.001

        sigma2 = D / theta * (1 - np.exp(- 2 * theta * (t - t0)))
        xs = (x0 - mu) * np.exp(-theta * (t - t0)) + mu

        return 1/np.sqrt(2 * np.pi * sigma2) * np.exp(-(x - xs)**2 / (2 * sigma2))

    def StationaryDensity(self, x):
        theta, mu, nu = self.params[0], self.params[1], self.params[2]
        
        D = nu**2/2

        sigma2 = D / theta
        xs = mu

        return 1/np.sqrt(2 * np.pi * sigma2) * np.exp(-(x - xs)**2 / (2 * sigma2))

    def SampleFromDensity(self, x0, t, t0):
        theta = self.params[0]
        mu = self.params[1]
        sigma = self.params[2]

        result = np.zeros(self.Nx)

        for j in range(0, self.Nx):
            m = mu + (x0[j] - mu) * np.exp(-theta * (t - t0))
            v = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * (t - t0)))
            result[j] = np.random.normal(m, np.sqrt(v))
        return result