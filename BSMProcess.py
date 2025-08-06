from StochasticProcess import StochasticProcess
import numpy as np
from numba import njit

class BSMProcess(StochasticProcess):
    __slots__ = ()
    def __init__(self, params, T, Nx, Nt, InitState = None):
        super().__init__(params, T, Nx, Nt, InitState)

    # params = [r, sigma]
    @staticmethod
    @njit
    def bxt(x, t, params):
        return params[0] * x

    @staticmethod
    @njit
    def sigmaxt(x, t, params):
        return params[1] * x

    @staticmethod
    @njit
    def bxt_x(x, t, params):
        return params[0] * np.ones_like(x)

    @staticmethod
    @njit
    def sigmaxt_x(x, t, params):
        return params[1] * np.ones_like(x)

    def DefaultInitState(self):
        return np.ones(self.Nx)

    def SampleFromDensity(self, x0, t, t0):
        r = self.params[0]
        sigma = self.params[1]

        result = np.zeros(self.Nx)

        m = (r - 1/2 * sigma**2) * (t - t0)
        v = sigma**2 * (t - t0)
        
        for j in range(0, self.Nx):
            result[j] = x0[j] * np.random.lognormal(m, np.sqrt(v))
        return result

    def SolutionExact(self, dwt = None):
        dt = self.T / self.Nt
        t_data = np.linspace(0, self.T, self.Nt + 1)
        x_data = np.zeros((self.Nt + 1, self.Nx))
        
        r = self.params[0]
        sigma = self.params[1]

        dwt = self.GenerateWienerProcess(dwt)

        if self.InitState is None:
            x_data[0] = self.DefaultInitState()
        else:
            x_data[0] = self.InitState

        for i in range(1, self.Nt + 1):
            x_data[i] = x_data[i - 1] * np.exp((r - sigma**2 / 2) * dt + sigma * dwt[i - 1]) 
        return t_data, x_data   