from StochasticProcess import StochasticProcess
import numpy as np
from numba import njit

class HWProcess(StochasticProcess):
    def __init__(self, params, T, Nx, Nt, InitState = None):
        super().__init__(params, T, Nx, Nt, InitState)

    # params = [theta1, theta2, theta3]
    @staticmethod
    @njit
    def bxt(x, t, params):
        return params[0] * t * (params[1] * np.sqrt(t) - x)

    @staticmethod
    @njit
    def sigmaxt(x, t, params):
        return params[2] * t * np.ones_like(x)
        
    @staticmethod
    @njit
    def bxt_x(x, t, params):
        return -params[0] * t * np.ones_like(x)
                     
    @staticmethod
    @njit
    def bxt_t(x, t, params):
        return params[0] * params[1] * 3/2 * np.sqrt(t) * np.ones_like(x)

    def DefaultInitState(self):
        return self.params[1] * np.ones(self.Nx)