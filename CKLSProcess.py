from StochasticProcess import StochasticProcess
import numpy as np
from numba import njit

class CKLSProcess(StochasticProcess):
    __slots__ = ()
    def __init__(self, params, T, Nx, Nt, InitState = None):
        super().__init__(params, T, Nx, Nt, InitState)

    # params = [theta1, theta2, theta3, theta4]
    @staticmethod
    @njit
    def bxt(x, t, params):
        return params[0] + params[1] * x

    @staticmethod
    @njit
    def sigmaxt(x, t, params):
        return params[2] * x**params[3]
        
    @staticmethod
    @njit
    def bxt_x(x, t, params):
        return params[1] * np.ones_like(x)
              
    @staticmethod
    @njit
    def sigmaxt_x(x, t, params):
        return params[2] * params[3] * x**(params[3] - 1)
        
    @staticmethod
    @njit
    def sigmaxt_xx(x, t, params):
        return params[2] * params[3] * (params[3] - 1) * x**(params[3] - 2)

    def DefaultInitState(self):
        return self.params[0] / self.params[1] * np.ones(self.Nx)
