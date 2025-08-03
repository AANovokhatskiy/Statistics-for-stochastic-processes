This project implements a number of numerical algorithms applied to diffusion random processes of the type
```math
dx_t = b(x_t, t)dt + \sigma(x_t, t)dW_t
```

The program code includes the implementation
    1. Numerical schemes of integration (Euler, Milstein 1, Milstein 2, Predictor-corrector)
    2. Sampling from transition density
    3. Parameter estimation (Euler, Ozaki, ShojiOzaki, Kessler)
    4. Non parametric stationary density estimation
    5. Non parametric drift and diffusion coefficients estimation
The following models are descibed:
    1. OU - Ornstein-Uhlenbeck
    ```math
        dx_t = \theta (\mu - x_t) dt + \nu dW_t
    ```
    2. CIR - Cox-Ingersoll-Ross
    ```math
        dx_t = (\theta_1 - \theta_2 x_t) dt + \theta_3 \sqrt(x_t)dW_t
    ```
    3. CKLS - Chan-Karolyi-Longstaff-Sanders
    ```math
        dxt = (\theta_1 + \theta_2 * x_t) dt + \theta_3 x_t^\theta_4 dW_t
    ```
    4. HW - Hull-White
    ```math
        dx_t = \theta_1 t (\theta_2 \sqrt(t) - x_t) dt + \theta_3 t dW_t
    ```
    5. BSM - Black–Scholes–Merton
    ```math
        dx_t = r x_t dt + \sigma x_t dW_t
    ```

Basic functionality (more complete description and examples are provided in the jupyter notebook example.ipynb)
```python
'''Init the process'''
'''time interval, number of trajectories, number of time steps'''
T, Nx, Nt = 1, 100, 300

theta, mu, nu = 10, 2, 1
params = [theta, mu, nu]

process = OUProcess(params, T, Nx, Nt)

'''Set stationary state as init state (if need)'''
process.init_state = process.StationaryState()

'''get the solution'''
'''set dwt to fix Wiener process (if need)'''
dwt = np.random.normal(0, 1, size = (Nt, Nx)) * np.sqrt(T/Nt)
t, xt = process.SolutionEuler(dwt)
```

The project is made in Sirius University, Sochi, Russia; "Financial mathematics and financial technologies" department as a part of Statistics for stochastic processes academical course.

For the matematical description see
M.E. Semenov, G.V. Fedorov, Statistics for stochastic processes
S.M. Iacus, Simulation and Inference for Stochastic Differential Equations with R Examples