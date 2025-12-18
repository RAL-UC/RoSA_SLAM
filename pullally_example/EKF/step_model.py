# step_model: integra el modelo de un proceso entregando
#             el estado luego de un tiempo "step size" a partir del tiempo
#             actual t0.  
#             La función es genérica y puede integrar cualquier modelo
#             La función puede optimizarse limitando los pasos intermedios
#             de integración.  Actualmente el intervalo "step size" se
#             halla dividido en 10 subpasos.
# 
# Inputs:
#   model (como un string entre ' ')
#   t0 initial time
#   step size
#   x0 intial state
# Output:
#   x(t+step size)
#   auxiliary outputs set by the user
#
# [t,x]=step_model('model_particle',[1;0.1],[0.1;0.05;pi])
#
# modificación de Miguel Torres-Torriti (c) 2012
#

import numpy as np
from scipy.integrate import solve_ivp
from timeit import default_timer as timer 
from functools import partial  # https://docs.python.org/3.6/library/functools.html
                               # required to pass additional parameters to solve_ivp


def step_model(model, u, t0, sigma_v, step_size, x0):

    tfinal = t0 + step_size
    Nsamples = 5 #10+1 changed 29 feb 24

    tX = np.linspace(t0, tfinal, Nsamples) # tiempos en los que se guarda la evaluación de las ode
    
    # solve_ivp
    x = solve_ivp(partial(model, u=u, sigma_v=sigma_v), (t0, tfinal), x0, method='RK45',teval=tX) #RK23
    # x.t --> tndarray, shape (n_points,) Time points.
    # x.y --> yndarray, shape (n, n_points) Values of the solution at t.

    return x.t, x.y
    
def main(): # Test step_model

    # Set the seed of the random number generator to zero
    # to have always the same sequence of random numbers.
    np.random.seed(0)

    t = 0.
#    x = np.array([1., 0., np.pi/4., 250.,    0., 0.01745329251994329576923690768489]) 
#                #[0 , 0 ,    45°  , 900 km/h, 0, 1°/s]

    x = np.array([0., 0., 0.]) #initial state
    #u = np.array([0.125, 0.] ) # v -->0.125 d = 0.25m in 2sec
    u = np.array([0., 0.7854] ) # w -->0.7854 theta = 90grad in 2sec
    sigma_v = np.array([0., 0.])
    
    DeltaT = 2.
    
    import model
    
    start = timer() 
    t, x = step_model(model.model_particle, u, t, sigma_v, DeltaT, x)
    #print(t.shape, x.shape)
    #print("final time: " + str(t[-1]) + "  state at final time: " + str(x[:,-1]))
    print("with CPU:", timer()-start) 
    
if __name__ == '__main__': main()