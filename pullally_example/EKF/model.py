# Modelo dinámico de una partícula (2D)
#
# Vector de estado robot
# x0: x (posicion en coordenadas del mundo) 
# x1: y (posicion en coordenadas del mundo) 
# x2: theta (orientación en coordenadas del mundo relativa al eje x) 

# Vector de estado lanmarks
# x3: m_x1 (landmark posicion en coordenadas x del mundo) 
# x4: m_y1 (landmark posicion en coordenadas y del mundo) 
# the same for each landmark
# ..................


# Vector de control
# u0: V
# u1: W

# Vector de ruido en actuadores y perturbaciones en estado
# v1: n_V   o n_T
# v2: n_W
#
#Ejemplo:
# xdot = modelo_particula(t0,[0., 0., 0.],[0.1, 0.])
#
#basado en Miguel Torres-Torriti (c) 2012
#

import numpy as np
from timeit import default_timer as timer 


# xdot = modelo_particula(x, t, u, sigma_v)

# Nota: Aunque no se use la variable tiempo t, la declaración del modelo
# debe contener al menos las variables t y x, de modo que:

# xdot = modelo_avion(t, x, u, sigma_v):

def model_particle(t, x, u, sigma_v):

    v = sigma_v*np.random.randn(len(sigma_v)) # '*' is the element-wise multiplication
                                              # when using NumPy arrays.
    c = np.cos(x[2])
    s = np.sin(x[2])
    xdot = np.array([
         (u[0] + v[0])*c,  # x_dot
         (u[0] + v[0])*s,  # y_dot
         (u[1] + v[1])   # theta_dot
         ])   
    return xdot

def main(): # Test modelo_avion

    # Set the seed of the random number generator to zero
    # to have always the same sequence of random numbers.
    np.random.seed(0)

    t = 0.
    #x = np.array([1, 0, np.pi/4.]) 
    x = np.array([0., 0., np.pi/2]) 
    u = np.array([2.5, 0] )
    sigma_v = np.array([0.001, 0.001])

    start = timer() 
    x_dot = model_particle(t, x, u, sigma_v)
    print(x_dot)
    print("with CPU:", timer()-start) 
    
if __name__ == '__main__': main()