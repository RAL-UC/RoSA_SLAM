# Jacobianos de una particula
# movimiento en el plano 2D 
#
# modificaciòn a Miguel Torres-Torriti (c) 2012
#

# Vector de estado particula
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
#
# Ejemplo:
# [A, B, G, C, D, H] = jacobianos_particle([1;0;pi/4;1000;0;0],[1;0.1])
#

import numpy as np
from timeit import default_timer as timer 

# A, B, G, C, D, H = jacobianos_avion(x,u)

def jacobianos_particle(x, u):
    c = np.cos(x[2])
    s = np.sin(x[2])
    
    A = np.array( # x1 =dstate/dx  x2 =dstate/dy  x3 =dstate/dtheta
        [[0, 0, -u[0]*s],
         [0, 0, u[0]*c],
         [0, 0,  0     ]])

    B = np.array( # u1 u2
        [[ c,   0   ],
         [ s,   0   ],
         [ 0,   1   ]])
    
    G = np.array( # v1 v2    ·· pregntar, serìa del algoritmo de hausdorff
        [[ 1,   0,   0  ],
         [ 0,   1,   0  ],
         [ 0,   0,   1  ]])
    """
    C = np.array( # x1  x2  x3
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
    """
    C = np.array( # x1  x2  x3
        [[1, 0 , 0],
         [0, 1 , 0],
         [0, 0 , 1]])

    D = np.array( # u1  u2
        [[0, 0 ],
         [0, 0 ],
         [0, 0 ]])
    
    H = np.array( # w1 w2 w3 --- seria x y theta del sensor
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])
    

    return A, B, G, C, D, H


def main(): # Test jacobianos_avion

    x = np.array([1., 0., np.pi/4.]) 
                #[0 , 0 ,    45°  , 900 km/h, 0, 1°/s]
    u = np.array([1., 0.])

    start = timer() 
    A, B, G, C, D, H = jacobianos_particle(x, u)
    print(A, B, G, C, D, H)
    print("with CPU:", timer()-start) 

if __name__ == '__main__': main()