from numba import jit
import numpy as np

from utils.param_pullally import SCALE
from hausdorff.njit_voronoi import jit_calcular_voronoi
from utils.pix2meters_v4 import x2screen, y2screen, screen2x, screen2y

from utils.match_common import filter_coord
from utils.param_pullally import THETA_MX, THETA_MN, S_MAX_RANGE, D_MN 

MATCH_MAX_V = 30

def change_int_cordinates (points, ang, l, m, Rs, As, escala = SCALE):
    angulos = ang + As
    points = points + Rs
    cs = points*np.cos(angulos)
    sn = points*np.sin(angulos)

    X = l + (cs * escala).astype(int)
    Y = m - (sn * escala).astype(int)
    return X, Y

# input  (x_points, y_points)--> lidar points in int xy cordiates
#        k --> punto de corte de los puntos más cercanos
#        V --> imagen de distancias encotrada con Voronoiç
# output maxDH --> distancia máxima al punto de corte k
#        lista_h--> lista de todas las distancias encontradas
#        prom--> promedio de todas las distancias hasta el punto de corte k
@jit(nopython=True)
def jit_hausdorff_int_xy(x_points, y_points, k, V):
    c,d = V.shape   #sup de voronoi
    longitud = len(x_points)
    Vec = np.zeros(longitud)
    err1, err2 = 0. , 0.
    for n in range(longitud):
        err1 = 0.
        err2 = 0.
        if x_points[n]>d:
            D = d
            err1 = x_points[n]-d
        else:
            if x_points[n]<1:
                D = 0
                err1 = np.absolute(1-x_points[n])
            else:
                D = x_points[n]
        
        if y_points[n]>c:
            C = c
            err2 = y_points[n]-c
        else:
            if y_points[n]<1:
                C = 1
                err2 = np.absolute(1-y_points[n])
            else:
                C = y_points[n]

        #Vec.append(V[int(C)-1][int(D)-1]+err1+err2) 
        Vec[n] = V[int(C) - 1][int(D) - 1] + err1 + err2

    #Vec = np.array(Vec)
    lista_h = np.sort(Vec) # lista de todas las DH ordenadas 
    DPH = int(lista_h.shape[0]*k)
    maxDH = lista_h[DPH] # distancia maxima de haussdorf
    prom = np.mean(lista_h[0:DPH]) # promedio de las k DH aceptadas.

    return maxDH, lista_h, prom




"""
P0 = [Xs0 Ys0 Rs0 As0] % Xs0 columnas, Ys0 filas
P(1) = posicion x en pixeles, obtenidas de la minimizacion del promedio DH
P(2) = posicion y en pixeles, obtenidas de la minimizacion del promedio DH
P(3) = correccion del rango, obtenidas de la minimizacion del promedio DH.
P(4) = correccion de rumbo, obtenidas de la minimizacion del promedio DH.
hk = lista de las distancias de Haussdorf de cada iteracion
prom_hk = promedio de las DH de cada iteracion.
"""

# input     points, ang --> filtered laser points that touch a surface in the maximum range of the sensor
#           P0 --> estimated point and pose where the robot is located in pixels
#           k --> value of porcentage of points used in the match
#           V --> voronoi image with the distances at the nearest line or point
# output    P--> estimated point after the match algorithm in pixels
#           prom_hk --> promedios de las distancias buscando el mejor match, el ultimo promedio indica que tan buen mtach se encontro
def min_MHD (points, ang, P0, k, V): #,tol,Rumbo
    W1 = 2**3 #4 # Ventana del gradiente,potencias de 2.
    minprom = 10000
    minDH = 10000
    PuntoP = P0
    prom_hk = []
    hk = []
    dataP = [0, 0, 0, 0]
    while W1 >= 1:
        P0 = np.array([0, 0, 0, 0])
        while not((P0[0] == PuntoP[0]) and (P0[1] == PuntoP[1]) and (P0[2] == PuntoP[2]) and (P0[3] == PuntoP[3])):
            P0 = PuntoP
            #print("P0"+str(P0))
            #for i in range(-1,2):
            #    for j in range(-1,2):
            #        for Ang in [-0.1*np.pi/18, 0. , 0.1*np.pi/18]:
            for i in range(-2,3):
                for j in range(-2,3):
                    #for Ang in [-0.4*np.pi/18, -0.3*np.pi/18, -0.2*np.pi/18, -0.1*np.pi/18, 0.0, 0.1*np.pi/18, 0.2*np.pi/18, 0.3*np.pi/18, 0.4*np.pi/18]: #[-0.1*np.pi/18, 0. , 0.1*np.pi/18]: #
                    #for Ang in [ -0.2*np.pi/18,-0.15*np.pi/18,-0.1*np.pi/18,-0.05*np.pi/18, 0.0, 0.05*np.pi/18, 0.1*np.pi/18, 0.15*np.pi/18, 0.2*np.pi/18]:
                    for Ang in range(-21, 22, 3):
                        AngCorregido = 0.95*(P0[3]+(Ang/100)*W1) #0.95
                        xpoints, ypoints = change_int_cordinates (points, ang, P0[0]+i*W1, P0[1]+j*W1, P0[2], AngCorregido, SCALE)
                        maxDH, lista_hk, prom = jit_hausdorff_int_xy(xpoints, ypoints, k, V) #, tol, Rumbo
                        if prom < minprom:
                            minprom = prom
                            minDH = maxDH
                            PuntoP = [P0[0]+i*W1, P0[1]+j*W1, P0[2], AngCorregido]
                            dataP = [i, j, W1, Ang]
                            prom_hk.append(minprom)
                            hk.append(lista_hk)
                        else:
                            if ((prom == minprom) and (maxDH < minDH)):
                                minDH = maxDH
                                PuntoP = [P0[0]+i*W1, P0[1]+j*W1, P0[2], AngCorregido]
                                dataP = [i, j, W1, Ang]
                                prom_hk.append(minprom)
                                hk.append(lista_hk)
        W1 = int(W1/2)
    P = PuntoP
    hk = np.array(hk)
    prom_hk = np.array(prom_hk)
    #print("mean_p1: " + str(np.mean(hk, axis = 1)) + "std_p1: " + str(np.std(hk, axis = 1)) + "mean:" + str(np.mean(prom_hk)) + "std: "+ str(np.std(prom_hk)))
    return P, prom_hk,dataP# np.array(hk), np.array(prom_hk)


# input     l_data7--> array[n,2] of filtered laser points that touch a surface in the maximum range of the sensor
#           v_distances--> voronoi distances of the previous map of probabililties (data between 0 and 1, 0,5 means no data, near to 1 is and object in that point)
#           Pose0 -->estimated point and pose where the robot is located in pixels (depents of the scale)
#           prom_list --> lista del promedio de matches
#           lambd --> 0.7 value of porcentage of points used in the match´
# outpt     xpose_m --> estimated point after the match algorithm in meters (depents of the scale)
#           prom_list --> lista del promedio de matches adding the new match value
def MHD_sensor(i, l_data7, v_distances, Pose0, prom_list, lambd = 0.7):#,i):
    #img2 = np.where(map_prob_occ > 0.7,255, 0)
    #dt_map = np.ones((img2.shape[0]+2, img2.shape[1]+2))
    #v_distances = jit_calcular_voronoi(img2, dt_map)
    #Pose0 = [x2screen(x_k_pred[0]), y2screen(x_k_pred [1]), 0 ,x_k_pred[2] ]  #[row col]=[y x range_bias heading]
                                                                            #where (x,y) are the position coordinates
                                                                            #of the robot in units of 'pixels', and
                                                                            #[range_bias heading] are in units of 'pixels' and 'degrees'.
    Pose1, prom_match, dataP = min_MHD(l_data7[:, 0], l_data7[:, 1], Pose0, lambd, v_distances)

    print("i : " + str(i) + "dataP : " +str(dataP))
    #if dataP[2] >= 8 and (dataP[0]!= 0 or dataP[1]!= 0 ):
    #    Pose1, prom_match, dataP = min_MHD (l_data7[:, 0], l_data7[:, 1], Pose1, lambd, v_distances)
    #    print("i : " + str(i) + "dataP : " +str(dataP))
    #Pose1, prom_match = min_MHD(l_data7[:, 0], l_data7[:, 1], Pose0, lambd, v_distances)
    prom_list.append(prom_match[-1])
    xpose_m = [screen2x(Pose1[0]), screen2y(Pose1[1]), Pose1[3]] 

    return xpose_m, prom_list
