import numpy as np
from utils.param_pullally import SCALE
from hausdorff.njit_hausdorff import change_int_cordinates

def hausdorff_xy_v3(x_points, y_points, k, V): #tol, Rumbo --se quita xq no se usan
    c,d = V.shape   #sup de voronoi
    longitud = len(x_points)
    Vec = np.zeros((longitud,2))
    err1, err2 = 0. , 0.
    
    for n in range(longitud):
        err1 = 0
        err2 = 0
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
        #Vec.append([V[int(C)-1][int(D)-1]+err1+err2, n]) # cambio de v3
        Vec[n][0] = V[int(C) - 1][int(D) - 1] + err1 + err2
        Vec[n][1] = n
    print(Vec.shape, Vec[:,0].shape)
    lista_h = Vec[Vec[:, 0].argsort()] # cambio de v3, lista de todas las DH ordenadas 
    #lista_h = np.sort(Vec[:,0]) 
    DPH = int(lista_h.shape[0]*k)
    print(DPH)
    maxDH = lista_h[DPH][0] # distancia maxima de haussdorf
    while(maxDH>30):
        k= k-0.02
        DPH = int(lista_h.shape[0]*(k))
        maxDH = lista_h[DPH][0] # distancia maxima de haussdorf
        if k <0.4: #before 0.6 change feb 2024
            break
    #prom = np.mean(lista_h[0:DPH, 0]) # promedio de las k DH aceptadas.
    #prom_off = np.mean(lista_h[DPH+1:, 0]) # promedio de las k DH aceptadas.
    n_off = lista_h[DPH:, 1]
    #print(DPH)
    return n_off.astype(int)

def filter_data_mhd(data, pose,img_distances, scale  = SCALE):
    xf_points, yf_points = change_int_cordinates (data[:, 0], data[:, 1], pose[0], pose[1], pose[2], pose[3], SCALE)
    n_off = hausdorff_xy_v3(xf_points, yf_points, 0.7, img_distances)
    n_data = np.delete(data, n_off, axis=0)
    return n_data