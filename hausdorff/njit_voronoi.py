from numba import jit
import numpy as np
import cv2

def find_voronoi_img (map_prob, treshold = 0.7):
    img2 = np.where(map_prob > treshold, 255, 0)
    img2 = np.array(img2,np.uint8)
    #kernel = np.ones((3, 3), np.uint8) 
    #kernel = np.ones((5, 5), np.uint8)
    #kernel = np.ones((4, 4), np.uint8) 
    kernel = np.ones((2, 2), np.uint8) 
    img_dilation = cv2.dilate(img2, kernel, iterations = 1) 
    dt_map = np.ones((img2.shape[0]+2, img2.shape[1]+2))
    img2 = np.array(img_dilation,np.int32)
    v_distances = jit_calcular_voronoi(img2, dt_map)
    #v_distances = jit_calcular_voronoi(img_dilation, dt_map)
    return v_distances

# in:   I is the image, the inside data must be in enteros int
#       dt_aux in an array of the size of I+2, it means (Irows+2, Icols+2), the data must be in enteros int
# out:  The result dt_aux is and Image of the same size of I with the distances between the points.
@jit(nopython=True)
def jit_calcular_voronoi(I, dt_aux):

    Irows, Icols = I.shape
    max_dist = Irows*Icols
    
    rect = [0, 0, Irows, Icols]
    AI = [rect[0], rect[1]]
    BI = [rect[0]+rect[2], rect[1]+rect[3]]

    dt_aux = dt_aux*max_dist
    for i in range(Irows):
        for j in range(Icols):
            if I[i,j] > 0:
                dt_aux[i+1,j+1] = 0

    for i in range(AI[0]+1, BI[0]+1):
        for j in range(AI[1]+1, BI[1]+1):
            dt_aux[i,j] = jit_update_distance(dt_aux[i-1:i+2,j-1:j+2])
            
    for i in range(BI[0], AI[0], -1):
        for j in range (BI[1], AI[1], -1):
            dt_aux[i,j] = jit_update_distance(dt_aux[i-1:i+2,j-1:j+2])

    return dt_aux[1:Irows+1, 1:Icols+1]


# in:   neighborts is a (2x2) matrix with elements (int)
# out:  distance of the element (int)
@jit(nopython=True)
def jit_update_distance(neighbors):

    min = int(999999999999999)

    if neighbors[0,1] < min:
        min = neighbors[0,1]

    if neighbors[1,0] < min:
        min = neighbors[1,0]

    if neighbors[1,2] < min:
        min = neighbors[1,2]

    if neighbors[2,1] < min:
        min = neighbors[2,1]
    
    if neighbors[1,1] > min:
        d = min+1
    else:
        d = neighbors[1,1]

    return d