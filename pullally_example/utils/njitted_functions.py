import numpy as np
from numba import jit, njit

#from voronio.calcular_v import Update_Distance

from utils.param_pullally import XMAX_pix, YMAX_pix
from utils.param_pullally import S_MAX_RANGE, ALPHA, BETA, L_0, L_OCC, L_FREE, THETA_MX, THETA_MN

from utils.pix2meters_v4 import x2screen, y2screen, screen2x, screen2y


def sensor_to_Cartesian_world(r_s, theta_s, x):
    theta_r = x[2]
    x_s_w = x[0] + r_s*np.cos(theta_s + theta_r)
    y_s_w = x[1] + r_s*np.sin(theta_s + theta_r)
    return x_s_w, y_s_w

def inverse_sensor_model(map_cell_position, x, z): #devuelve la probabilidad viene de lo de inverse sensor model stachniss  
    x_r = x[0]
    y_r = x[1]
    theta_r = x[2]
    
    r_s = z[:, 0] #z[0,:]
    theta_s = z[:, 1] #z[1,:]


    m_x = map_cell_position[0]
    m_y = map_cell_position[1]
    
    # Compute the robot-to-cell distance
    r = np.sqrt((m_x-x_r)**2 + (m_y-y_r)**2) 
    
    # Compute the robot-to-cell direction
    phi = np.arctan2(m_y - y_r, m_x - x_r) - theta_r
    # agregado por Pao para mantener los angulos entre -pi y pi
    if phi < -np.pi:
        phi = phi + (2*np.pi )
    if phi > np.pi:
        phi = phi - (2*np.pi)
    
    # Find closest sensor beam to the robot-to-cell direction
    # solving k = arg min_j |phi - theta_list_j|
    #    
    angles_diff = np.abs(phi - theta_s)
    k = np.where( angles_diff == min(angles_diff) )[0][0]    
    # 
    theta_k = theta_s[k]

    """
    r_k = r_s[k] 
    cell_log_odds = L_0 #l
    if (r > min(S_MAX_RANGE, r_k + ALPHA/2)) or (np.abs(phi-theta_k) > BETA/2):
        cell_log_odds = L_0
    elif (r_k < S_MAX_RANGE) and (np.abs(r - r_k) < ALPHA/2):
        cell_log_odds = L_OCC
    elif (r <= r_k-ALPHA/2):
        cell_log_odds = L_FREE
    """
    r_k = r_s[k] + ALPHA/2
    cell_log_odds = L_0 #l
    if (r > min(S_MAX_RANGE, r_k + ALPHA)) or (np.abs(phi-theta_k) > BETA/2):#if theta_k<= THETA_MX and theta_k >= THETA_MN:
        cell_log_odds = L_0
    elif (r_k < S_MAX_RANGE) and (np.abs(r - r_k) < ALPHA):
        cell_log_odds = L_OCC
    elif (r <= r_k-ALPHA):
        cell_log_odds = L_FREE
    
    return cell_log_odds

def convert_map_log_odds_to_map_prob_occ(map_log_odds, map_prob_occ):
    for i in range(map_log_odds.shape[0]):
        for j in range(map_log_odds.shape[1]):
            e = np.exp(map_log_odds[i,j])
            map_prob_occ[i,j] = e/(1+e)
    return map_prob_occ


@jit(target_backend='cuda')
def njitted_convert_maplog2mapocc(map_log_odds, map_prob_occ):
    for i in range(map_log_odds.shape[0]):
        for j in range(map_log_odds.shape[1]):
            e = np.exp(map_log_odds[i,j])
            map_prob_occ[i,j] = e/(1+e)
    return map_prob_occ

njitted_x2screen = njit()(x2screen)
njitted_y2screen = njit()(y2screen)
njitted_screen2x = njit()(screen2x)
njitted_screen2y = njit()(screen2y)
njitted_sensor2cartesian_world = njit()(sensor_to_Cartesian_world)
njitted_inverse_sensor_model = njit()(inverse_sensor_model)
#njitted_convert_maplog2mapocc = njit()(convert_map_log_odds_to_map_prob_occ)
#njitted_update_distance = njit()(Update_Distance)

# Perceputal field is defined by 3 corners x_r, x_s_1, x_s_2
# To speed up the search, the map is scanned for all cells within the bounding box
# To find find the containing bounding box, the center and circle arc points are checked to find the bounding points
#output: pixels: x_min, x_max, y_min, y_max
#@jit(nopython=True)
@jit(target_backend='cuda')
def njitted_compute_perceptual_field(x, z):
    # Values are computed in matrix coordinates
    x_min, y_min = min(njitted_x2screen(x[0]), XMAX_pix), min(njitted_y2screen(x[1]), YMAX_pix)
    x_max, y_max = max(njitted_x2screen(x[0]), 0), max(njitted_y2screen(x[1]), 0)
    #for k in range(len(z[1,:])):
    for k in range(len(z[:, 1])):
        x_s_w, y_s_w = njitted_sensor2cartesian_world(S_MAX_RANGE+2*ALPHA, z[k][1], x)# z[1,k], x) #
        #print(z[1,k])
        x_min, y_min = min(njitted_x2screen(x_s_w), x_min), min(njitted_y2screen(y_s_w), y_min)
        x_max, y_max = max(njitted_x2screen(x_s_w), x_max), max(njitted_y2screen(y_s_w), y_max)
    # Clip to valid matrix indeces
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(XMAX_pix, int(x_max)), min(YMAX_pix, int(y_max)) 
    return x_min, x_max, y_min, y_max

#@jit(nopython=True)
@jit(target_backend='cuda')
def njitted_occ_grid_v0(map_log, x_m, z_m): 
    x_min, x_max, y_min, y_max = njitted_compute_perceptual_field(x_m, z_m)
    #print(x_min, x_max, y_min, y_max)
    # For cells in perceptual field
    for i in range(y_min, y_max+1):
        m_y = njitted_screen2y(i)#i*map_res 
        for j in range(x_min,x_max+1):
            m_x = njitted_screen2x(j)#j*map_res
            map_log[i,j] = map_log[i,j] + njitted_inverse_sensor_model([m_x,m_y], x_m, z_m) #probabilidad en escala lograritmica o plausibilidad
    return map_log, x_min, x_max, y_min, y_max



@jit(nopython=True)
def njitted_calcular_voronoi(map_prob_occ):
    n_max = np.max(map_prob_occ)*0.7 #values of treshold
    I_ = np.where(map_prob_occ > n_max, 1, 0)

    I = I_*255 #include by Pao... formato de imagen para cálculo

    #I = np.array(I, dtype = np.double)
    Irows, Icols = I.shape
    max_dist = Irows*Icols
    
    rect = [0, 0, Irows, Icols]
    AI=[rect[0], rect[1]]
    BI=[rect[0]+rect[2], rect[1]+rect[3]]

    dt_aux = np.ones((Irows+2, Icols+2))*max_dist
    
    for i in range(Irows):
        for j in range(Icols):
            if I[i,j] > 0:
                dt_aux[i+1,j+1] = 0

    for i in range(AI[0]+1, BI[0]+1):
        for j in range(AI[1]+1, BI[1]+1):
            dt_aux[i,j] = njitted_update_distance(dt_aux[i-1:i+2,j-1:j+2])
            
    for i in range(BI[0], AI[0], -1):
        for j in range (BI[1], AI[1], -1):
            dt_aux[i,j] = njitted_update_distance(dt_aux[i-1:i+2,j-1:j+2])

    return dt_aux[1:Irows+1, 1:Icols+1]