import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, cuda

from EKF.step_model import step_model
#from hausdorff.min_hd_v3 import Min_DH_v2

from utils.param_pullally  import STEP_SCAN_VELODYNE, MAP_RES, ALPHA
from utils.pix2meters_v4 import x2screen, y2screen, screen2x, screen2y
from utils.njitted_functions import njitted_occ_grid_v0, njitted_calcular_voronoi, njitted_convert_maplog2mapocc

def join_scan2map_min(scan0, scan1):
    new_scan = []
    for i in range(len(scan0)):
        if scan0[i] == scan1[i]:
            new_scan.append(scan0[i])
        else:
            if scan0[i] == float("inf"):
                new_scan.append(scan1[i])
            else:
                if scan0[i]>scan1[i]:
                    new_scan.append(scan1[i])
                else:
                    if scan1[i] == float("inf"):
                        new_scan.append(scan0[i])
                    else:
                        if scan1[i]>scan0[i]:
                            new_scan.append(scan0[i])
    return np.array(new_scan)

# function converts csv velodyne scan data to polar cordinates
# and at the same time it filters the data in angle and range distance
# inputs:  points - list of lidar points in one lecture or step 
#          theta_max, theta_min - limits to filter the data in angle (persons next to go1)
#          d_max, d_min - limits to filter longitud of data (max ray or min ray)
#          example: theta_max, theta_min, d_min, d_max = 2.6, -2.6, 6.5,  0.5
# output: 1 array shape[n,2] with polar data of the lidar rays ([radios, angles])
#         1 array shape[len(points),2] with polar data of the lidar rays ([radios, angles])
def make_data_polar_mhd_map(points, theta_max, theta_min, d_max, d_min):
    read_scan = []
    map_data = []
    angle_scan_start = -np.pi #value given of velodyne datasheet
    for i in range (len(points)):
        thetha_velodine = angle_scan_start + STEP_SCAN_VELODYNE*i
        if thetha_velodine >= theta_min and thetha_velodine <= theta_max:
            if points[i] >= d_min and points[i] < d_max:
                read_scan.append ([points[i], thetha_velodine])
                map_data.append ([points[i], thetha_velodine])
            if points[i] >= d_max:
                map_data.append([d_max, thetha_velodine])
    return np.array(read_scan), np.array(map_data)


def join_scan_data_plusv6(points0, points1, theta_max, theta_min, d_max, d_min):
    points_map = join_scan2map_min(points0, points1)
    lidar_data, mapa_data = make_data_polar_mhd_map(points_map, theta_max, theta_min, d_max, d_min)
    return lidar_data, mapa_data


#@jit(nopython=True)
@jit(target_backend ='cuda')   
def upload_mapv6(l_data, x_k, map_prob_occ, map_log_odds):
    map_log_odds, xmin, xmax, ymin, ymax = njitted_occ_grid_v0(map_log_odds, x_k, l_data)
    map_prob_occ = njitted_convert_maplog2mapocc(map_log_odds, map_prob_occ)
    #suma  = sum(sum(x) for x in map_prob_occ[ymin : ymax, xmin : xmax])
    return map_prob_occ, map_log_odds#, suma