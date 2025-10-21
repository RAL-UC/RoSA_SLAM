import numpy as np
import matplotlib.pyplot as plt

from EKF.step_model import step_model
#from hausdorff.min_hd_v3 import Min_DH_v2

from utils.param_pullally  import STEP_SCAN_VELODYNE, MAP_RES, ALPHA
from utils.pix2meters_v4 import x2screen, y2screen, screen2x, screen2y
from utils.njitted_functions import njitted_occ_grid_v0, njitted_calcular_voronoi, njitted_convert_maplog2mapocc



def match_voronoi_v00 (V_img_g, laser_points, P0, lambd):#, XMAX, YMAX):

    # === Parameters for the Hausdorff Matching Process ===
    #lambd = 0.7 # Selection ration of partial Hausdorff distances, h_K(T(B),A):
                # satisfying lambda=K/N, for 1<K<q, where q is the number of
                # points in the reference image.         
    # --- Hausdorff Matching Tolerance ---
    #tol=2
    #--------------------------------------------------------------------------
    # calcular voronoi(imagen)
    #img_g=(current_map.astype(int))*255
    #V_img_g = calcular_voronoi(img_g)
    
    # --- Initial Hausdorff Matching Point ---
    #P0=[x_0, y_0]#punto de medición del robot
    MP0=[P0[0],P0[1],0,P0[3]]   #[row col]=[y x range_bias heading]
                            #where (x,y) are the position coordinates
                            #of the robot in units of 'pixels', and
                            #[range_bias heading] are in units of 'pixels' and 'degrees'.  
    
    MP1,_,prom= Min_DH_v2(laser_points[:, 0], laser_points[:, 1], MP0, lambd, V_img_g) #, tol, 0
    mean_prom = np.mean(np.array(prom))
    MP1[3]= ((MP1[3]*3)/np.pi) #))#
    
    return MP1, prom[-1]#mean_prom #prom 

def find_error(x_1, x_2 ):
    e_x = np.abs(x_1 - x_2)

    print(" values with variable sample time T ")
    print("x  component mean: " + "%.4f"%np.mean(e_x[:,0]) + "  std: " + "%.4f"%np.std(e_x[:,0]))
    print("y  component mean: " + "%.4f"%np.mean(e_x[:,1]) + "  std: " + "%.4f"%np.std(e_x[:,1])) 
    print("theta  component mean: " + "%.4f"%np.mean(e_x[:,2]) + "  std: " + "%.4f"%np.std(e_x[:,2])) 
    fig, axs = plt.subplots(3)
    fig.suptitle('Error comparison')
    axs[0].plot(e_x[:,0], label= 'x[m] T variable ')
    axs[0].legend(loc='lower right')
    axs[1].plot(e_x[:,1], label= 'y[m] T variable')
    axs[1].legend(loc='lower right')
    axs[2].plot(e_x[:,2], label= 'theta[rad] T variable')
    axs[2].legend(loc='lower right')
    plt.show()
    return 

# function converts csv velodyne scan data to polar cordinates
# and at the same time it filters the data in angle and range distance
# inputs:  points - list of lidar points in one lecture or step 
#          theta_max, theta_min - limits to filter the data in angle (persons next to go1)
#          d_max, d_min - limits to filter longitud of data (max ray or min ray)
#          example: theta_max, theta_min, d_min, d_max = 2.6, -2.6, 6.5,  0.5
# output: 1 array shape[n,2] with polar data of the lidar rays ([radios, angles])
def make_data_polar_filter_coord(points, theta_max, theta_min, d_max, d_min):
    read_scan = []
    angle_scan_start = -np.pi #value given of velodyne datasheet
    for i in range (len(points)):
        thetha_velodine = angle_scan_start + STEP_SCAN_VELODYNE*i
        if thetha_velodine >= theta_min and thetha_velodine <= theta_max:
            if points[i] >= d_min and points[i] < d_max:
                read_scan.append ([points[i], thetha_velodine])
    return np.array(read_scan)


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

def filter_coord(points, theta_max, theta_min, d_max, d_min):
    read_scan = []
    for r, tetha in zip(points[:, 0], points[:, 1]):
        if r >= d_min and r < d_max:
                read_scan.append ([r, tetha])
    return np.array(read_scan)

def make_data_for_occmap(points, theta_max, theta_min, d_max, d_min):
    read_scan = []
    angle_scan_start = -np.pi #value given of velodyne datasheet
    for i in range (len(points)):
        thetha_velodine = angle_scan_start + STEP_SCAN_VELODYNE*i
        if thetha_velodine >= theta_min and thetha_velodine <= theta_max:
            if points[i] >= d_min and points[i] < d_max:
                read_scan.append ([points[i], thetha_velodine])
            if points[i] >= d_max:
                read_scan.append ([d_max, thetha_velodine])
    return np.array(read_scan)

def make_data_for_occmap_plus(points, theta_max, theta_min, d_max, d_min):
    read_scan = []
    angle_scan_start = -np.pi #value given of velodyne datasheet
    for i in range (len(points)):
        thetha_velodine = angle_scan_start + STEP_SCAN_VELODYNE*i
        if thetha_velodine >= theta_min and thetha_velodine <= theta_max:
            if points[i] >= d_min and points[i] < d_max:
                read_scan.append ([points[i], thetha_velodine - (STEP_SCAN_VELODYNE)])
                read_scan.append ([points[i], thetha_velodine])
                read_scan.append ([points[i], thetha_velodine + (STEP_SCAN_VELODYNE)])
            if points[i] >= d_max:
                read_scan.append ([d_max, thetha_velodine])
    return np.array(read_scan)

def map_initial_estimate_v0(size_x_pix, size_y_pix):
    Ns_x = size_x_pix+1 # columns
    Ns_y = size_y_pix+1 # rows
    map_prob_occ = 0.5*np.ones((Ns_y,Ns_x))
    map_log_odds = np.zeros((Ns_y,Ns_x))
    return map_prob_occ, map_log_odds



def find_next_state (model, x_estados, x_predichos, sig_v, delta_t, u, time_before):
    last_position = x_estados[-1]
    _, x_aux = step_model(model, u, time_before, sig_v, delta_t, last_position)
    x_predichos = np.vstack((x_predichos,(x_aux.T)[-1]))
    return x_predichos

def hausdorff_sensor(l_data7, map_prob_occ, x_k_pred, prom_list):#,i):
    v_distances = njitted_calcular_voronoi(map_prob_occ)
    #Pose0=[x2screen(x_k_pred[i] [0]), y2screen(x_k_pred[i] [1]), 0 ,x_k_pred[i] [2] ]
    Pose0=[x2screen(x_k_pred[0]), y2screen(x_k_pred [1]), 0 ,x_k_pred[2] ]
    #Pose1, prom_match = filtered_and_match(v_distances, l_data7, Pose0, False)
    Pose1, prom_match = match_voronoi_v00(v_distances, l_data7, Pose0, 0.7)
    prom_list.append(prom_match)
    xpose_m = [screen2x(Pose1[0]), screen2y(Pose1[1]), Pose1[3]] 
    return xpose_m, prom_list

def upload_map(l_data, x_k, map_prob_occ, map_log_odds):
    map_log_odds, xmin, xmax, ymin, ymax = njitted_occ_grid_v0(map_log_odds, x_k, l_data)
    map_prob_occ = njitted_convert_maplog2mapocc(map_log_odds, map_prob_occ)
    suma  = sum(sum(x) for x in map_prob_occ[ymin : ymax, xmin : xmax])
    return map_prob_occ, map_log_odds, suma



def join_scan2map(scan0, scan1):
    new_scan = []
    for i in range(len(scan0)):
        if scan0[i] == scan1[i]:
            new_scan.append(scan0[i])
        else:
            if scan0[i] == float("inf"):
                new_scan.append(scan1[i])
            else:
                if scan0[i]>scan1[i]:
                    new_scan.append(scan0[i])
                else:
                    if scan1[i] == float("inf"):
                        new_scan.append(scan0[i])
                    else:
                        if scan1[i]>scan0[i]:
                            new_scan.append(scan1[i])
    return np.array(new_scan)

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

def join_scan_data(points0, points1, theta_max, theta_min, d_max, d_min):
    lidar_scan0 = make_data_polar_filter_coord(points0, theta_max, theta_min, d_max, d_min)
    lidar_scan1 = make_data_polar_filter_coord(points1, theta_max, theta_min, d_max, d_min)
    lidar_data = np.concatenate((lidar_scan0, lidar_scan1))
    points_map = join_scan2map(points0, points1)
    mapa_data = make_data_for_occmap(points_map, theta_max, theta_min, d_max, d_min)
    return lidar_data, mapa_data


def join_scan_data_plus(points0, points1, theta_max, theta_min, d_max, d_min):
    lidar_scan0 = make_data_polar_filter_coord(points0, theta_max, theta_min, d_max, d_min)
    lidar_scan1 = make_data_polar_filter_coord(points1, theta_max, theta_min, d_max, d_min)
    lidar_data = np.concatenate((lidar_scan0, lidar_scan1))
    points_map = join_scan2map_min(points0, points1)
    mapa_data = make_data_for_occmap_plus(points_map, theta_max, theta_min, d_max, d_min)
    return lidar_data, mapa_data

def join_scan_data_plus_1(points0, theta_max, theta_min, d_max, d_min):
    lidar_scan0 = make_data_polar_filter_coord(points0, theta_max, theta_min, d_max, d_min)
    mapa_data = make_data_for_occmap_plus(points0, theta_max, theta_min, d_max, d_min)
    return lidar_scan0, mapa_data