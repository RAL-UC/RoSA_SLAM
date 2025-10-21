import numpy as np

#variables Mapa or World in meters
XMAX_m = 66#134 # 90[m] world width
YMAX_m = 40#90 # 22[m] world height
#XMAX_m = 4.4#134 # 90[m] world width
#YMAX_m = 4.4#90 # 22[m] world height
MAP_RES = 0.01 # [m] is the map resolution

#variables Mapa pixels
SCALE = 1/MAP_RES # when map_res = 0.01 m it means 1 pixel per 1 cm
XMAX_pix = int(XMAX_m*SCALE)
YMAX_pix = int(YMAX_m*SCALE)

#variables Lidar
STEP_SCAN_VELODYNE = (2*np.pi)/898 #0.007#0.005800 #  #value from datasheet and ros

#variables grid mapping
S_MAX_RANGE = 12.0 #d_mx# d_mx # maximum measurment distance
#S_MAX_RANGE = 2.0 #d_mx# d_mx # maximum measurment distance
#ALPHA = 0.04          # [m] obstacle thickness   4cm o 4pix
ALPHA = 0.06 
BETA  = (1/180)*np.pi # [rad] beam width   
L_0    = 0               # log-odds l_0   = log( p(m=1)/p(m=0) ) = log( 0.5/0.5 )  # probabilidad inicial
L_OCC  = np.log(0.8/0.2) # log-odds l_occ = log( p(m=1)/p(m=0) ) = log( 0.8/0.2 )  # probabilidad de lectura del sensor
L_FREE = np.log(0.2/0.8) # log-odds l_occ = log( p(m=1)/p(m=0) ) = log( 0.2/0.8 )  # probabilidad de q el sensor mida una celda libre y en realidad este ocupada

#variables data filter
THETA_MX, THETA_MN, D_MX, D_MN = 3.0, -3.0, 30.0, 0.50 #np.pi, -np.pi, 30.0, 0.50#2.45, -2.45, 100.0,  0.5

#THETA_MX, THETA_MN, D_MX, D_MN = np.pi, -np.pi, 30.0, 0.50 