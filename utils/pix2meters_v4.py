from utils.param_pullally  import XMAX_pix, YMAX_pix, SCALE, S_MAX_RANGE

s_pix_range = 1.5*int(S_MAX_RANGE*SCALE) 
#s_pix_range = 1.1*int(S_MAX_RANGE*SCALE) 

def x2screen(x):
    return int(SCALE*x + (s_pix_range))# int(SCALE*x + XMAX_pix/2)#

def y2screen(y):
    return int(-SCALE*y + (YMAX_pix - s_pix_range )) # int(-SCALE*y + YMAX_pix/2) #

def screen2x(px):
    return (px - (s_pix_range)) / SCALE # (px - XMAX_pix/2) / SCALE #

def screen2y(py):
    return (- py + (YMAX_pix - s_pix_range)) / SCALE # (YMAX_pix/2 - py) / SCALE #
'''
def x_m2pix(x_array):
    return (SCALE*x_array).astype(int)    # no XMAX/2 xq ya esta incluido en P0

def y_m2pix(y_array):
    return (-SCALE*y_array).astype(int)  # no YMAX/2 xq ya esta incluido en P0
'''
# x_k:      input, pose in meters
# Pose0:    output, pose in pixels
def find_pose(x_k):
    Pose0 = [x2screen(x_k[0]), y2screen(x_k[1]), 0 , x_k[2] ]  
    #[row col]=[y x range_bias heading]
    #where (x,y) are the position coordinates
    #of the robot in units of 'pixels', and
    #[range_bias heading] are in units of 'pixels' and 'degrees'.
    return Pose0
