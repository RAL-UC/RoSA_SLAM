import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm

from EKF.model import model_particle
from EKF.step_model import step_model
from EKF.jacobianos import jacobianos_particle

from utils.param_pullally import S_MAX_RANGE
from utils.param_pullally import XMAX_pix, YMAX_pix
from utils.param_pullally import THETA_MX, THETA_MN, D_MN 

from utils.match_common import map_initial_estimate_v0, upload_map
from utils.match_common import join_scan_data_plus

from hausdorff.njit_hausdorff import MHD_sensor
from hausdorff.njit_voronoi import find_voronoi_img
from hausdorff.filter_mhd import filter_data_mhd

from utils.arrange_data import arrange_npz_v4, arrange_results, arrange_resultsv6
from utils.pix2meters_v4 import find_pose
from utils.init_data import init_sigmas, init_data_p0, init_memories

from sklearn.linear_model import LinearRegression


def main():

    sigma_P, P_k0, Q_k, R_k = init_sigmas()
    # Input data of lidar, u[v,w], time, delta theta, gnss, etc
    file_path_data = input ("Ingrese el path de los datos uv npz ")
    data_C1t1 = np.load(str(file_path_data))
    print("Los elementos dentro del archivo son: " + str(data_C1t1.files))

    scan0_d, scan1_d, u_const_array, u_v_array, time_array, ts_array, x_k_gnss, u_datas_array = arrange_npz_v4 (data_C1t1)

    option_data = input ("Si desea cargar resultados anteriores presione Y")

    if option_data == 'Y' or option_data == 'y':
        # Input data of lidar, u[v,w], time, delta theta, gnss, etc
        file_result_data = input ("Ingrese el path de los resultados anteriores (npz): ")
        results_C1t1 = np.load(str(file_result_data), allow_pickle=True)
        gnss_data0, z_k, x_k_k, x_k1_k, x, m_log_odds, m_prob_occ, lista_hausdorff_prom, P_k_k, lista_errores, lista_sumll, tresholds, es_lost, counter, lista_segment, dict_segments, aux_dic, lista_e_ok = arrange_resultsv6 (results_C1t1)
        #gnss_data0, z_k, x_k_k, x_k1_k, x, m_log_odds, m_prob_occ, lista_hausdorff_prom, P_k_k, lista_errores, lista_sumll, tresholds, es_lost, counter, lista_e_ok = arrange_results (results_C1t1)
        init_val = len(x_k_k)
        linear_regressor = LinearRegression()
        array_dict = {index: value for index, value in enumerate(dict_segments.flatten())}
        dict_segments = array_dict[0]
        tresholds = np.array([1.2, 0.4, 2])
        ref_m, slope, prop = 0 , 0, 0.8
        print("len dict: "+ str (len (dict_segments)) )

    else:
        x_k0_m = np.array([0.0, 0.0, 0.0]) # measured value [x, y, theta]
        x, x_k_k, x_k1_k, z_k, P_k_k = init_data_p0 (x_k0_m, P_k0)
        lista_hausdorff_prom, lista_errores, lista_sumll, es_lost, std_list = init_memories ()
        x = x_k0_m.reshape((1, len(x_k0_m)))        # real
        m_prob_occ, m_log_odds = map_initial_estimate_v0(XMAX_pix, YMAX_pix)
        tresholds = np.array([1.2, 0.4, 2]) #np.array([0.25, 1.5, 0.3, 3]) # filter_d_error_data, map_update, max_angle_error, max_miss_maps
        init_val =0
        counter = 0
        linear_regressor = LinearRegression()
        lista_segment = []
        lista_ms = []
        lista_e_ok = []
        dict_segments ={}
        aux_dic = 0
        ref_m, slope, prop = 0 , 0, 0.8
    

    val = init_val + 120

    print(init_val, val)
    for k in range(init_val, val):
        lidar_data, mapa_data = join_scan_data_plus(scan0_d[k],scan1_d[k], THETA_MX, THETA_MN, S_MAX_RANGE, D_MN)

        if k == 0:  
            m_prob_occ, m_log_odds, sum_ll = upload_map(lidar_data, x[k,:], m_prob_occ, m_log_odds)
            m_prob_occ, m_log_odds, sum_ll = upload_map(mapa_data, x[k,:], m_prob_occ, m_log_odds)
            #lista_hausdorff_prom.append(5)
            lista_sumll.append(sum_ll)
            print("creo mapa")

        else:                   
            u = u_const_array[k-1]
            time_ini = time_array[k-1]
            Ts = ts_array[k-1]

            A, B, G, C, D, H = jacobianos_particle(x_k_k[k-1,:], u )
            Ad = expm(A*Ts) # Ojo usar "expm" no "exp"!!!
            Bd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(B) # Bd = Ad*Ts*(eye(size(Ad))-A*Ts/2)*B;
            Gd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(G) # Gd = Ad*Ts*(eye(size(Ad))-A*Ts/2)*G;
            A, B, G = Ad, Bd, Gd

            # -- Predicción por integración del modelo no-lineal
            _, x_aux = step_model(model_particle, u, time_ini, sigma_P, Ts, x_k_k[k-1,:])
            x_k1_k = np.vstack((x_k1_k,(x_aux.T)[-1]))                             # Store x_{k+1|k}
            z_k1_k_aux = C.dot(x_k1_k[k,:]) + D.dot(u)                   # <-- z_{k+1|k}

            P_k1_k_aux = (A.dot(P_k_k[k-1,:,:])).dot(A.T) + (G.dot(Q_k)).dot(G.T) # <-- P_{k+1|k}
            S_k1_k_aux = (C.dot(P_k1_k_aux)).dot(C.T) + (H.dot(R_k)).dot(H.T)   # <-- S_{k+1|k}

            v_distances, pose_0 = find_voronoi_img (m_prob_occ), find_pose(x_k1_k[k,:])
            z_k1_aux, lista_hausdorff_prom = MHD_sensor(k,lidar_data, v_distances, pose_0, lista_hausdorff_prom)
                        
            e = z_k1_aux - z_k1_k_aux
            distace_e = math.sqrt(e[0]**2 + e[1]**2)

            K = (P_k1_k_aux.dot(C.T)).dot(np.linalg.inv(S_k1_k_aux))# <-- K_{k+1}
            
            radio = np.abs (u[0]/u[1])


            if distace_e > tresholds[1] :#and distace_e < tresholds[0] :#or np.abs(e[2]) > tresholds[0]:#0.32:#0.4:
                print("k: " + str(k) +"error: " + str(e) + " d_error: " + "%.4f"%(distace_e) )
                filter_data = filter_data_mhd(lidar_data, pose_0,v_distances)
                z_k1_aux, lista_hausdorff_prom = MHD_sensor(k,filter_data, v_distances, pose_0, lista_hausdorff_prom[:-1])
                print("k: " + str(k) + " hausdorff2: " +str(z_k1_aux), " mhd: " +str(lista_hausdorff_prom[-1]))
                lidar_data = filter_data
                e = z_k1_aux - z_k1_k_aux

            """
                if math.sqrt(e[0]**2 + e[1]**2) > distace_e :
                    distace_e = 2.0
                else:
                    lidar_data = filter_data
                    #if radio>3 and len (dict_segments) != 0 :
                    #    ref_m = dict_segments["slope0"]
                    #e[2] = (ref_m - x_k1_k[k][2])
                    #distace_e = math.sqrt(e[0]**2 + e[1]**2)
                    #print("errorh: " + str(e) + " d_error: " + "%.4f"%(distace_e) )
                    
            if distace_e >= tresholds[0]:
                print("ojo error")
                print("k: " + str(k) + "error2: " + str(e) + " d_error: " + "%.4f"%(distace_e) )
                e = np.array([0.06456889, 0.05008964, 0.12027746])
                print("k: " + str(k) + "error2: " + str(e) + " d_error: " + "%.4f"%(distace_e) )
            """
                

            #radio = np.abs (u[0]/u[1])

            if k> 3 :
                if radio > 9: #15: #20:
                    xk1_ = x_k1_k[k,:] + K.dot(e) 
                    lista_segment.append([xk1_[0], xk1_[1]])
                    if len (dict_segments) != 0 and len(lista_segment)>2:
                        ref_m = dict_segments["slope0"]
                        elems_s0 = np.array (lista_segment)
                        if len(lista_segment)> 4:
                            linear_regressor.fit(elems_s0[-5:,0].reshape(-1,1), elems_s0[-5:,1].reshape(-1,1))
                            #print(elems_s0[-5:,0])
                        else :
                            linear_regressor.fit(elems_s0[:,0].reshape(-1,1), elems_s0[:,1].reshape(-1,1))
                        slope = linear_regressor.coef_[0][0]
                        if np.abs(ref_m - slope)<0.02: #0.035
                            ref_m, slope = 0, 0
                        print(ref_m - slope)
                    else:
                        print("no")
                else :
                    if len(lista_segment)!= 0:
                        s1 = np.array(lista_segment)
                        linear_regressor.fit(s1[:,0].reshape(-1,1), s1[:,1].reshape(-1,1))
                        name_key1 = "slope"+str(aux_dic)
                        name_key2 = "segment"+str(aux_dic)
                        dict_segments[name_key1] = linear_regressor.coef_[0][0]
                        dict_segments[name_key2] = s1
                        aux_dic =  aux_dic+1
                        lista_segment, ref_m, slope = [], 0, 0
                    
            e[2] = e[2] + (prop*(ref_m - slope))
            lista_errores.append([e[0], e[1], e[2], distace_e])
                
            x_k_k_aux = x_k1_k[k,:] + K.dot(e) 
            P_k_k_aux = P_k1_k_aux - (K.dot(S_k1_k_aux)).dot(K.T)   # <-- P_{k+1|k+1}
            z_k = np.vstack((z_k, z_k1_aux))                         # Store z_{k+1}
            x_k_k = np.vstack((x_k_k, x_k_k_aux))                    # Store x_{k+1|k+1}
            P_k_k = np.vstack((P_k_k, P_k_k_aux[np.newaxis,...]))    # Store P_{k+1|k+1}


            m_prob_occ, m_log_odds, sum_ll = upload_map(lidar_data, x_k_k_aux, m_prob_occ, m_log_odds)
                    
            if distace_e < tresholds[1] and np.abs(e[2]) < 0.1: #lista_hausdorff_prom [-1] < 90:#treshold_map:
                m_prob_occ, m_log_odds, sum_ll = upload_map(mapa_data, x_k_k_aux, m_prob_occ, m_log_odds)
                lista_sumll.append (sum_ll)
                lista_e_ok.append([e[0], e[1], e[2], distace_e])
                counter = 0
                print(counter)
            else:
                lista_sumll.append(lista_sumll[-1])
                es_lost.append(k)
                counter = counter + 1
                if counter > tresholds[2]:
                    m_prob_occ, m_log_odds, sum_ll = upload_map(lidar_data, x_k_k_aux, m_prob_occ, m_log_odds)
                    m_prob_occ, m_log_odds, sum_ll = upload_map(mapa_data, x_k_k_aux, m_prob_occ, m_log_odds)
                    counter = 0
                    print(counter)
        
    
    plt.imshow(m_log_odds)#,interpolation='nearest')#, extent = [0, XMAX, 0, YMAX])
    plt.show()
    plt.imshow(m_prob_occ)#,interpolation='nearest')#, extent = [0, XMAX, 0, YMAX])
    plt.show()

    q = -1
    plt.plot(data_C1t1["gnss_rotated_c"][:q,0], data_C1t1["gnss_rotated_c"][:q,1], label= 'x vs y data gnss rotated')
    plt.plot(z_k[:q,:1], z_k[:q,1:2], label= 'hausdorff medido')
    plt.plot(x_k_k[:q,:1], x_k_k[:q,1:2], label= 'estimado')
    plt.plot(x_k1_k[:q,:1], x_k1_k[:q,1:2], label= 'predicho')
    plt.plot(x[:q,:1], x[:q,1:2], label= 'real')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.show()

    """
    save = input("If you want to save the data, please write Y: ")
    if save == "Y" or save =="y":
        name_output = input("please write the name of the file : ")
        output_path = "results_v6/" + str(name_output) +".npz"
        
        np.savez_compressed(output_path, gnss_cones = x_k_gnss, hausdorff_medido = z_k, estimado = x_k_k,
                    predicho = x_k1_k, real = x, map_log = m_log_odds, map_occ = m_prob_occ, 
                    mean_hauss = lista_hausdorff_prom, error_x = None, pk = P_k_k, lista_e = lista_errores,
                    lista_ll = lista_sumll, limits_filters = tresholds, k_losts = es_lost, count_n = counter, 
                    segment_list = lista_segment, dictionary = dict_segments, dic_aux = aux_dic, reference = ref_m,
                    m_slope = slope, P_filter = prop, l_error_ok = lista_e_ok)
    """
    #name_output = "exp1_2slam_195_11oct"
    #output_path = "results_v7/" + str(name_output) +".npz"
        
    np.savez_compressed("results_v7/exp_16oct_350.npz", gnss_cones = x_k_gnss, hausdorff_medido = z_k, estimado = x_k_k,
                    predicho = x_k1_k, real = x, map_log = m_log_odds, map_occ = m_prob_occ, 
                    mean_hauss = lista_hausdorff_prom, error_x = None, pk = P_k_k, lista_e = lista_errores,
                    lista_ll = 0, limits_filters = tresholds, k_losts = es_lost, count_n = counter, 
                    segment_list = lista_segment, dictionary = dict_segments, dic_aux = aux_dic, reference = ref_m,
                    m_slope = slope, P_filter = prop, l_error_ok = lista_e_ok)

    return

if __name__ == '__main__': main()

"""
results_v7\exp_14oct_350.npz - > perfecto paso a paso en jupyter

results_v7\exp_16oct_350_no.npz -> finaliza la trayectoria pero los errores en trayectoria son mayores, sigue la aleatroriedad
"""