import numpy as np

def arrange_npz_v4 (file_npz):
    x_k_gnss = np.vstack((file_npz["gnss_rotated_c"].T, file_npz["angles"].T)).T
    print(x_k_gnss.shape)
    scan0_data = file_npz["scan0"]
    scan1_data = file_npz["scan1"]
    u_const_array = file_npz["u_constant"]
    time_array = file_npz["time"]
    ts_array = np.full((len(file_npz["delta_time"])),np.mean(file_npz["delta_time"]))
    print("constant time sample : " + str (ts_array[0]))
    u_var_array = file_npz["u_var"]
    u_dataset_array = file_npz["u_datos"]
    print(scan0_data.shape, u_const_array.shape, time_array.shape, ts_array.shape, np.mean(ts_array), time_array[0])
    return scan0_data,scan1_data, u_const_array,u_var_array, time_array, ts_array, x_k_gnss, u_dataset_array

def arrange_npz_pullally_v0 (file_npz):
    x_k_gnss = np.vstack((file_npz["gnss_rotated_c"].T, file_npz["angles"].T)).T
    print(x_k_gnss.shape)
    scan0_data = file_npz["scan0"]
    #scan1_data = file_npz["scan1"]
    u_const_array = file_npz["u_constant"]
    time_array = file_npz["time"]
    ts_array = np.full((len(file_npz["delta_time"])),np.mean(file_npz["delta_time"]))
    print("constant time sample : " + str (ts_array[0]))
    u_var_array = file_npz["u_var"]
    #u_dataset_array = file_npz["u_datos"]
    print(scan0_data.shape, u_const_array.shape, time_array.shape, ts_array.shape, np.mean(ts_array), time_array[0])
    return scan0_data, u_const_array,u_var_array, time_array, ts_array, x_k_gnss


def arrange_results (file_npz):
    print(file_npz.files)
    gnss = file_npz["gnss_cones"]
    hausdorff_x = file_npz["hausdorff_medido"]
    x_k_k = file_npz["estimado"]
    x_k1_k = file_npz["predicho"]
    x = file_npz["real"]
    m_log = file_npz["map_log"]
    m_prob = file_npz["map_occ"]
    hausdorff_prom = file_npz["mean_hauss"]
    #e_x = file_npz["error_x"]
    p_k = file_npz["pk"]
    error_list = file_npz["lista_e"]
    """
    ll_list = file_npz["lista_ll"]
    l1 = file_npz["limit_filter"]
    l2 = file_npz["limit_map"]
    """
    ll_list = file_npz["lista_ll"]
    limits = file_npz["limits_filters"]
    ks_lost = file_npz["k_losts"]
    count = file_npz["count_n"]
    #gnss, hausdorff_x, x_k_k, x_k1_k, x, m_log, m_prob, hausdorff_prom.tolist(), e_x, p_k, error_list.tolist(), ll_list.tolist(), limits, ks_lost.tolist(), count # l1,l2#
    return gnss, hausdorff_x, x_k_k, x_k1_k, x, m_log, m_prob, hausdorff_prom.tolist(), p_k, error_list.tolist(), ll_list.tolist(), limits, ks_lost.tolist(), count # l1,l2#

def arrange_results_list (file_npz):
    #print(file_npz.files)
    gnss = file_npz["gnss_cones"]
    hausdorff_x = file_npz["hausdorff_medido"]
    x_k_k = file_npz["estimado"]
    x_k1_k = file_npz["predicho"]
    x = file_npz["real"]
    m_log = file_npz["map_log"]
    m_prob = file_npz["map_occ"]
    hausdorff_prom = file_npz["mean_hauss"]
    p_k = file_npz["pk"]
    error_list = file_npz["lista_e"]
    ll_list = file_npz["lista_ll"]
    limits = file_npz["limits_filters"]
    ks_lost = file_npz["k_losts"]
    count = file_npz["count_n"]
    return [gnss, hausdorff_x, x_k_k, x_k1_k, x, m_log, m_prob, hausdorff_prom.tolist(), p_k, error_list.tolist(), ll_list.tolist(), limits, ks_lost.tolist(), count] # l1,l2#

def arrange_resultsv6 (file_npz):
    print(file_npz.files)
    gnss = file_npz["gnss_cones"]
    hausdorff_x = file_npz["hausdorff_medido"]
    x_k_k = file_npz["estimado"]
    x_k1_k = file_npz["predicho"]
    x = file_npz["real"]
    m_log = file_npz["map_log"]
    m_prob = file_npz["map_occ"]
    hausdorff_prom = file_npz["mean_hauss"]
    #e_x = file_npz["error_x"]
    p_k = file_npz["pk"]
    error_list = file_npz["lista_e"]
    ll_list = file_npz["lista_ll"]
    limits = file_npz["limits_filters"]
    ks_lost = file_npz["k_losts"]
    count = file_npz["count_n"]
    lista_segment = file_npz['segment_list']
    dict_segments =file_npz['dictionary']
    aux_dic = file_npz['dic_aux']
    e_list_ok = file_npz["l_error_ok"]
    return gnss, hausdorff_x, x_k_k, x_k1_k, x, m_log, m_prob, hausdorff_prom.tolist(), p_k, error_list.tolist(), ll_list.tolist(), limits, ks_lost.tolist(), count, lista_segment.tolist(), dict_segments, aux_dic, e_list_ok.tolist()
