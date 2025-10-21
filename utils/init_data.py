import numpy as np

def init_sigmas():
    sigma_p = np.array([0.035, 0.035, 0.07])
    p_k0 = np.diag(sigma_p**2) #covariance matrix of the state
    sigma_q = np.array([0.12, 0.37, 0.08])
    q_k = np.diag(sigma_q**2)  
    sigma_r = np.array([0.07, 0.07, 0.1])
    r_k = np.diag(sigma_r**2) 
    return sigma_p, p_k0, q_k, r_k

def init_data_p0 (x_k0_m, p_k0):
    xi = x_k0_m.reshape((1, len(x_k0_m)))        # real
    xi_k_k = x_k0_m.reshape((1, len(x_k0_m)))    # estimado
    xi_k1_k = x_k0_m.reshape((1, len(x_k0_m)))    # predicho
    zi_k = x_k0_m.reshape((1, len(x_k0_m))) 
    p_k_k = p_k0[np.newaxis,...]  # P0 transformed to a stack of 6x6 arrays
    return xi, xi_k_k, xi_k1_k, zi_k, p_k_k

def init_memories ():
    ilista_hausdorff_prom = []
    lista_erroresi =[]
    lista_sumlli = []
    es_losti = []
    std_listi = []
    return ilista_hausdorff_prom, lista_erroresi, lista_sumlli, es_losti, std_listi

"""
def init_sigmas():
    #sigma_p = np.array([0.035, 0.035, 0.07]) # 0.01-si 0.03-results 0.05-no 0.1-no 0.2
    sigma_p = np.array([0.04, 0.04, 0.08])
    #sigma_p = np.array([0.07, 0.07, 0.08])
    p_k0 = np.diag(sigma_p**2) #covariance matrix of the state
    #sigma_q = np.array([0.07, 0.07, 0.1]) 
    #sigma_q = np.array([0.15, 0.15, 0.1]) 
    sigma_q = np.array([0.07, 0.37, 0.12]) 
    q_k = np.diag(sigma_q**2)  
    #sigma_r = np.array([0.05, 0.05, 0.08])
    #sigma_r = np.array([0.15, 0.75, 0.08])
    #sigma_r = np.array([0.15, 0.37, 0.08])
    sigma_r = np.array([0.2, 0.2, 0.1])
    r_k = np.diag(sigma_r**2) 
    return sigma_p, p_k0, q_k, r_k
"""