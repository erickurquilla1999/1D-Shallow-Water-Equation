import numpy as np
import inputs

def compute_M_matrix_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad):

    print('Computing M inverse matrix ... ')
    
    # in element k: M_ij = integral phi_i(x) phi_j(x) dx inside the element domain
    M = []
    M_inverse = []

    #Lopp over all element
    for n in elmnt_numb:
        phi = np.array(basis_values_at_gauss_quad[n])
        weights = gauss_weights[n]
        delta_x = element_lgth[n]
        
        # Compute M for the current element
        M_in_element_n = 0.5 * delta_x * np.dot(phi.T * weights, phi)
        # Append M to the list
        M.append(M_in_element_n)

        # Compute the inverse of M for the current element
        M_inv_in_element_n = np.linalg.inv(M_in_element_n)

        # Append the inverse of M to the list
        M_inverse.append(M_inv_in_element_n)

    return M_inverse

def compute_N_matrix(elem_num, basis_vals_at_gauss_quad_elements, basis_vals_time_derivative_at_gauss_quad_elements,gauss_weights_elmts,elmnt_l):

    print('Computing N inverse matrix ... ')
    
    # N_ij = integral dphi_i_dx(x) phi_j(x) dx
    N_matrix=[]

    for n in elem_num:
        phi = np.array(basis_vals_at_gauss_quad_elements[n])
        dphi_dx = np.array(basis_vals_time_derivative_at_gauss_quad_elements[n])
        weights = gauss_weights_elmts[n]
        delta_x = elmnt_l[n]

        # Compute N for the current element
        N_matrix.append(0.5 * delta_x * np.dot(dphi_dx.T * weights, phi))

    return N_matrix

def compute_residual_vector(element_n,u1,u2,f1,f2,basis_values_at_nods,N_matx):

    print('Computing residual vector ... ')

    R_f1=[]
    R_f2=[]

    for n in element_n:

        # Compute R for fist term this is: integral dphi_i_dx(x) phi_j(x) dx f_j
        R1_f1_in_element_n = np.dot(N_matx[n],f1[n])
        R1_f2_in_element_n = np.dot(N_matx[n],f2[n])
        
        # computing Roe flux
        roe_flux_1_left = roe_flux_1_right = roe_flux_2_left = roe_flux_2_right = 0

        if n == 0:
            roe_flux_2_left = inputs.g * u1[n][n]**2 / 2
            Jacobian_right = -inputs.g * 0.5 * (u1[n][-1] + u1[n + 1][0]) + ((f1[n][-1] + f1[n + 1][0]) / (u1[n][-1] + u1[n + 1][0]))**2
            roe_flux_1_right = 0.5 * (f1[n][-1] + f1[n + 1][0]) - 0.5 * Jacobian_right * (u1[n][-1] + u1[n + 1][0])
            roe_flux_2_right = 0.5 * (f2[n][-1] + f2[n + 1][0]) - 0.5 * Jacobian_right * (u2[n][-1] + u2[n + 1][0])
        elif n == len(element_n) - 1:
            Jacobian_left = -inputs.g * 0.5 * (u1[n - 1][-1] + u1[n][0]) + ((f1[n - 1][-1] + f1[n][0]) / (u1[n - 1][-1] + u1[n][0]))**2
            roe_flux_1_left = 0.5 * (f1[n - 1][-1] + f1[n][0]) - 0.5 * Jacobian_left * (u1[n - 1][-1] + u1[n][0])
            roe_flux_2_left = 0.5 * (f2[n - 1][-1] + f2[n][0]) - 0.5 * Jacobian_left * (u2[n - 1][-1] + u2[n][0])
            roe_flux_2_right = inputs.g * u1[n][len(f1[n]) - 1]**2 / 2
        else:
            Jacobian_left = -inputs.g * 0.5 * (u1[n - 1][-1] + u1[n][0]) + ((f1[n - 1][-1] + f1[n][0]) / (u1[n - 1][-1] + u1[n][0]))**2
            roe_flux_1_left = 0.5 * (f1[n - 1][-1] + f1[n][0]) - 0.5 * Jacobian_left * (u1[n - 1][-1] + u1[n][0])
            roe_flux_2_left = 0.5 * (f2[n - 1][-1] + f2[n][0]) - 0.5 * Jacobian_left * (u2[n - 1][-1] + u2[n][0])
            Jacobian_right = -inputs.g * 0.5 * (u1[n][-1] + u1[n + 1][0]) + ((f1[n][-1] + f1[n + 1][0]) / (u1[n][-1] + u1[n + 1][0]))**2
            roe_flux_1_right = 0.5 * (f1[n][-1] + f1[n + 1][0]) - 0.5 * Jacobian_right * (u1[n][-1] + u1[n + 1][0])
            roe_flux_2_right = 0.5 * (f2[n][-1] + f2[n + 1][0]) - 0.5 * Jacobian_right * (u2[n][-1] + u2[n + 1][0])

        # residual vector 2
        R2_f1_in_element_n = -(np.array(basis_values_at_nods[n][0]) * roe_flux_1_right - np.array(basis_values_at_nods[n][-1]) * roe_flux_1_left)
        R2_f2_in_element_n = -(np.array(basis_values_at_nods[n][0]) * roe_flux_2_right - np.array(basis_values_at_nods[n][-1]) * roe_flux_2_right)

        # adding residual vector 1 and 2
        R_f1.append(np.array(R1_f1_in_element_n)+np.array(R2_f1_in_element_n))
        R_f2.append(np.array(R1_f2_in_element_n)+np.array(R2_f2_in_element_n))

    return R_f1, R_f2

def compute_time_derivates(element_num,M_inv, Rf1, Rf2 ):

    print('Computing time derivatives of u_1 and u_2 ... ')

    du1dt=[]
    du2dt=[]

    #Lopp over all element
    for n in element_num:
        du1dt.append(np.dot(M_inv[n],Rf1[n]))    
        du2dt.append(np.dot(M_inv[n],Rf2[n])) 

    return du1dt, du2dt