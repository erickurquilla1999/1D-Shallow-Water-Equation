import numpy as np
import inputs

def compute_mass_matrix_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad):

    # print('Computing mass inverse matrix ... ')
    
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

def compute_stiffness_matrix(elem_num, basis_vals_at_gauss_quad_elements, basis_vals_x_derivative_at_gauss_quad_elements,gauss_weights_elmts,elmnt_l):

    # print('Computing stiffness matrix ... ')
    
    # S_ij = integral dphi_i_dx(x) phi_j(x) dx
    stff_matrix=[]

    for n in elem_num:

        phi = np.array(basis_vals_at_gauss_quad_elements[n])
        dphi_dx = np.array(basis_vals_x_derivative_at_gauss_quad_elements[n])
        weights = gauss_weights_elmts[n]
        delta_x = elmnt_l[n]

        # Compute N for the current element
        stff_matrix.append(0.5 * delta_x * np.dot(dphi_dx.T * weights, phi))

    return stff_matrix

def compute_numerical_flux_vector_1(element_n, basis_values_at_nods, h_, u_):

    # computing roe flux
    roe_flux = []

    # looping over each element (except the final element) to compute the roe flux at the right
    for n in element_n[:-1]:

        # computing average value in the right border of element n and n+1
        h_average = 0.5*(h_[n][-1]+h_[n+1][0])
        u_average = 0.5*(u_[n][-1]+u_[n+1][0])

        # compute the jacobian evaluated in the border between elements n and n+1
        jacobian = [ [ 0 , 1 ] , [ inputs.g * h_average - u_average**2, 2 * u_average ] ]

        # compute eigenvalues and eigenvector of the jacobian
        eigenvalues_jacobian, eigenvectors_jacobian = np.linalg.eig(jacobian)

        # biulds abs_A matrix
        abs_A = eigenvectors_jacobian @ np.diag(np.abs(eigenvalues_jacobian)) @ np.linalg.inv(eigenvectors_jacobian)

        # for the border between element n and n+1 compute the f1 on the left and the right
        f1_left  = h_[n] * u_[n] 
        f1_right = h_[n+1] * u_[n+1]

        # for the border between element n and n+1 compute the u1 on the left and the right
        u1_left  = h_[n]
        u1_right = h_[n+1]

        # for the border between element n and n+1 compute the u2 on the left and the right
        u2_left  = h_[n] * u_[n] 
        u2_right = h_[n+1] * u_[n+1]

        # compute roe flux
        roe_flux.append( 0.5 * ( f1_left + f1_right ) - 0.5 * abs_A[0][0] * ( u1_right - u1_left ) - 0.5 * abs_A[0][1] * ( u2_right - u2_left ) )

    # creating matrix P_ij=phi_i*phi_j
    # Pa is evaluated in x = a. This is the begining of the element
    P_a=np.outer(basis_values_at_nods[0][0],basis_values_at_nods[0][0])
    # Pb is evaluated in x = b. This is the end of the element
    P_b=np.outer(basis_values_at_nods[0][-1],basis_values_at_nods[0][-1])

    # computing the difference between the numerical fluxe in limits of the element
    difference_numerical_flux = []

    #looping over all element
    for n in element_n:
        # compute differences between flux: right numerical flux - left numerical flux
        if n == 0:               difference_numerical_flux.append( P_b @ roe_flux[n]     - 0                   )
        elif n == element_n[-1]: difference_numerical_flux.append( 0                     - P_a @ roe_flux[n-1] )
        else:                    difference_numerical_flux.append( P_b @ roe_flux[n]     - P_a @ roe_flux[n-1] )

    return np.array(difference_numerical_flux)

def compute_stiffness_vector_1(ele_n, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_weights, ele_lengths, h_, u_):
    
    # compute values of height h at gauss quadrature
    h_at_gau_quad = [ bas_at_gau_quad @ h__ for bas_at_gau_quad, h__ in zip(bas_vals_at_gau_quad,h_)]
    
    # compute values of velocity u at gauss quadrature
    u_at_gau_quad = [ bas_at_gau_quad @ u__ for bas_at_gau_quad, u__ in zip(bas_vals_at_gau_quad,u_)]

    # integrate over each element : int ( d_dx phi_i ) * u * h dx
    stiff_vec_1 = [[0.5 * ele_lengths[n] * np.sum( np.array(bas_vals_x_der_at_gau_quad[n])[:,m] * gau_weights[n] * h_at_gau_quad[n] * u_at_gau_quad[n] ) for m in range(len(bas_vals_x_der_at_gau_quad[n][0]))] for n in ele_n]

    return stiff_vec_1