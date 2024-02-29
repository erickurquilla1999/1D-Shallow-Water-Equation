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

def compute_numerical_flux_vector(element_n,u1,u2,f1,f2,basis_values_at_nods):

    roe_fluxex_1 = []
    roe_fluxex_2 = []

    # computing roe fluxex
    for n in element_n[:-1]:

        u1_average = 0.5*(u1[n][-1]+u1[n+1][0])
        u2_average = 0.5*(u2[n][-1]+u2[n+1][0])

        jacobian = [ [ 0 , 1 ] , [ inputs.g * u1_average - ( u2_average / u1_average )**2, 2 * u2_average / u1_average ] ]

        eigenvalues_jacobian, eigenvectors_jacobian = np.linalg.eig(jacobian)

        # Construct the modified Roe matrix
        abs_A = eigenvectors_jacobian @ np.diag(np.abs(eigenvalues_jacobian)) @ np.linalg.inv(eigenvectors_jacobian)

        roe_fluxex_1.append( 0.5 * ( f1[n] + f1[n+1] ) - 0.5 * abs_A[0][0] * ( u1[n+1] - u1[n] ) - 0.5 * abs_A[0][1] * ( u2[n+1] - u2[n] ) )
        roe_fluxex_2.append( 0.5 * ( f2[n] + f2[n+1] ) - 0.5 * abs_A[1][0] * ( u1[n+1] - u1[n] ) - 0.5 * abs_A[1][1] * ( u2[n+1] - u2[n] ) )

    # creating matrix P_ij=phi_i*phi_j
    P_a=np.outer(basis_values_at_nods[0][0],basis_values_at_nods[0][0])
    P_b=np.outer(basis_values_at_nods[0][-1],basis_values_at_nods[0][-1])

    # computing the difference between the numerical fluxe in limits of the element

    numerical_flux_1=[]
    numerical_flux_2=[]

    for n in element_n:
        
        if n == 0:

            numerical_flux_1.append( P_b @ roe_fluxex_1[n] - 0 )
            numerical_flux_2.append( P_b @ roe_fluxex_2[n] - P_a @ (0.5 * inputs.g * u1[n]**2 ) )

        elif n == element_n[-1]:

            numerical_flux_1.append( 0 - P_a @ roe_fluxex_1[-1] )
            numerical_flux_2.append( P_b @ ( 0.5 * inputs.g * u1[n]**2 ) - P_a @ roe_fluxex_2[-1] )

        else:

            numerical_flux_1.append( P_b @ roe_fluxex_1[n] - P_a @ roe_fluxex_1[n-1] )
            numerical_flux_2.append( P_b @ roe_fluxex_2[n] - P_a @ roe_fluxex_2[n-1] )

    return np.array(numerical_flux_1), np.array(numerical_flux_2)

def compute_stiffness_vectors(e_numb,e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature, _h, _u):
    
    # number of basis or nodes in each element
    number_of_basis = len(bas_vals_at_gauss_quadrature[0][0])

    # interpolate h from nodes to quadrature points
    _h_at_gau_quad = [ bas_at_gau_quad @ __h for bas_at_gau_quad, __h in zip(bas_vals_at_gauss_quadrature, _h)]

    # interpolate u from nodes to quadrature points
    _u_at_gau_quad = [ bas_at_gau_quad @ __u for bas_at_gau_quad, __u in zip(bas_vals_at_gauss_quadrature, _u)]

    # compute stiffness vectors
    stiff_vec_1 = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_x_der_at_gauss_quadrature[n][:,i] * ( _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])                                                   ) ) for i in range(number_of_basis) ] for n in e_numb]
    stiff_vec_2 = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_x_der_at_gauss_quadrature[n][:,i] * ( _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])**2 + 0.5 * inputs.g * np.array(_h_at_gau_quad[n])**2 ) ) for i in range(number_of_basis) ] for n in e_numb]
    
    return stiff_vec_1, stiff_vec_2
