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

def compute_numerical_flux_vector(element_n,u1,u2,f1,f2,basis_values_at_nods):

    # computing roe flux
    roe_flux_1 = []
    roe_flux_2 = []

    # looping over each element (except the final element) to compute the roe flux at the right
    for n in element_n[:-1]:

        # computing average value in the right border of element n and n+1
        u1_average = 0.5*(u1[n][-1]+u1[n+1][0])
        u2_average = 0.5*(u2[n][-1]+u2[n+1][0])
        
        # compute the jacobian evaluated in the border between elements n and n+1
        jacobian = [ [ 0 , 1 ] , [ inputs.g * u1_average - ( u2_average / u1_average )**2, 2 * u2_average / u1_average ] ]

        # compute eigenvalues and eigenvector of the jacobian
        eigenvalues_jacobian, eigenvectors_jacobian = np.linalg.eig(jacobian)

        # biulds abs_A matrix
        abs_A = eigenvectors_jacobian @ np.diag(np.abs(eigenvalues_jacobian)) @ np.linalg.inv(eigenvectors_jacobian)

        # compute roe flux
        roe_flux_1.append( 0.5 * ( f1[n][-1] + f1[n+1][0] ) - 0.5 * abs_A[0][0] * ( u1[n+1][0] - u1[n][-1] ) - 0.5 * abs_A[0][1] * ( u2[n+1][0] - u2[n][-1] ) )
        roe_flux_2.append( 0.5 * ( f2[n][-1] + f2[n+1][0] ) - 0.5 * abs_A[1][0] * ( u1[n+1][0] - u1[n][-1] ) - 0.5 * abs_A[1][1] * ( u2[n+1][0] - u2[n][-1] ) )

    # computing the difference between the numerical fluxe in limits of the element
    difference_numerical_flux_1 = []
    difference_numerical_flux_2 = []

    #looping over all element
    for n in element_n:
        # compute differences between flux: right numerical flux - left numerical flux
        if n == 0:               
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * roe_flux_1[n] - basis_values_at_nods[n][:,0] * 0 )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * roe_flux_2[n] - basis_values_at_nods[n][:,0] * ( 0.5 * inputs.g * u1[n][0]**2 ) )
        elif n == element_n[-1]: 
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * 0 - basis_values_at_nods[n][:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * ( 0.5 * inputs.g * u1[n][-1]**2 ) - basis_values_at_nods[n][:,0] * roe_flux_2[n-1] )
        else:                    
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * roe_flux_1[n] - basis_values_at_nods[n][:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * roe_flux_2[n] - basis_values_at_nods[n][:,0] * roe_flux_2[n-1] )

    return np.array(difference_numerical_flux_1), np.array(difference_numerical_flux_2)

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
