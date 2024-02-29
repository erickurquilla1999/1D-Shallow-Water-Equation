import numpy as np
import inputs

def compute_mass_matrix_1_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad):

    # number of basis or nodes in each element
    number_of_basis = len(basis_values_at_gauss_quad[0][0])

    # compute mass matrix 1
    M1 = [ [ [ 0.5 * element_lgth[n] * np.sum( basis_values_at_gauss_quad[n][:,i] * gauss_weights[n] * basis_values_at_gauss_quad[n][:,j] ) for j in range(number_of_basis) ] for i in range(number_of_basis) ] for n in elmnt_numb]

    # compute inverse of mass matrix 2
    M1_inverse = [ np.linalg.inv(M1[n]) for n in elmnt_numb]
    
    return M1_inverse

def compute_stiffness_vector_1(ele_n, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_weights, ele_lengths, h_, u_):
    
    # compute values of height h at gauss quadrature
    h_at_gau_quad = [ bas_at_gau_quad @ h__ for bas_at_gau_quad, h__ in zip(bas_vals_at_gau_quad,h_)]
    
    # compute values of velocity u at gauss quadrature
    u_at_gau_quad = [ bas_at_gau_quad @ u__ for bas_at_gau_quad, u__ in zip(bas_vals_at_gau_quad,u_)]

    # integrate over each element : int ( d_dx phi_i ) * u * h dx
    stiff_vec_1 = [[0.5 * ele_lengths[n] * np.sum( np.array(bas_vals_x_der_at_gau_quad[n])[:,m] * gau_weights[n] * h_at_gau_quad[n] * u_at_gau_quad[n] ) for m in range(len(bas_vals_x_der_at_gau_quad[n][0]))] for n in ele_n]

    return stiff_vec_1

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

def compute_mass_matrix_2_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad, _h):
    
    # number of basis or nodes in each element
    number_of_basis = len(basis_values_at_gauss_quad[0][0])

    # interpolate h from nodes to quadrature points
    _h_at_gau_quad = [ bas_at_gau_quad @ __h for bas_at_gau_quad, __h in zip(basis_values_at_gauss_quad, _h)]

    # compute mass matrix 2
    M2 = [ [ [ 0.5 * element_lgth[n] * np.sum( basis_values_at_gauss_quad[n][:,i] * _h_at_gau_quad[n] * gauss_weights[n] * basis_values_at_gauss_quad[n][:,j] ) for j in range(number_of_basis) ] for i in range(number_of_basis) ] for n in elmnt_numb]

    # compute inverse of mass matrix 2
    M2_inverse = [ np.linalg.inv(M2[n]) for n in elmnt_numb]
    
    return M2_inverse

def compute_mass_vector_2_complement(e_numb,e_lgth, g_weights, bas_vals_at_gauss_quadrature, _dh_dt, _u):
    
    # number of basis or nodes in each element
    number_of_basis = len(bas_vals_at_gauss_quadrature[0][0])

    # interpolate u from nodes to quadrature points
    _u_at_gau_quad = [ bas_at_gau_quad @ __u for bas_at_gau_quad, __u in zip(bas_vals_at_gauss_quadrature, _u)]

    # interpolate dh_dt from nodes to quadrature points
    _dh_dt_at_gau_quad = [ bas_at_gau_quad @ __dh_dt for bas_at_gau_quad, __dh_dt in zip(bas_vals_at_gauss_quadrature, _dh_dt)]

    # compute mass vector 2 complement
    M2_vec_comp = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_at_gauss_quadrature[n][:,i] * _dh_dt_at_gau_quad[n] * _u_at_gau_quad[n] ) for i in range(number_of_basis) ] for n in e_numb]
    
    return M2_vec_comp



def compute_stiffness_vector_2(e_numb,e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature, _h, _u):
    
    # number of basis or nodes in each element
    number_of_basis = len(bas_vals_at_gauss_quadrature[0][0])

    # interpolate h from nodes to quadrature points
    _h_at_gau_quad = [ bas_at_gau_quad @ __h for bas_at_gau_quad, __h in zip(bas_vals_at_gauss_quadrature, _h)]

    # interpolate u from nodes to quadrature points
    _u_at_gau_quad = [ bas_at_gau_quad @ __u for bas_at_gau_quad, __u in zip(bas_vals_at_gauss_quadrature, _u)]

    # compute stiffness vector 2
    stiff_vec_2 = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_x_der_at_gauss_quadrature[n][:,i] * ( _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])**2 + 0.5 * inputs.g * np.array(_h_at_gau_quad)**2 ) ) for i in range(number_of_basis) ] for n in e_numb]
    
    return stiff_vec_2
