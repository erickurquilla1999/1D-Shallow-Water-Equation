import numpy as np

def compute_numerical_flux_vectors(h_, u_):

    # basis_func_values_at_nodes_in_phys_space = [ [phi_1(x_node_1), phi_2(x_node_1) , ... , phi_p(x_node_1)] , 
    #                                              [phi_1(x_node_2), phi_2(x_node_2) , ... , phi_p(x_node_2)], ... , ]
    # basis_values_at_nods = 1 (identity matrix of size number of nodes per element
    basis_values_at_nods = np.eye(len(h_[0])) # basis function evaluated at nodes in physical space

    Num_elements = len(h_) # number of elements

    # computing roe flux
    roe_flux_1 = []
    roe_flux_2 = []

    # looping over each element (except the final element) to compute the roe flux at the right
    for n in np.arange(Num_elements-1):

        # computing average value in the right border of element n and n+1
        h_average = 0.5*(h_[n][-1]+h_[n+1][0])
        u_average = 0.5*(u_[n][-1]+u_[n+1][0])

        # compute the jacobian evaluated in the border between elements n and n+1
        jacobian = [ [ 0 , 1 ] , [ 9.8 * h_average - u_average**2, 2 * u_average ] ]

        # compute eigenvalues and eigenvector of the jacobian
        eigenvalues_jacobian, eigenvectors_jacobian = np.linalg.eig(jacobian)

        # biulds abs_A matrix
        abs_A = eigenvectors_jacobian @ np.diag(np.abs(eigenvalues_jacobian)) @ np.linalg.inv(eigenvectors_jacobian)

        # for the border between element n and n+1 compute the f1 on the left and the right
        f1_left  = h_[n][-1] * u_[n][-1] 
        f1_right = h_[n+1][0] * u_[n+1][0]

        # for the border between element n and n+1 compute the f2 on the left and the right
        f2_left  = h_[n][-1] * u_[n][-1]**2 + 0.5 * 9.8 * h_[n][-1]**2
        f2_right = h_[n+1][0] * u_[n+1][0]**2 + 0.5 * 9.8 * h_[n+1][0]**2

        # for the border between element n and n+1 compute the u1 on the left and the right
        u1_left  = h_[n][-1]
        u1_right = h_[n+1][0]

        # for the border between element n and n+1 compute the u2 on the left and the right
        u2_left  = h_[n][-1] * u_[n][-1] 
        u2_right = h_[n+1][0] * u_[n+1][0]

        # compute roe flux
        roe_flux_1.append( 0.5 * ( f1_left + f1_right ) - 0.5 * abs_A[0][0] * ( u1_right - u1_left ) - 0.5 * abs_A[0][1] * ( u2_right - u2_left ) )
        roe_flux_2.append( 0.5 * ( f2_left + f2_right ) - 0.5 * abs_A[1][0] * ( u1_right - u1_left ) - 0.5 * abs_A[1][1] * ( u2_right - u2_left ) )

    # computing the difference between the numerical fluxe in limits of the element
    difference_numerical_flux_1 = []
    difference_numerical_flux_2 = []

    #looping over all element
    for n in np.arange(Num_elements):
        # compute differences between flux: right numerical flux - left numerical flux
        if n == 0:               
            difference_numerical_flux_1.append( basis_values_at_nods[:,-1] * roe_flux_1[n] - basis_values_at_nods[:,0] * 0 )
            difference_numerical_flux_2.append( basis_values_at_nods[:,-1] * roe_flux_2[n] - basis_values_at_nods[:,0] * ( 0.5 * 9.8 * h_[n][0]**2 ) )
        elif n == Num_elements-1: 
            difference_numerical_flux_1.append( basis_values_at_nods[:,-1] * 0 - basis_values_at_nods[:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[:,-1] * ( 0.5 * 9.8 * h_[n][-1]**2 ) - basis_values_at_nods[:,0] * roe_flux_2[n-1] )
        else:                    
            difference_numerical_flux_1.append( basis_values_at_nods[:,-1] * roe_flux_1[n] - basis_values_at_nods[:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[:,-1] * roe_flux_2[n] - basis_values_at_nods[:,0] * roe_flux_2[n-1] )

    return np.array(difference_numerical_flux_1), np.array(difference_numerical_flux_2)

def calcula_matrix_de_rigidez(longitud_elemento_, pesos_de_gauss_, polinomios_de_lagrange_en_cuadratura_de_gauss, derivada_x_polinomios_de_lagrange_en_cuadratura_de_gauss):

    # S_ij = integral (dphi_i(x)/dx)phi_j(x) dx
    matriz_de_rigidez = 0.5 * longitud_elemento_ * np.dot(derivada_x_polinomios_de_lagrange_en_cuadratura_de_gauss * pesos_de_gauss_, polinomios_de_lagrange_en_cuadratura_de_gauss.T)

    return matriz_de_rigidez

def compute_stiffness_vectors(_h, _u, _matriz_de_rigidez):
    
    e_numb = np.arange(len(_h)) # elements number

    stiff_vec_1 = np.array([ _matriz_de_rigidez @ ( _h[n] * np.array(_u[n]) ) for n in e_numb])
    stiff_vec_2 = np.array([ _matriz_de_rigidez @ ( _h[n] * np.array(_u[n])**2 + 0.5 * 9.8 * np.array(_h[n])**2 ) for n in e_numb])

    return stiff_vec_1, stiff_vec_2