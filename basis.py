import numpy as np

def lagrange_basis_derivative(nodes, i, x):
    """
    Compute the derivative of the Lagrange basis function for a given node and value of x.

    Parameters:
        nodes (array-like): Array of nodes in the element.
        i (int): Index of the node for which to compute the derivative.
        x (float): Value of x for which to compute the derivative.

    Returns:
        float: The derivative of the Lagrange basis function at the given node and value of x.
    """
    basis_derivative = 0
    for j in range(len(nodes)):
        if j != i:
            pc = 1
            for k in range(len(nodes)):
                if k != j and k != i:
                    pc *= (x-nodes[k])/(nodes[i]-nodes[k])
            basis_derivative += pc/(nodes[i]-nodes[j])
    return basis_derivative

def generate_reference_space(nodes_phys_space, n_gauss_quad_points, polinomios_de_lagrange):

    # print(f'Generating reference space information ... \nNumber of Gauss quadrature points: {n_gauss_quad_points}')

    number_of_elements = len(nodes_phys_space)

    # generate Gauss cuadrature and weights in reference space
    gauss_coords_ref_space, gauss_quad_weights = np.polynomial.legendre.leggauss(n_gauss_quad_points)

    # saving Gauss cuadrature in physical space
    gauss_coords_phys_space = [ 0.5 * ( nodes_phys_space[n][-1] - nodes_phys_space[n][0] ) * gauss_coords_ref_space + 0.5 * ( nodes_phys_space[n][-1] + nodes_phys_space[n][0]) for n in np.arange(number_of_elements)]

    # saving gauss coordinates and weigths all of them are the same for each element
    gauss_quad_weights = [gauss_quad_weights for _ in np.arange(number_of_elements)]

    # evaluating the basis function in the gauss quadrature points
    # basis_func_values_at_gauss_quad_in_phys_space = [ [phi_1(gauss_coords_1), phi_2(gauss_coords_1) , ... , phi_p(gauss_coords_1)] , 
    #                                                   [phi_1(gauss_coords_2), phi_2(gauss_coords_2) , ... , phi_p(gauss_coords_2)] , ... , ]
    basis_func_values_at_gauss_quad_in_phys_space = np.array([
        [
            [polinomios_de_lagrange(nodes, base_index, x) for base_index in range(len(nodes))]
            for x in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_phys_space, gauss_coords_phys_space)
    ])

    # evaluating the derivative in x of basis function evaluated in the gauss quadrature points
    # x_derivative_of_basis_func_at_gauss_quad_in_phys_space = [ [phi'_1(gauss_coords_1), phi'_2(gauss_coords_1) , ... , phi'_p(gauss_coords_1)], 
    #                                                               [phi'_1(gauss_coords_2), phi'_2(gauss_coords_2) , ... , phi'_p(gauss_coords_2)], ... , ]
    x_derivative_of_basis_func_at_gauss_quad_in_phys_space = np.array([
        [
            [lagrange_basis_derivative(nodes, base_index, x) for base_index in range(len(nodes))]
            for x in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_phys_space, gauss_coords_phys_space)
    ])

    return gauss_quad_weights, basis_func_values_at_gauss_quad_in_phys_space, x_derivative_of_basis_func_at_gauss_quad_in_phys_space