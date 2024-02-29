import numpy as np
import utilities

def lagrange_basis(nodes, i, x):
    """
    Compute the Lagrange basis function corresponding to node k.

    Parameters:
        nodes (numpy.ndarray): Array of Lagrange nodes.
        i (int): Index of the Lagrange node for which to compute the basis function.
        x (float): Point at which to evaluate the basis function.

    Returns:
        float: Value of the Lagrange basis function at the point x.
    """
    n = len(nodes)
    basis = 1.0
    for j in range(n):
        if j != i:
            basis *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return basis

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

def generate_reference_space(elements, nodes_phys_space, n_gauss_quad_points,l_elem_coordinates, r_elem_coordinates):

    # print(f'Generating reference space information ... \nNumber of Gauss quadrature points: {n_gauss_quad_points}')

    # saving basis function evaluated at nodes in physical space
    # basis_func_values_at_nodes_in_phys_space = [ [phi_1(x_node_1), phi_2(x_node_1) , ... , phi_p(x_node_1)] , 
    #                                              [phi_1(x_node_2), phi_2(x_node_2) , ... , phi_p(x_node_2)], ... , ]
    basis_func_values_at_nodes_in_phys_space = [
        [
            [lagrange_basis(nodes, base_index, x) for base_index in range(len(nodes))]
            for x in nodes
        ]
        for nodes in nodes_phys_space
    ]

    # generate Gauss cuadrature and weights in reference space
    gauss_coords_ref_space, gauss_quad_weights = np.polynomial.legendre.leggauss(n_gauss_quad_points)

    # saving Gauss cuadrature in physical space
    gauss_coords_phys_space = [ 0.5 * ( r_elem_coordinates[n] - l_elem_coordinates[n] ) * gauss_coords_ref_space + 0.5 * ( r_elem_coordinates[n] + l_elem_coordinates[n]) for n in elements]

    # saving gauss coordinates and weigths all of them are the same for each element
    gauss_coords_ref_space = [gauss_coords_ref_space for _ in elements]
    gauss_quad_weights = [gauss_quad_weights for _ in elements]

    # evaluating the basis function in the gauss quadrature points
    # basis_func_values_at_gauss_quad_in_phys_space = [ [phi_1(gauss_coords_1), phi_2(gauss_coords_1) , ... , phi_p(gauss_coords_1)] , 
    #                                                   [phi_1(gauss_coords_2), phi_2(gauss_coords_2) , ... , phi_p(gauss_coords_2)] , ... , ]
    basis_func_values_at_gauss_quad_in_phys_space = [
        [
            [lagrange_basis(nodes, base_index, x) for base_index in range(len(nodes))]
            for x in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_phys_space, gauss_coords_phys_space)
    ]    

    # evaluating the derivative in x of basis function evaluated in the gauss quadrature points
    # x_derivative_of_basis_func_at_gauss_quad_in_phys_space = [ [phi'_1(gauss_coords_1), phi'_2(gauss_coords_1) , ... , phi'_p(gauss_coords_1)], 
    #                                                               [phi'_1(gauss_coords_2), phi'_2(gauss_coords_2) , ... , phi'_p(gauss_coords_2)], ... , ]
    x_derivative_of_basis_func_at_gauss_quad_in_phys_space = [
        [
            [lagrange_basis_derivative(nodes, base_index, x) for base_index in range(len(nodes))]
            for x in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_phys_space, gauss_coords_phys_space)
    ]

    # saving this information in generatedfiles/reference_space.h5
    utilities.save_data_to_hdf5([elements,nodes_phys_space,basis_func_values_at_nodes_in_phys_space,gauss_coords_ref_space,gauss_coords_phys_space,gauss_quad_weights,basis_func_values_at_gauss_quad_in_phys_space,x_derivative_of_basis_func_at_gauss_quad_in_phys_space],
                                ['elements','nodes_phys_space','basis_func_values_at_nodes_in_phys_space','gauss_coords_ref_space','gauss_coords_phys_space','gauss_quad_weights','basis_func_values_at_gauss_quad_in_phys_space','x_derivative_of_basis_func_at_gauss_quad_in_phys_space'],
                                'generatedfiles/reference_space.h5')

    return gauss_quad_weights, np.array(basis_func_values_at_gauss_quad_in_phys_space), np.array(x_derivative_of_basis_func_at_gauss_quad_in_phys_space), np.array(basis_func_values_at_nodes_in_phys_space)