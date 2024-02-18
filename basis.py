import numpy as np
import utilities

def lagrange_basis(nodes, k, x):
    """
    Compute the Lagrange basis function corresponding to node k.

    Parameters:
        nodes (numpy.ndarray): Array of Lagrange nodes.
        k (int): Index of the Lagrange node for which to compute the basis function.
        x (float): Point at which to evaluate the basis function.

    Returns:
        float: Value of the Lagrange basis function at the point x.
    """
    n = len(nodes)
    basis = 1.0
    for j in range(n):
        if j != k:
            basis *= (x - nodes[j]) / (nodes[k] - nodes[j])
    return basis

def lagrange_interpolation(nodes, values, x):
    """
    Perform Lagrange interpolation of a function.

    Parameters:
        nodes (numpy.ndarray): Array of Lagrange nodes.
        values (numpy.ndarray): Array of function values at the Lagrange nodes.
        x (float): Point at which to interpolate the function.

    Returns:
        float: Interpolated function value at the point x.
    """
    n = len(nodes)
    result = 0.0
    for k in range(n):
        result += values[k] * lagrange_basis(nodes, k, x)
    return result

def lagrange_basis_derivative(nodes, node_index, x):
    """
    Compute the derivative of the Lagrange basis function for a given node and value of x.

    Parameters:
        nodes (array-like): Array of nodes in the element.
        node_index (int): Index of the node for which to compute the derivative.
        x (float): Value of x for which to compute the derivative.

    Returns:
        float: The derivative of the Lagrange basis function at the given node and value of x.
    """
    basis_derivative = 0
    for i, node in enumerate(nodes):
        if i != node_index:
            numerator = 1
            denominator = 1
            for j, other_node in enumerate(nodes):
                if j != node_index and j != i:
                    numerator *= (x - other_node)
                    denominator *= (node - other_node)
            basis_derivative += numerator / denominator

    return basis_derivative

def generate_reference_space(elements, nodes_ref_space, n_gauss_quad_points):

    print(f'\nGenerating reference space information ... \nNumber of Gauss quadrature points: {n_gauss_quad_points}\n')

    # saving basis function evaluated at nodes in reference space
    # basis_func_values_at_nodes_in_ref_space = [ [phi_1(x_node_1), phi_2(x_node_1) , ... , phi_p(x_node_1)] , 
    #                                             [phi_1(x_node_2), phi_2(x_node_2) , ... , phi_p(x_node_2)], ... , ]
    basis_func_values_at_nodes_in_ref_space = [
        [
            [lagrange_basis(nodes, bas, e) for bas in range(len(nodes))]
            for e in nodes
        ]
        for nodes in nodes_ref_space
    ]    

    # generate Gauss cuadrature and weights in reference space
    gauss_coords_ref_space, gauss_weights_ref_space = np.polynomial.legendre.leggauss(n_gauss_quad_points)
    
    # saving gauss coordinates and weigths all of them are the same for each element
    gauss_coords_ref_space = [gauss_coords_ref_space for _ in elements]
    gauss_weights_ref_space = [gauss_weights_ref_space for _ in elements]

    # evaluating the basis function in the gauss quadrature points
    # basis_func_values_at_gauss_quad_in_ref_space = [ [phi_1(gauss_coords_1), phi_2(gauss_coords_1) , ... , phi_p(gauss_coords_1)] , 
    #                                                  [phi_1(gauss_coords_2), phi_2(gauss_coords_2) , ... , phi_p(gauss_coords_2)] , ... , ]
    basis_func_values_at_gauss_quad_in_ref_space = [
        [
            [lagrange_basis(nodes, bas, e) for bas in range(len(nodes))]
            for e in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_ref_space, gauss_coords_ref_space)
    ]    

    # evaluating the derivative in x of basis function evaluated in the gauss quadrature points
    # time_derivative_of_basis_func_at_gauss_quad_in_ref_space = [ [phi'_1(gauss_coords_1), phi'_2(gauss_coords_1) , ... , phi'_p(gauss_coords_1)], 
    #                                                              [phi'_1(gauss_coords_2), phi'_2(gauss_coords_2) , ... , phi'_p(gauss_coords_2)], ... , ]
    time_derivative_of_basis_func_at_gauss_quad_in_ref_space = [
        [
            [lagrange_basis_derivative(nodes, bas, e) for bas in range(len(nodes))]
            for e in gauss_coords
        ]
        for nodes, gauss_coords in zip(nodes_ref_space, gauss_coords_ref_space)
    ]

    # saving this information in generatedfiles/reference_space.h5
    utilities.save_data_to_hdf5([elements,nodes_ref_space,basis_func_values_at_nodes_in_ref_space,gauss_coords_ref_space,gauss_weights_ref_space,basis_func_values_at_gauss_quad_in_ref_space,time_derivative_of_basis_func_at_gauss_quad_in_ref_space],
                                ['elements','nodes_ref_space','basis_func_values_at_nodes_in_ref_space','gauss_coords_ref_space','gauss_weights_ref_space','basis_func_values_at_gauss_quad_in_ref_space','time_derivative_of_basis_func_at_gauss_quad_in_ref_space'],
                                'generatedfiles/reference_space.h5')

    # return basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords_in_elements,basis_values_at_gauss_coords,gauss_weights_in_elements,basis_derivative_values_at_gauss_coords