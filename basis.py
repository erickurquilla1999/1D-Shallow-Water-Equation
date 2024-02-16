import numpy as np
import basis
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


def generate_reference_space(N_elements,p_basis_order,out_x_points_per_element,n_gauss_poins, nodes_coord_ref_space):

    # element number
    element_number=np.arange(N_elements)

    # saving basis function evaluated at nodes in reference space
    basis_values_at_ref_coords=[]

    # loop over each element
    for i in element_number:
        basis_values_at_ele_i=[]
        # loop over the coordinates in reference space of the nodes in this element
        for e in nodes_coord_ref_space[i]:
            basis_in_e=[]
            # loop over the lagrange basis functions in this element and save the lagrange basis functions evaluated in e
            for k in np.arange(p_basis_order+1):
                basis_in_e.append(basis.lagrange_basis(nodes_coord_ref_space[i], k, e))
            basis_values_at_ele_i.append(basis_in_e)                
        basis_values_at_ref_coords.append(basis_values_at_ele_i)                

    # generating lagrange basis values in reference space [-1,1] to store the output data
    ref_coords_to_save_data=[]
    ref_coords_to_save_data = [np.linspace(-1, 1, out_x_points_per_element + 1) for _ in range(N_elements)]

    basis_values_at_the_point_to_save_data=[]

    # loop over each element
    for i in element_number:
        basis_values_at_ele_i=[]
        # loop over the coordinates in reference space of the nodes of the data i will storage for this element
        for e in ref_coords_to_save_data[i]:
            basis_in_e=[]
            # loop over the basis function and save the lagrange basis functions evaluated in e
            for k in np.arange(p_basis_order+1):
                basis_in_e.append(basis.lagrange_basis(nodes_coord_ref_space[i], k, e))
            basis_values_at_ele_i.append(basis_in_e)                
        basis_values_at_the_point_to_save_data.append(basis_values_at_ele_i)

    # generate Gauss cuadrature and weights
    gauss_coords, gauss_weights = np.polynomial.legendre.leggauss(n_gauss_poins)
    
    # saving gauss coordinates and weigths all of them are the same for each element
    gauss_coords_in_elements=[]
    gauss_weights_in_elements=[]
    gauss_coords_in_elements.extend([gauss_coords] * N_elements)
    gauss_weights_in_elements.extend([gauss_weights] * N_elements)

    # evaluating the basis function in the gauss quadrature points
    basis_values_at_gauss_coords=[]
    # loop over each element
    for i in element_number:
        basis_values_at_ele_i=[]
        # loop over the coordinates in reference space of the nodes in this element
        for e in gauss_coords_in_elements[i]:
            basis_in_e=[]
            # loop over the lagrange basis functions in this element and save the lagrange basis functions evaluated in e in the quarature points
            for k in np.arange(p_basis_order+1):
                basis_in_e.append(basis.lagrange_basis(nodes_coord_ref_space[i], k, e))
            basis_values_at_ele_i.append(basis_in_e)                
        basis_values_at_gauss_coords.append(basis_values_at_ele_i)
    # basis_values_at_gauss_coords contains [ [gauss coords1 phi1, gauss coord1 phi2 , ... , gauss coord1 phin], [gauss coords2 phi1, gauss coord2 phi2 , ... , gauss coord2 phin] ... , ]

    # saving this information in generatedfiles/reference_space.h5
    utilities.save_data_to_hdf5([element_number,nodes_coord_ref_space,basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords_in_elements,basis_values_at_gauss_coords,gauss_weights_in_elements],
                                ['element_number','nodes_coord_ref_space','basis_values_at_ref_coords','ref_coords_to_save_data','basis_values_at_the_point_to_save_data','gauss_coords_in_elements','basis_values_at_gauss_coords','gauss_weights_in_elements'],
                                'generatedfiles/reference_space.h5')

    return element_number,nodes_coord_ref_space,basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords_in_elements,basis_values_at_gauss_coords,gauss_weights_in_elements