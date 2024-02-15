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


def generate_reference_space(p_basis_order,out_x_points_per_element,n_gauss_poins):

    # generating lagrange basis values in reference space [-1,1]
    ref_coords=np.linspace(-1,1,p_basis_order+1)
    basis_number=np.arange(p_basis_order+1)
    basis_values_at_ref_coords=[]

    for k in range(p_basis_order+1):
        basis_k=[]
        for e in ref_coords:
            basis_k.append(basis.lagrange_basis(ref_coords, k, e))
        basis_values_at_ref_coords.append(basis_k)

    # generating lagrange basis values in reference space [-1,1] for store the output data
    ref_coords_to_save_data=np.linspace(-1,1,out_x_points_per_element+1)
    basis_values_at_the_point_to_save_data=[]

    for k in range(p_basis_order+1):
        basis_k=[]
        for e in ref_coords_to_save_data:
            basis_k.append(basis.lagrange_basis(ref_coords, k, e))
        basis_values_at_the_point_to_save_data.append(basis_k)

    # reference space for Gauss cuadrature
    gauss_coords, gauss_weights = np.polynomial.legendre.leggauss(n_gauss_poins)
    basis_values_at_gauss_coords=[]

    for k in range(p_basis_order+1):
        basis_k=[]
        for e in gauss_coords:
            basis_k.append(basis.lagrange_basis(ref_coords, k, e))
        basis_values_at_gauss_coords.append(basis_k)

    # saving this information in generatedfiles/reference_space.h5
    utilities.save_data_to_hdf5([basis_number,ref_coords,basis_values_at_ref_coords,ref_coords_to_save_data,basis_values_at_the_point_to_save_data,gauss_coords,gauss_weights,basis_values_at_gauss_coords],['basis_number','ref_coords','basis_values_at_ref_coords','ref_coords_to_save_data','basis_values_at_the_point_to_save_data','gauss_coords','gauss_weights','basis_values_at_gauss_coords'],'generatedfiles/reference_space.h5')
