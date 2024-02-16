import numpy as np
import initial_conditions
import utilities
import inputs
import basis
import os


def compute_M_matrix_inverse(element_number,basis_values_at_gauss_coords,gauss_weights_in_elements,left_node_coords, right_node_coords):
    #Lopp over all element
    M = []
    M_inverse = []

    for n in element_number:
        phi = np.array(basis_values_at_gauss_coords[n])
        weights = gauss_weights_in_elements[n]
        delta_x = right_node_coords[n] - left_node_coords[n]
        
        # Compute M for the current element
        M_ele_n = 0.5 * delta_x * np.dot(phi.T * weights, phi)
        # Append M to the list
        M.append(M_ele_n)

        # Compute the inverse of M for the current element
        M_inv_ele_n = np.linalg.inv(M_ele_n)

        # Append the inverse of M to the list
        M_inverse.append(M_inv_ele_n)

    return M_inverse

def compute_time_derivates(f1, f2, element_number):
    #Lopp over all element
    for n in element_number:
        a=0




