import utilities
import numpy as np
import os

def generate_1d_mesh(initial_coord, final_coord, num_elements, basis_order):

    print(f'Generating mesh \nPhysical domain: [{initial_coord},{final_coord}] meters\nNumber of elements: {num_elements}\nNodes per element: {basis_order+1}\nLagrange basis order: {basis_order}')

    # Generate elements coordinates
    elements_division = np.linspace(initial_coord, final_coord, num_elements + 1)

    # Compute element lengths
    element_length = np.diff(elements_division)

    # Compute nodes physical space inside each element
    if basis_order != 0 :
        nodes_coord_phys_space = [np.linspace(elements_division[i], elements_division[i + 1], basis_order + 1) for i in np.arange(num_elements)]
    else:
        nodes_coord_phys_space = [ [ elements_division[i] + 0.5 * ( elements_division[i+1] - elements_division[i]) ] for i in np.arange(num_elements)]

    return nodes_coord_phys_space, element_length