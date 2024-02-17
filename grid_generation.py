import utilities
import numpy as np
import os

def generate_1d_mesh(initial_coord, final_coord, num_elements, basis_order):
    # Generate elements coordinates
    elements_division = np.linspace(initial_coord, final_coord, num_elements + 1)

    # Generate element numbers
    elements_numb = np.arange(num_elements)
    
    # Compute coordinates on the left and right sides of each element
    left_node_coords = elements_division[:-1]
    right_node_coords = elements_division[1:]
    
    # Compute element lengths
    element_lengths = np.diff(elements_division)

    # Compute nodes physical space inside each element
    nodes_coord_phys_space = [np.linspace(elements_division[i], elements_division[i + 1], basis_order + 1) for i in elements_numb]

    # Compute nodes refrecne space inside each element
    nodes_coord_ref_space = [np.linspace(-1, 1, basis_order + 1) for _ in elements_numb]

    # Check if the directory exists
    directory = 'generatedfiles'
    if os.path.exists(directory):
        # If the directory exists, remove all files inside it
        file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
        for f in file_list:
            os.remove(f)
    else:
        # If the directory does not exist, create it
        os.makedirs(directory)

    # save mesh information in 'generatedfiles/grid.h5'
    utilities.save_data_to_hdf5([elements_numb, left_node_coords, right_node_coords, element_lengths, nodes_coord_phys_space, nodes_coord_ref_space],
                                ['element_number','left_node_coords','right_node_coords','element_lengths','nodes_coord_phys_space', 'nodes_coord_ref_space'],
                                'generatedfiles/grid.h5')

    return elements_numb, left_node_coords, right_node_coords, nodes_coord_phys_space, nodes_coord_ref_space