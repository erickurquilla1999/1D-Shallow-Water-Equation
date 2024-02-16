import utilities
import numpy as np

def generate_1d_mesh(initial_coord, final_coord, num_elements, nodes_per_elements):
    # Generate elements coordinates
    elements_division = np.linspace(initial_coord, final_coord, num_elements + 1)

    # Generate element numbers
    elements = np.arange(num_elements)
    
    # Compute coordinates on the left and right sides of each element
    left_node_coords = elements_division[:-1]
    right_node_coords = elements_division[1:]
    
    # Compute element lengths
    element_lengths = np.diff(elements_division)

    # Compute nodes coordinates inside each element
    nodes_coord = []
    nodes_coord_ref_space = []

    for i in range(num_elements):
        nodes_coord.append(np.linspace(elements_division[i], elements_division[i + 1], nodes_per_elements + 1))
        nodes_coord_ref_space.append(np.linspace(-1,1,nodes_per_elements + 1))

    utilities.save_data_to_hdf5([elements, left_node_coords, right_node_coords, element_lengths, nodes_coord, nodes_coord_ref_space],['element_number','left_node_coords','right_node_coords','element_lengths','nodes_coords', 'nodes_coord_ref_space'],'generatedfiles/grid.h5')

    return elements, nodes_coord, left_node_coords, right_node_coords, nodes_coord_ref_space