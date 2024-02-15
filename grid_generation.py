import utilities
import numpy as np

def generate_1d_mesh(initial_coord, final_coord, num_elements,nodes_per_elements):
    # Generate node coordinates
    elements_division = np.linspace(initial_coord, final_coord, num_elements + 1)

    # Generate element numbers
    elements = np.arange(num_elements)
    
    # Compute node coordinates on the left and right sides of each element
    left_node_coords = elements_division[:-1]
    right_node_coords = elements_division[1:]
    
    # Compute element lengths
    element_lengths = np.diff(elements_division)

    # Compute subnode coordinates inside each element
    nodes_coord = []
    for i in range(num_elements):
        nodes = np.linspace(elements_division[i], elements_division[i + 1], nodes_per_elements)
        nodes_coord.append(nodes)

    utilities.save_data_to_hdf5([elements, left_node_coords, right_node_coords, element_lengths, nodes_coord],['element_number','left_node_coords','right_node_coords','element_lengths','nodes_coords'],'generatedfiles/grid.h5')