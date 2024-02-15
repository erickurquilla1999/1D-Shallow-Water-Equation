import utilities
import numpy as np

def generate_1d_mesh(initial_coord, final_coord, num_elements):
    # Generate node coordinates
    nodes = np.linspace(initial_coord, final_coord, num_elements + 1)
    
    # Generate element numbers
    elements = np.arange(1, num_elements + 1)
    
    # Compute node coordinates on the left and right sides of each element
    left_node_coords = nodes[:-1]
    right_node_coords = nodes[1:]
    
    # Compute element lengths
    element_lengths = np.diff(nodes)

    utilities.save_data_to_hdf5([elements, left_node_coords, right_node_coords, element_lengths],['element_number','left_node_coords','right_node_coords','element_lengths'],'generatedfiles/grid.h5')